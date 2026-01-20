import gc
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import optax
from jax import Array, random
from PIL import Image

from jaxsvg import draw
from jaxsvg.blobs import sample_blob_distribution


ASSETS = Path(__file__).parent / "assets" / "noto"
VIDEO_RESOLUTION = 128


@dataclass
class CurriculumStage:
    resolution: int
    num_steps: int
    lr: float
    boundary_width: float


DEFAULT_CURRICULUM = [
    CurriculumStage(resolution=64, num_steps=400, lr=1e-3, boundary_width=0.1),
    CurriculumStage(resolution=64, num_steps=400, lr=5e-3, boundary_width=0.1),
    *[
        CurriculumStage(resolution=64, num_steps=100, lr=5e-3, boundary_width=0.05),
        CurriculumStage(resolution=64, num_steps=100, lr=5e-3, boundary_width=0.025),
    ]
    * 8,
    *[
        CurriculumStage(resolution=64, num_steps=100, lr=5e-3, boundary_width=0.03),
        CurriculumStage(resolution=64, num_steps=100, lr=5e-3, boundary_width=0.015),
    ]
    * 8,
    CurriculumStage(resolution=96, num_steps=300, lr=3e-3, boundary_width=0.04),
    CurriculumStage(resolution=96, num_steps=200, lr=1e-3, boundary_width=0.02),
    CurriculumStage(resolution=96, num_steps=200, lr=1e-3, boundary_width=0.01),
]


def load_target(name: str, size: int, padding: int = 0) -> Array:
    """Load emoji at specified resolution."""
    inner_size = size - 2 * padding if padding > 0 else size
    path = ASSETS / f"{name}.png"
    img = Image.open(path).convert("RGBA").resize((inner_size, inner_size))

    arr = jnp.array(img, dtype=jnp.float32) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    composited = rgb * alpha + (1 - alpha)

    if padding > 0:
        composited = jnp.pad(
            composited,
            ((padding, padding), (padding, padding), (0, 0)),
            mode="constant",
            constant_values=1.0,
        )
    return composited


def init_shapes(
    key: random.PRNGKey, num_shapes: int, num_curves: int = 3
) -> tuple[Array, Array]:
    keys = random.split(key, num_shapes)

    bezier_params = jax.vmap(
        partial(
            sample_blob_distribution,
            num_curves=num_curves,
            center_mean=(0.5, 0.5),
            center_std=0.1,
            scale_mean=0.25,
            scale_std=0.05,
            blobiness_mean=0.5,
            blobiness_std=0.1,
        )
    )(keys)

    color_key = random.fold_in(key, 999)
    colors = random.uniform(color_key, (num_shapes, 4), minval=-1.0, maxval=1.0)
    colors = colors.at[:, 3].set(0.0)
    return bezier_params, colors


def promote_best_shape(bezier_params, colors, target, size, boundary_width):
    def compute_loss(bp, c):
        colors_normalized = jax.nn.sigmoid(c)
        rendered = draw.draw_shapes(
            bp,
            colors_normalized,
            width=size,
            height=size,
            boundary_width=boundary_width,
        )
        return ((rendered - target) ** 2).mean()

    current_loss = compute_loss(bezier_params, colors)
    num_shapes = bezier_params.shape[0]
    best_idx = -1
    best_loss = current_loss

    for i in range(num_shapes - 1):
        others_bp = jnp.concatenate([bezier_params[:i], bezier_params[i + 1 :]], axis=0)
        others_c = jnp.concatenate([colors[:i], colors[i + 1 :]], axis=0)
        new_bp = jnp.concatenate([others_bp, bezier_params[i : i + 1]], axis=0)
        new_c = jnp.concatenate([others_c, colors[i : i + 1]], axis=0)
        loss = compute_loss(new_bp, new_c)
        if loss < best_loss:
            best_loss = loss
            best_idx = i

    if best_idx >= 0:
        others_bp = jnp.concatenate(
            [bezier_params[:best_idx], bezier_params[best_idx + 1 :]], axis=0
        )
        others_c = jnp.concatenate([colors[:best_idx], colors[best_idx + 1 :]], axis=0)
        return (
            jnp.concatenate(
                [others_bp, bezier_params[best_idx : best_idx + 1]], axis=0
            ),
            jnp.concatenate([others_c, colors[best_idx : best_idx + 1]], axis=0),
            True,
        )
    return bezier_params, colors, False


@partial(jax.jit, static_argnames=["size", "sampling_res", "boundary_width"])
def render(bezier_params, colors, size, boundary_width, sampling_res=64):
    colors_normalized = jax.nn.sigmoid(colors)
    return draw.draw_shapes(
        bezier_params,
        colors_normalized,
        width=size,
        height=size,
        sampling_res=sampling_res,
        boundary_width=boundary_width,
    )


def save_frame(frame: Array, path: Path) -> None:
    frame_np = np.array(frame)
    frame_uint8 = (np.clip(frame_np, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(frame_uint8)
    img.save(path, "PNG")



@partial(jax.jit, static_argnames=["optimizer_def", "size", "boundary_width"])
def global_train_step(params, opt_state, target, optimizer_def, size, boundary_width):
    def loss_fn(p):
        bezier_params, colors = p
        rendered = draw.draw_shapes(
            bezier_params,
            jax.nn.sigmoid(colors),
            width=size,
            height=size,
            boundary_width=boundary_width,
        )
        reconstruction_loss = ((rendered - target) ** 2).mean()
        return reconstruction_loss, rendered

    (loss, rendered), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_def.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, rendered


def optimize_curriculum_with_frames(
    key: random.PRNGKey,
    target_name: str,
    initial_shapes: int,
    num_curves: int = 3,
    curriculum: list[CurriculumStage] = DEFAULT_CURRICULUM,
    verbose: bool = True,
    frame_every: int = 10,
    video_resolution: int = VIDEO_RESOLUTION,
    output_dir: str = "examples/assets",
) -> dict:
    key, init_key = random.split(key)
    bezier_params, colors = init_shapes(init_key, initial_shapes, num_curves)
    params = (bezier_params, colors)

    history = {"losses": [], "num_shapes": [], "stages": [], "renders": []}

    frames_dir = Path(output_dir) / target_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    frame_count = 0

    for stage_idx, stage in enumerate(curriculum):
        if verbose:
            print(
                f"\n=== Stage {stage_idx + 1}: {stage.resolution}x{stage.resolution} ==="
            )

        padding = stage.resolution // 8
        target = load_target(target_name, stage.resolution, padding=padding)

        optimizer_def = optax.adam(stage.lr)
        opt_state = optimizer_def.init(params)

        for step in range(stage.num_steps):
            params, opt_state, loss, rendered = global_train_step(
                params,
                opt_state,
                target,
                optimizer_def,
                stage.resolution,
                stage.boundary_width,
            )

            history["losses"].append(float(loss))
            history["num_shapes"].append(params[0].shape[0])
            history["stages"].append(stage_idx)

            if global_step % frame_every == 0:
                frame = render(
                    params[0], params[1], video_resolution, stage.boundary_width
                )
                frame.block_until_ready()
                save_frame(frame, frames_dir / f"frame_{frame_count:04d}.png")
                frame_count += 1

            if verbose and (step + 1) % 50 == 0:
                loss.block_until_ready()
                print(f"    Step {step + 1}: loss={float(loss):.4f}")

            global_step += 1

        rendered.block_until_ready()
        history["renders"].append(np.array(rendered))

        bezier_params, colors = params
        bezier_params = jnp.clip(bezier_params, 0.01, 0.99)

        if stage.resolution <= 128:
            promoted = True
            shapes_promoted = 0
            while promoted and shapes_promoted < 4:
                bezier_params, colors, promoted = promote_best_shape(
                    bezier_params,
                    colors,
                    target,
                    stage.resolution,
                    stage.boundary_width,
                )
                shapes_promoted += 1
            if verbose:
                print(f"    Promoted {shapes_promoted} shape(s)")

        params = (bezier_params, colors)

    final_render = render(
        params[0], params[1], curriculum[-1].resolution, curriculum[-1].boundary_width
    )
    return {
        "params": params,
        "history": history,
        "final_render": final_render,
        "target": load_target(target_name, curriculum[-1].resolution),
        "frames_dir": str(frames_dir),
        "num_frames": frame_count,
    }


def main():
    key = random.PRNGKey(42)
    emojis = [
        "cat",
        "penguin",
        "turtle",
        "duck",
        "bee",
        "bird",
        "butterfly",
        "chipmunk",
        "dog",
        "dolphin",
        "flamingo",
        "fox",
        "koala",
        "hedgehog",
        "octopus",
        "otter",
        "panda",
        "peacock",
        "rabbit",
        "raccoon",
        "seal",
        "snail",
        "tropical-fish",
        "spouting-whale",
    ]
    output_dir = "examples/assets/frames"

    for name in emojis:
        print(f"\n{'=' * 50}\nOptimizing: {name}\n{'=' * 50}")
        key, subkey = random.split(key)

        jax.clear_caches()
        gc.collect()

        results = optimize_curriculum_with_frames(
            subkey, name, initial_shapes=24, num_curves=4, output_dir=output_dir
        )

        print(f"Done. Frames saved to {results['frames_dir']}")


if __name__ == "__main__":
    main()
