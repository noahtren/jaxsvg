"""Minimal example: grow a shape to fill a target using direct parameter optimization."""

import jax
import jax.numpy as np
import optax
from jax import Array
import matplotlib.pyplot as plt

from jaxsvg import draw


def init_bezier_params(num_curves: int = 3, scale: float = 0.1) -> Array:
    """Initialize small bezier shape centered at (0.5, 0.5)."""
    angles = np.linspace(0, 2 * np.pi, num_curves, endpoint=False)

    starts = 0.5 + scale * np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    angle_offsets = angles + np.pi / 2
    c1 = starts + scale * 0.5 * np.stack(
        [np.cos(angle_offsets), np.sin(angle_offsets)], axis=-1
    )
    c2 = np.roll(starts, -1, axis=0) - scale * 0.5 * np.stack(
        [np.cos(np.roll(angle_offsets, -1)), np.sin(np.roll(angle_offsets, -1))],
        axis=-1,
    )

    return np.stack([starts, c1, c2], axis=1)


def grow_shape(num_steps: int = 64, num_curves: int = 3, lr: float = 1e-2) -> dict:
    """Grow a shape from small to fill the canvas."""
    params = init_bezier_params(num_curves=num_curves)
    target = np.ones((256, 256), dtype=np.float32)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(p):
        raster = draw.draw_path(p)
        return ((raster - target) ** 2).mean(), raster

    losses = []
    for i in range(num_steps):
        (loss, raster), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(float(loss))
        if i == 0:
            initial = raster

    return {"initial": initial, "final": raster, "losses": losses}


if __name__ == "__main__":
    results = grow_shape()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(results["initial"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Initial (loss={results['losses'][0]:.3f})")
    axes[1].imshow(results["final"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Final (loss={results['losses'][-1]:.3f})")
    axes[2].plot(results["losses"])
    axes[2].set_title("Loss")
    plt.savefig("examples/assets/grown_shape.png", dpi=150)
    plt.show()
