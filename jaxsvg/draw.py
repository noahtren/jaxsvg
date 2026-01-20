import functools

import jax
import jax.numpy as np
from jax import Array

from jaxsvg import geometry


def safe_norm(x, axis=-1, eps=1e-8):
    """Norm with safe gradient at zero."""
    return np.sqrt(np.sum(x**2, axis=axis) + eps)


def compute_polyline_sdf(points: Array, width: int, height: int) -> Array:
    """Compute unsigned distance field to polyline segments."""
    y_coords, x_coords = np.meshgrid(
        (np.arange(height) + 0.5) / height,
        (np.arange(width) + 0.5) / width,
        indexing="ij",
    )
    grid = np.stack([x_coords, y_coords], axis=-1)  # [H, W, 2]

    p0 = points  # [N, 2]
    p1 = np.roll(points, -1, axis=0)  # [N, 2]

    grid_exp = grid[:, :, np.newaxis, :]
    p0_exp = p0[np.newaxis, np.newaxis, :, :]
    p1_exp = p1[np.newaxis, np.newaxis, :, :]

    seg = p1_exp - p0_exp  # [1, 1, N, 2]
    w = grid_exp - p0_exp  # [H, W, N, 2]

    seg_len_sq = np.sum(seg * seg, axis=-1, keepdims=True)  # [1, 1, N, 1]
    t = np.sum(w * seg, axis=-1, keepdims=True) / np.maximum(seg_len_sq, 1e-10)
    t = np.clip(t, 0, 1)  # [H, W, N, 1]

    closest = p0_exp + t * seg  # [H, W, N, 2]

    dist = safe_norm(grid_exp - closest, axis=-1)  # [H, W, N]

    return np.min(dist, axis=-1)  # [H, W]


def draw_path(
    bezier_parameters: Array,
    width: int = 256,
    height: int = 256,
    sampling_res: int = 32,
    boundary_width: float = 0.025,  # in normalized [0, 1] coordinates
    filled: bool = True,
) -> Array:
    """Draw a filled path mask with anti-aliasing according to Bezier curves."""
    start = bezier_parameters[:, 0]
    c1 = bezier_parameters[:, 1]
    c2 = bezier_parameters[:, 2]
    end = np.roll(start, -1, axis=-2)
    points = geometry.cubic(start, end, c1, c2, res=sampling_res)
    points = np.reshape(points, [-1, 2])

    dist = compute_polyline_sdf(points, width, height)

    sigma = boundary_width / 2.5
    outline = np.exp(-0.5 * (dist / sigma) ** 2)

    if not filled:
        return outline

    x_loc = geometry.get_raytrace_points(points, dim=width)
    idxs_x = np.sort(
        np.where(
            x_loc,
            np.tile(np.arange(x_loc.shape[0])[:, np.newaxis], [1, x_loc.shape[1]]),
            np.ones(x_loc.shape, dtype=np.int32) * -1,
        ),
        axis=0,
    )[::-1][:4]
    sorted_x = np.sort(
        x_loc[idxs_x, np.tile(np.arange(width)[np.newaxis], [4, 1])], axis=0
    )[::-1]
    fill_mask = geometry.fill_shape(sorted_x, height)

    return np.stack([outline, fill_mask]).max(axis=0)


def draw_shapes(
    bezier_params: Array,
    colors: Array = None,
    width: int = 256,
    height: int = 256,
    sampling_res: int = 32,
    background_color: Array = np.array([1.0, 1.0, 1.0]),
    boundary_width: float = 0.025,
    filled: bool = True,
) -> Array:
    """Render multiple shapes and composite them with implicit z-ordering."""
    masks = jax.vmap(
        functools.partial(
            draw_path,
            width=width,
            height=height,
            sampling_res=sampling_res,
            boundary_width=boundary_width,
            filled=filled,
        )
    )(bezier_params)

    if colors is None:
        return masks.max(axis=0)

    canvas = np.zeros([height, width, 3], dtype=np.float32)
    remaining_alpha = np.ones([height, width], dtype=np.float32)
    num_shapes = masks.shape[0]
    for i in range(num_shapes):
        alpha = colors[i, 3]
        effective_alpha = remaining_alpha * masks[i] * alpha
        remaining_alpha -= effective_alpha
        canvas += effective_alpha[..., np.newaxis] * colors[i, :3]

    canvas += (
        np.ones([height, width, 1])
        * background_color
        * remaining_alpha[..., np.newaxis]
    )
    return canvas
