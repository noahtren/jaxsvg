"""Stochastic blob shape initializers with controllable properties."""

import jax.numpy as np
from jax import Array
from jax.random import PRNGKey, split, uniform, normal


def sample_blob(
    key: PRNGKey,
    num_curves: int = 3,
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 0.25,
    blobiness: float = 0.3,
) -> Array:
    """Sample a convex blob with explicit parameters.

    Arguments:
        key: JAX random key
        num_curves: number of cubic bezier curves (2, 3, or 4)
        center: (x, y) center position in [0, 1] space
        scale: base radius of the blob
        blobiness: per-anchor radius variance (0 = circle, higher = more irregular)

    Returns:
        Bezier parameters, shape [num_curves, 3, 2]
    """
    cx, cy = center

    angles = np.linspace(0, 2 * np.pi, num_curves, endpoint=False)
    max_radius_factor = 0.5 / 1.4  # ~0.36
    normalized_scale = scale * max_radius_factor

    radius_noise = uniform(key, shape=(num_curves,), minval=-blobiness, maxval=blobiness)
    radii = normalized_scale * (1 + radius_noise)
    anchors = np.stack(
        [cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=-1
    )

    arc_angle = 2 * np.pi / num_curves
    tension = (4 / 3) * np.tan(arc_angle / 4)
    tangent_angles = angles + np.pi / 2
    tangents = np.stack([np.cos(tangent_angles), np.sin(tangent_angles)], axis=-1)

    c1 = anchors + tension * radii[:, np.newaxis] * tangents

    next_radii = np.roll(radii, -1)
    next_anchors = np.roll(anchors, -1, axis=0)
    next_tangents = np.roll(tangents, -1, axis=0)
    c2 = next_anchors - tension * next_radii[:, np.newaxis] * next_tangents

    return np.stack([anchors, c1, c2], axis=1)


def sample_blob_distribution(
    key: PRNGKey,
    num_curves: int = 3,
    center_mean: tuple[float, float] = (0.5, 0.5),
    center_std: float = 0.2,
    scale_mean: float = 0.25,
    scale_std: float = 0.05,
    blobiness_mean: float = 1.0,
    blobiness_std: float = 0.5,
) -> Array:
    """Sample a blob with parameters drawn from distributions.

    Arguments:
        key: JAX random key
        num_curves: number of cubic bezier curves (2, 3, or 4)
        center_mean: mean (x, y) center position
        center_std: standard deviation for center position
        scale_mean: mean blob radius
        scale_std: standard deviation for scale
        blobiness_mean: mean blobiness
        blobiness_std: standard deviation for blobiness

    Returns:
        Bezier parameters, shape [num_curves, 3, 2]
    """
    k1, k2, k3, k4 = split(key, 4)

    center_offset = normal(k1, shape=(2,)) * center_std
    center = (center_mean[0] + center_offset[0], center_mean[1] + center_offset[1])

    scale = scale_mean + normal(k2, shape=()) * scale_std
    scale = np.maximum(scale, 0.01)

    blobiness = blobiness_mean + normal(k3, shape=()) * blobiness_std
    blobiness = np.clip(blobiness, 0.0, 1.0)

    return sample_blob(k4, num_curves=num_curves, center=center, scale=scale, blobiness=blobiness)


def max_safe_scale(center: tuple[float, float], blobiness: float, margin: float = 0.02) -> float:
    """Compute maximum scale that keeps blob within [0, 1] bounds.
    
    Arguments:
        center: (x, y) center position
        blobiness: blobiness parameter (affects max radius)
        margin: extra margin from edge
        
    Returns:
        Maximum safe scale value (in normalized units where 1.0 ≈ fills canvas)
    """
    cx, cy = center
    dist_to_edge = min(cx, cy, 1 - cx, 1 - cy)
    safe_dist = (dist_to_edge - margin) / (1 + blobiness)
    max_radius_factor = 0.5 / 1.4
    safe_scale = safe_dist / max_radius_factor
    return max(safe_scale, 0.01)
