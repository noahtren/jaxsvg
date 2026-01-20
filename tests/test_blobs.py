"""Tests for blob initializers."""

import jax.numpy as np
from jax import random

from jaxsvg.blobs import sample_blob, sample_blob_distribution
from jaxsvg import draw


def is_convex(params) -> bool:
    """Check if a bezier shape is convex by verifying anchor points are ordered angularly."""
    anchors = params[:, 0, :]  # shape [num_curves, 2]
    center = anchors.mean(axis=0)

    deltas = anchors - center
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])

    sorted_angles = np.sort(angles)
    diffs = np.diff(sorted_angles)

    return bool(np.all(diffs > 0) and np.all(diffs < np.pi))


def get_centroid(params) -> tuple[float, float]:
    """Compute centroid of anchor points."""
    anchors = params[:, 0, :]
    return float(anchors[:, 0].mean()), float(anchors[:, 1].mean())


def get_avg_radius(params) -> float:
    """Compute average distance from centroid to anchors."""
    anchors = params[:, 0, :]
    center = anchors.mean(axis=0)
    distances = np.sqrt(((anchors - center) ** 2).sum(axis=1))
    return float(distances.mean())


def get_radius_variance(params) -> float:
    """Compute variance of anchor distances from center."""
    anchors = params[:, 0, :]
    center = anchors.mean(axis=0)
    distances = np.sqrt(((anchors - center) ** 2).sum(axis=1))
    return float(distances.var())


def test_convex_blobiness_zero():
    """With blobiness=0, shape should be convex (it's a circle)."""
    key = random.PRNGKey(0)
    params = sample_blob(key, blobiness=0.0)
    assert is_convex(params)


def test_convex_blobiness_moderate():
    """With moderate blobiness, shape should still be convex."""
    for seed in range(10):
        key = random.PRNGKey(seed)
        params = sample_blob(key, blobiness=0.3)
        assert is_convex(params), f"Failed for seed {seed}"


def test_center_affects_position():
    """Changing center should move the shape."""
    key = random.PRNGKey(42)
    params1 = sample_blob(key, center=(0.3, 0.3), blobiness=0.0)
    params2 = sample_blob(key, center=(0.7, 0.7), blobiness=0.0)

    c1 = get_centroid(params1)
    c2 = get_centroid(params2)

    assert abs(c1[0] - 0.3) < 0.01
    assert abs(c1[1] - 0.3) < 0.01
    assert abs(c2[0] - 0.7) < 0.01
    assert abs(c2[1] - 0.7) < 0.01


def test_default_center():
    """Default center should be (0.5, 0.5)."""
    key = random.PRNGKey(0)
    params = sample_blob(key, blobiness=0.0)
    cx, cy = get_centroid(params)
    assert abs(cx - 0.5) < 0.01
    assert abs(cy - 0.5) < 0.01


def test_scale_affects_size():
    """Larger scale should produce larger radius."""
    key = random.PRNGKey(42)
    params_small = sample_blob(key, scale=0.3, blobiness=0.0)
    params_large = sample_blob(key, scale=0.9, blobiness=0.0)

    r_small = get_avg_radius(params_small)
    r_large = get_avg_radius(params_large)

    assert r_large > r_small * 2  # Should be 3x larger


def test_scale_approximately_matches():
    """Average radius should scale proportionally."""
    key = random.PRNGKey(0)
    params1 = sample_blob(key, scale=0.5, blobiness=0.0)
    params2 = sample_blob(key, scale=1.0, blobiness=0.0)
    r1 = get_avg_radius(params1)
    r2 = get_avg_radius(params2)
    # r2 should be ~2x r1
    assert 1.8 < r2 / r1 < 2.2


def test_blobiness_zero_is_circle():
    """With blobiness=0, all anchors should be equidistant (circle)."""
    key = random.PRNGKey(0)
    params = sample_blob(key, blobiness=0.0)
    variance = get_radius_variance(params)
    assert variance < 1e-6


def test_blobiness_increases_variance():
    """Higher blobiness should produce more radius variance."""
    key = random.PRNGKey(42)
    var_low = get_radius_variance(sample_blob(key, blobiness=0.1))
    var_high = get_radius_variance(sample_blob(key, blobiness=0.5))
    assert var_high > var_low


def test_blobiness_stochastic():
    """Different keys should produce different shapes."""
    params1 = sample_blob(random.PRNGKey(0), blobiness=0.3)
    params2 = sample_blob(random.PRNGKey(1), blobiness=0.3)
    assert not np.allclose(params1, params2)


def test_distribution_produces_variety():
    """Sampling from distribution should produce different shapes."""
    shapes = [
        sample_blob_distribution(random.PRNGKey(i)) for i in range(5)
    ]
    # Check that not all shapes are identical
    for i in range(1, 5):
        assert not np.allclose(shapes[0], shapes[i])


def test_distribution_center_variance():
    """Centers should vary when center_std > 0."""
    centers = []
    for i in range(20):
        params = sample_blob_distribution(
            random.PRNGKey(i), center_std=0.2, blobiness_mean=0.0, blobiness_std=0.0
        )
        centers.append(get_centroid(params))

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    assert max(xs) - min(xs) > 0.1  # Should have spread
    assert max(ys) - min(ys) > 0.1


def test_distribution_scale_variance():
    """Scales should vary when scale_std > 0."""
    radii = []
    for i in range(20):
        params = sample_blob_distribution(
            random.PRNGKey(i),
            center_std=0.0,
            scale_mean=0.6,
            scale_std=0.2,
            blobiness_mean=0.0,
            blobiness_std=0.0,
        )
        radii.append(get_avg_radius(params))

    assert max(radii) - min(radii) > 0.02  # Should have spread


# --- Rendering tests ---


def test_blob_renders():
    """Blob should produce a valid raster."""
    key = random.PRNGKey(0)
    params = sample_blob(key)
    img = draw.draw_shapes(params[np.newaxis], filled=True)
    assert img.shape == (256, 256)
    assert img.min() >= 0 and img.max() <= 1


def test_blob_has_pixels():
    """Rendered blob should have filled pixels."""
    key = random.PRNGKey(0)
    params = sample_blob(key, scale=0.8)
    img = draw.draw_shapes(params[np.newaxis], filled=True)
    filled_count = int((img > 0.5).sum())
    assert filled_count > 1000  # Should have substantial coverage
