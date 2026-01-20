"""Tests for grow_shape example."""

from examples.optim_grow_shape import grow_shape


def test_initial_loss_high():
    """Initial shape is small, so loss should be near 1.0."""
    results = grow_shape(num_steps=1)
    assert results["losses"][0] > 0.9


def test_final_loss_low():
    """After optimization, loss should be near zero."""
    results = grow_shape(num_steps=64)
    assert results["losses"][-1] < 0.05


def test_shape_fills_canvas():
    """Final shape should cover nearly the entire canvas."""
    results = grow_shape(num_steps=64)
    final_coverage = (results["final"] > 0.5).mean()
    assert final_coverage > 0.95


def test_loss_decreases_overall():
    """Loss should decrease significantly from start to end."""
    results = grow_shape(num_steps=64)
    assert results["losses"][-1] < results["losses"][0] / 10
