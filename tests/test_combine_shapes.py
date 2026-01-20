import jax.numpy as np
from jaxsvg import draw
from examples.combine_shapes import (
    get_shape_params,
    get_x_extents,
    count_filled_pixels,
)

shape1_params, shape2_params = get_shape_params()


def test_shape1_x_extents():
    """shape 1 should be on the left side."""
    img = draw.draw_shapes(shape1_params[np.newaxis])
    x_min, x_max = get_x_extents(img)
    assert x_min < 128 and x_max < 200


def test_shape2_x_extents():
    """shape 2 should be on the right side."""
    img = draw.draw_shapes(shape2_params[np.newaxis])
    x_min, x_max = get_x_extents(img)
    assert x_min > 50 and x_max > 128


def test_combined_wider_than_individuals():
    """Combined shape should span wider than either individual."""
    img1 = draw.draw_shapes(shape1_params[np.newaxis])
    img2 = draw.draw_shapes(shape2_params[np.newaxis])
    combined = draw.draw_shapes(np.stack([shape1_params, shape2_params]))

    w1 = get_x_extents(img1)[1] - get_x_extents(img1)[0]
    w2 = get_x_extents(img2)[1] - get_x_extents(img2)[0]
    w_combined = get_x_extents(combined)[1] - get_x_extents(combined)[0]

    assert w_combined >= max(w1, w2)


def test_filled_pixel_counts():
    """Each shape should have reasonable pixel count."""
    img1 = draw.draw_shapes(shape1_params[np.newaxis])
    img2 = draw.draw_shapes(shape2_params[np.newaxis])

    assert 5000 < count_filled_pixels(img1) < 40000
    assert 5000 < count_filled_pixels(img2) < 40000


def test_shapes_overlap():
    """Combined pixels should be less than sum (they overlap)."""
    img1 = draw.draw_shapes(shape1_params[np.newaxis])
    img2 = draw.draw_shapes(shape2_params[np.newaxis])
    combined = draw.draw_shapes(np.stack([shape1_params, shape2_params]))

    assert count_filled_pixels(combined) < count_filled_pixels(
        img1
    ) + count_filled_pixels(img2)


def test_outline_fewer_than_filled():
    """Outline should have fewer pixels than filled."""
    outline = draw.draw_shapes(np.stack([shape1_params, shape2_params]), filled=False)
    filled = draw.draw_shapes(np.stack([shape1_params, shape2_params]))

    assert count_filled_pixels(outline) < count_filled_pixels(filled)
