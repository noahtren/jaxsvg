"""Draw two shapes separately and then overlapping to demonstrate composition"""

import jax.numpy as np
import matplotlib.pyplot as plt
from jax import Array

from jaxsvg import draw


def get_shape_params():
    """Return bezier parameters for two petal shapes."""
    petal1 = np.array(
        [
            [[0.55, 0.55], [0.15, 0.45], [0.1, 0.2]],
            [[0.25, 0.1], [0.45, 0.15], [0.55, 0.55]],
        ]
    )
    petal2 = np.array(
        [
            [[0.45, 0.55], [0.55, 0.15], [0.75, 0.1]],
            [[0.9, 0.2], [0.85, 0.45], [0.45, 0.55]],
        ]
    )
    return petal1, petal2


def get_x_extents(img: Array, threshold: float = 0.5) -> tuple[int, int] | None:
    """Get (x_min, x_max) of pixels above threshold."""
    mask = img > threshold
    cols_with_pixels = np.any(mask, axis=0)

    if not cols_with_pixels.any():
        return None

    indices = np.arange(img.shape[1])
    x_min = int(np.where(cols_with_pixels, indices, img.shape[1]).min())
    x_max = int(np.where(cols_with_pixels, indices, -1).max())
    return (x_min, x_max)


def count_filled_pixels(img: Array, threshold: float = 0.5) -> int:
    """Count pixels above threshold."""
    return int((img > threshold).sum())


def show_visual():
    """Display the interactive matplotlib visualization."""
    petal1_params, petal2_params = get_shape_params()

    samples = [
        draw.draw_shapes(petal1_params[np.newaxis], filled=False),
        draw.draw_shapes(petal2_params[np.newaxis], filled=False),
        draw.draw_shapes(np.stack([petal1_params, petal2_params]), filled=False),
        draw.draw_shapes(np.stack([petal1_params, petal2_params]), filled=True),
    ]
    labels = [
        "Shape 1 (outline)",
        "Shape 2 (outline)",
        "Both (outlines)",
        "Both (filled)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, img, label in zip(axes.flatten(), samples, labels):
        ax.imshow(img, cmap="gray")
        ax.set_xlabel(label)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("examples/assets/combined_shapes.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    show_visual()
