"""Visual gallery showing blob parameter effects."""

import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random

from jaxsvg import draw
from jaxsvg.blobs import sample_blob, sample_blob_distribution


def show_gallery():
    """Display a grid showing blob parameter effects."""
    key = random.PRNGKey(42)

    fig, axes = plt.subplots(3, 5, figsize=(12, 7))

    # Row 1: Varying blobiness (0 to 1.0) at fixed scale/center
    blobiness_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    for i, blobiness in enumerate(blobiness_values):
        key, subkey = random.split(key)
        params = sample_blob(subkey, blobiness=blobiness, scale=0.7)
        img = draw.draw_shapes(params[np.newaxis])
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"blobiness={blobiness}")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
    axes[0, 0].set_ylabel("Blobiness", fontsize=11)

    # Row 2: Varying scale (now normalized: 1.0 ≈ fills canvas)
    scale_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    for i, scale in enumerate(scale_values):
        key, subkey = random.split(key)
        params = sample_blob(subkey, scale=scale, blobiness=0.3)
        img = draw.draw_shapes(params[np.newaxis])
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title(f"scale={scale}")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    axes[1, 0].set_ylabel("Scale", fontsize=11)

    # Row 3: Multiple random samples per cell (outlines)
    samples_per_cell = [3, 4, 5, 6, 8]
    for i, n_samples in enumerate(samples_per_cell):
        params_list = []
        for j in range(n_samples):
            key, subkey = random.split(key)
            params = sample_blob_distribution(
                subkey,
                center_std=0.12,
                scale_mean=0.35,
                scale_std=0.1,
                blobiness_mean=0.4,
                blobiness_std=0.15,
            )
            params_list.append(params)
        img = draw.draw_shapes(np.stack(params_list))
        axes[2, i].imshow(img, cmap="gray")
        axes[2, i].set_title(f"n={n_samples}")
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])
    axes[2, 0].set_ylabel("Distribution", fontsize=11)

    plt.suptitle("Blob Parameter Gallery", fontsize=14)
    plt.tight_layout()
    plt.savefig("examples/assets/blob_gallery.png", dpi=150)
    plt.show()


def show_curve_counts():
    """Compare blobs with 2, 3, and 4 curves."""
    key = random.PRNGKey(123)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    for col, num_curves in enumerate([2, 3, 4]):
        key, k1, k2 = random.split(key, 3)

        # Filled
        params = sample_blob(k1, num_curves=num_curves, blobiness=0.4, scale=0.7)
        img = draw.draw_shapes(params[np.newaxis])
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"{num_curves} curves")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # Multiple outlines
        params_list = []
        for _ in range(5):
            key, subkey = random.split(key)
            params = sample_blob(subkey, num_curves=num_curves, blobiness=0.4, scale=0.5)
            params_list.append(params)
        img = draw.draw_shapes(np.stack(params_list))
        axes[1, col].imshow(img, cmap="gray")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel("Filled", fontsize=11)
    axes[1, 0].set_ylabel("Outlines (5x)", fontsize=11)

    plt.suptitle("Curve Count Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("examples/assets/blob_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    show_gallery()
    show_curve_counts()