"""Embedding utilities for ARC grids."""

from __future__ import annotations

import numpy as np

from arc_fiftyone.render import MAX_GRID_SIZE, grid_to_array

EMBEDDINGS_FIELD = "arc_embeddings"


def compute_grid_embedding(grid: list[list[int]]) -> np.ndarray:
    """Build a structural embedding from grid content and shape."""
    arr = grid_to_array(grid)
    h, w = arr.shape

    padded = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
    padded[:h, :w] = arr.astype(np.float32) / 9.0

    flat = arr.ravel()
    hist = np.bincount(flat, minlength=10).astype(np.float32)
    hist /= max(h * w, 1)

    unique_colors = np.unique(flat)
    shape = np.array(
        [
            h / MAX_GRID_SIZE,
            w / MAX_GRID_SIZE,
            (h * w) / (MAX_GRID_SIZE * MAX_GRID_SIZE),
            len(unique_colors) / 10.0,
            np.mean(flat == 0),
            np.std(flat.astype(np.float32) / 9.0),
        ],
        dtype=np.float32,
    )

    return np.concatenate([padded.ravel(), hist, shape])


def try_load_visual_model(model_name: str = "mobilenet-v2-imagenet-torch"):
    """Return a FiftyOne zoo model if torch and the model are available."""
    try:
        import fiftyone.zoo as foz

        return foz.load_zoo_model(model_name)
    except Exception:
        return None
