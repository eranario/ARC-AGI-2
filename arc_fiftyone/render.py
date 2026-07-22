"""Render ARC grids as PNG images for FiftyOne."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from arc_fiftyone.colors import ARC_COLORS

MAX_GRID_SIZE = 30


def grid_to_array(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.uint8)


def grid_signature(grid: list[list[int]]) -> str:
    return json.dumps(grid, separators=(",", ":"))


def grid_stats(grid: list[list[int]]) -> dict:
    arr = grid_to_array(grid)
    h, w = arr.shape
    flat = arr.ravel()
    unique, counts = np.unique(flat, return_counts=True)
    color_counts = {int(c): int(n) for c, n in zip(unique, counts)}
    nonzero = flat[flat != 0]
    return {
        "grid_height": int(h),
        "grid_width": int(w),
        "grid_area": int(h * w),
        "aspect_ratio": float(w / h),
        "num_colors": int(len(unique)),
        "num_nonzero_colors": int(len(np.unique(nonzero))) if nonzero.size else 0,
        "background_ratio": float(np.mean(flat == 0)),
        "color_counts": color_counts,
    }


def pair_aspect_stats(
    input_grid: list[list[int]], output_grid: list[list[int]]
) -> dict:
    """Aspect-ratio fields shared by both grids in an input/output pair."""
    input_stats = grid_stats(input_grid)
    output_stats = grid_stats(output_grid)
    input_ar = input_stats["aspect_ratio"]
    output_ar = output_stats["aspect_ratio"]
    return {
        "pair_input_aspect_ratio": input_ar,
        "pair_output_aspect_ratio": output_ar,
        "aspect_ratio_delta": output_ar - input_ar,
        "aspect_ratio_ratio": output_ar / input_ar if input_ar else 0.0,
    }


def render_grid(
    grid: list[list[int]],
    *,
    cell_size: int = 30,
    border: int = 1,
    border_color: tuple[int, int, int] = (64, 64, 64),
) -> Image.Image:
    arr = grid_to_array(grid)
    h, w = arr.shape
    img_h = h * cell_size + (h + 1) * border
    img_w = w * cell_size + (w + 1) * border
    image = Image.new("RGB", (img_w, img_h), border_color)
    draw = ImageDraw.Draw(image)

    for row in range(h):
        for col in range(w):
            color = ARC_COLORS[int(arr[row, col])]
            x0 = col * cell_size + (col + 1) * border
            y0 = row * cell_size + (row + 1) * border
            draw.rectangle(
                [x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                fill=color,
            )
    return image


def save_grid_image(
    grid: list[list[int]],
    path: Path,
    *,
    cell_size: int = 30,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    render_grid(grid, cell_size=cell_size).save(path)
    return path
