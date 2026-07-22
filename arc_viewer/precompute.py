"""Precompute structural embeddings and t-SNE coordinates for the viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from arc_fiftyone.embeddings import compute_grid_embedding

DEFAULT_OUTPUT = Path("artifacts/viewer/tsne.json")


def _iter_task_files(data_dir: Path, split: str) -> list[Path]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(split_dir.glob("*.json"))


def _collect_grids(
    data_dir: Path,
    splits: tuple[str, ...],
) -> tuple[list[dict], np.ndarray]:
    records: list[dict] = []
    embeddings: list[np.ndarray] = []

    for split in splits:
        for task_path in _iter_task_files(data_dir, split):
            task_id = task_path.stem
            with task_path.open() as f:
                task = json.load(f)

            for pair_type in ("train", "test"):
                for pair_index, pair in enumerate(task[pair_type]):
                    for io_type in ("input", "output"):
                        grid = pair[io_type]
                        records.append(
                            {
                                "task_id": task_id,
                                "split": split,
                                "pair_type": pair_type,
                                "pair_index": pair_index,
                                "io_type": io_type,
                                "height": len(grid),
                                "width": len(grid[0]) if grid else 0,
                                "grid": grid,
                            }
                        )
                        embeddings.append(compute_grid_embedding(grid))

    return records, np.stack(embeddings, axis=0)


def precompute_tsne(
    data_dir: Path | str = Path("data"),
    *,
    splits: tuple[str, ...] = ("training",),
    output: Path | str = DEFAULT_OUTPUT,
    perplexity: float = 30.0,
    pca_dims: int = 50,
    seed: int = 42,
) -> Path:
    """Compute t-SNE of ARC grid embeddings and write a viewer JSON file."""
    data_dir = Path(data_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading grids from {', '.join(splits)}...")
    records, embeddings = _collect_grids(data_dir, splits)
    n = len(records)
    print(f"  {n} grids, embedding dim={embeddings.shape[1]}")

    pca_dims = min(pca_dims, n - 1, embeddings.shape[1])
    print(f"Running PCA -> {pca_dims} dims...")
    reduced = PCA(n_components=pca_dims, random_state=seed).fit_transform(
        embeddings
    )

    # t-SNE requires perplexity < n_samples
    effective_perplexity = min(perplexity, max(5.0, (n - 1) / 3.0))
    print(
        f"Running t-SNE (perplexity={effective_perplexity:.1f}, seed={seed})..."
    )
    coords = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(reduced)

    points = []
    for record, (x, y) in zip(records, coords, strict=True):
        points.append(
            {
                **record,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
            }
        )

    task_ids = sorted({p["task_id"] for p in points if p["pair_type"] == "train"})
    payload = {
        "meta": {
            "splits": list(splits),
            "n_points": n,
            "n_tasks": len(task_ids),
            "embedding": "structural_arc_embeddings",
            "method": "pca+tsne",
            "pca_dims": pca_dims,
            "perplexity": effective_perplexity,
            "seed": seed,
            "has_grids": True,
        },
        "task_ids": task_ids,
        "points": points,
    }

    output.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {output.resolve()} ({output.stat().st_size / 1024:.1f} KB)")
    return output.resolve()
