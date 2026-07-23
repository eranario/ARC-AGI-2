"""Precompute structural / DINOv3 embeddings, t-SNE, and I/O trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from arc_fiftyone.embeddings import compute_grid_embedding
from arc_fiftyone.render import render_grid
from arc_viewer.dino import DEFAULT_DINO_MODEL, embed_images, load_dinov3

DEFAULT_OUTPUT = Path("artifacts/viewer/tsne.json")
DEFAULT_DINO_OUTPUT = Path("artifacts/viewer/tsne_dino.json")
DEFAULT_STRUCTURAL_EMB = Path("artifacts/viewer/embeddings_structural.npz")
DEFAULT_DINO_EMB = Path("artifacts/viewer/embeddings_dinov3.npz")
DEFAULT_TRAJ_STRUCTURAL = Path("artifacts/viewer/traj_structural.json")
DEFAULT_TRAJ_DINO = Path("artifacts/viewer/traj_dinov3.json")


def _iter_task_files(data_dir: Path, split: str) -> list[Path]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(split_dir.glob("*.json"))


def collect_records(
    data_dir: Path,
    splits: tuple[str, ...],
) -> list[dict]:
    """Load ARC task grids into viewer record dicts (including raw grids)."""
    records: list[dict] = []
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
    return records


def structural_embeddings(records: list[dict]) -> np.ndarray:
    return np.stack(
        [compute_grid_embedding(r["grid"]) for r in records],
        axis=0,
    )


def dinov3_embeddings(
    records: list[dict],
    *,
    model_name: str = DEFAULT_DINO_MODEL,
    batch_size: int = 32,
    cell_size: int = 16,
    device: str | None = None,
) -> tuple[np.ndarray, str]:
    """Render grids as RGB images and embed with DINOv3."""
    processor, model, device = load_dinov3(model_name, device=device)
    print(f"Rendering {len(records)} grids as images (cell_size={cell_size})...")
    images = [
        render_grid(r["grid"], cell_size=cell_size).convert("RGB")
        for r in records
    ]
    print("Running DINOv3 forward passes...")
    feats = embed_images(
        images,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        processor=processor,
        model=model,
    )
    return feats, model_name


def save_raw_embeddings(
    path: Path | str,
    records: list[dict],
    embeddings: np.ndarray,
    *,
    meta: dict | None = None,
) -> Path:
    """Persist raw embedding matrix aligned with records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        task_id=np.array([r["task_id"] for r in records], dtype=object),
        split=np.array([r["split"] for r in records], dtype=object),
        pair_type=np.array([r["pair_type"] for r in records], dtype=object),
        pair_index=np.array([r["pair_index"] for r in records], dtype=np.int32),
        io_type=np.array([r["io_type"] for r in records], dtype=object),
        height=np.array([r["height"] for r in records], dtype=np.int32),
        width=np.array([r["width"] for r in records], dtype=np.int32),
        # grids stored as JSON strings to keep npz manageable
        grid_json=np.array(
            [json.dumps(r["grid"], separators=(",", ":")) for r in records],
            dtype=object,
        ),
        meta_json=np.array(json.dumps(meta or {}), dtype=object),
    )
    print(
        f"Wrote raw embeddings {path.resolve()} "
        f"({path.stat().st_size / 1024:.1f} KB, shape={embeddings.shape})"
    )
    return path.resolve()


def load_raw_embeddings(path: Path | str) -> tuple[list[dict], np.ndarray, dict]:
    """Load records + embedding matrix previously saved by save_raw_embeddings."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    meta = json.loads(str(data["meta_json"].item()))
    records = []
    for i in range(len(embeddings)):
        records.append(
            {
                "task_id": str(data["task_id"][i]),
                "split": str(data["split"][i]),
                "pair_type": str(data["pair_type"][i]),
                "pair_index": int(data["pair_index"][i]),
                "io_type": str(data["io_type"][i]),
                "height": int(data["height"][i]),
                "width": int(data["width"][i]),
                "grid": json.loads(str(data["grid_json"][i])),
            }
        )
    return records, embeddings, meta


def pair_key(record: dict) -> tuple:
    return (
        record["task_id"],
        record["split"],
        record["pair_type"],
        int(record["pair_index"]),
    )


def compute_io_deltas(
    records: list[dict],
    embeddings: np.ndarray,
    *,
    pair_types: tuple[str, ...] = ("train",),
) -> tuple[list[dict], np.ndarray]:
    """Δ = emb(output) − emb(input) in raw embedding space, one row per pair."""
    by_key: dict[tuple, dict[str, int]] = {}
    for i, record in enumerate(records):
        if record["pair_type"] not in pair_types:
            continue
        key = pair_key(record)
        by_key.setdefault(key, {})[record["io_type"]] = i

    pair_records: list[dict] = []
    deltas: list[np.ndarray] = []
    for key, ios in sorted(by_key.items()):
        if "input" not in ios or "output" not in ios:
            continue
        i_in = ios["input"]
        i_out = ios["output"]
        delta = embeddings[i_out] - embeddings[i_in]
        magnitude = float(np.linalg.norm(delta))
        inp = records[i_in]
        out = records[i_out]
        pair_records.append(
            {
                "task_id": inp["task_id"],
                "split": inp["split"],
                "pair_type": inp["pair_type"],
                "pair_index": inp["pair_index"],
                "input": {
                    "height": inp["height"],
                    "width": inp["width"],
                    "grid": inp["grid"],
                },
                "output": {
                    "height": out["height"],
                    "width": out["width"],
                    "grid": out["grid"],
                },
                "magnitude": round(magnitude, 6),
            }
        )
        deltas.append(delta.astype(np.float32))

    if not deltas:
        return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    return pair_records, np.stack(deltas, axis=0)


def run_pca_tsne(
    embeddings: np.ndarray,
    *,
    perplexity: float = 30.0,
    pca_dims: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    n, dim = embeddings.shape
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "method": "pca+tsne",
            "pca_dims": 0,
            "perplexity": 0.0,
            "seed": seed,
            "embedding_dim": int(dim),
        }

    pca_dims = min(pca_dims, n - 1, dim)
    print(f"Running PCA -> {pca_dims} dims...")
    reduced = PCA(n_components=pca_dims, random_state=seed).fit_transform(
        embeddings
    )

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

    meta = {
        "method": "pca+tsne",
        "pca_dims": pca_dims,
        "perplexity": effective_perplexity,
        "seed": seed,
        "embedding_dim": int(dim),
    }
    return coords, meta


def write_viewer_json(
    records: list[dict],
    coords: np.ndarray,
    *,
    output: Path,
    splits: tuple[str, ...],
    embedding_name: str,
    extra_meta: dict | None = None,
) -> Path:
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
            "n_points": len(points),
            "n_tasks": len(task_ids),
            "embedding": embedding_name,
            "has_grids": True,
            **(extra_meta or {}),
        },
        "task_ids": task_ids,
        "points": points,
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {output.resolve()} ({output.stat().st_size / 1024:.1f} KB)")
    return output.resolve()


def write_trajectory_json(
    pair_records: list[dict],
    coords: np.ndarray,
    *,
    output: Path,
    splits: tuple[str, ...],
    embedding_name: str,
    extra_meta: dict | None = None,
) -> Path:
    """Write I/O delta trajectories projected to 2D."""
    points = []
    for record, (x, y) in zip(pair_records, coords, strict=True):
        points.append(
            {
                **record,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
            }
        )

    task_ids = sorted({p["task_id"] for p in points})
    payload = {
        "meta": {
            "kind": "trajectories",
            "splits": list(splits),
            "n_points": len(points),
            "n_tasks": len(task_ids),
            "embedding": embedding_name,
            "delta": "output_minus_input_raw",
            "has_grids": True,
            **(extra_meta or {}),
        },
        "task_ids": task_ids,
        "points": points,
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"Wrote trajectories {output.resolve()} "
        f"({output.stat().st_size / 1024:.1f} KB, {len(points)} pairs)"
    )
    return output.resolve()


def precompute_trajectories(
    records: list[dict],
    embeddings: np.ndarray,
    *,
    output: Path | str,
    splits: tuple[str, ...],
    embedding_name: str,
    perplexity: float = 30.0,
    pca_dims: int = 50,
    seed: int = 42,
    extra_meta: dict | None = None,
) -> Path:
    """Build Δ=out−in in raw space, then PCA+t-SNE the deltas."""
    print("Computing raw input→output embedding deltas...")
    pair_records, deltas = compute_io_deltas(records, embeddings)
    print(f"  {len(pair_records)} train pairs, delta dim={deltas.shape[1]}")
    coords, meta = run_pca_tsne(
        deltas, perplexity=perplexity, pca_dims=pca_dims, seed=seed
    )
    meta = {
        **meta,
        **(extra_meta or {}),
        "raw_delta_dim": int(deltas.shape[1]),
    }
    return write_trajectory_json(
        pair_records,
        coords,
        output=Path(output),
        splits=splits,
        embedding_name=embedding_name,
        extra_meta=meta,
    )


def precompute_tsne(
    data_dir: Path | str = Path("data"),
    *,
    splits: tuple[str, ...] = ("training",),
    output: Path | str = DEFAULT_OUTPUT,
    embeddings_output: Path | str = DEFAULT_STRUCTURAL_EMB,
    trajectory_output: Path | str = DEFAULT_TRAJ_STRUCTURAL,
    perplexity: float = 30.0,
    pca_dims: int = 50,
    seed: int = 42,
) -> Path:
    """Compute t-SNE of structural ARC grid embeddings (+ raw + trajectories)."""
    data_dir = Path(data_dir)
    print(f"Loading grids from {', '.join(splits)}...")
    records = collect_records(data_dir, splits)
    print(f"  {len(records)} grids")
    embeddings = structural_embeddings(records)
    print(f"  structural embedding dim={embeddings.shape[1]}")
    save_raw_embeddings(
        embeddings_output,
        records,
        embeddings,
        meta={"embedding": "structural_arc_embeddings", "splits": list(splits)},
    )
    coords, meta = run_pca_tsne(
        embeddings, perplexity=perplexity, pca_dims=pca_dims, seed=seed
    )
    path = write_viewer_json(
        records,
        coords,
        output=Path(output),
        splits=splits,
        embedding_name="structural_arc_embeddings",
        extra_meta=meta,
    )
    precompute_trajectories(
        records,
        embeddings,
        output=trajectory_output,
        splits=splits,
        embedding_name="structural_arc_embeddings",
        perplexity=perplexity,
        pca_dims=pca_dims,
        seed=seed,
    )
    return path


def precompute_dino_tsne(
    data_dir: Path | str = Path("data"),
    *,
    splits: tuple[str, ...] = ("training",),
    output: Path | str = DEFAULT_DINO_OUTPUT,
    embeddings_output: Path | str = DEFAULT_DINO_EMB,
    trajectory_output: Path | str = DEFAULT_TRAJ_DINO,
    model_name: str = DEFAULT_DINO_MODEL,
    batch_size: int = 32,
    cell_size: int = 16,
    device: str | None = None,
    perplexity: float = 30.0,
    pca_dims: int = 50,
    seed: int = 42,
) -> Path:
    """Render grids, embed with DINOv3, then PCA + t-SNE (+ raw + trajectories)."""
    data_dir = Path(data_dir)
    print(f"Loading grids from {', '.join(splits)}...")
    records = collect_records(data_dir, splits)
    print(f"  {len(records)} grids")
    embeddings, used_model = dinov3_embeddings(
        records,
        model_name=model_name,
        batch_size=batch_size,
        cell_size=cell_size,
        device=device,
    )
    print(f"  DINOv3 embedding dim={embeddings.shape[1]}")
    emb_meta = {
        "embedding": "dinov3",
        "dino_model": used_model,
        "cell_size": cell_size,
        "batch_size": batch_size,
        "splits": list(splits),
    }
    save_raw_embeddings(
        embeddings_output, records, embeddings, meta=emb_meta
    )
    coords, meta = run_pca_tsne(
        embeddings, perplexity=perplexity, pca_dims=pca_dims, seed=seed
    )
    meta = {**meta, **emb_meta}
    path = write_viewer_json(
        records,
        coords,
        output=Path(output),
        splits=splits,
        embedding_name="dinov3",
        extra_meta=meta,
    )
    precompute_trajectories(
        records,
        embeddings,
        output=trajectory_output,
        splits=splits,
        embedding_name="dinov3",
        perplexity=perplexity,
        pca_dims=pca_dims,
        seed=seed,
        extra_meta=emb_meta,
    )
    return path


def precompute_trajectories_from_saved(
    embeddings_path: Path | str,
    *,
    output: Path | str,
    perplexity: float = 30.0,
    pca_dims: int = 50,
    seed: int = 42,
) -> Path:
    """Rebuild trajectory JSON from a saved raw-embeddings npz (no model reload)."""
    records, embeddings, meta = load_raw_embeddings(embeddings_path)
    splits = tuple(meta.get("splits") or sorted({r["split"] for r in records}))
    embedding_name = str(meta.get("embedding") or "unknown")
    return precompute_trajectories(
        records,
        embeddings,
        output=output,
        splits=splits,
        embedding_name=embedding_name,
        perplexity=perplexity,
        pca_dims=pca_dims,
        seed=seed,
        extra_meta={k: v for k, v in meta.items() if k != "splits"},
    )
