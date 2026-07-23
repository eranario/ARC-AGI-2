"""Registry of precomputed embedding spaces for the latent viewer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VIEWER_DIR = Path("artifacts/viewer")

# Stable ids used by the viewer UI / API. filenames are under viewer_dir.
KNOWN_SOURCES: tuple[tuple[str, str, str], ...] = (
    # id, filename, human label
    ("structural", "tsne.json", "structural"),
    ("dinov3", "tsne_dino.json", "DINOv3"),
)

KNOWN_TRAJECTORIES: tuple[tuple[str, str, str], ...] = (
    ("structural", "traj_structural.json", "structural"),
    ("dinov3", "traj_dinov3.json", "DINOv3"),
)


@dataclass(frozen=True)
class EmbeddingSource:
    id: str
    label: str
    path: Path
    available: bool
    kind: str = "grids"  # "grids" | "trajectories"
    meta: dict | None = None

    @property
    def url(self) -> str:
        if self.kind == "trajectories":
            return f"/data/traj/{self.id}.json"
        return f"/data/{self.id}.json"


def _read_meta(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    meta = payload.get("meta")
    return meta if isinstance(meta, dict) else None


def _label_for(source_id: str, fallback: str, meta: dict | None) -> str:
    if meta:
        emb = meta.get("embedding")
        if source_id == "dinov3" or emb == "dinov3":
            model = meta.get("dino_model")
            if isinstance(model, str) and model:
                return f"DINOv3 ({model.rsplit('/', 1)[-1]})"
            return "DINOv3"
        if isinstance(emb, str) and emb and emb != "structural_arc_embeddings":
            return emb
    return fallback


def _discover_named(
    viewer_dir: Path,
    known: tuple[tuple[str, str, str], ...],
    *,
    kind: str,
    glob_pattern: str,
    skip_names: set[str],
    overrides: dict[str, Path | None] | None = None,
) -> list[EmbeddingSource]:
    overrides = overrides or {}
    sources: list[EmbeddingSource] = []
    seen: set[str] = set()

    for source_id, filename, fallback_label in known:
        path = overrides.get(source_id) or (viewer_dir / filename)
        path = Path(path).resolve()
        available = path.is_file()
        meta = _read_meta(path) if available else None
        sources.append(
            EmbeddingSource(
                id=source_id,
                label=_label_for(source_id, fallback_label, meta),
                path=path,
                available=available,
                kind=kind,
                meta=meta,
            )
        )
        seen.add(source_id)

    if viewer_dir.is_dir():
        for path in sorted(viewer_dir.glob(glob_pattern)):
            if path.name in skip_names:
                continue
            # tsne_<id>.json / traj_<id>.json
            prefix = "tsne_" if kind == "grids" else "traj_"
            if not path.stem.startswith(prefix):
                continue
            source_id = path.stem.removeprefix(prefix)
            if not source_id or source_id in seen:
                continue
            meta = _read_meta(path)
            sources.append(
                EmbeddingSource(
                    id=source_id,
                    label=_label_for(source_id, source_id, meta),
                    path=path.resolve(),
                    available=True,
                    kind=kind,
                    meta=meta,
                )
            )
            seen.add(source_id)

    return sources


def discover_sources(
    viewer_dir: Path | str = DEFAULT_VIEWER_DIR,
    *,
    structural_path: Path | str | None = None,
    dino_path: Path | str | None = None,
) -> list[EmbeddingSource]:
    """Return known + auto-discovered grid embedding JSON sources."""
    return _discover_named(
        Path(viewer_dir),
        KNOWN_SOURCES,
        kind="grids",
        glob_pattern="tsne_*.json",
        skip_names={"tsne_dino.json"},
        overrides={
            "structural": Path(structural_path) if structural_path else None,
            "dinov3": Path(dino_path) if dino_path else None,
        },
    )


def discover_trajectories(
    viewer_dir: Path | str = DEFAULT_VIEWER_DIR,
) -> list[EmbeddingSource]:
    """Return known + auto-discovered I/O trajectory JSON sources."""
    return _discover_named(
        Path(viewer_dir),
        KNOWN_TRAJECTORIES,
        kind="trajectories",
        glob_pattern="traj_*.json",
        skip_names=set(),
    )


def sources_catalog(sources: list[EmbeddingSource]) -> dict:
    """JSON payload for GET /data/sources.json."""
    return {
        "sources": [
            {
                "id": s.id,
                "label": s.label,
                "url": s.url,
                "available": s.available,
                "kind": s.kind,
                "meta": s.meta or {},
            }
            for s in sources
        ]
    }


def path_map(sources: list[EmbeddingSource]) -> dict[str, Path]:
    """id -> filesystem path for available sources only."""
    return {s.id: s.path for s in sources if s.available}
