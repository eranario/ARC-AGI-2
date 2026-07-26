"""Standalone latent-space viewer for ARC-AGI-2 grids."""

from arc_viewer.precompute import (
    DEFAULT_DINO_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_TRAJ_DINO,
    DEFAULT_TRAJ_STRUCTURAL,
    ensure_clusters_on_trajectory_json,
    precompute_dino_tsne,
    precompute_tsne,
)
from arc_viewer.sources import discover_sources, discover_trajectories

__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_DINO_OUTPUT",
    "DEFAULT_TRAJ_STRUCTURAL",
    "DEFAULT_TRAJ_DINO",
    "precompute_tsne",
    "precompute_dino_tsne",
    "ensure_clusters_on_trajectory_json",
    "discover_sources",
    "discover_trajectories",
]
