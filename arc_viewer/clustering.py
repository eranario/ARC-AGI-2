"""Unsupervised clustering on PCA-reduced embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# Fixed k grid shared by every parametric method.
K_VALUES: tuple[int, ...] = (8, 16, 32)

METHOD_FAMILIES: tuple[tuple[str, str], ...] = (
    ("kmeans", "KMeans"),
    ("gmm", "Gaussian Mixture"),
    ("agglomerative", "Agglomerative Ward"),
    ("birch", "Birch"),
)


@dataclass(frozen=True)
class ClusterResult:
    id: str
    family: str
    label: str
    labels: np.ndarray
    n_clusters: int
    k: int
    params: dict


def _count_clusters(labels: np.ndarray) -> int:
    return int(len({int(x) for x in labels if int(x) >= 0}))


def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float | None:
    mask = labels >= 0
    if mask.sum() < 3:
        return None
    labs = labels[mask]
    if len(set(labs.tolist())) < 2:
        return None
    n = int(mask.sum())
    sample = min(n, 4000)
    try:
        return float(
            silhouette_score(
                X[mask],
                labs,
                sample_size=sample if sample < n else None,
                random_state=0,
            )
        )
    except ValueError:
        return None


def _fit_labels(X: np.ndarray, family: str, k: int, seed: int) -> np.ndarray:
    if family == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    if family == "gmm":
        return GaussianMixture(
            n_components=k, covariance_type="full", random_state=seed
        ).fit_predict(X)
    if family == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if family == "birch":
        return Birch(n_clusters=k).fit_predict(X)
    raise ValueError(f"Unknown clustering family: {family}")


def method_id(family: str, k: int) -> str:
    return f"{family}_{k}"


def cluster_pca(
    X_pca: np.ndarray,
    *,
    seed: int = 42,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[str, ClusterResult]:
    """Run each method at each fixed k; return id -> result (e.g. kmeans_16)."""
    X = np.asarray(X_pca, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return {}

    usable_ks = [k for k in k_values if 2 <= k < n]
    if not usable_ks:
        usable_ks = [max(2, min(8, n - 1))] if n > 2 else []
    if not usable_ks:
        return {}

    print(
        f"Clustering PCA features ({n} × {X.shape[1]}) "
        f"at k={list(usable_ks)}..."
    )
    results: dict[str, ClusterResult] = {}

    for family, family_label in METHOD_FAMILIES:
        if family == "agglomerative" and n > 12_000:
            print("  agglomerative: skipped (n too large)")
            continue
        for k in usable_ks:
            mid = method_id(family, k)
            labels = _fit_labels(X, family, k, seed).astype(np.int32)
            sil = _silhouette_safe(X, labels)
            params: dict = {"k": int(k), "family": family}
            if sil is not None:
                params["silhouette"] = round(sil, 4)
            if family == "gmm":
                params["covariance"] = "full"
            if family == "agglomerative":
                params["linkage"] = "ward"
            if family == "birch":
                params["threshold"] = 0.5

            results[mid] = ClusterResult(
                id=mid,
                family=family,
                label=f"{family_label} (k={k})",
                labels=labels,
                n_clusters=_count_clusters(labels),
                k=int(k),
                params=params,
            )
            sil_txt = f"{sil:.3f}" if sil is not None else "n/a"
            print(f"  {mid}: clusters={results[mid].n_clusters}, silhouette={sil_txt}")

    return results


def clustering_meta(
    results: dict[str, ClusterResult],
    *,
    pca_dims: int,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict:
    families_present = []
    seen: set[str] = set()
    for family, family_label in METHOD_FAMILIES:
        if any(r.family == family for r in results.values()) and family not in seen:
            families_present.append({"id": family, "label": family_label})
            seen.add(family)

    return {
        "space": "pca",
        "pca_dims": int(pca_dims),
        "k_values": [int(k) for k in k_values],
        "families": families_present,
        "methods": [
            {
                "id": r.id,
                "family": r.family,
                "k": r.k,
                "label": r.label,
                "n_clusters": r.n_clusters,
                "params": r.params,
            }
            for r in results.values()
        ],
    }


def labels_by_point(results: dict[str, ClusterResult], index: int) -> dict[str, int]:
    return {cid: int(r.labels[index]) for cid, r in results.items()}
