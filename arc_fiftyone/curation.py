"""FiftyOne Brain curation workflows for ARC-AGI-2."""

from __future__ import annotations

from dataclasses import dataclass

import fiftyone as fo
import fiftyone.brain as fob

from arc_fiftyone.embeddings import EMBEDDINGS_FIELD, try_load_visual_model


@dataclass
class CurationConfig:
    embeddings_field: str = EMBEDDINGS_FIELD
    use_visual_model: bool = False
    visual_model: str = "mobilenet-v2-imagenet-torch"
    similarity_model: str = "clip-vit-base32-torch"
    run_visualization: bool = True
    run_similarity: bool = True
    run_uniqueness: bool = True
    run_near_duplicates: bool = True
    run_exact_duplicates: bool = True
    near_duplicate_threshold: float = 0.15
    brain_keys: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.brain_keys is None:
            self.brain_keys = {
                "visualization": "arc_viz",
                "similarity": "arc_sim",
                "uniqueness": "arc_uniqueness",
                "near_duplicates": "arc_near_dups",
            }


def _resolve_embeddings(dataset: fo.Dataset, config: CurationConfig):
    if config.use_visual_model:
        model = try_load_visual_model(config.visual_model)
        if model is not None:
            print(f"Computing visual embeddings with {config.visual_model}...")
            return dataset.compute_embeddings(model)

    print(f"Using structural embeddings from `{config.embeddings_field}`...")
    return config.embeddings_field


def _resolve_similarity_source(config: CurationConfig):
    if config.use_visual_model:
        model = try_load_visual_model(config.similarity_model)
        if model is not None:
            return {"model": model}
    return {"embeddings": config.embeddings_field}


def run_curation(
    dataset: fo.Dataset,
    config: CurationConfig | None = None,
) -> dict:
    """Run pre-annotation curation brain methods on an ARC dataset."""
    config = config or CurationConfig()
    keys = config.brain_keys or {}
    results: dict = {}

    embeddings = _resolve_embeddings(dataset, config)
    similarity_source = _resolve_similarity_source(config)

    if config.run_visualization:
        print("Computing embedding visualization (UMAP)...")
        results["visualization"] = fob.compute_visualization(
            dataset,
            embeddings=embeddings,
            brain_key=keys["visualization"],
            create_index=True,
            seed=51,
        )

    if config.run_similarity:
        print("Building similarity index...")
        results["similarity"] = fob.compute_similarity(
            dataset,
            brain_key=keys["similarity"],
            **similarity_source,
        )

    if config.run_uniqueness:
        print("Computing uniqueness scores...")
        similarity_index = results.get("similarity")
        fob.compute_uniqueness(
            dataset,
            embeddings=embeddings,
            similarity_index=similarity_index,
            uniqueness_field="uniqueness",
        )

    if config.run_near_duplicates:
        print("Scanning for near-duplicate grids...")
        similarity_index = results.get("similarity")
        results["near_duplicates"] = fob.compute_near_duplicates(
            dataset,
            embeddings=embeddings,
            similarity_index=similarity_index,
            threshold=config.near_duplicate_threshold,
        )
        dup_ids = results["near_duplicates"].duplicate_ids
        print(f"  Found {len(dup_ids)} near-duplicate samples")

    if config.run_exact_duplicates:
        print("Scanning for exact duplicate images...")
        results["exact_duplicates"] = fob.compute_exact_duplicates(dataset)
        num_exact = sum(
            1 for ids in results["exact_duplicates"].values() if ids
        )
        print(f"  Found {num_exact} samples with exact duplicates")

    dataset.save()
    return results
