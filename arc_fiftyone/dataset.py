"""Load ARC-AGI-2 tasks into a FiftyOne dataset."""

from __future__ import annotations

import json
from pathlib import Path

import fiftyone as fo

from arc_fiftyone.embeddings import EMBEDDINGS_FIELD, compute_grid_embedding
from arc_fiftyone.render import (
    grid_signature,
    grid_stats,
    pair_aspect_stats,
    save_grid_image,
)

DEFAULT_DATASET_NAME = "arc-agi-2"
DEFAULT_IMAGE_DIR = Path("artifacts/fiftyone/images")


def _iter_task_files(data_dir: Path, split: str) -> list[Path]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(split_dir.glob("*.json"))


def load_arc_dataset(
    data_dir: Path | str = Path("data"),
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    splits: tuple[str, ...] = ("training", "evaluation"),
    image_dir: Path | str = DEFAULT_IMAGE_DIR,
    cell_size: int = 30,
    recreate: bool = False,
    persistent: bool = True,
) -> fo.Dataset:
    """Create or refresh a FiftyOne dataset from ARC JSON task files."""
    data_dir = Path(data_dir)
    image_dir = Path(image_dir)

    if fo.dataset_exists(dataset_name):
        if recreate:
            fo.delete_dataset(dataset_name)
        else:
            return fo.load_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name, persistent=persistent)
    dataset.add_sample_field("task_id", fo.StringField)
    dataset.add_sample_field("split", fo.StringField)
    dataset.add_sample_field("pair_type", fo.StringField)
    dataset.add_sample_field("io_type", fo.StringField)
    dataset.add_sample_field("pair_index", fo.IntField)
    dataset.add_sample_field("grid_height", fo.IntField)
    dataset.add_sample_field("grid_width", fo.IntField)
    dataset.add_sample_field("grid_area", fo.IntField)
    dataset.add_sample_field("aspect_ratio", fo.FloatField)
    dataset.add_sample_field("pair_input_aspect_ratio", fo.FloatField)
    dataset.add_sample_field("pair_output_aspect_ratio", fo.FloatField)
    dataset.add_sample_field("aspect_ratio_delta", fo.FloatField)
    dataset.add_sample_field("aspect_ratio_ratio", fo.FloatField)
    dataset.add_sample_field("num_colors", fo.IntField)
    dataset.add_sample_field("num_nonzero_colors", fo.IntField)
    dataset.add_sample_field("background_ratio", fo.FloatField)
    dataset.add_sample_field("grid_signature", fo.StringField)
    dataset.add_sample_field(EMBEDDINGS_FIELD, fo.VectorField)

    samples: list[fo.Sample] = []

    for split in splits:
        for task_path in _iter_task_files(data_dir, split):
            task_id = task_path.stem
            with task_path.open() as f:
                task = json.load(f)

            for pair_type in ("train", "test"):
                for pair_index, pair in enumerate(task[pair_type]):
                    pair_aspects = pair_aspect_stats(pair["input"], pair["output"])
                    for io_type in ("input", "output"):
                        grid = pair[io_type]
                        stats = grid_stats(grid)
                        rel_path = (
                            Path(split)
                            / task_id
                            / f"{pair_type}_{pair_index}_{io_type}.png"
                        )
                        filepath = image_dir / rel_path
                        save_grid_image(grid, filepath, cell_size=cell_size)

                        sample = fo.Sample(
                            filepath=str(filepath.resolve()),
                            task_id=task_id,
                            split=split,
                            pair_type=pair_type,
                            io_type=io_type,
                            pair_index=pair_index,
                            grid_signature=grid_signature(grid),
                            grid_height=stats["grid_height"],
                            grid_width=stats["grid_width"],
                            grid_area=stats["grid_area"],
                            aspect_ratio=stats["aspect_ratio"],
                            pair_input_aspect_ratio=pair_aspects[
                                "pair_input_aspect_ratio"
                            ],
                            pair_output_aspect_ratio=pair_aspects[
                                "pair_output_aspect_ratio"
                            ],
                            aspect_ratio_delta=pair_aspects["aspect_ratio_delta"],
                            aspect_ratio_ratio=pair_aspects["aspect_ratio_ratio"],
                            num_colors=stats["num_colors"],
                            num_nonzero_colors=stats["num_nonzero_colors"],
                            background_ratio=stats["background_ratio"],
                        )
                        sample[EMBEDDINGS_FIELD] = compute_grid_embedding(grid)
                        samples.append(sample)

    dataset.add_samples(samples)
    dataset.compute_metadata()
    dataset.save()
    return dataset
