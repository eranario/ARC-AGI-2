"""CLI for exploring ARC-AGI-2 with FiftyOne."""

from __future__ import annotations

import argparse
from pathlib import Path

import fiftyone as fo

from arc_fiftyone.analyze import run_analysis
from arc_fiftyone.curation import CurationConfig, run_curation
from arc_fiftyone.dataset import DEFAULT_DATASET_NAME, load_arc_dataset
from arc_viewer.dino import DEFAULT_DINO_MODEL
from arc_viewer.precompute import DEFAULT_DINO_EMB
from arc_viewer.precompute import DEFAULT_DINO_OUTPUT as VIEWER_DEFAULT_DINO_OUTPUT
from arc_viewer.precompute import DEFAULT_OUTPUT as VIEWER_DEFAULT_OUTPUT
from arc_viewer.precompute import DEFAULT_STRUCTURAL_EMB
from arc_viewer.precompute import DEFAULT_TRAJ_DINO
from arc_viewer.precompute import DEFAULT_TRAJ_STRUCTURAL
from arc_viewer.precompute import (
    precompute_dino_tsne,
    precompute_trajectories_from_saved,
    precompute_tsne,
)
from arc_viewer.serve import serve_viewer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore ARC-AGI-2 with FiftyOne curation workflows.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing training/ and evaluation/ splits",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="FiftyOne dataset name",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=30,
        help="Pixels per ARC grid cell when rendering images",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port for the FiftyOne App",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    load_parser = subparsers.add_parser(
        "load",
        help="Load ARC tasks into a FiftyOne dataset",
    )
    load_parser.add_argument(
        "--split",
        choices=("training", "evaluation", "all"),
        default="all",
    )
    load_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the dataset if it already exists",
    )

    curate_parser = subparsers.add_parser(
        "curate",
        help="Run pre-annotation brain methods (embeddings, similarity, etc.)",
    )
    curate_parser.add_argument(
        "--visual-model",
        action="store_true",
        help="Use a FiftyOne zoo model for embeddings when torch is installed",
    )
    curate_parser.add_argument(
        "--skip-visualization",
        action="store_true",
    )
    curate_parser.add_argument(
        "--skip-similarity",
        action="store_true",
    )
    curate_parser.add_argument(
        "--skip-uniqueness",
        action="store_true",
    )
    curate_parser.add_argument(
        "--skip-near-duplicates",
        action="store_true",
    )
    curate_parser.add_argument(
        "--skip-exact-duplicates",
        action="store_true",
    )

    subparsers.add_parser("app", help="Launch the FiftyOne App")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Plot uniqueness vs area and input/output aspect ratios",
    )
    analyze_parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("artifacts/fiftyone/plots"),
        help="Directory to write plot PNG files",
    )
    analyze_parser.add_argument(
        "--show",
        action="store_true",
        help="Also open interactive plot windows",
    )
    analyze_parser.add_argument(
        "--app",
        action="store_true",
        help="Also launch the FiftyOne App after saving plots",
    )
    analyze_parser.add_argument(
        "--backend",
        choices=("plotly", "matplotlib"),
        default="matplotlib",
        help="Plotting backend for saved images",
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Load, curate, and launch the FiftyOne App",
    )
    all_parser.add_argument(
        "--split",
        choices=("training", "evaluation", "all"),
        default="all",
    )
    all_parser.add_argument("--recreate", action="store_true")
    all_parser.add_argument("--visual-model", action="store_true")
    all_parser.add_argument("--skip-visualization", action="store_true")
    all_parser.add_argument("--skip-similarity", action="store_true")
    all_parser.add_argument("--skip-uniqueness", action="store_true")
    all_parser.add_argument("--skip-near-duplicates", action="store_true")
    all_parser.add_argument("--skip-exact-duplicates", action="store_true")

    viewer_parser = subparsers.add_parser(
        "viewer",
        help="Precompute t-SNE and launch the latent-space viewer",
    )
    viewer_parser.add_argument(
        "--split",
        choices=("training", "evaluation", "all"),
        default="training",
        help="Which ARC split(s) to embed",
    )
    viewer_parser.add_argument(
        "--output",
        type=Path,
        default=VIEWER_DEFAULT_OUTPUT,
        help="Path for structural t-SNE JSON",
    )
    viewer_parser.add_argument(
        "--dino-output",
        type=Path,
        default=VIEWER_DEFAULT_DINO_OUTPUT,
        help="Path for DINOv3 t-SNE JSON",
    )
    viewer_parser.add_argument(
        "--embedding",
        choices=("structural", "dinov3", "all"),
        default="structural",
        help=(
            "Which embedding space(s) to precompute. "
            "DINOv3 is only computed for dinov3/all when missing or --precompute "
            "(gated HF weights; not downloaded on plain serve)."
        ),
    )
    viewer_parser.add_argument(
        "--dino-model",
        default=DEFAULT_DINO_MODEL,
        help="Hugging Face DINOv3 model id",
    )
    viewer_parser.add_argument(
        "--dino-batch-size",
        type=int,
        default=32,
        help="Batch size for DINOv3 inference",
    )
    viewer_parser.add_argument(
        "--dino-cell-size",
        type=int,
        default=16,
        help="Pixels per ARC cell when rendering images for DINOv3",
    )
    viewer_parser.add_argument(
        "--device",
        default=None,
        help="Torch device for DINOv3 (default: cuda if available else cpu)",
    )
    viewer_parser.add_argument(
        "--precompute",
        action="store_true",
        help="Recompute selected embeddings even if outputs already exist",
    )
    viewer_parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="Only write precomputed JSON; do not serve the viewer",
    )
    viewer_parser.add_argument(
        "--viewer-port",
        type=int,
        default=8765,
        help="Port for the latent-space viewer",
    )
    viewer_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser when serving",
    )
    viewer_parser.add_argument("--perplexity", type=float, default=30.0)
    viewer_parser.add_argument("--pca-dims", type=int, default=50)
    viewer_parser.add_argument("--seed", type=int, default=42)

    return parser


def _split_tuple(split: str) -> tuple[str, ...]:
    if split == "all":
        return ("training", "evaluation")
    return (split,)


def _load_dataset(args: argparse.Namespace) -> fo.Dataset:
    return load_arc_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        splits=_split_tuple(args.split),
        cell_size=args.cell_size,
        recreate=args.recreate,
    )


def _curate_dataset(args: argparse.Namespace, dataset: fo.Dataset) -> None:
    config = CurationConfig(
        use_visual_model=getattr(args, "visual_model", False),
        run_visualization=not getattr(args, "skip_visualization", False),
        run_similarity=not getattr(args, "skip_similarity", False),
        run_uniqueness=not getattr(args, "skip_uniqueness", False),
        run_near_duplicates=not getattr(args, "skip_near_duplicates", False),
        run_exact_duplicates=not getattr(args, "skip_exact_duplicates", False),
    )
    run_curation(dataset, config)


def _launch_app(dataset: fo.Dataset, port: int) -> None:
    session = fo.launch_app(dataset, port=port)
    print(f"FiftyOne App running at {session.url}")
    print("Press Ctrl+C to exit.")
    session.wait()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "load":
        dataset = _load_dataset(args)
        print(f"Loaded {len(dataset)} samples into dataset '{args.dataset_name}'")
        return

    if args.command == "curate":
        if not fo.dataset_exists(args.dataset_name):
            raise SystemExit(
                f"Dataset '{args.dataset_name}' not found. Run `load` first."
            )
        dataset = fo.load_dataset(args.dataset_name)
        _curate_dataset(args, dataset)
        print("Curation complete.")
        return

    if args.command == "app":
        if not fo.dataset_exists(args.dataset_name):
            raise SystemExit(
                f"Dataset '{args.dataset_name}' not found. Run `load` first."
            )
        dataset = fo.load_dataset(args.dataset_name)
        _launch_app(dataset, args.port)
        return

    if args.command == "analyze":
        if not fo.dataset_exists(args.dataset_name):
            raise SystemExit(
                f"Dataset '{args.dataset_name}' not found. Run `load` first."
            )
        dataset = fo.load_dataset(args.dataset_name)
        if "uniqueness" not in dataset.get_field_schema():
            raise SystemExit(
                "Dataset has no `uniqueness` field. Run `curate` first."
            )
        run_analysis(
            dataset,
            save_dir=args.save_dir,
            launch_app=args.app,
            show=args.show,
            port=args.port,
            backend=args.backend,
        )
        return

    if args.command == "all":
        dataset = _load_dataset(args)
        print(f"Loaded {len(dataset)} samples into dataset '{args.dataset_name}'")
        _curate_dataset(args, dataset)
        _launch_app(dataset, args.port)
        return

    if args.command == "viewer":
        want_structural = args.embedding in ("structural", "all")
        want_dino = args.embedding in ("dinov3", "all")
        force = args.precompute or args.precompute_only

        if want_structural:
            if force or not args.output.is_file():
                precompute_tsne(
                    data_dir=args.data_dir,
                    splits=_split_tuple(args.split),
                    output=args.output,
                    perplexity=args.perplexity,
                    pca_dims=args.pca_dims,
                    seed=args.seed,
                )
            else:
                print(f"Using existing {args.output.resolve()}")

        if want_dino:
            # Never auto-download gated DINOv3 weights on a plain `viewer` serve.
            # Compute only when --embedding dinov3|all and (missing or --precompute*).
            should_dino = args.embedding in ("dinov3", "all") and (
                force or not args.dino_output.is_file()
            )
            # For --embedding all without --precompute, skip creating missing dino data.
            if args.embedding == "all" and not force and not args.dino_output.is_file():
                should_dino = False

            if should_dino:
                precompute_dino_tsne(
                    data_dir=args.data_dir,
                    splits=_split_tuple(args.split),
                    output=args.dino_output,
                    model_name=args.dino_model,
                    batch_size=args.dino_batch_size,
                    cell_size=args.dino_cell_size,
                    device=args.device,
                    perplexity=args.perplexity,
                    pca_dims=args.pca_dims,
                    seed=args.seed,
                )
            elif args.dino_output.is_file():
                print(f"Using existing {args.dino_output.resolve()}")
            else:
                print(
                    f"DINOv3 data not found at {args.dino_output} "
                    "(run: uv run python main.py viewer --embedding dinov3 --precompute-only)"
                )

        # Rebuild missing trajectory JSONs from saved raw embeddings (no model reload).
        if DEFAULT_STRUCTURAL_EMB.is_file() and not DEFAULT_TRAJ_STRUCTURAL.is_file():
            precompute_trajectories_from_saved(
                DEFAULT_STRUCTURAL_EMB,
                output=DEFAULT_TRAJ_STRUCTURAL,
                perplexity=args.perplexity,
                pca_dims=args.pca_dims,
                seed=args.seed,
            )
        if DEFAULT_DINO_EMB.is_file() and not DEFAULT_TRAJ_DINO.is_file():
            precompute_trajectories_from_saved(
                DEFAULT_DINO_EMB,
                output=DEFAULT_TRAJ_DINO,
                perplexity=args.perplexity,
                pca_dims=args.pca_dims,
                seed=args.seed,
            )

        if args.precompute_only:
            return

        if not args.output.is_file():
            raise SystemExit(
                f"Structural viewer data not found at {args.output}. "
                "Run with --embedding structural|all --precompute first."
            )

        serve_viewer(
            data_path=args.output,
            dino_data_path=args.dino_output,
            port=args.viewer_port,
            open_browser=not args.no_open,
        )
        return


if __name__ == "__main__":
    main()
