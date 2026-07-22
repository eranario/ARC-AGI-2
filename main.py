"""CLI for exploring ARC-AGI-2 with FiftyOne."""

from __future__ import annotations

import argparse
from pathlib import Path

import fiftyone as fo

from arc_fiftyone.analyze import run_analysis
from arc_fiftyone.curation import CurationConfig, run_curation
from arc_fiftyone.dataset import DEFAULT_DATASET_NAME, load_arc_dataset
from arc_viewer.precompute import DEFAULT_OUTPUT as VIEWER_DEFAULT_OUTPUT
from arc_viewer.precompute import precompute_tsne
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
        help="Path for precomputed t-SNE JSON",
    )
    viewer_parser.add_argument(
        "--precompute",
        action="store_true",
        help="Recompute t-SNE even if output already exists",
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
        needs_precompute = (
            args.precompute
            or args.precompute_only
            or not args.output.is_file()
        )
        if needs_precompute:
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

        if args.precompute_only:
            return

        serve_viewer(
            data_path=args.output,
            port=args.viewer_port,
            open_browser=not args.no_open,
        )
        return


if __name__ == "__main__":
    main()
