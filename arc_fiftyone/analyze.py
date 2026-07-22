"""Custom scatterplot analysis for ARC-AGI-2 FiftyOne datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import fiftyone as fo
import fiftyone.core.plots as fop
from fiftyone import ViewField as F

DEFAULT_PLOT_DIR = Path("artifacts/fiftyone/plots")


def _require_field(dataset: fo.Dataset, field: str) -> None:
    if field not in dataset.get_field_schema():
        raise ValueError(
            f"Dataset is missing `{field}`. "
            f"Run `uv run python main.py curate` first."
        )


def _samples_with_field(dataset: fo.Dataset, field: str) -> fo.DatasetView:
    return dataset.match(F(field).exists())


def _scatter_kwargs(backend: str, *, equal_axes: bool = False) -> dict:
    if backend == "matplotlib":
        return {"ax_equal": equal_axes}
    return {"axis_equal": equal_axes}


def _apply_axis_labels(
    plot: fop.InteractivePlot,
    *,
    backend: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if backend == "matplotlib":
        plot.ax.set(xlabel=xlabel, ylabel=ylabel)
        return

    if hasattr(plot, "_figure"):
        plot._figure.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)


def plot_uniqueness_vs_grid_area(
    dataset: fo.Dataset | fo.DatasetView,
    *,
    backend: str = "matplotlib",
) -> fop.InteractivePlot:
    """Scatter plot of uniqueness vs grid area, linked to dataset samples."""
    root = dataset._root_dataset if hasattr(dataset, "_root_dataset") else dataset
    _require_field(root, "uniqueness")

    view = _samples_with_field(dataset, "uniqueness")
    if len(view) == 0:
        raise ValueError(
            "No samples with `uniqueness` scores. Run `main.py curate` first."
        )

    points = np.column_stack(
        [
            view.values("grid_area"),
            view.values("uniqueness"),
        ]
    )

    plot = fop.scatterplot(
        points,
        samples=view,
        labels="io_type",
        title="Uniqueness vs grid area",
        backend=backend,
        **_scatter_kwargs(backend, equal_axes=False),
    )
    _apply_axis_labels(
        plot,
        backend=backend,
        xlabel="Grid area (cells)",
        ylabel="Uniqueness",
    )
    return plot


def plot_input_output_aspect_ratio(
    dataset: fo.Dataset | fo.DatasetView,
    *,
    backend: str = "matplotlib",
) -> fop.InteractivePlot:
    """Scatter plot of input vs output aspect ratio, one point per pair."""
    _require_field(dataset, "pair_input_aspect_ratio")

    view = dataset.match(F("io_type") == "output")
    if len(view) == 0:
        raise ValueError("No output samples found in dataset.")

    points = np.column_stack(
        [
            view.values("pair_input_aspect_ratio"),
            view.values("pair_output_aspect_ratio"),
        ]
    )

    plot = fop.scatterplot(
        points,
        samples=view,
        labels="split",
        title="Input vs output aspect ratio",
        backend=backend,
        **_scatter_kwargs(backend, equal_axes=True),
    )
    _apply_axis_labels(
        plot,
        backend=backend,
        xlabel="Input aspect ratio (width / height)",
        ylabel="Output aspect ratio (width / height)",
    )
    return plot


def save_analysis_plots(
    dataset: fo.Dataset,
    save_dir: Path | str = DEFAULT_PLOT_DIR,
    *,
    backend: str = "matplotlib",
    dpi: int = 200,
) -> dict[str, Path]:
    """Save analysis scatter plots as PNG files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        "uniqueness_vs_area": plot_uniqueness_vs_grid_area(
            dataset, backend=backend
        ),
        "input_output_aspect_ratio": plot_input_output_aspect_ratio(
            dataset, backend=backend
        ),
    }

    paths: dict[str, Path] = {}
    for name, plot in plots.items():
        path = save_dir / f"{name}.png"
        plot.save(path, dpi=dpi)
        paths[name] = path.resolve()

    return paths


def run_analysis(
    dataset: fo.Dataset,
    *,
    save_dir: Path | str = DEFAULT_PLOT_DIR,
    launch_app: bool = False,
    show: bool = False,
    port: int = 5151,
    backend: str = "matplotlib",
    dpi: int = 200,
) -> fo.Session | dict[str, Path]:
    """Save analysis plots and optionally show them or launch the FiftyOne App."""
    paths = save_analysis_plots(
        dataset, save_dir, backend=backend, dpi=dpi
    )

    for name, path in paths.items():
        print(f"Saved {name} -> {path}")

    if show:
        plots = {
            "uniqueness_vs_area": plot_uniqueness_vs_grid_area(
                dataset, backend=backend
            ),
            "input_output_aspect_ratio": plot_input_output_aspect_ratio(
                dataset, backend=backend
            ),
        }
        for plot in plots.values():
            plot.show()

    if not launch_app:
        return paths

    session = fo.launch_app(dataset, port=port)
    print(f"FiftyOne App running at {session.url}")
    print("Press Ctrl+C to exit.")
    session.wait()
    return session
