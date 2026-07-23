"""Serve the ARC latent-space viewer over HTTP."""

from __future__ import annotations

import functools
import http.server
import json
import socketserver
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

from arc_viewer.sources import (
    DEFAULT_VIEWER_DIR,
    discover_sources,
    discover_trajectories,
    path_map,
    sources_catalog,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DATA = Path("artifacts/viewer/tsne.json")
DEFAULT_DINO_DATA = Path("artifacts/viewer/tsne_dino.json")


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        directory: str,
        catalog: dict,
        traj_catalog: dict,
        files: dict[str, Path],
        traj_files: dict[str, Path],
        **kwargs,
    ):
        self.catalog = catalog
        self.traj_catalog = traj_catalog
        self.files = files
        self.traj_files = traj_files
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in ("/data/sources.json", "/sources.json"):
            self._serve_json(self.catalog)
            return
        if path in ("/data/trajectories.json", "/trajectories.json"):
            self._serve_json(self.traj_catalog)
            return

        legacy = {
            "/data/tsne.json": ("grids", "structural"),
            "/tsne.json": ("grids", "structural"),
            "/data/tsne_dino.json": ("grids", "dinov3"),
            "/tsne_dino.json": ("grids", "dinov3"),
        }
        kind_id = legacy.get(path)
        if kind_id is None and path.startswith("/data/traj/") and path.endswith(".json"):
            kind_id = ("trajectories", path[len("/data/traj/") : -len(".json")])
        elif kind_id is None and path.startswith("/data/") and path.endswith(".json"):
            kind_id = ("grids", path[len("/data/") : -len(".json")])

        if kind_id is not None:
            kind, source_id = kind_id
            file_map = self.traj_files if kind == "trajectories" else self.files
            file_path = file_map.get(source_id)
            if file_path is None:
                self.send_error(404, f"Unknown or unavailable source: {source_id}")
                return
            self._serve_file(file_path, "application/json")
            return

        if path in ("/", "/index.html"):
            self.path = "/index.html"
        super().do_GET()

    def _serve_json(self, payload: dict) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(404, f"Missing precomputed data: {path}")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        print(f"[viewer] {args[0]}")


def serve_viewer(
    *,
    data_path: Path | str = DEFAULT_DATA,
    dino_data_path: Path | str | None = DEFAULT_DINO_DATA,
    viewer_dir: Path | str = DEFAULT_VIEWER_DIR,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Serve the static viewer and all discovered embedding / trajectory JSONs."""
    sources = discover_sources(
        viewer_dir,
        structural_path=data_path,
        dino_path=dino_data_path,
    )
    trajectories = discover_trajectories(viewer_dir)
    files = path_map(sources)
    traj_files = path_map(trajectories)
    if "structural" not in files:
        raise FileNotFoundError(
            f"Structural data not found.\n"
            "Run: uv run python main.py viewer --embedding structural --precompute"
        )

    catalog = sources_catalog(sources)
    traj_catalog = sources_catalog(trajectories)

    handler = functools.partial(
        ViewerHandler,
        directory=str(STATIC_DIR),
        catalog=catalog,
        traj_catalog=traj_catalog,
        files=files,
        traj_files=traj_files,
    )
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"ARC viewer at {url}")
        for src in sources:
            status = src.path if src.available else f"unavailable ({src.path.name})"
            print(f"  [grids:{src.id}] {status}")
        for src in trajectories:
            status = src.path if src.available else f"unavailable ({src.path.name})"
            print(f"  [traj:{src.id}] {status}")
        print("Press Ctrl+C to stop.")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
