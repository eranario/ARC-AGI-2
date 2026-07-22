"""Serve the ARC latent-space viewer over HTTP."""

from __future__ import annotations

import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DATA = Path("artifacts/viewer/tsne.json")


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, data_path: Path, **kwargs):
        self.data_path = data_path
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        if self.path in ("/data/tsne.json", "/tsne.json"):
            self._serve_file(self.data_path, "application/json")
            return
        if self.path in ("/", "/index.html"):
            self.path = "/index.html"
        super().do_GET()

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
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Serve the static viewer and precomputed t-SNE JSON."""
    data_path = Path(data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Precomputed data not found: {data_path}\n"
            "Run: uv run python main.py viewer --precompute"
        )

    handler = functools.partial(
        ViewerHandler,
        directory=str(STATIC_DIR),
        data_path=data_path,
    )
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"ARC viewer at {url}")
        print(f"Data: {data_path}")
        print("Press Ctrl+C to stop.")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
