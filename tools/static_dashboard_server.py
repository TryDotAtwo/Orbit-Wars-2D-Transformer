from __future__ import annotations

import argparse
import functools
import http.server
import pathlib
import socketserver


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, dist_dir: pathlib.Path, telemetry_dir: pathlib.Path, **kwargs):
        self.dist_dir = dist_dir
        self.telemetry_dir = telemetry_dir
        super().__init__(*args, directory=str(dist_dir), **kwargs)

    def translate_path(self, path: str) -> str:
        if path.startswith("/telemetry/"):
            suffix = path.removeprefix("/telemetry/").split("?", 1)[0].split("#", 1)[0]
            return str((self.telemetry_dir / suffix).resolve())
        translated = pathlib.Path(super().translate_path(path))
        if translated.is_dir():
            return str(translated / "index.html")
        if not translated.exists() and "." not in pathlib.PurePosixPath(path).name:
            return str(self.dist_dir / "index.html")
        return str(translated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--dist-dir", type=pathlib.Path, default=pathlib.Path("dashboard/dist"))
    parser.add_argument("--telemetry-dir", type=pathlib.Path, default=pathlib.Path("dashboard/public/telemetry"))
    args = parser.parse_args()
    handler = functools.partial(
        DashboardRequestHandler,
        dist_dir=args.dist_dir.resolve(),
        telemetry_dir=args.telemetry_dir.resolve(),
    )
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as server:
        server.allow_reuse_address = True
        print(f"dashboard_static_server=http://{args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
