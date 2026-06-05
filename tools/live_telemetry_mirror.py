from __future__ import annotations

import argparse
import pathlib
import shutil
import time


def latest_telemetry(run_dir: pathlib.Path) -> pathlib.Path | None:
    files = list(run_dir.glob("generation-*/*.telemetry.json"))
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--target", type=pathlib.Path, default=pathlib.Path("dashboard/public/telemetry/latest.json"))
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    last_source: pathlib.Path | None = None
    while True:
        source = latest_telemetry(args.run_dir)
        if source and source != last_source:
            shutil.copyfile(source, args.target)
            print(f"mirrored={source} -> {args.target}", flush=True)
            last_source = source
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
