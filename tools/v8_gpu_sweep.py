#!/usr/bin/env python3
"""Record GPU sweep rows for v8 training runs."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

SWEEP_FIELDS = [
    "config_id",
    "gpu",
    "arch",
    "batch",
    "workers",
    "max_fleets",
    "memory_gib",
    "stage_ms",
    "wall_s",
    "throughput",
    "status",
    "notes",
]


def probe_gpu() -> tuple[str, float]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        first = output.splitlines()[0]
        name, memory_mib = [part.strip() for part in first.split(",", 1)]
        return name, float(memory_mib) / 1024.0
    except Exception:
        return "unknown", 0.0


def record_sweep_row(
    output: Path,
    *,
    config_id: str,
    arch: str,
    batch: int,
    workers: int,
    max_fleets: int,
    stage_ms: float,
    wall_s: float,
    throughput: float,
    status: str,
    notes: str = "",
) -> dict[str, object]:
    gpu, memory_gib = probe_gpu()
    row = {
        "config_id": config_id,
        "gpu": gpu,
        "arch": arch,
        "batch": batch,
        "workers": workers,
        "max_fleets": max_fleets,
        "memory_gib": round(memory_gib, 3),
        "stage_ms": stage_ms,
        "wall_s": wall_s,
        "throughput": throughput,
        "status": status,
        "notes": notes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS)
        if not existing:
            writer.writeheader()
        writer.writerow(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("test_results/v8_gpu_sweeps.csv"))
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-fleets", type=int, default=640)
    parser.add_argument("--stage-ms", type=float, default=0.0)
    parser.add_argument("--wall-s", type=float, default=0.0)
    parser.add_argument("--throughput", type=float, default=0.0)
    parser.add_argument("--status", default="recorded")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    row = record_sweep_row(
        args.output,
        config_id=args.config_id,
        arch=args.arch,
        batch=args.batch,
        workers=args.workers,
        max_fleets=args.max_fleets,
        stage_ms=args.stage_ms,
        wall_s=args.wall_s,
        throughput=args.throughput,
        status=args.status,
        notes=args.notes,
    )
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
