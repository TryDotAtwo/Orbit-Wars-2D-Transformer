from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ITERATIONS = 2_000
WARMUP_ITERATIONS = 100
MILLISECONDS_PER_SECOND = 1_000.0

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "kaggle_submission"
sys.path.insert(0, str(SUBMISSION_DIR))

import main  # noqa: E402


OBSERVATION = {
    "player": 0,
    "angular_velocity": 0.03,
    "planets": [
        [1, 0, 10, 10, 1.0, 80, 1],
        [2, -1, 20, 14, 1.7, 12, 2],
        [3, -1, 30, 30, 2.2, 31, 3],
        [4, 1, 82, 72, 2.6, 44, 4],
        [5, 0, 51, 28, 1.0, 15, 1],
        [6, -1, 74, 35, 1.7, 20, 2],
    ],
    "initial_planets": [
        [1, 0, 10, 10, 1.0, 10, 1],
        [2, -1, 20, 14, 1.7, 12, 2],
        [3, -1, 30, 30, 2.2, 31, 3],
        [4, 1, 82, 72, 2.6, 10, 4],
        [5, -1, 51, 28, 1.0, 15, 1],
        [6, -1, 74, 35, 1.7, 20, 2],
    ],
    "comet_planet_ids": [],
    "comets": [],
}


def main_benchmark() -> None:
    for _ in range(WARMUP_ITERATIONS):
        main.agent(OBSERVATION)

    timings_ms: list[float] = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        main.agent(OBSERVATION)
        timings_ms.append((time.perf_counter() - start) * MILLISECONDS_PER_SECOND)

    timings_ms.sort()
    p95_index = int(len(timings_ms) * 0.95)
    print(f"iterations={ITERATIONS}")
    print(f"mean_ms={statistics.mean(timings_ms):.6f}")
    print(f"p50_ms={statistics.median(timings_ms):.6f}")
    print(f"p95_ms={timings_ms[p95_index]:.6f}")
    print(f"max_ms={max(timings_ms):.6f}")


if __name__ == "__main__":
    main_benchmark()

