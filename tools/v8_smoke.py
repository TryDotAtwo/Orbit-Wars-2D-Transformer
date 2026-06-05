#!/usr/bin/env python3
"""Small end-to-end smoke for dataset, tiny training export, and Kaggle wrapper input."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.leaderboard_v8_dataset import convert_replay, write_dataset_shards
from tools.v8_model import V8ModelConfig
from tools.v8_train import train_imitation


def run_smoke(replays: list[Path], output_dir: Path, *, device: str = "cpu") -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for replay in replays:
        replay_samples, _stats = convert_replay(replay)
        samples.extend(replay_samples[:32])
    dataset_dir = output_dir / "dataset"
    metadata = write_dataset_shards(samples, dataset_dir, shard_size=64)
    checkpoint = train_imitation(
        dataset_dir,
        output_dir / "run",
        epochs=1,
        batch_size=8,
        device=device,
        config=V8ModelConfig(d_model=16, encoder_layers=1, decoder_layers=1, heads=4),
    )
    model_bin = output_dir / "run" / "model.bin"
    submission_model = Path("kaggle_submission/model.bin")
    shutil.copyfile(model_bin, submission_model)
    observation = _first_observation(replays[0])
    return {
        "samples": metadata["samples"],
        "checkpoint": str(checkpoint),
        "model_bin": str(model_bin),
        "submission_model": str(submission_model),
        "observation_planets": len(observation.get("planets") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("test_results/v8_smoke"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_smoke(args.replays[:2], args.output_dir, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _first_observation(replay_path: Path) -> dict[str, object]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    for step in replay.get("steps") or []:
        if isinstance(step, list):
            for agent in step:
                if isinstance(agent, dict) and agent.get("observation"):
                    return agent["observation"]
    raise ValueError(f"no observation in {replay_path}")


if __name__ == "__main__":
    main()
