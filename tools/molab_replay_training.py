#!/usr/bin/env python3
"""Molab replay-training helpers extracted from nb_uAtQZzbx95XsvTVVqeN32M.

This keeps the supervised replay-training launch/eval cells that were used in
Molab, without credential or replay-download cells.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import time


def tail_train_log(base_dir: pathlib.Path, run_name: str, lines: int = 50) -> None:
    log = base_dir / "runs" / run_name / "train.log"
    print("\n".join(log.read_text(errors="replace").splitlines()[-lines:]))


def run_v8_epoch020_eval_diag(base_dir: pathlib.Path) -> pathlib.Path:
    import torch
    from torch.utils.data import DataLoader, Subset

    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    from tools.v8_metrics import compute_action_metrics
    from tools.v8_model import V8ActionSlotTransformer, V8ModelConfig
    from tools.v8_train import (
        V8ShardDataset,
        collate_samples,
        compute_supervised_loss,
        predictions_from_outputs,
        truth_from_batch,
    )

    dataset_dir = base_dir / "data" / "v8_leader_trace"
    run_dir = base_dir / "runs" / "v8-molab-trace-20e"
    ckpt_path = run_dir / "checkpoint.pt"
    out_path = run_dir / "eval_epoch020.json"

    started = time.perf_counter()
    payload = torch.load(ckpt_path, map_location="cpu")
    config = V8ModelConfig(**payload["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = V8ActionSlotTransformer(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    dataset = V8ShardDataset(dataset_dir)
    episode_ids = []
    for index in range(len(dataset)):
        item = dataset[index]
        episode_id = int(item["episode_id"]) if "episode_id" in item else index
        episode_ids.append(episode_id)
    unique_episodes = sorted(set(episode_ids))
    eval_episodes = set(unique_episodes[-max(1, len(unique_episodes) // 10) :])
    eval_indices = [index for index, episode_id in enumerate(episode_ids) if episode_id in eval_episodes]
    max_samples = 32768
    if len(eval_indices) > max_samples:
        rng = random.Random(8)
        eval_indices = sorted(rng.sample(eval_indices, max_samples))

    loader = DataLoader(
        Subset(dataset, eval_indices),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_samples,
    )
    total_loss = 0.0
    total_batches = 0
    parts_sum = {"fire": 0.0, "source": 0.0, "target": 0.0, "amount": 0.0}
    truth_all = []
    pred_all_05 = []
    pred_all_02 = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                batch_dev["tokens"],
                batch_dev["token_type_ids"],
                batch_dev["owner_ids"],
                padding_mask=batch_dev["padding_mask"],
                planet_mask=batch_dev["planet_mask"],
            )
            loss, parts = compute_supervised_loss(outputs, batch_dev)
            total_loss += float(loss.detach().cpu())
            for key in parts_sum:
                parts_sum[key] += float(parts.get(key, 0.0))
            total_batches += 1
            truth = truth_from_batch(batch)
            truth_all.extend(truth)
            pred_all_05.extend(predictions_from_outputs(outputs, fire_threshold=0.5))
            pred_all_02.extend(predictions_from_outputs(outputs, fire_threshold=0.2))

    metrics_05 = compute_action_metrics(truth_all, pred_all_05, fire_threshold=0.5)
    metrics_02 = compute_action_metrics(truth_all, pred_all_02, fire_threshold=0.2)
    result = {
        "event": "eval_complete",
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch_completed_0based": int(payload.get("epoch_completed", -1)),
        "checkpoint_epochs_requested": int(payload.get("epochs_requested", -1)),
        "dataset_samples": len(dataset),
        "eval_samples": len(eval_indices),
        "eval_episodes": len(eval_episodes),
        "device": device,
        "loss": total_loss / max(1, total_batches),
        "loss_parts": {key: value / max(1, total_batches) for key, value in parts_sum.items()},
        "metrics_fire_0_5": metrics_05,
        "metrics_fire_0_2": metrics_02,
        "seconds": time.perf_counter() - started,
        "note": "diagnostic episode-split eval; 20e training used the full dataset, so this is not a clean unseen holdout",
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return out_path


def launch_v8_continue_from019_20e(base_dir: pathlib.Path) -> dict[str, object]:
    source_ckpt = base_dir / "runs" / "v8-molab-trace-20e" / "checkpoint-epoch-019.pt"
    run_dir = base_dir / "runs" / "v8-molab-trace-from019-plus20e-lr3e4"
    return _launch_training(
        base_dir=base_dir,
        source_ckpt=source_ckpt,
        run_dir=run_dir,
        event="continuation_started",
        epochs=39,
        batch_size=256,
        lr="3.0e-4",
        log_every_steps=50,
        start_new_session=False,
        write_probe=True,
    )


def launch_v8_continue_from038_b1024_lr1e5(base_dir: pathlib.Path) -> dict[str, object]:
    source_ckpt = (
        base_dir
        / "runs"
        / "v8-molab-trace-from019-plus20e-lr3e4"
        / "checkpoint-epoch-038.pt"
    )
    run_dir = base_dir / "runs" / "v8-molab-trace-from038-b1024-lr1e5-plus20e"
    return _launch_training(
        base_dir=base_dir,
        source_ckpt=source_ckpt,
        run_dir=run_dir,
        event="b1024_lr1e5_started",
        epochs=58,
        batch_size=1024,
        lr="1.0e-5",
        log_every_steps=10,
        start_new_session=True,
        write_probe=False,
        extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


def _launch_training(
    *,
    base_dir: pathlib.Path,
    source_ckpt: pathlib.Path,
    run_dir: pathlib.Path,
    event: str,
    epochs: int,
    batch_size: int,
    lr: str,
    log_every_steps: int,
    start_new_session: bool,
    write_probe: bool,
    extra_env: dict[str, str] | None = None,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    pid_path = run_dir / "train.pid"
    status_path = run_dir / "launch_status.json"
    if not source_ckpt.exists():
        raise FileNotFoundError(source_ckpt)

    cmd = [
        sys.executable,
        "-u",
        "tools/v8_train.py",
        "--dataset",
        "data/v8_leader_trace",
        "--run-dir",
        str(run_dir.relative_to(base_dir)),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda",
        "--lr",
        lr,
        "--resume-checkpoint",
        str(source_ckpt),
        "--log-every-steps",
        str(log_every_steps),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_env:
        env.update(extra_env)

    probe: dict[str, object] = {
        "event": "continue_launch_probe",
        "base_dir": str(base_dir),
        "run_dir": str(run_dir),
        "source_checkpoint": str(source_ckpt),
        "target_epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }
    if write_probe:
        try:
            import torch

            probe["torch"] = torch.__version__
            probe["cuda_available"] = bool(torch.cuda.is_available())
            probe["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                probe["cuda_mem_total_gib"] = round(total / 1024**3, 2)
                probe["cuda_mem_free_gib"] = round(free / 1024**3, 2)
        except Exception as exc:
            probe["torch_error"] = repr(exc)

    with log_path.open("ab") as log_file:
        if write_probe:
            log_file.write((json.dumps(probe, ensure_ascii=False) + "\n").encode("utf-8"))
            log_file.write((json.dumps({"event": "launch_command", "cmd": cmd}, ensure_ascii=False) + "\n").encode("utf-8"))
            log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=base_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=start_new_session,
        )

    pid_path.write_text(str(proc.pid), encoding="utf-8")
    status = {
        "event": event,
        "pid": proc.pid,
        "log_path": str(log_path),
        "pid_path": str(pid_path),
        "source_checkpoint": str(source_ckpt),
        "run_dir": str(run_dir),
        "cmd": cmd,
        "probe": probe,
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=pathlib.Path, default=pathlib.Path("/marimo"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    tail_parser = subparsers.add_parser("tail-log")
    tail_parser.add_argument("--run-name", default="v8-molab-trace-20e")
    tail_parser.add_argument("--lines", type=int, default=50)
    subparsers.add_parser("eval-epoch020")
    subparsers.add_parser("continue-from019")
    subparsers.add_parser("continue-from038-b1024-lr1e5")
    args = parser.parse_args()

    if args.command == "tail-log":
        tail_train_log(args.base_dir, args.run_name, args.lines)
    elif args.command == "eval-epoch020":
        run_v8_epoch020_eval_diag(args.base_dir)
    elif args.command == "continue-from019":
        launch_v8_continue_from019_20e(args.base_dir)
    elif args.command == "continue-from038-b1024-lr1e5":
        launch_v8_continue_from038_b1024_lr1e5(args.base_dir)


if __name__ == "__main__":
    main()
