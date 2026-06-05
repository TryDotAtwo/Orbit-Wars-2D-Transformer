#!/usr/bin/env python3
"""Supervised imitation training for v8 action-slot models."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.leaderboard_v8_dataset import ACTION_SLOTS, AMOUNT_CLASSES
from tools.v8_metrics import compute_action_metrics
from tools.v8_model import V8ActionSlotTransformer, V8ModelConfig, export_model_bin


class V8ShardDataset(Dataset):
    def __init__(self, dataset_dir: Path):
        metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
        self.arrays: list[dict[str, np.ndarray]] = []
        self.index: list[tuple[int, int]] = []
        for shard_index, shard_meta in enumerate(metadata["shards"]):
            shard = np.load(dataset_dir / shard_meta["file"])
            arrays = {name: shard[name] for name in shard.files}
            shard.close()
            self.arrays.append(arrays)
            for row in range(arrays["tokens"].shape[0]):
                self.index.append((shard_index, row))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        shard_index, row = self.index[item]
        arrays = self.arrays[shard_index]
        return {key: arrays[key][row] for key in arrays}


def collate_samples(samples: Iterable[dict[str, Any]]) -> dict[str, torch.Tensor]:
    rows = list(samples)
    lengths = [_sample_token_length(row) for row in rows]
    max_tokens = max(lengths)
    token_features = int(rows[0]["tokens"].shape[-1])
    batch_size = len(rows)
    tokens = torch.zeros((batch_size, max_tokens, token_features), dtype=torch.float32)
    token_type_ids = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    owner_ids = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    padding_mask = torch.ones((batch_size, max_tokens), dtype=torch.bool)
    planet_mask = torch.zeros((batch_size, 64), dtype=torch.bool)
    labels_fire = torch.zeros((batch_size, ACTION_SLOTS), dtype=torch.float32)
    labels_source = torch.full((batch_size, ACTION_SLOTS), -1, dtype=torch.long)
    labels_target = torch.full((batch_size, ACTION_SLOTS), -1, dtype=torch.long)
    labels_amount = torch.full((batch_size, ACTION_SLOTS), -1, dtype=torch.long)

    for index, row in enumerate(rows):
        length = lengths[index]
        tokens[index, :length] = torch.as_tensor(row["tokens"][:length], dtype=torch.float32)
        token_type_ids[index, :length] = torch.as_tensor(row["token_type_ids"][:length], dtype=torch.long)
        owner_ids[index, :length] = torch.as_tensor(row["owner_ids"][:length], dtype=torch.long)
        padding_mask[index, :length] = False
        planet_mask[index] = torch.as_tensor(row["planet_mask"], dtype=torch.bool)
        labels_fire[index] = torch.as_tensor(row["labels_fire"], dtype=torch.float32)
        labels_source[index] = torch.as_tensor(row["labels_source"], dtype=torch.long)
        labels_target[index] = torch.as_tensor(row["labels_target"], dtype=torch.long)
        labels_amount[index] = torch.as_tensor(row["labels_amount"], dtype=torch.long)

    return {
        "tokens": tokens,
        "token_type_ids": token_type_ids,
        "owner_ids": owner_ids,
        "padding_mask": padding_mask,
        "planet_mask": planet_mask,
        "labels_fire": labels_fire,
        "labels_source": labels_source,
        "labels_target": labels_target,
        "labels_amount": labels_amount,
    }


def _sample_token_length(row: dict[str, Any]) -> int:
    offsets = row.get("sample_offsets")
    if offsets is None:
        return int(row["tokens"].shape[0])
    return int(offsets[1])


def compute_supervised_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    fire_loss = nn.functional.binary_cross_entropy_with_logits(outputs["fire_logits"], batch["labels_fire"])
    active = batch["labels_fire"].bool()
    if active.any():
        source_loss = nn.functional.cross_entropy(outputs["source_logits"][active], batch["labels_source"][active])
        target_loss = nn.functional.cross_entropy(outputs["target_logits"][active], batch["labels_target"][active])
        amount_loss = nn.functional.cross_entropy(outputs["amount_logits"][active], batch["labels_amount"][active])
    else:
        zero = outputs["fire_logits"].sum() * 0.0
        source_loss = target_loss = amount_loss = zero
    loss = fire_loss + source_loss + target_loss + amount_loss
    return loss, {
        "fire": float(fire_loss.detach().cpu()),
        "source": float(source_loss.detach().cpu()),
        "target": float(target_loss.detach().cpu()),
        "amount": float(amount_loss.detach().cpu()),
    }


def train_imitation(
    dataset_dir: Path,
    run_dir: Path,
    *,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 3.0e-4,
    device: str | None = None,
    config: V8ModelConfig | None = None,
    log_every_steps: int = 250,
    resume_checkpoint: Path | str | None = None,
) -> Path:
    explicit_config = config
    config = config or V8ModelConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = _resolve_resume_checkpoint(run_dir, resume_checkpoint)
    checkpoint_payload: dict[str, Any] | None = None
    if resume_path is not None:
        checkpoint_payload = torch.load(resume_path, map_location="cpu")
        if "config" in checkpoint_payload and explicit_config is None:
            config = V8ModelConfig(**checkpoint_payload["config"])
    dataset = V8ShardDataset(dataset_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_samples)
    model = V8ActionSlotTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    start_epoch = 0
    last_metrics: dict[str, float] = {}
    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model"])
        if "optimizer" in checkpoint_payload:
            optimizer.load_state_dict(checkpoint_payload["optimizer"])
            _move_optimizer_state(optimizer, device)
        start_epoch = int(checkpoint_payload.get("epoch_completed", checkpoint_payload.get("epoch", -1))) + 1
        last_metrics = dict(checkpoint_payload.get("metrics") or {})
        print(json.dumps({
            "event": "resume_checkpoint",
            "checkpoint": str(resume_path),
            "start_epoch": start_epoch + 1,
            "target_epochs": epochs,
            "loaded_metrics": last_metrics,
        }), flush=True)
    else:
        print(json.dumps({
            "event": "resume_checkpoint_missing",
            "checkpoint": str(run_dir / "checkpoint.pt") if resume_checkpoint in ("auto", True) else str(resume_checkpoint),
            "start_epoch": 1,
            "target_epochs": epochs,
        }), flush=True)

    if start_epoch >= epochs:
        export_model_bin(model.cpu().eval(), config, run_dir / "model.bin")
        return run_dir / "checkpoint.pt"

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        epoch_started = time.perf_counter()
        epoch_steps = len(loader)
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["tokens"], batch["token_type_ids"], batch["owner_ids"], padding_mask=batch["padding_mask"], planet_mask=batch["planet_mask"])
            loss, parts = compute_supervised_loss(outputs, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
            last_metrics = parts
            if log_every_steps > 0 and (steps == 1 or steps % log_every_steps == 0):
                elapsed = max(1.0e-6, time.perf_counter() - epoch_started)
                print(json.dumps({
                    "event": "train_step",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "step": steps,
                    "steps": epoch_steps,
                    "loss": float(loss.detach().cpu()),
                    "avg_loss": total_loss / steps,
                    "samples_per_sec": (steps * batch["tokens"].shape[0]) / elapsed,
                }), flush=True)
        last_metrics["loss"] = total_loss / max(1, steps)
        last_metrics["epoch"] = float(epoch)
        write_telemetry(run_dir / "telemetry", run_id=run_dir.name, generation=epoch, metrics=_metrics_for_dashboard(last_metrics))
        checkpoint = _save_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            config=config,
            metrics=last_metrics,
            epoch=epoch,
            epochs=epochs,
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            lr=lr,
        )
        print(json.dumps({
            "event": "epoch_complete",
            "epoch": epoch + 1,
            "epochs": epochs,
            "checkpoint": str(checkpoint),
            "loss": last_metrics["loss"],
            "fire_loss": last_metrics.get("fire", 0.0),
            "source_loss": last_metrics.get("source", 0.0),
            "target_loss": last_metrics.get("target", 0.0),
            "amount_loss": last_metrics.get("amount", 0.0),
        }), flush=True)

    export_model_bin(model.cpu().eval(), config, run_dir / "model.bin")
    return run_dir / "checkpoint.pt"


def _resolve_resume_checkpoint(run_dir: Path, resume_checkpoint: Path | str | None) -> Path | None:
    if resume_checkpoint is None or resume_checkpoint in ("", "none", "false", False):
        return None
    if resume_checkpoint in ("auto", True):
        checkpoint = run_dir / "checkpoint.pt"
        return checkpoint if checkpoint.exists() else None
    checkpoint = Path(resume_checkpoint)
    return checkpoint if checkpoint.exists() else None


def _save_checkpoint(
    run_dir: Path,
    *,
    model: V8ActionSlotTransformer,
    optimizer: torch.optim.Optimizer,
    config: V8ModelConfig,
    metrics: dict[str, float],
    epoch: int,
    epochs: int,
    dataset_dir: Path,
    batch_size: int,
    lr: float,
) -> Path:
    payload = {
        "config": config.__dict__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "epoch_completed": epoch,
        "epochs_requested": epochs,
        "dataset": str(dataset_dir),
        "batch_size": batch_size,
        "lr": lr,
    }
    epoch_checkpoint = run_dir / f"checkpoint-epoch-{epoch + 1:03d}.pt"
    torch.save(payload, epoch_checkpoint)
    torch.save(payload, run_dir / "checkpoint.pt")
    return epoch_checkpoint


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(target)


def write_telemetry(run_dir: Path, *, run_id: str, generation: int, metrics: dict[str, float]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = {
        "runId": run_id,
        "currentGeneration": generation,
        "generations": [
            {
                "generation": generation,
                "models": 1,
                "games": int(metrics.get("games", 0)),
                "winRate": float(metrics.get("winRate", 0.0)),
                "actionF1": float(metrics.get("actionF1", 0.0)),
                "sourceTop1": float(metrics.get("sourceTop1", 0.0)),
                "targetTop1": float(metrics.get("targetTop1", 0.0)),
                "targetTop2": float(metrics.get("targetTop2", 0.0)),
                "targetTop3": float(metrics.get("targetTop3", 0.0)),
                "amountAccuracy": float(metrics.get("amountAccuracy", 0.0)),
                "amountNonAllInAccuracy": float(metrics.get("amountNonAllInAccuracy", 0.0)),
                "duplicateSourceRate": float(metrics.get("duplicateSourceRate", 0.0)),
                "fireRate": float(metrics.get("fireRate", 0.0)),
            }
        ],
        "models": [],
        "amountHistogram": [{"className": name, "count": 0} for name in AMOUNT_CLASSES],
        "sweeps": [],
    }
    path = run_dir / "latest.json"
    path.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def predictions_from_outputs(outputs: dict[str, torch.Tensor], *, fire_threshold: float = 0.5) -> list[list[dict[str, Any]]]:
    fire_probs = outputs["fire_logits"].sigmoid().detach().cpu()
    source = outputs["source_logits"].argmax(dim=-1).detach().cpu()
    target_logits = outputs["target_logits"].detach().cpu()
    target = target_logits.argmax(dim=-1)
    amount = outputs["amount_logits"].argmax(dim=-1).detach().cpu()
    batches = []
    for batch_index in range(fire_probs.shape[0]):
        rows = []
        for slot_index in range(fire_probs.shape[1]):
            topk = torch.topk(target_logits[batch_index, slot_index], k=min(3, target_logits.shape[-1])).indices.tolist()
            rows.append({
                "fire": float(fire_probs[batch_index, slot_index]),
                "source": int(source[batch_index, slot_index]),
                "target": int(target[batch_index, slot_index]),
                "target_topk": [int(value) for value in topk],
                "amount": int(amount[batch_index, slot_index]),
            })
        batches.append([row for row in rows if row["fire"] >= fire_threshold])
    return batches


def truth_from_batch(batch: dict[str, torch.Tensor]) -> list[list[dict[str, Any]]]:
    truth_batches = []
    labels_fire = batch["labels_fire"].detach().cpu()
    for batch_index in range(labels_fire.shape[0]):
        truth = []
        for slot_index in range(labels_fire.shape[1]):
            if labels_fire[batch_index, slot_index] > 0:
                truth.append({
                    "source": int(batch["labels_source"][batch_index, slot_index]),
                    "target": int(batch["labels_target"][batch_index, slot_index]),
                    "amount": int(batch["labels_amount"][batch_index, slot_index]),
                })
        truth_batches.append(truth)
    return truth_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v8-imitation"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every-steps", type=int, default=250)
    parser.add_argument("--resume-checkpoint", default="auto", help="Checkpoint path, 'auto' for run-dir/checkpoint.pt, or 'none'.")
    args = parser.parse_args()
    checkpoint = train_imitation(
        args.dataset,
        args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        log_every_steps=args.log_every_steps,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps({"checkpoint": str(checkpoint), "model_bin": str(args.run_dir / "model.bin")}))


def _metrics_for_dashboard(loss_parts: dict[str, float]) -> dict[str, float]:
    # Training loss telemetry is intentionally separate from evaluation metrics;
    # absent evaluation fields stay zero so the dashboard remains schema-stable.
    return {
        "actionF1": 0.0,
        "sourceTop1": 0.0,
        "targetTop1": 0.0,
        "targetTop2": 0.0,
        "targetTop3": 0.0,
        "amountAccuracy": 0.0,
        "amountNonAllInAccuracy": 0.0,
        "duplicateSourceRate": 0.0,
        "fireRate": 0.0,
        "loss": float(loss_parts.get("loss", 0.0)),
    }


if __name__ == "__main__":
    main()
