#!/usr/bin/env python3
"""GPU-resident single-model RL loop for OWV8.

CPU responsibilities are intentionally limited to setup, final status/metrics,
and checkpoint/export.  The steady state uses native CUDA for simulation/action
decode and PyTorch CUDA tensors for policy loss/backprop.
"""
from __future__ import annotations

import argparse
import math
import json
import time
from pathlib import Path

import torch

from tools.v8_cuda_ctypes import V8CudaRuntime
from tools.v8_gpu_policy_loss import compute_resident_policy_loss
from tools.v8_model import V8ActionSlotTransformer, export_model_bin, load_model_bin_state
from tools.v8_resident_device_batch import ResidentBatchTensorView, resident_batch_view_to_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-bin", type=Path, required=True)
    parser.add_argument("--cuda-lib", type=Path, default=Path("target/liborbit_wars_v8_cuda.so"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v8-gpu-resident-rl"))
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--train-batch-requests", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    run_gpu_resident_rl(args)


def run_gpu_resident_rl(args: argparse.Namespace) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config, state = load_model_bin_state(args.base_model_bin)
    model = V8ActionSlotTransformer(config).cuda()
    model.load_state_dict(state)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    runtime = V8CudaRuntime(args.cuda_lib)
    runtime.status()
    print(json.dumps({
        "event": "gpu_resident_rl_start",
        "base_model": str(args.base_model_bin),
        "cuda_lib": str(args.cuda_lib),
        "run_dir": str(args.run_dir),
        "generations": args.generations,
        "games": args.games,
        "players": args.players,
        "steps": args.steps,
        "lr": args.lr,
        "train_batch_requests": args.train_batch_requests,
        "bf16": args.bf16,
    }), flush=True)

    for generation in range(1, args.generations + 1):
        started = time.perf_counter()
        native_model = runtime.create_model_from_state({key: value.detach().cpu() for key, value in model.state_dict().items()})
        sim = runtime.create_sim(games=args.games, players=args.players, step_limit=args.steps)
        runtime.load_simple_games(sim)
        request_players = torch.as_tensor(sim.request_players, device="cuda", dtype=torch.long)
        step_batches: list[ResidentBatchTensorView] = []
        step_fire: list[torch.Tensor] = []
        try:
            for step in range(args.steps):
                view = runtime.decode_model_actions(native_model, sim, step)
                batch = resident_batch_view_to_tensors(view)
                step_batches.append(clone_resident_batch(batch))
                step_fire.append(batch.labels_fire.float().mean().detach())
                runtime.step_device_actions(sim, step)
            statuses, stats = runtime.read_status_stats(sim)
        finally:
            runtime.destroy_model(native_model)
            runtime.destroy_sim(sim)

        winners = torch.tensor([status.winner for status in statuses], device="cuda", dtype=torch.long)
        game_rewards = torch.full((args.games, args.players), -1.0, device="cuda")
        valid = winners >= 0
        if valid.any():
            game_rewards[torch.arange(args.games, device="cuda")[valid], winners[valid]] = 1.0
        request_game_ids = torch.arange(args.games, device="cuda", dtype=torch.long).repeat_interleave(args.players)
        request_rewards = game_rewards[request_game_ids, request_players]
        fire_rate = torch.stack(step_fire, dim=0).mean()
        total_chunks = sum(
            math.ceil(int(batch.tokens.shape[0]) / max(1, args.train_batch_requests))
            for batch in step_batches
        )
        loss_total = 0.0
        active_slots_total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch in step_batches:
            request_count = int(batch.tokens.shape[0])
            for start in range(0, request_count, args.train_batch_requests):
                stop = min(start + args.train_batch_requests, request_count)
                mini = slice_resident_batch(batch, start, stop)
                mini_rewards = request_rewards[start:stop]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.bf16)):
                    outputs = model(
                        mini.tokens,
                        mini.token_type_ids,
                        mini.owner_ids,
                        padding_mask=mini.padding_mask,
                        planet_mask=mini.planet_mask,
                    )
                    loss, metrics = compute_resident_policy_loss(outputs, mini, mini_rewards)
                (loss / max(1, total_chunks)).backward()
                loss_total += float(loss.detach().cpu())
                active_slots_total += float(metrics["active_slots"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        del step_batches
        torch.cuda.synchronize()
        mean_loss = loss_total / max(1, total_chunks)
        mean_active_slots = active_slots_total / max(1, total_chunks)

        generation_dir = args.run_dir / f"generation-{generation:04d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = generation_dir / f"checkpoint-generation-{generation:04d}.pt"
        torch.save({
            "generation": generation,
            "config": config.__dict__,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "metrics": {
                "loss": mean_loss,
                "fire_rate": float(fire_rate.detach().cpu()),
                "active_slots": mean_active_slots,
            },
        }, checkpoint)
        export_model_bin(model, config, args.run_dir / "model.bin")
        wins = int(valid.sum().detach().cpu())
        draws = int((~valid).sum().detach().cpu())
        elapsed = max(1.0e-6, time.perf_counter() - started)
        print(json.dumps({
            "event": "gpu_resident_rl_generation_complete",
            "generation": generation,
            "seconds": round(elapsed, 3),
            "games_per_second": round(args.games / elapsed, 3),
            "loss": mean_loss,
            "fire_rate": float(fire_rate.detach().cpu()),
            "active_slots": mean_active_slots,
            "train_chunks": total_chunks,
            "decisive_games": wins,
            "draws_as_losses": draws,
            "checkpoint": str(checkpoint),
            "model_bin": str(args.run_dir / "model.bin"),
        }), flush=True)


def clone_resident_batch(batch: ResidentBatchTensorView) -> ResidentBatchTensorView:
    return ResidentBatchTensorView(
        tokens=batch.tokens.detach().clone(),
        token_type_ids=batch.token_type_ids.detach().clone(),
        owner_ids=batch.owner_ids.detach().clone(),
        padding_mask=batch.padding_mask.detach().clone(),
        planet_mask=batch.planet_mask.detach().clone(),
        labels_fire=batch.labels_fire.detach().clone(),
        labels_source=batch.labels_source.detach().clone(),
        labels_target=batch.labels_target.detach().clone(),
        labels_amount=batch.labels_amount.detach().clone(),
        labels_confidence=batch.labels_confidence.detach().clone(),
    )


def slice_resident_batch(batch: ResidentBatchTensorView, start: int, stop: int) -> ResidentBatchTensorView:
    return ResidentBatchTensorView(
        tokens=batch.tokens[start:stop],
        token_type_ids=batch.token_type_ids[start:stop],
        owner_ids=batch.owner_ids[start:stop],
        padding_mask=batch.padding_mask[start:stop],
        planet_mask=batch.planet_mask[start:stop],
        labels_fire=batch.labels_fire[start:stop],
        labels_source=batch.labels_source[start:stop],
        labels_target=batch.labels_target[start:stop],
        labels_amount=batch.labels_amount[start:stop],
        labels_confidence=batch.labels_confidence[start:stop],
    )


if __name__ == "__main__":
    main()
