#!/usr/bin/env python3
"""GPU-resident single-model RL loop for OWV8.

CPU responsibilities are intentionally limited to setup, final status/metrics,
and checkpoint/export.  The steady state uses native CUDA for simulation/action
decode and PyTorch CUDA tensors for policy loss/backprop.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tools.v8_cuda_ctypes import V8CudaRuntime
from tools.v8_gpu_policy_loss import compute_resident_selected_logprob
from tools.v8_model import V8ActionSlotTransformer, export_model_bin, load_model_bin_state
from tools.v8_resident_device_batch import resident_batch_view_to_tensors


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
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    run_gpu_resident_rl(args)


def run_gpu_resident_rl(args: argparse.Namespace) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config, state = load_model_bin_state(args.base_model_bin)
    model = V8ActionSlotTransformer(config).cuda()
    model.load_state_dict(state)
    model.train()
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
        "bf16": args.bf16,
    }), flush=True)

    for generation in range(1, args.generations + 1):
        started = time.perf_counter()
        native_model = runtime.create_model_from_state({key: value.detach().cpu() for key, value in model.state_dict().items()})
        sim = runtime.create_sim(games=args.games, players=args.players, step_limit=args.steps)
        runtime.load_simple_games(sim)
        request_players = torch.as_tensor(sim.request_players, device="cuda", dtype=torch.long)
        step_logprobs: list[torch.Tensor] = []
        step_fire: list[torch.Tensor] = []
        try:
            for step in range(args.steps):
                view = runtime.decode_model_actions(native_model, sim, step)
                batch = resident_batch_view_to_tensors(view)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.bf16)):
                    outputs = model(
                        batch.tokens,
                        batch.token_type_ids,
                        batch.owner_ids,
                        padding_mask=batch.padding_mask,
                        planet_mask=batch.planet_mask,
                    )
                    selected_logprob, labels_fire = compute_resident_selected_logprob(outputs, batch)
                step_logprobs.append(selected_logprob.mean(dim=1))
                step_fire.append(labels_fire.detach().mean(dim=1))
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
        logprob = torch.stack(step_logprobs, dim=0)
        fire_rate = torch.stack(step_fire, dim=0).mean()
        loss = -(logprob * request_rewards[None, :]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()

        generation_dir = args.run_dir / f"generation-{generation:04d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = generation_dir / f"checkpoint-generation-{generation:04d}.pt"
        torch.save({
            "generation": generation,
            "config": config.__dict__,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "metrics": {
                "loss": float(loss.detach().cpu()),
                "fire_rate": float(fire_rate.detach().cpu()),
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
            "loss": float(loss.detach().cpu()),
            "fire_rate": float(fire_rate.detach().cpu()),
            "decisive_games": wins,
            "draws_as_losses": draws,
            "checkpoint": str(checkpoint),
            "model_bin": str(args.run_dir / "model.bin"),
        }), flush=True)


if __name__ == "__main__":
    main()

