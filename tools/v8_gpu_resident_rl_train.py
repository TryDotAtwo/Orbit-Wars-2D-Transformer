#!/usr/bin/env python3
"""GPU-resident single-model RL loop for OWV8.

CPU responsibilities are intentionally limited to setup, final status/metrics,
and checkpoint/export.  The steady state uses native CUDA for simulation/action
decode and PyTorch CUDA tensors for policy loss/backprop.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import torch
from torch import nn

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
    parser.add_argument("--generation-tournament-games", type=int, default=0)
    parser.add_argument("--arena-bin", type=Path, default=Path("target/release/orbit-wars-arena-v8"))
    parser.add_argument("--progress-steps", type=int, default=100)
    parser.add_argument("--debug-nan", action="store_true")
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
        "generation_tournament_games": args.generation_tournament_games,
        "debug_nan": args.debug_nan,
        "bf16": args.bf16,
    }), flush=True)

    for generation in range(1, args.generations + 1):
        started = time.perf_counter()
        native_model = runtime.create_model_from_state({key: value.detach().cpu() for key, value in model.state_dict().items()})
        sim = runtime.create_sim(games=args.games, players=args.players, step_limit=args.steps)
        runtime.load_official_like_games(sim, generation=generation)
        request_players = torch.as_tensor(sim.request_players, device="cuda", dtype=torch.long)
        step_batches: list[ResidentBatchTensorView] = []
        step_fire: list[torch.Tensor] = []
        progress_interval = max(0, int(args.progress_steps))
        try:
            for step in range(args.steps):
                view = runtime.decode_model_actions(native_model, sim, step)
                batch = resident_batch_view_to_tensors(view)
                step_batches.append(clone_resident_batch(batch))
                step_fire.append(batch.labels_fire.float().mean().detach())
                runtime.step_device_actions(sim, step)
                completed_steps = step + 1
                if progress_interval > 0 and (
                    completed_steps % progress_interval == 0
                    or completed_steps == args.steps
                ):
                    elapsed = max(1.0e-6, time.perf_counter() - started)
                    steps_left = max(0, args.steps - completed_steps)
                    seconds_per_step = elapsed / completed_steps
                    print(json.dumps({
                        "event": "gpu_resident_rl_step_progress",
                        "generation": generation,
                        "games": args.games,
                        "players": args.players,
                        "step": completed_steps,
                        "steps": args.steps,
                        "steps_left": steps_left,
                        "seconds": round(elapsed, 3),
                        "eta_seconds": round(seconds_per_step * steps_left, 3),
                    }), flush=True)
            statuses, stats = runtime.read_status_stats(sim)
            planets, fleets, _next_ids, _state_stats = runtime.read_state(sim)
        finally:
            runtime.destroy_model(native_model)
            runtime.destroy_sim(sim)

        rewards_cpu, margins_cpu = outcome_rewards(
            statuses,
            planets,
            fleets,
            games=args.games,
            players=args.players,
            planet_count=sim.config.planet_count,
            max_fleets=sim.config.max_fleets_per_game,
        )
        game_rewards = torch.as_tensor(rewards_cpu, device="cuda", dtype=torch.float32)
        valid = torch.tensor([status.winner >= 0 for status in statuses], device="cuda", dtype=torch.bool)
        request_game_ids = torch.arange(args.games, device="cuda", dtype=torch.long).repeat_interleave(args.players)
        request_rewards = game_rewards[request_game_ids, request_players]
        fire_rate = torch.stack(step_fire, dim=0).mean()
        total_chunks = sum(
            math.ceil(int(batch.tokens.shape[0]) / max(1, args.train_batch_requests))
            for batch in step_batches
        )
        loss_total = 0.0
        active_slots_total = 0.0
        finite_chunks = 0
        skipped_nan_chunks = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(step_batches):
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
                if args.debug_nan and not torch.isfinite(loss).item():
                    skipped_nan_chunks += 1
                    print(json.dumps(nan_debug_payload(
                        generation=generation,
                        batch_index=batch_index,
                        chunk_start=start,
                        chunk_stop=stop,
                        outputs=outputs,
                        batch=mini,
                        rewards=mini_rewards,
                        loss=loss,
                    )), flush=True)
                    continue
                (loss / max(1, total_chunks)).backward()
                loss_total += float(loss.detach().cpu())
                active_slots_total += float(metrics["active_slots"])
                finite_chunks += 1
        grad_debug = gradient_debug_payload(model) if args.debug_nan else None
        if args.debug_nan and grad_debug is not None:
            print(json.dumps({
                "event": "gpu_resident_rl_grad_debug",
                "generation": generation,
                **grad_debug,
            }), flush=True)
        if args.debug_nan and grad_debug is not None and grad_debug["bad_tensors"] > 0:
            optimizer.zero_grad(set_to_none=True)
            update_applied = False
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_applied = True
        del step_batches
        torch.cuda.synchronize()
        mean_loss = loss_total / max(1, finite_chunks)
        mean_active_slots = active_slots_total / max(1, finite_chunks)

        generation_dir = args.run_dir / f"generation-{generation:04d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = generation_dir / f"checkpoint-generation-{generation:04d}.pt"
        generation_model_bin = generation_dir / "model.bin"
        torch.save({
            "generation": generation,
            "config": config.__dict__,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "metrics": {
                "loss": mean_loss,
                "fire_rate": float(fire_rate.detach().cpu()),
                "active_slots": mean_active_slots,
                "mean_winner_margin": sum(margins_cpu) / max(1, len(margins_cpu)),
                "finite_chunks": finite_chunks,
                "skipped_nan_chunks": skipped_nan_chunks,
                "update_applied": update_applied,
            },
        }, checkpoint)
        export_model_bin(model, config, generation_model_bin)
        shutil.copy2(generation_model_bin, args.run_dir / "model.bin")
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
            "finite_chunks": finite_chunks,
            "skipped_nan_chunks": skipped_nan_chunks,
            "update_applied": update_applied,
            "decisive_games": wins,
            "draws_as_losses": draws,
            "mean_winner_margin": sum(margins_cpu) / max(1, len(margins_cpu)),
            "checkpoint": str(checkpoint),
            "generation_model_bin": str(generation_model_bin),
            "model_bin": str(args.run_dir / "model.bin"),
        }), flush=True)

    if args.generation_tournament_games > 0:
        run_generation_tournament(args)


def outcome_rewards(statuses, planets, fleets, *, games: int, players: int, planet_count: int, max_fleets: int) -> tuple[list[list[float]], list[float]]:
    rewards = [[-0.2 for _ in range(players)] for _ in range(games)]
    margins: list[float] = []
    for game in range(games):
        winner = int(statuses[game].winner)
        scores = [0.0 for _ in range(players)]
        planet_base = game * planet_count
        for row in range(planet_count):
            planet = planets[planet_base + row]
            owner = int(planet.owner)
            if 0 <= owner < players:
                scores[owner] += math.floor(float(planet.ships))
        fleet_base = game * max_fleets
        for row in range(max_fleets):
            fleet = fleets[fleet_base + row]
            owner = int(fleet.owner)
            if int(fleet.alive) and 0 <= owner < players:
                scores[owner] += math.floor(float(fleet.ships))
        if winner < 0:
            margins.append(0.0)
            continue
        winner_score = scores[winner]
        runner_up = max((score for player, score in enumerate(scores) if player != winner), default=0.0)
        margin = max(0.0, winner_score - runner_up)
        scale = max(1.0, winner_score + runner_up)
        bonus = min(0.1, 0.1 * margin / scale)
        rewards[game][winner] = 1.0 + bonus
        margins.append(margin)
    return rewards, margins


def run_generation_tournament(args: argparse.Namespace) -> None:
    model_paths = sorted(args.run_dir.glob("generation-*/model.bin"))
    if not model_paths:
        raise RuntimeError(f"no generation model bins in {args.run_dir}")
    model_list = args.run_dir / "generation_tournament_models.txt"
    telemetry = args.run_dir / "generation_tournament.json"
    model_list.write_text("\n".join(str(path) for path in model_paths) + "\n", encoding="utf-8")
    cmd = [
        str(args.arena_bin),
        "--cuda-v8",
        "--gpu-sim",
        f"--model-list={model_list}",
        f"--games-per-model={args.generation_tournament_games}",
        f"--players={args.players}",
        f"--workers={max(args.generation_tournament_games, len(model_paths))}",
        f"--output={telemetry}",
        "--replay-stride=500",
    ]
    print(json.dumps({
        "event": "generation_tournament_start",
        "models": len(model_paths),
        "games_per_model": args.generation_tournament_games,
        "cmd": cmd,
    }), flush=True)
    subprocess.run(cmd, cwd=args.arena_bin.parent.parent.parent, check=True)
    winner_index, winner_row = read_generation_tournament_winner(telemetry)
    winner_model = model_paths[winner_index]
    shutil.copy2(winner_model, args.run_dir / "model.bin")
    print(json.dumps({
        "event": "generation_tournament_complete",
        "winner_index": winner_index,
        "winner_model": str(winner_model),
        "winner": winner_row,
        "model_bin": str(args.run_dir / "model.bin"),
        "telemetry": str(telemetry),
    }), flush=True)


def read_generation_tournament_winner(path: Path) -> tuple[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("models") or []
    if not rows:
        raise RuntimeError(f"generation tournament telemetry has no models: {path}")
    best_index = 0
    best_key = None
    for index, row in enumerate(rows):
        games = max(1, int(row.get("games", 0)))
        wins = int(row.get("wins", 0))
        losses = int(row.get("losses", 0))
        draws = int(row.get("draws", 0))
        key = (wins / games, wins - losses - draws, wins, -losses)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
    return best_index, rows[best_index]


def nan_debug_payload(
    *,
    generation: int,
    batch_index: int,
    chunk_start: int,
    chunk_stop: int,
    outputs: dict[str, torch.Tensor],
    batch: ResidentBatchTensorView,
    rewards: torch.Tensor,
    loss: torch.Tensor,
) -> dict:
    return {
        "event": "gpu_resident_rl_nan_debug",
        "generation": generation,
        "batch_index": batch_index,
        "chunk_start": chunk_start,
        "chunk_stop": chunk_stop,
        "loss": scalar_debug(loss),
        "rewards": tensor_debug(rewards),
        "outputs": {name: tensor_debug(value) for name, value in outputs.items()},
        "labels": {
            "fire": tensor_debug(batch.labels_fire),
            "source": tensor_debug(batch.labels_source),
            "target": tensor_debug(batch.labels_target),
            "amount": tensor_debug(batch.labels_amount),
            "confidence": tensor_debug(batch.labels_confidence),
        },
        "loss_parts": policy_loss_parts_debug(outputs, batch, rewards),
    }


def policy_loss_parts_debug(
    outputs: dict[str, torch.Tensor],
    batch: ResidentBatchTensorView,
    rewards: torch.Tensor,
) -> dict:
    device = outputs["fire_logits"].device
    labels_fire = batch.labels_fire.to(device=device, dtype=torch.float32)
    labels_source = batch.labels_source.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=max(0, int(outputs["source_logits"].shape[-1]) - 1),
    )
    labels_target = batch.labels_target.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=max(0, int(outputs["target_logits"].shape[-1]) - 1),
    )
    labels_amount = batch.labels_amount.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=max(0, int(outputs["amount_logits"].shape[-1]) - 1),
    )
    rewards = rewards.to(device=device, dtype=torch.float32)
    if rewards.ndim == 1:
        rewards = rewards[:, None].expand_as(labels_fire)
    active = labels_fire > 0.5

    fire_logprob = -nn.functional.binary_cross_entropy_with_logits(
        outputs["fire_logits"],
        labels_fire,
        reduction="none",
    )
    parts = {
        "fire": scalar_debug(-(rewards * 0.35 * fire_logprob).mean()),
        "active_slots": int(active.sum().detach().cpu()),
    }
    if active.any():
        source_logprob = outputs["source_logits"].log_softmax(dim=-1)
        target_logprob = outputs["target_logits"].log_softmax(dim=-1)
        amount_logprob = outputs["amount_logits"].log_softmax(dim=-1)
        parts.update({
            "source": scalar_debug(-(
                rewards
                * active
                * source_logprob.gather(-1, labels_source.unsqueeze(-1)).squeeze(-1)
            ).mean()),
            "target": scalar_debug(-(
                rewards
                * active
                * target_logprob.gather(-1, labels_target.unsqueeze(-1)).squeeze(-1)
            ).mean()),
            "amount": scalar_debug(-(
                rewards
                * active
                * 0.75
                * amount_logprob.gather(-1, labels_amount.unsqueeze(-1)).squeeze(-1)
            ).mean()),
        })
    return parts


def gradient_debug_payload(model: torch.nn.Module) -> dict:
    total_sq = 0.0
    max_abs = 0.0
    bad_tensors = 0
    bad_names: list[str] = []
    tensor_count = 0
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        tensor_count += 1
        finite = torch.isfinite(grad)
        if not finite.all().item():
            bad_tensors += 1
            if len(bad_names) < 16:
                bad_names.append(name)
            finite_grad = grad[finite]
        else:
            finite_grad = grad
        if finite_grad.numel() == 0:
            continue
        total_sq += float(finite_grad.detach().float().pow(2).sum().cpu())
        max_abs = max(max_abs, float(finite_grad.detach().float().abs().max().cpu()))
    return {
        "tensors": tensor_count,
        "bad_tensors": bad_tensors,
        "bad_names": bad_names,
        "global_norm": math.sqrt(total_sq),
        "max_abs": max_abs,
    }


def tensor_debug(value: torch.Tensor) -> dict:
    detached = value.detach()
    finite = torch.isfinite(detached) if detached.is_floating_point() or detached.is_complex() else torch.ones_like(detached, dtype=torch.bool)
    finite_count = int(finite.sum().cpu())
    total = detached.numel()
    payload = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": finite_count,
        "total": total,
        "nan": int(torch.isnan(detached).sum().cpu()) if detached.is_floating_point() else 0,
        "inf": int(torch.isinf(detached).sum().cpu()) if detached.is_floating_point() else 0,
    }
    if total > 0 and finite_count > 0:
        finite_values = detached[finite].float()
        payload.update({
            "min": float(finite_values.min().cpu()),
            "max": float(finite_values.max().cpu()),
            "mean": float(finite_values.mean().cpu()),
        })
    return payload


def scalar_debug(value: torch.Tensor) -> dict:
    debug = tensor_debug(value.reshape(()))
    if debug["finite"] == 1:
        debug["value"] = float(value.detach().float().cpu())
    return debug


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
