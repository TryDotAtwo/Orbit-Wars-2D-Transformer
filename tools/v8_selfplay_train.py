#!/usr/bin/env python3
"""v8 self-play relaxation with Rust arena rollouts, PyTorch backprop, and GA.

Python stays at the orchestration boundary:
- mutate/export a small population of OWV8 model bins;
- run the native Rust arena for evaluation and compact training traces;
- turn winning self-play actions into action-slot training samples;
- backprop the selected checkpoint on GPU;
- carry the best checkpoint into the next generation.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.leaderboard_v8_dataset import ACTION_SLOTS, build_sample
from tools.v8_model import V8ActionSlotTransformer, V8ModelConfig, export_model_bin, load_model_bin_state
from tools.v8_train import collate_samples
from tools.v8_gpu_selfplay import run_gpu_selfplay


@dataclass(frozen=True)
class SelfPlayConfig:
    generations: int = 32
    population: int = 64
    elites: int = 4
    mutations_per_elite: int = 3
    mutation_std: float = 0.015
    games_per_candidate: int = 20
    players: int = 4
    replay_stride: int = 500
    gpu_status_poll_stride: int | None = None
    arena_step_limit: int | None = None
    arena_gpu_sim: bool = True
    training_trace_stride: int = 1
    backprop_epochs: int = 1
    backprop_batch_size: int = 128
    backprop_lr: float = 1.0e-5
    target_loss_weight: float = 2.0
    source_loss_weight: float = 1.0
    amount_loss_weight: float = 0.75
    fire_loss_weight: float = 0.35
    max_selfplay_samples: int | None = None

    def training_trace_capacity(self) -> int:
        if self.max_selfplay_samples is not None and self.max_selfplay_samples > 0:
            return self.max_selfplay_samples
        step_count = self.arena_step_limit or 500
        sampled_steps = max(1, math.ceil(step_count / max(1, self.training_trace_stride)))
        game_count = max(1, self.population * self.games_per_candidate // max(1, self.players))
        return game_count * self.players * sampled_steps

    def single_rl_training_trace_capacity(self) -> int:
        if self.max_selfplay_samples is not None and self.max_selfplay_samples > 0:
            return self.max_selfplay_samples
        step_count = self.arena_step_limit or 500
        sampled_steps = max(1, math.ceil(step_count / max(1, self.training_trace_stride)))
        return max(1, self.games_per_candidate) * max(1, self.players) * sampled_steps


@dataclass(frozen=True)
class CandidateResult:
    generation: int
    candidate_id: str
    parent_id: str
    role: str
    mutation_std: float
    model_bin: str
    telemetry: str
    score: float
    win_rate: float
    wins: int
    draws: int
    losses: int
    launch_actions: int
    captures: int
    fleet_hits: int
    sun_destroyed_fleets: int
    selfplay_samples: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-model-bin", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v8-selfplay-gpu"))
    parser.add_argument("--arena-bin", type=Path, default=Path("target/release/orbit-wars-arena-v8"))
    parser.add_argument("--arena-backend", choices=["gpu-python", "rust-cpu", "rust-cuda", "rust-cuda-population"], default="rust-cuda-population")
    parser.add_argument("--arena-workers", type=int, default=0, help="0 = auto; for GPU population arena this batches all generation games together")
    parser.add_argument("--arena-cpu-workers", type=int, default=1)
    parser.add_argument("--arena-gpu-sim", dest="arena_gpu_sim", action="store_true", default=True)
    parser.add_argument("--no-arena-gpu-sim", dest="arena_gpu_sim", action="store_false")
    parser.add_argument("--arena-via-docker-compose", action="store_true")
    parser.add_argument("--docker-project", default="orbitwars")
    parser.add_argument("--docker-service", default="trainer")
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=4)
    parser.add_argument("--mutations-per-elite", type=int, default=3)
    parser.add_argument("--mutation-std", type=float, default=0.015)
    parser.add_argument("--games-per-candidate", type=int, default=20)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--arena-step-limit", type=int)
    parser.add_argument("--backprop-epochs", type=int, default=1)
    parser.add_argument("--backprop-batch-size", type=int, default=128)
    parser.add_argument("--backprop-lr", type=float, default=1.0e-5)
    parser.add_argument("--target-loss-weight", type=float, default=2.0)
    parser.add_argument("--replay-stride", type=int, default=500)
    parser.add_argument("--gpu-status-poll-stride", type=int)
    parser.add_argument("--training-trace-stride", type=int, default=1)
    parser.add_argument("--max-selfplay-samples", type=int)
    parser.add_argument("--dashboard-telemetry", type=Path, default=Path("dashboard/public/telemetry/latest.json"))
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rl-single-model", action="store_true", help="Train one model from self-play outcomes only; no GA, no mutations.")
    parser.add_argument(
        "--cpu-training-trace",
        action="store_true",
        help="Debug-only: copy per-tick training samples from the arena to CPU. This is intentionally off for GPU-resident RL.",
    )
    args = parser.parse_args()
    if not args.base_checkpoint and not args.base_model_bin:
        parser.error("one of --base-checkpoint or --base-model-bin is required")

    config = SelfPlayConfig(
        generations=args.generations,
        population=args.population,
        elites=args.elites,
        mutations_per_elite=args.mutations_per_elite,
        mutation_std=args.mutation_std,
        games_per_candidate=args.games_per_candidate,
        players=args.players,
        arena_step_limit=args.arena_step_limit,
        arena_gpu_sim=args.arena_gpu_sim,
        training_trace_stride=args.training_trace_stride,
        max_selfplay_samples=args.max_selfplay_samples,
        backprop_epochs=args.backprop_epochs,
        backprop_batch_size=args.backprop_batch_size,
        backprop_lr=args.backprop_lr,
        target_loss_weight=args.target_loss_weight,
        replay_stride=args.replay_stride,
        gpu_status_poll_stride=args.gpu_status_poll_stride,
    )
    if args.rl_single_model:
        run_single_model_rl(
            base_checkpoint=args.base_checkpoint,
            base_model_bin=args.base_model_bin,
            run_dir=args.run_dir,
            arena_bin=args.arena_bin,
            arena_backend=args.arena_backend,
            arena_prefix=arena_command_prefix(args),
            arena_workers=args.arena_workers,
            arena_cpu_workers=args.arena_cpu_workers,
            config=config,
            dashboard_telemetry=None if args.no_dashboard else args.dashboard_telemetry,
            device=args.device,
            cpu_training_trace=args.cpu_training_trace,
        )
    else:
        run_selfplay(
            base_checkpoint=args.base_checkpoint,
            base_model_bin=args.base_model_bin,
            run_dir=args.run_dir,
            arena_bin=args.arena_bin,
            arena_backend=args.arena_backend,
            arena_prefix=arena_command_prefix(args),
            arena_workers=args.arena_workers,
            arena_cpu_workers=args.arena_cpu_workers,
            config=config,
            dashboard_telemetry=None if args.no_dashboard else args.dashboard_telemetry,
            device=args.device,
        )


def arena_command_prefix(args: argparse.Namespace) -> list[str]:
    if not args.arena_via_docker_compose:
        return []
    return [
        "docker",
        "compose",
        "-p",
        args.docker_project,
        "run",
        "--rm",
        args.docker_service,
    ]


def run_selfplay(
    base_checkpoint: Path | None,
    base_model_bin: Path | None,
    run_dir: Path,
    *,
    arena_bin: Path,
    arena_backend: str,
    arena_prefix: list[str],
    arena_workers: int,
    arena_cpu_workers: int,
    config: SelfPlayConfig,
    dashboard_telemetry: Path | None,
    device: str | None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_config, champion_state, optimizer_state, base_source = load_base_weights(base_checkpoint, base_model_bin)
    elite_bank: list[tuple[str, dict[str, torch.Tensor]]] = [("champion", clone_state(champion_state))]
    history: list[CandidateResult] = []
    print(json.dumps({
        "event": "selfplay_start",
        "base_source": base_source,
        "run_dir": str(run_dir),
        "arena_bin": str(arena_bin),
        "arena_backend": arena_backend,
        "arena_prefix": arena_prefix,
        "arena_workers": arena_workers,
        "arena_cpu_workers": arena_cpu_workers,
        "device": device,
        "config": asdict(config),
        "training_trace_capacity": config.training_trace_capacity(),
    }), flush=True)

    for generation in range(1, config.generations + 1):
        generation_dir = run_dir / f"generation-{generation:04d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        candidates = make_population(elite_bank, model_config, generation_dir, generation, config)
        candidate_states = {
            str(candidate["id"]): candidate["state"]
            for candidate in candidates
        }
        results: list[CandidateResult] = []
        selfplay_samples: list[dict[str, Any]] = []
        if arena_backend == "rust-cuda-population":
            telemetry_path = generation_dir / f"generation-{generation:04d}.population.telemetry.json"
            training_trace_path = generation_dir / f"generation-{generation:04d}.training_trace.jsonl"
            effective_arena_workers = arena_workers
            if effective_arena_workers <= 0:
                if config.arena_gpu_sim:
                    generation_games = math.ceil(len(candidates) * config.games_per_candidate / max(1, config.players))
                    effective_arena_workers = max(1, generation_games)
                else:
                    effective_arena_workers = 1
            run_population_arena(
                arena_bin,
                arena_prefix=arena_prefix,
                candidates=candidates,
                generation_dir=generation_dir,
                telemetry_path=telemetry_path,
                games_per_model=config.games_per_candidate,
                players=config.players,
                replay_stride=config.replay_stride,
                step_limit=config.arena_step_limit,
                workers=effective_arena_workers,
                cpu_workers=arena_cpu_workers,
                gpu_sim=config.arena_gpu_sim,
                gpu_status_poll_stride=config.gpu_status_poll_stride,
                training_trace_path=training_trace_path,
                training_trace_capacity=config.training_trace_capacity(),
                training_trace_stride=config.training_trace_stride,
            )
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            selfplay_samples.extend(
                samples_from_training_trace(training_trace_path)
            )
            for index, candidate in enumerate(candidates):
                model_row = (telemetry.get("models") or [{}])[index] if index < len(telemetry.get("models") or []) else {}
                result = candidate_result_from_model_row(
                    generation,
                    candidate,
                    telemetry_path,
                    model_row,
                    len(selfplay_samples),
                )
                results.append(result)
                print(json.dumps({"event": "candidate_complete", **asdict(result)}, ensure_ascii=False), flush=True)
        else:
            for candidate in candidates:
                telemetry_path = generation_dir / f"{candidate['id']}.telemetry.json"
                if arena_backend == "gpu-python":
                    run_gpu_candidate(
                        candidate["state"],
                        model_config,
                        telemetry_path=telemetry_path,
                        games=config.games_per_candidate,
                        players=config.players,
                        device=device,
                        seed=generation * 10_000 + len(results) * 257,
                    )
                else:
                    run_arena(
                        arena_bin,
                        arena_prefix=arena_prefix,
                        model_bin=Path(candidate["model_bin"]),
                        telemetry_path=telemetry_path,
                        games=config.games_per_candidate,
                        players=config.players,
                        replay_stride=config.replay_stride,
                        step_limit=config.arena_step_limit,
                        workers=arena_workers,
                        cuda_v8=arena_backend == "rust-cuda",
                    )
                telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
                samples = samples_from_telemetry(telemetry, f"selfplay:{generation}:{candidate['id']}")
                selfplay_samples.extend(samples)
                result = candidate_result(generation, candidate, telemetry_path, telemetry, len(samples))
                results.append(result)
                print(json.dumps({"event": "candidate_complete", **asdict(result)}, ensure_ascii=False), flush=True)

        results.sort(key=lambda row: row.score, reverse=True)
        history.extend(results)
        winner = results[0]
        champion_state = clone_state(candidate_states[winner.candidate_id])
        elite_bank = [
            (result.candidate_id, clone_state(candidate_states[result.candidate_id]))
            for result in results[: max(1, min(config.elites, len(results)))]
        ]
        if dashboard_telemetry:
            dashboard_telemetry.parent.mkdir(parents=True, exist_ok=True)
            write_dashboard_summary(
                dashboard_telemetry,
                run_dir=run_dir,
                history=history,
                generation=generation,
                winner=winner,
                backprop_metrics={"samples": 0.0, "loss": 0.0},
                config=config,
            )
        if selfplay_samples:
            champion_state, optimizer_state, backprop_metrics = backprop_on_selfplay(
                champion_state,
                optimizer_state,
                model_config,
                selfplay_samples[: config.training_trace_capacity()],
                config,
                device,
            )
        else:
            backprop_metrics = {"samples": 0, "loss": 0.0}

        checkpoint_path = save_generation_checkpoint(
            run_dir,
            generation,
            champion_state,
            optimizer_state,
            model_config,
            config,
            winner,
            backprop_metrics,
        )
        export_champion(run_dir, champion_state, model_config)
        write_history(run_dir, history, generation, winner, backprop_metrics, config)
        if dashboard_telemetry:
            write_dashboard_summary(
                dashboard_telemetry,
                run_dir=run_dir,
                history=history,
                generation=generation,
                winner=winner,
                backprop_metrics=backprop_metrics,
                config=config,
            )
        print(json.dumps({
            "event": "generation_complete",
            "generation": generation,
            "checkpoint": str(checkpoint_path),
            "winner": winner.candidate_id,
            "winner_score": winner.score,
            "backprop": backprop_metrics,
        }), flush=True)
    return run_dir / "checkpoint.pt"


def run_single_model_rl(
    base_checkpoint: Path | None,
    base_model_bin: Path | None,
    run_dir: Path,
    *,
    arena_bin: Path,
    arena_backend: str,
    arena_prefix: list[str],
    arena_workers: int,
    arena_cpu_workers: int,
    config: SelfPlayConfig,
    dashboard_telemetry: Path | None,
    device: str | None,
    cpu_training_trace: bool,
) -> Path:
    if arena_backend != "rust-cuda-population":
        raise ValueError("--rl-single-model currently requires --arena-backend=rust-cuda-population")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_config, current_state, optimizer_state, base_source = load_base_weights(base_checkpoint, base_model_bin)
    history: list[CandidateResult] = []
    trace_capacity = config.single_rl_training_trace_capacity()
    print(json.dumps({
        "event": "single_rl_start",
        "base_source": base_source,
        "run_dir": str(run_dir),
        "arena_bin": str(arena_bin),
        "arena_backend": arena_backend,
        "arena_prefix": arena_prefix,
        "arena_workers": arena_workers,
        "arena_cpu_workers": arena_cpu_workers,
        "device": device,
        "reward": {"win": 1.0, "loss": -1.0, "draw": -1.0},
        "config": asdict(config),
        "arena_games_per_generation": config.games_per_candidate,
        "cpu_training_trace": cpu_training_trace,
        "training_trace_capacity": trace_capacity if cpu_training_trace else 0,
    }), flush=True)

    for generation in range(1, config.generations + 1):
        generation_dir = run_dir / f"generation-{generation:04d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        candidates = make_single_model_players(current_state, model_config, generation_dir, generation, config.players)
        telemetry_path = generation_dir / f"generation-{generation:04d}.single_rl.telemetry.json"
        training_trace_path = generation_dir / f"generation-{generation:04d}.single_rl.training_trace.jsonl" if cpu_training_trace else None
        effective_arena_workers = arena_workers
        if effective_arena_workers <= 0:
            effective_arena_workers = max(1, config.games_per_candidate)
        started = time.perf_counter()
        run_population_arena(
            arena_bin,
            arena_prefix=arena_prefix,
            candidates=candidates,
            generation_dir=generation_dir,
            telemetry_path=telemetry_path,
            games_per_model=config.games_per_candidate,
            players=config.players,
            replay_stride=config.replay_stride,
            step_limit=config.arena_step_limit,
            workers=effective_arena_workers,
            cpu_workers=arena_cpu_workers,
            gpu_sim=config.arena_gpu_sim,
            gpu_status_poll_stride=config.gpu_status_poll_stride or (config.arena_step_limit or 500),
            training_trace_path=training_trace_path,
            training_trace_capacity=trace_capacity if cpu_training_trace else None,
            training_trace_stride=config.training_trace_stride if cpu_training_trace else None,
        )
        arena_seconds = time.perf_counter() - started
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        selfplay_samples = samples_from_training_trace(training_trace_path) if training_trace_path else []
        aggregate = single_rl_candidate_result(
            generation,
            telemetry_path,
            telemetry,
            len(selfplay_samples),
        )
        history.append(aggregate)
        print(json.dumps({
            "event": "single_rl_arena_complete",
            "generation": generation,
            "games": config.games_per_candidate,
            "players": config.players,
            "seconds": round(arena_seconds, 3),
            "games_per_second": round(config.games_per_candidate / max(1.0e-9, arena_seconds), 3),
            "samples": len(selfplay_samples),
            "result": asdict(aggregate),
        }, ensure_ascii=False), flush=True)

        if selfplay_samples:
            current_state, optimizer_state, backprop_metrics = backprop_on_selfplay(
                current_state,
                optimizer_state,
                model_config,
                selfplay_samples,
                config,
                device,
            )
        else:
            backprop_metrics = {
                "samples": 0,
                "loss": 0.0,
                "gpu_resident_rollout": True,
                "cpu_training_trace": False,
                "note": "CPU trace disabled; device-side RL loss/backprop is required for true GPU-only training.",
            }
        backprop_metrics["arena_seconds"] = float(arena_seconds)
        backprop_metrics["arena_games_per_second"] = config.games_per_candidate / max(1.0e-9, arena_seconds)

        checkpoint_path = save_generation_checkpoint(
            run_dir,
            generation,
            current_state,
            optimizer_state,
            model_config,
            config,
            aggregate,
            backprop_metrics,
        )
        export_champion(run_dir, current_state, model_config)
        write_history(run_dir, history, generation, aggregate, backprop_metrics, config)
        if dashboard_telemetry:
            dashboard_telemetry.parent.mkdir(parents=True, exist_ok=True)
            write_dashboard_summary(
                dashboard_telemetry,
                run_dir=run_dir,
                history=history,
                generation=generation,
                winner=aggregate,
                backprop_metrics=backprop_metrics,
                config=config,
            )
        print(json.dumps({
            "event": "single_rl_generation_complete",
            "generation": generation,
            "checkpoint": str(checkpoint_path),
            "model_bin": str(run_dir / "model.bin"),
            "backprop": backprop_metrics,
            "result": asdict(aggregate),
        }, ensure_ascii=False), flush=True)
    return run_dir / "checkpoint.pt"


def load_base_weights(
    base_checkpoint: Path | None,
    base_model_bin: Path | None,
) -> tuple[V8ModelConfig, dict[str, torch.Tensor], dict[str, Any] | None, str]:
    if base_checkpoint:
        payload = torch.load(base_checkpoint, map_location="cpu")
        config = V8ModelConfig(**payload["config"])
        state = {key: value.detach().cpu().clone() for key, value in payload["model"].items()}
        return config, state, payload.get("optimizer"), str(base_checkpoint)
    assert base_model_bin is not None
    config, state = load_model_bin_state(base_model_bin)
    return config, state, None, str(base_model_bin)


def make_population(
    elite_bank: list[tuple[str, dict[str, torch.Tensor]]],
    model_config: V8ModelConfig,
    generation_dir: Path,
    generation: int,
    config: SelfPlayConfig,
) -> list[dict[str, Any]]:
    candidates = []
    elite_count = max(1, min(config.elites, len(elite_bank), config.population))
    for index, (parent_id, parent_state) in enumerate(elite_bank[:elite_count]):
        state = clone_state(parent_state)
        model_bin = generation_dir / f"candidate-{index:03d}.bin"
        export_state_dict(state, model_config, model_bin)
        candidates.append({
            "id": f"g{generation:04d}-m{index:03d}",
            "parent": parent_id,
            "role": "elite",
            "mutation_std": 0.0,
            "model_bin": str(model_bin),
            "state": state,
        })
    index = len(candidates)
    mutation_round = 0
    while index < config.population:
        for parent_slot, (parent_id, parent_state) in enumerate(elite_bank[:elite_count]):
            if index >= config.population:
                break
            seed = generation * 100_000 + index * 9973 + mutation_round * 31
            if elite_count >= 2 and mutation_round % 2 == 1:
                other_slot = (parent_slot + mutation_round) % elite_count
                if other_slot == parent_slot:
                    other_slot = (other_slot + 1) % elite_count
                other_id, other_state = elite_bank[other_slot]
                state = crossover_state(parent_state, other_state, config.mutation_std, seed)
                parent_label = f"{parent_id}+{other_id}"
                role = "crossover_mutation"
            else:
                state = mutate_state(parent_state, config.mutation_std, seed)
                parent_label = parent_id
                role = "mutation"
            model_bin = generation_dir / f"candidate-{index:03d}.bin"
            export_state_dict(state, model_config, model_bin)
            candidates.append({
                "id": f"g{generation:04d}-m{index:03d}",
                "parent": parent_label,
                "role": role,
                "mutation_std": config.mutation_std,
                "model_bin": str(model_bin),
                "state": state,
            })
            index += 1
        mutation_round += 1
    return candidates


def make_single_model_players(
    state: dict[str, torch.Tensor],
    model_config: V8ModelConfig,
    generation_dir: Path,
    generation: int,
    players: int,
) -> list[dict[str, Any]]:
    candidates = []
    player_count = max(1, players)
    for index in range(player_count):
        model_bin = generation_dir / f"single-player-{index:03d}.bin"
        player_state = clone_state(state)
        export_state_dict(player_state, model_config, model_bin)
        candidates.append({
            "id": f"g{generation:04d}-self-{index:03d}",
            "parent": "single_model",
            "role": "self_copy",
            "mutation_std": 0.0,
            "model_bin": str(model_bin),
            "state": player_state,
        })
    return candidates


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def mutate_state(state: dict[str, torch.Tensor], mutation_std: float, seed: int) -> dict[str, torch.Tensor]:
    if mutation_std <= 0:
        return {key: value.clone() for key, value in state.items()}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mutated: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        tensor = value.detach().cpu().clone()
        if tensor.is_floating_point():
            scale = tensor.float().std(unbiased=False).item()
            if not math.isfinite(scale) or scale <= 1.0e-6:
                scale = tensor.float().abs().mean().item()
            if not math.isfinite(scale) or scale <= 1.0e-6:
                scale = 1.0
            noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * (mutation_std * scale)
            tensor = tensor + noise
        mutated[key] = tensor
    return mutated


def crossover_state(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    mutation_std: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    crossed: dict[str, torch.Tensor] = {}
    for key, left_value in left.items():
        left_tensor = left_value.detach().cpu()
        right_tensor = right.get(key, left_value).detach().cpu()
        if left_tensor.shape != right_tensor.shape:
            crossed[key] = left_tensor.clone()
            continue
        if left_tensor.is_floating_point():
            mask = torch.rand(left_tensor.shape, generator=generator) < 0.5
            tensor = torch.where(mask, left_tensor, right_tensor).clone()
        else:
            tensor = left_tensor.clone()
        crossed[key] = tensor
    return mutate_state(crossed, mutation_std, seed + 17)


def export_state_dict(state: dict[str, torch.Tensor], config: V8ModelConfig, path: Path) -> None:
    model = V8ActionSlotTransformer(config)
    model.load_state_dict(state, strict=False)
    model.eval()
    export_model_bin(model, config, path)


def run_gpu_candidate(
    state: dict[str, torch.Tensor],
    config: V8ModelConfig,
    *,
    telemetry_path: Path,
    games: int,
    players: int,
    device: str,
    seed: int,
) -> None:
    started = time.perf_counter()
    model = V8ActionSlotTransformer(config)
    model.load_state_dict(state, strict=False)
    run_gpu_selfplay(
        model,
        output_path=telemetry_path,
        games=games,
        players=players,
        batch_games=games,
        seed=seed,
        device=device,
    )
    print(json.dumps({
        "event": "gpu_arena_eval",
        "telemetry": str(telemetry_path),
        "games": games,
        "players": players,
        "seconds": round(time.perf_counter() - started, 3),
    }), flush=True)


def run_arena(
    arena_bin: Path,
    *,
    arena_prefix: list[str],
    model_bin: Path,
    telemetry_path: Path,
    games: int,
    players: int,
    replay_stride: int,
    step_limit: int | None,
    workers: int,
    cuda_v8: bool,
) -> None:
    arena_arg = command_path(arena_bin, docker_style=bool(arena_prefix))
    model_arg = command_path(model_bin, docker_style=bool(arena_prefix))
    telemetry_arg = command_path(telemetry_path, docker_style=bool(arena_prefix))
    command = [*arena_prefix,
        arena_arg,
        f"--games={games}",
        f"--players={players}",
        f"--replay-stride={replay_stride}",
        f"--workers={workers}",
        f"--model={model_arg}",
        f"--output={telemetry_arg}",
    ]
    if cuda_v8:
        command.append("--cuda-v8")
    if step_limit is not None:
        command.append(f"--step-limit={step_limit}")
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"arena_failed rc={completed.returncode} cmd={command} output={completed.stdout[-4000:]}")
    print(json.dumps({
        "event": "arena_eval",
        "model": str(model_bin),
        "telemetry": str(telemetry_path),
        "seconds": round(time.perf_counter() - started, 3),
    }), flush=True)


def run_population_arena(
    arena_bin: Path,
    *,
    arena_prefix: list[str],
    candidates: list[dict[str, Any]],
    generation_dir: Path,
    telemetry_path: Path,
    games_per_model: int,
    players: int,
    replay_stride: int,
    step_limit: int | None,
    workers: int,
    cpu_workers: int,
    gpu_sim: bool,
    gpu_status_poll_stride: int | None,
    training_trace_path: Path | None,
    training_trace_capacity: int | None,
    training_trace_stride: int | None,
) -> None:
    model_list = generation_dir / "models.txt"
    model_list.write_text(
        "\n".join(Path(str(candidate["model_bin"])).name for candidate in candidates) + "\n",
        encoding="utf-8",
    )
    arena_arg = command_path(arena_bin, docker_style=bool(arena_prefix))
    model_list_arg = command_path(model_list, docker_style=bool(arena_prefix))
    telemetry_arg = command_path(telemetry_path, docker_style=bool(arena_prefix))
    command = [
        *arena_prefix,
        arena_arg,
        "--cuda-v8",
        f"--model-list={model_list_arg}",
        f"--games-per-model={games_per_model}",
        f"--players={players}",
        f"--replay-stride={replay_stride}",
        f"--workers={workers}",
        f"--cpu-workers={cpu_workers}",
        f"--output={telemetry_arg}",
    ]
    if gpu_status_poll_stride is not None:
        command.append(f"--gpu-status-poll-stride={gpu_status_poll_stride}")
    if training_trace_path is not None:
        training_trace_arg = command_path(training_trace_path, docker_style=bool(arena_prefix))
        command.extend([
            f"--training-trace={training_trace_arg}",
            f"--training-trace-capacity={training_trace_capacity}",
            f"--training-trace-stride={training_trace_stride}",
        ])
    if step_limit is not None:
        command.append(f"--step-limit={step_limit}")
    if gpu_sim:
        command.append("--gpu-sim")
    started = time.perf_counter()
    arena_lines: list[str] = []
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        arena_lines.append(line)
        if len(arena_lines) > 20:
            arena_lines = arena_lines[-20:]
        print(line, flush=True)
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"population_arena_failed rc={returncode} cmd={command} output={chr(10).join(arena_lines[-20:])}")
    print(json.dumps({
        "event": "population_arena_eval",
        "models": len(candidates),
        "games_per_model": games_per_model,
        "players": players,
        "workers": workers,
        "cpu_workers": cpu_workers,
        "gpu_sim": gpu_sim,
        "telemetry": str(telemetry_path),
        "training_trace": str(training_trace_path) if training_trace_path else None,
        "seconds": round(time.perf_counter() - started, 3),
        "arena_tail": arena_lines[-3:],
    }, ensure_ascii=False), flush=True)


def command_path(path: Path, *, docker_style: bool) -> str:
    if not docker_style:
        return str(path)
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved.as_posix()
    return "/workspace/" + relative.as_posix()


def candidate_result(
    generation: int,
    candidate: dict[str, Any],
    telemetry_path: Path,
    telemetry: dict[str, Any],
    sample_count: int,
) -> CandidateResult:
    models = telemetry.get("models") or []
    selected = models[0] if models else {}
    latest = (telemetry.get("metrics") or [{}])[-1]
    wins = int(selected.get("wins", 0))
    draws = int(selected.get("draws", 0))
    losses = int(selected.get("losses", 0))
    captures = int(selected.get("captures", latest.get("captures", 0)))
    fleet_hits = int(selected.get("fleetHits", latest.get("fleetHits", 0)))
    launch_actions = int(selected.get("launchActions", latest.get("launchActions", 0)))
    sun = int(selected.get("sunDestroyedFleets", latest.get("sunDestroyedFleets", 0)))
    win_rate = float(latest.get("winRate", 0.0))
    score = win_rate * 1000.0 + captures * 2.0 + fleet_hits * 0.25 + launch_actions * 0.02 - sun * 3.0
    return CandidateResult(
        generation=generation,
        candidate_id=str(candidate["id"]),
        parent_id=str(candidate["parent"]),
        role=str(candidate["role"]),
        mutation_std=float(candidate["mutation_std"]),
        model_bin=str(candidate["model_bin"]),
        telemetry=str(telemetry_path),
        score=float(score),
        win_rate=win_rate,
        wins=wins,
        draws=draws,
        losses=losses,
        launch_actions=launch_actions,
        captures=captures,
        fleet_hits=fleet_hits,
        sun_destroyed_fleets=sun,
        selfplay_samples=sample_count,
    )


def candidate_result_from_model_row(
    generation: int,
    candidate: dict[str, Any],
    telemetry_path: Path,
    model_row: dict[str, Any],
    sample_count: int,
) -> CandidateResult:
    wins = int(model_row.get("wins", 0))
    draws = int(model_row.get("draws", 0))
    losses = int(model_row.get("losses", 0))
    games = max(1, wins + draws + losses)
    captures = int(model_row.get("captures", 0))
    fleet_hits = int(model_row.get("fleetHits", 0))
    launch_actions = int(model_row.get("launchActions", 0))
    sun = int(model_row.get("sunDestroyedFleets", 0))
    win_rate = wins / games
    score = win_rate * 1000.0 + captures * 2.0 + fleet_hits * 0.25 + launch_actions * 0.02 - sun * 3.0
    return CandidateResult(
        generation=generation,
        candidate_id=str(candidate["id"]),
        parent_id=str(candidate["parent"]),
        role=str(candidate["role"]),
        mutation_std=float(candidate["mutation_std"]),
        model_bin=str(candidate["model_bin"]),
        telemetry=str(telemetry_path),
        score=float(score),
        win_rate=float(win_rate),
        wins=wins,
        draws=draws,
        losses=losses,
        launch_actions=launch_actions,
        captures=captures,
        fleet_hits=fleet_hits,
        sun_destroyed_fleets=sun,
        selfplay_samples=sample_count,
    )


def single_rl_candidate_result(
    generation: int,
    telemetry_path: Path,
    telemetry: dict[str, Any],
    sample_count: int,
) -> CandidateResult:
    models = telemetry.get("models") or []
    wins = sum(int(model.get("wins", 0)) for model in models)
    draws = sum(int(model.get("draws", 0)) for model in models)
    losses = sum(int(model.get("losses", 0)) for model in models)
    games = max(1, wins + draws + losses)
    captures = sum(int(model.get("captures", 0)) for model in models)
    fleet_hits = sum(int(model.get("fleetHits", 0)) for model in models)
    launch_actions = sum(int(model.get("launchActions", 0)) for model in models)
    sun = sum(int(model.get("sunDestroyedFleets", 0)) for model in models)
    win_rate = wins / games
    # Selection is irrelevant in single-RL, but score remains useful in history/dashboard.
    score = (wins - losses - draws) / games
    return CandidateResult(
        generation=generation,
        candidate_id=f"single-rl-g{generation:04d}",
        parent_id="single_model",
        role="single_rl",
        mutation_std=0.0,
        model_bin=str(telemetry_path.parent / "single-player-000.bin"),
        telemetry=str(telemetry_path),
        score=float(score),
        win_rate=float(win_rate),
        wins=wins,
        draws=draws,
        losses=losses,
        launch_actions=launch_actions,
        captures=captures,
        fleet_hits=fleet_hits,
        sun_destroyed_fleets=sun,
        selfplay_samples=sample_count,
    )


def samples_from_population_telemetry(telemetry: dict[str, Any], source_file: str) -> list[dict[str, Any]]:
    models = telemetry.get("models") or []
    winner_model_ids = {
        index
        for index, model in enumerate(models)
        if int(model.get("wins", 0)) > 0
    }
    if not winner_model_ids:
        return []
    samples: list[dict[str, Any]] = []
    for replay_index, frames in telemetry_frame_sets(telemetry):
        if not frames:
            continue
        initial_planets = frame_planet_rows(frames[0])
        for frame in frames:
            actions_by_player = frame.get("actions") or []
            model_ids = frame.get("modelIds") or []
            for player, model_id in enumerate(model_ids):
                if int(model_id) not in winner_model_ids:
                    continue
                if player >= len(actions_by_player) or not actions_by_player[player]:
                    continue
                observation = {
                    "player": player,
                    "step": int(frame.get("step", 0)),
                    "angular_velocity": float(telemetry.get("angularVelocity", 0.03)),
                    "planets": frame_planet_rows(frame),
                    "initial_planets": initial_planets,
                    "fleets": frame_fleet_rows(frame),
                }
                actions = [action_row(action) for action in actions_by_player[player]]
                sample = build_sample(observation, actions, f"{source_file}:replay:{replay_index}", target_traces=None)
                if int(sample["labels_fire"].sum()) > 0:
                    samples.append(sample)
    return samples


def samples_from_training_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    samples: list[dict[str, Any]] = []
    token_count = 1 + 64 + 640
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "sample":
            continue
        labels_fire = np.asarray(row["labels_fire"], dtype=np.float32)
        if int(labels_fire.sum()) <= 0:
            continue
        tokens = np.asarray(row["tokens"], dtype=np.float32).reshape(token_count, 14)
        padding = np.asarray(row["padding_mask"], dtype=np.bool_)
        valid_length = int(np.argmax(padding)) if padding.any() else token_count
        if valid_length <= 0:
            valid_length = token_count
        samples.append({
            "tokens": tokens[:valid_length],
            "token_type_ids": np.asarray(row["token_type_ids"], dtype=np.int64)[:valid_length],
            "owner_ids": np.asarray(row["owner_ids"], dtype=np.int64)[:valid_length],
            "sample_offsets": np.asarray([0, valid_length], dtype=np.int64),
            "planet_mask": np.asarray(row["planet_mask"], dtype=np.bool_),
            "labels_fire": labels_fire,
            "labels_source": np.asarray(row["labels_source"], dtype=np.int64),
            "labels_target": np.asarray(row["labels_target"], dtype=np.int64),
            "labels_amount": np.asarray(row["labels_amount"], dtype=np.int64),
            "outcome_reward": float(row.get("outcome_reward", 1.0)),
            "episode_id": f"selfplay-trace-{row.get('game', 0)}",
            "step": int(row.get("step", 0)),
            "player": int(row.get("player", 0)),
        })
    return samples


def samples_from_telemetry(telemetry: dict[str, Any], source_file: str) -> list[dict[str, Any]]:
    models = telemetry.get("models") or []
    winner_slots = [index for index, model in enumerate(models) if int(model.get("wins", 0)) > 0]
    if not winner_slots:
        winner_slots = [0]
    samples: list[dict[str, Any]] = []
    for replay_index, frames in telemetry_frame_sets(telemetry):
        if not frames:
            continue
        initial_planets = frame_planet_rows(frames[0])
        for frame in frames:
            actions_by_player = frame.get("actions") or []
            for player in winner_slots:
                if player >= len(actions_by_player) or not actions_by_player[player]:
                    continue
                observation = {
                    "player": player,
                    "step": int(frame.get("step", 0)),
                    "angular_velocity": float(telemetry.get("angularVelocity", 0.03)),
                    "planets": frame_planet_rows(frame),
                    "initial_planets": initial_planets,
                    "fleets": frame_fleet_rows(frame),
                }
                actions = [action_row(action) for action in actions_by_player[player]]
                sample = build_sample(observation, actions, f"{source_file}:replay:{replay_index}", target_traces=None)
                if int(sample["labels_fire"].sum()) > 0:
                    samples.append(sample)
    return samples


def telemetry_frame_sets(telemetry: dict[str, Any]) -> list[tuple[int, list[dict[str, Any]]]]:
    replay_games = telemetry.get("replayGames") or []
    frame_sets = [
        (index, list(game.get("frames") or []))
        for index, game in enumerate(replay_games)
        if game.get("frames")
    ]
    if frame_sets:
        return frame_sets
    return [(0, list(telemetry.get("frames") or []))]


def action_row(action: Any) -> list[float]:
    if isinstance(action, dict):
        return [action["source"], action["angle"], action["ships"]]
    return [action[0], action[1], action[2]]


def frame_planet_rows(frame: dict[str, Any]) -> list[list[float]]:
    return [
        [
            planet.get("id", 0),
            planet.get("owner", -1),
            planet.get("x", 0.0),
            planet.get("y", 0.0),
            planet.get("radius", 1.0),
            planet.get("ships", 0.0),
            planet.get("production", 0.0),
            planet.get("velocity_x", 0.0),
            planet.get("velocity_y", 0.0),
        ]
        for planet in frame.get("planets", [])
    ]


def frame_fleet_rows(frame: dict[str, Any]) -> list[list[float]]:
    return [
        [
            fleet.get("id", 0),
            fleet.get("owner", -1),
            fleet.get("x", 0.0),
            fleet.get("y", 0.0),
            fleet.get("angle", 0.0),
            fleet.get("from_planet_id", -1),
            fleet.get("ships", 0.0),
        ]
        for fleet in frame.get("fleets", [])
    ]


def backprop_on_selfplay(
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any] | None,
    model_config: V8ModelConfig,
    samples: list[dict[str, Any]],
    config: SelfPlayConfig,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float]]:
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    model = V8ActionSlotTransformer(model_config).to(device)
    model.load_state_dict(state, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.backprop_lr, weight_decay=0.01)
    use_bf16 = device.startswith("cuda")
    if optimizer_state:
        try:
            optimizer.load_state_dict(optimizer_state)
            move_optimizer_state(optimizer, device)
        except ValueError:
            pass
    total_loss = 0.0
    steps = 0
    model.train()
    for _epoch in range(config.backprop_epochs):
        for start in range(0, len(samples), config.backprop_batch_size):
            batch_samples = samples[start:start + config.backprop_batch_size]
            batch = {key: value.to(device) for key, value in collate_samples(batch_samples).items()}
            outcome_rewards = torch.as_tensor(
                [float(sample.get("outcome_reward", 1.0)) for sample in batch_samples],
                dtype=torch.float32,
                device=device,
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(
                    batch["tokens"],
                    batch["token_type_ids"],
                    batch["owner_ids"],
                    padding_mask=batch["padding_mask"],
                    planet_mask=batch["planet_mask"],
                )
                loss, parts = weighted_action_loss(outputs, batch, config, outcome_rewards)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
    new_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    metrics = {
        "samples": float(len(samples)),
        "steps": float(steps),
        "loss": total_loss / max(1, steps),
    }
    metrics.update(parts if steps else {})
    return new_state, optimizer.state_dict(), metrics


def weighted_action_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: SelfPlayConfig,
    outcome_rewards: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    fire_slot_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["fire_logits"],
        batch["labels_fire"].float(),
        reduction="none",
    )
    fire_loss_by_sample = fire_slot_loss.float().mean(dim=1)
    active = batch["labels_fire"].bool()
    batch_size = int(batch["labels_fire"].shape[0])
    source_loss_by_sample = torch.zeros(batch_size, dtype=torch.float32, device=outputs["fire_logits"].device)
    target_loss_by_sample = torch.zeros_like(source_loss_by_sample)
    amount_loss_by_sample = torch.zeros_like(source_loss_by_sample)
    if active.any():
        active_rows = active.nonzero(as_tuple=False)[:, 0]
        source_slot_loss = nn.functional.cross_entropy(
            outputs["source_logits"][active],
            batch["labels_source"][active],
            reduction="none",
        )
        target_slot_loss = nn.functional.cross_entropy(
            outputs["target_logits"][active],
            batch["labels_target"][active],
            reduction="none",
        )
        amount_slot_loss = nn.functional.cross_entropy(
            outputs["amount_logits"][active],
            batch["labels_amount"][active],
            reduction="none",
        )
        source_slot_loss = source_slot_loss.float()
        target_slot_loss = target_slot_loss.float()
        amount_slot_loss = amount_slot_loss.float()
        active_counts = torch.zeros(batch_size, dtype=torch.float32, device=outputs["fire_logits"].device)
        active_counts.index_add_(0, active_rows, torch.ones_like(source_slot_loss))
        active_counts = active_counts.clamp_min(1.0)
        source_loss_by_sample.index_add_(0, active_rows, source_slot_loss)
        target_loss_by_sample.index_add_(0, active_rows, target_slot_loss)
        amount_loss_by_sample.index_add_(0, active_rows, amount_slot_loss)
        source_loss_by_sample = source_loss_by_sample / active_counts
        target_loss_by_sample = target_loss_by_sample / active_counts
        amount_loss_by_sample = amount_loss_by_sample / active_counts
    else:
        zero = outputs["fire_logits"].sum() * 0.0
        source_loss_by_sample = source_loss_by_sample + zero
        target_loss_by_sample = target_loss_by_sample + zero
        amount_loss_by_sample = amount_loss_by_sample + zero
    loss_by_sample = (
        fire_loss_by_sample * config.fire_loss_weight
        + source_loss_by_sample * config.source_loss_weight
        + target_loss_by_sample * config.target_loss_weight
        + amount_loss_by_sample * config.amount_loss_weight
    )
    if outcome_rewards is None:
        outcome_rewards = torch.ones_like(loss_by_sample)
    outcome_rewards = outcome_rewards.to(loss_by_sample.device, dtype=loss_by_sample.dtype)
    loss = (loss_by_sample * outcome_rewards).mean()
    return loss, {
        "fire_loss": float(fire_loss_by_sample.mean().detach().cpu()),
        "source_loss": float(source_loss_by_sample.mean().detach().cpu()),
        "target_loss": float(target_loss_by_sample.mean().detach().cpu()),
        "amount_loss": float(amount_loss_by_sample.mean().detach().cpu()),
        "positive_samples": float((outcome_rewards > 0).sum().detach().cpu()),
        "negative_samples": float((outcome_rewards < 0).sum().detach().cpu()),
    }


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(target)


def save_generation_checkpoint(
    run_dir: Path,
    generation: int,
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any] | None,
    model_config: V8ModelConfig,
    config: SelfPlayConfig,
    winner: CandidateResult,
    backprop_metrics: dict[str, float],
) -> Path:
    payload = {
        "config": model_config.__dict__,
        "model": state,
        "optimizer": optimizer_state,
        "selfplay_config": asdict(config),
        "winner": asdict(winner),
        "metrics": backprop_metrics,
        "epoch_completed": generation - 1,
    }
    checkpoint = run_dir / f"checkpoint-generation-{generation:04d}.pt"
    torch.save(payload, checkpoint)
    torch.save(payload, run_dir / "checkpoint.pt")
    return checkpoint


def export_champion(run_dir: Path, state: dict[str, torch.Tensor], config: V8ModelConfig) -> Path:
    path = run_dir / "model.bin"
    export_state_dict(state, config, path)
    return path


def write_history(
    run_dir: Path,
    history: list[CandidateResult],
    generation: int,
    winner: CandidateResult,
    backprop_metrics: dict[str, float],
    config: SelfPlayConfig,
) -> None:
    path = run_dir / "selfplay_history.json"
    payload = {
        "generation": generation,
        "winner": asdict(winner),
        "backprop": backprop_metrics,
        "config": asdict(config),
        "candidates": [asdict(row) for row in history],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_dashboard_summary(
    path: Path,
    *,
    run_dir: Path,
    history: list[CandidateResult],
    generation: int,
    winner: CandidateResult,
    backprop_metrics: dict[str, float],
    config: SelfPlayConfig,
) -> None:
    source = json.loads(Path(winner.telemetry).read_text(encoding="utf-8"))
    latest_by_generation: list[dict[str, Any]] = []
    for gen in range(1, generation + 1):
        rows = [row for row in history if row.generation == gen]
        if not rows:
            continue
        best = max(rows, key=lambda row: row.score)
        evaluated_games = sum(row.wins + row.draws + row.losses for row in rows)
        latest_by_generation.append({
            "generation": gen,
            "winRate": best.win_rate,
            "gamesPerSecond": (source.get("gamesPerSecond") or 0.0),
            "turnsPerSecond": (source.get("turnsPerSecond") or 0.0),
            "p95LatencyMs": 0.0,
            "gpuUtilization": 0,
            "evaluatedGames": evaluated_games,
            "sampledReplayGames": min(4, len(source.get("replayGames") or [])),
            "modelActionCalls": (source.get("metrics") or [{}])[-1].get("modelActionCalls", 0),
            "launchActions": sum(row.launch_actions for row in rows),
            "launchedShips": (source.get("metrics") or [{}])[-1].get("launchedShips", 0),
            "captures": sum(row.captures for row in rows),
            "fleetHits": sum(row.fleet_hits for row in rows),
            "hitShips": (source.get("metrics") or [{}])[-1].get("hitShips", 0),
            "sunDestroyedFleets": sum(row.sun_destroyed_fleets for row in rows),
            "sunDestroyedShips": (source.get("metrics") or [{}])[-1].get("sunDestroyedShips", 0),
            "avgFleetSize": 0.0,
            "avgLaunchActionsPerTurn": 0.0,
            "avgLaunchedShipsPerTurn": 0.0,
            "avgModelActionMs": 0.0,
            "inferenceBatchCalls": (source.get("metrics") or [{}])[-1].get("inferenceBatchCalls", 0),
            "maxInferenceBatchSize": (source.get("metrics") or [{}])[-1].get("maxInferenceBatchSize", 0),
            "simultaneousGames": config.population,
            "modelActionSeconds": 0.0,
            "simulationStepSeconds": 0.0,
            "evaluationSeconds": 0.0,
            "replaySeconds": 0.0,
            "replayWriteSeconds": 0.0,
            "generationValidationGames": evaluated_games,
            "generationValidationSeconds": 0.0,
            "backpropSamples": backprop_metrics.get("samples", 0.0) if gen == generation else 0.0,
            "backpropModels": 1 if gen == generation and backprop_metrics.get("samples", 0.0) else 0,
            "backpropSeconds": 0.0,
            "reproductionSeconds": 0.0,
            "generationSeconds": 0.0,
        })
    source["runId"] = run_dir.name
    source["activeGeneration"] = generation
    source["populationSize"] = config.population
    source["eliteCount"] = config.elites
    source["gamesPerModel"] = config.games_per_candidate
    source["replaysPerModel"] = 4
    source["generationReplayGameCount"] = 4
    source["storedReplayGames"] = min(4, len(source.get("replayGames") or []))
    source["metrics"] = latest_by_generation
    source["generationWinRates"] = [
        {
            "validationGeneration": row.generation,
            "evaluatedGeneration": row.generation,
            "modelCount": config.population,
            "games": row.wins + row.draws + row.losses,
            "wins": row.wins,
            "draws": row.draws,
            "losses": row.losses,
            "winRate": row.win_rate,
        }
        for row in history
    ]
    source["models"] = [
        {
            "id": row.candidate_id,
            "parent": row.parent_id,
            "rating": int(600 + row.score),
            "wins": row.wins,
            "draws": row.draws,
            "losses": row.losses,
            "games": row.wins + row.draws + row.losses,
            "captures": row.captures,
            "fleetHits": row.fleet_hits,
            "hitShips": 0,
            "sunDestroyedFleets": row.sun_destroyed_fleets,
            "sunDestroyedShips": 0,
            "outOfBoundsFleets": 0,
            "outOfBoundsShips": 0,
            "launchActions": row.launch_actions,
            "launchedShips": 0,
            "avgFleetSize": 0.0,
            "mutation": row.role,
            "selected": row.candidate_id == winner.candidate_id,
        }
        for row in sorted([row for row in history if row.generation == generation], key=lambda item: item.score, reverse=True)
    ]
    path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
