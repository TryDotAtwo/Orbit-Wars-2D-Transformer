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
import tempfile
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
    parser.add_argument("--baseline-model-bin", type=Path, default=None)
    parser.add_argument("--cuda-lib", type=Path, default=Path("target/liborbit_wars_v8_cuda.so"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v8-gpu-resident-rl"))
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--train-batch-requests", type=int, default=512)
    parser.add_argument("--generation-tournament-games", type=int, default=0)
    parser.add_argument("--baseline-players", type=int, default=2)
    parser.add_argument("--baseline-eval-interval", type=int, default=1)
    parser.add_argument("--external-baseline-eval-interval", type=int, default=0)
    parser.add_argument("--external-baseline-eval-games", type=int, default=0)
    parser.add_argument("--external-exp50-main", type=Path, default=None)
    parser.add_argument("--external-candidate-vs-exp50-only", action="store_true")
    parser.add_argument("--external-env-src", type=Path, default=Path(".external/kaggle-env-src/kaggle_environments/envs/orbit_wars"))
    parser.add_argument("--submission-template-dir", type=Path, default=Path("kaggle_submission"))
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=1)
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
    model.load_state_dict(state, strict=False)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    baseline_config = None
    baseline_state = None
    baseline_model_bin = args.baseline_model_bin
    external_eval_requested = args.external_baseline_eval_interval > 0 and args.external_baseline_eval_games > 0
    if baseline_model_bin is None and (args.baseline_players > 0 or external_eval_requested):
        baseline_model_bin = args.base_model_bin
    if baseline_model_bin is not None and (args.baseline_players > 0 or external_eval_requested):
        baseline_config, baseline_state = load_model_bin_state(baseline_model_bin)
        if baseline_config != config:
            raise ValueError(f"baseline config mismatch: {baseline_config} != {config}")
    runtime = V8CudaRuntime(args.cuda_lib)
    runtime.status()
    print(json.dumps({
        "event": "gpu_resident_rl_start",
        "base_model": str(args.base_model_bin),
        "baseline_model": str(baseline_model_bin) if baseline_model_bin else None,
        "cuda_lib": str(args.cuda_lib),
        "run_dir": str(args.run_dir),
        "generations": args.generations,
        "games": args.games,
        "players": args.players,
        "steps": args.steps,
        "lr": args.lr,
        "train_batch_requests": args.train_batch_requests,
        "generation_tournament_games": args.generation_tournament_games,
        "baseline_players": args.baseline_players,
        "baseline_eval_interval": args.baseline_eval_interval,
        "external_baseline_eval_interval": args.external_baseline_eval_interval,
        "external_baseline_eval_games": args.external_baseline_eval_games,
        "external_exp50_main": str(args.external_exp50_main) if args.external_exp50_main else None,
        "external_candidate_vs_exp50_only": args.external_candidate_vs_exp50_only,
        "ppo_clip": args.ppo_clip,
        "ppo_epochs": 1,
        "requested_ppo_epochs_ignored": args.ppo_epochs,
        "debug_nan": args.debug_nan,
        "bf16": args.bf16,
    }), flush=True)

    for generation in range(1, args.generations + 1):
        started = time.perf_counter()
        native_model = runtime.create_model_from_state({key: value.detach().cpu() for key, value in model.state_dict().items()})
        native_baseline_model = None
        baseline_eval_interval = max(1, int(args.baseline_eval_interval))
        baseline_eval_active = baseline_state is not None and generation % baseline_eval_interval == 0
        if baseline_eval_active:
            native_baseline_model = runtime.create_model_from_state(baseline_state)
        sim = runtime.create_sim(games=args.games, players=args.players, step_limit=args.steps)
        runtime.load_official_like_games(sim, generation=generation)
        baseline_player_count = min(max(0, int(args.baseline_players)), max(0, args.players - 1)) if baseline_eval_active else 0
        current_player_count = args.players - baseline_player_count if native_baseline_model is not None else args.players
        current_mask = sim.request_players < current_player_count
        baseline_mask = ~current_mask
        current_request_games = sim.request_games[current_mask]
        current_request_players = sim.request_players[current_mask]
        baseline_request_games = sim.request_games[baseline_mask]
        baseline_request_players = sim.request_players[baseline_mask]
        request_players = torch.as_tensor(current_request_players, device="cuda", dtype=torch.long)
        step_batches: list[ResidentBatchTensorView] = []
        step_fire: list[torch.Tensor] = []
        progress_interval = max(0, int(args.progress_steps))
        try:
            for step in range(args.steps):
                runtime.require(runtime.lib.orbit_wars_cuda_sim_clear_actions(sim.ptr), "sim_clear_actions")
                view = runtime.decode_model_actions_for_requests(
                    native_model,
                    sim,
                    step,
                    current_request_games,
                    current_request_players,
                )
                batch = resident_batch_view_to_tensors(view)
                step_batches.append(clone_resident_batch(batch))
                step_fire.append(batch.labels_fire.float().mean().detach())
                if native_baseline_model is not None and baseline_request_games.size > 0:
                    runtime.decode_model_actions_for_requests(
                        native_baseline_model,
                        sim,
                        step,
                        baseline_request_games,
                        baseline_request_players,
                    )
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
            game_rewards = runtime.score_diff_rewards(sim)
            score_margins = game_rewards.max(dim=1).values
            statuses, stats = runtime.read_status_stats(sim)
        finally:
            runtime.destroy_model(native_model)
            if native_baseline_model is not None:
                runtime.destroy_model(native_baseline_model)
            runtime.destroy_sim(sim)

        current_wins = 0
        baseline_wins = 0
        other_results = 0
        for status in statuses:
            winner = int(status.winner)
            if 0 <= winner < current_player_count:
                current_wins += 1
            elif native_baseline_model is not None and current_player_count <= winner < args.players:
                baseline_wins += 1
            else:
                other_results += 1
        valid = torch.tensor([status.winner >= 0 for status in statuses], device="cuda", dtype=torch.bool)
        request_game_ids = torch.as_tensor(current_request_games, device="cuda", dtype=torch.long)
        request_rewards = game_rewards[request_game_ids, request_players]
        fire_rate = torch.stack(step_fire, dim=0).mean()
        total_chunks = sum(
            math.ceil(int(batch.tokens.shape[0]) / max(1, args.train_batch_requests))
            for batch in step_batches
        )
        loss_total = 0.0
        active_slots_total = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        invalid_loss_total = 0.0
        approx_kl_total = 0.0
        clip_fraction_total = 0.0
        invalid_slots_total = 0.0
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
                policy_loss_total += float(metrics["policy_loss"])
                value_loss_total += float(metrics["value_loss"])
                invalid_loss_total += float(metrics["invalid_loss"])
                approx_kl_total += float(metrics["approx_kl"])
                clip_fraction_total += float(metrics["clip_fraction"])
                invalid_slots_total += float(metrics["invalid_slots"])
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
        mean_policy_loss = policy_loss_total / max(1, finite_chunks)
        mean_value_loss = value_loss_total / max(1, finite_chunks)
        mean_invalid_loss = invalid_loss_total / max(1, finite_chunks)
        mean_approx_kl = approx_kl_total / max(1, finite_chunks)
        mean_clip_fraction = clip_fraction_total / max(1, finite_chunks)
        mean_invalid_slots = invalid_slots_total / max(1, finite_chunks)
        mean_score_margin = float(score_margins.detach().mean().cpu())

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
                "policy_loss": mean_policy_loss,
                "value_loss": mean_value_loss,
                "invalid_loss": mean_invalid_loss,
                "approx_kl": mean_approx_kl,
                "clip_fraction": mean_clip_fraction,
                "invalid_slots": mean_invalid_slots,
                "fire_rate": float(fire_rate.detach().cpu()),
                "active_slots": mean_active_slots,
                "mean_winner_margin": mean_score_margin,
                "mean_score_margin": mean_score_margin,
                "finite_chunks": finite_chunks,
                "skipped_nan_chunks": skipped_nan_chunks,
                "update_applied": update_applied,
                "ppo_clip": args.ppo_clip,
                "ppo_epochs": 1,
                "requested_ppo_epochs_ignored": args.ppo_epochs,
            },
        }, checkpoint)
        export_model_bin(model, config, generation_model_bin)
        shutil.copy2(generation_model_bin, args.run_dir / "model.bin")
        baseline_updated = False
        external_eval_result = None
        if baseline_eval_active and baseline_state is not None and current_wins > baseline_wins:
            baseline_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            baseline_model_bin = generation_model_bin
            baseline_updated = True
        external_interval = max(0, int(args.external_baseline_eval_interval))
        external_games = max(0, int(args.external_baseline_eval_games))
        if external_interval > 0 and external_games > 0 and generation % external_interval == 0:
            if baseline_model_bin is None:
                raise RuntimeError("external baseline eval requires a baseline model")
            external_eval_result = run_external_baseline_eval(
                args=args,
                generation=generation,
                baseline_model=baseline_model_bin,
                candidate_model=generation_model_bin,
                games=external_games,
            )
            if (
                not args.external_candidate_vs_exp50_only
                and external_eval_result["candidate_wins"] >= external_eval_result["baseline_wins"]
            ):
                baseline_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                baseline_model_bin = generation_model_bin
                baseline_updated = True
                external_eval_result["baseline_updated"] = True
            else:
                external_eval_result["baseline_updated"] = False
        wins = int(valid.sum().detach().cpu())
        draws = int((~valid).sum().detach().cpu())
        elapsed = max(1.0e-6, time.perf_counter() - started)
        print(json.dumps({
            "event": "gpu_resident_rl_generation_complete",
            "generation": generation,
            "seconds": round(elapsed, 3),
            "games_per_second": round(args.games / elapsed, 3),
            "loss": mean_loss,
            "policy_loss": mean_policy_loss,
            "value_loss": mean_value_loss,
            "invalid_loss": mean_invalid_loss,
            "approx_kl": mean_approx_kl,
            "clip_fraction": mean_clip_fraction,
            "invalid_slots": mean_invalid_slots,
            "fire_rate": float(fire_rate.detach().cpu()),
            "active_slots": mean_active_slots,
            "train_chunks": total_chunks,
            "ppo_epochs": 1,
            "ppo_clip": args.ppo_clip,
            "requested_ppo_epochs_ignored": args.ppo_epochs,
            "finite_chunks": finite_chunks,
            "skipped_nan_chunks": skipped_nan_chunks,
            "update_applied": update_applied,
            "decisive_games": wins,
            "draws_as_losses": draws,
            "current_model_wins": current_wins,
            "baseline_model_wins": baseline_wins,
            "baseline_eval_active": baseline_eval_active,
            "baseline_eval_interval": baseline_eval_interval,
            "baseline_updated": baseline_updated,
            "external_baseline_eval": external_eval_result,
            "other_results": other_results,
            "mean_winner_margin": mean_score_margin,
            "mean_score_margin": mean_score_margin,
            "checkpoint": str(checkpoint),
            "generation_model_bin": str(generation_model_bin),
            "model_bin": str(args.run_dir / "model.bin"),
        }), flush=True)

    if args.generation_tournament_games > 0:
        run_generation_tournament(args)


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


def run_external_baseline_eval(
    *,
    args: argparse.Namespace,
    generation: int,
    baseline_model: Path,
    candidate_model: Path,
    games: int,
) -> dict:
    if args.external_exp50_main is None:
        raise RuntimeError("--external-exp50-main is required for external baseline eval")
    exp50_main = args.external_exp50_main
    if not exp50_main.exists():
        raise FileNotFoundError(f"external exp50 main missing: {exp50_main}")
    template_dir = args.submission_template_dir
    for name in ("main.py", "liborbit_wars_agent.so"):
        if not (template_dir / name).exists():
            raise FileNotFoundError(f"submission template missing {name}: {template_dir}")

    make = load_kaggle_make_for_external_eval(args.external_env_src)
    baseline_wins = 0
    candidate_wins = 0
    exp50_wins = 0
    other_results = 0
    baseline_score_diff = 0.0
    candidate_score_diff = 0.0
    rows = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="orbitwars_external_eval_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        baseline_main = None
        if not args.external_candidate_vs_exp50_only:
            baseline_main = prepare_external_eval_agent(template_dir, baseline_model, tmp_dir / "baseline")
        candidate_main = prepare_external_eval_agent(template_dir, candidate_model, tmp_dir / "candidate")
        for game in range(games):
            if args.external_candidate_vs_exp50_only:
                baseline_seat = -1
                candidate_seat = game % 4
            else:
                baseline_seat = game % 4
                candidate_seat = (baseline_seat + 1) % 4
            agents = [str(exp50_main), str(exp50_main), str(exp50_main), str(exp50_main)]
            if baseline_main is not None:
                agents[baseline_seat] = str(baseline_main)
            agents[candidate_seat] = str(candidate_main)
            env = make("orbit_wars", configuration={"seed": 100000 + generation * 10000 + game}, debug=False)
            env.run(agents)
            scores = external_eval_scores(env)
            best_score = max(scores)
            winner_seats = [index for index, score in enumerate(scores) if score == best_score]
            if baseline_seat >= 0 and baseline_seat in winner_seats and candidate_seat not in winner_seats:
                baseline_wins += 1
                winner = "baseline"
            elif candidate_seat in winner_seats and baseline_seat not in winner_seats:
                candidate_wins += 1
                winner = "candidate"
            elif any(index not in (baseline_seat, candidate_seat) for index in winner_seats):
                exp50_wins += 1
                winner = "exp50"
            else:
                other_results += 1
                winner = "draw"
            exp50_best = max(score for index, score in enumerate(scores) if index not in (baseline_seat, candidate_seat))
            if baseline_seat >= 0:
                baseline_score_diff += scores[baseline_seat] - max(scores[candidate_seat], exp50_best)
                candidate_score_diff += scores[candidate_seat] - max(scores[baseline_seat], exp50_best)
            else:
                candidate_score_diff += scores[candidate_seat] - exp50_best
            rows.append({
                "game": game,
                "baseline_seat": baseline_seat,
                "candidate_seat": candidate_seat,
                "scores": scores,
                "winner": winner,
            })

    elapsed = max(1.0e-6, time.perf_counter() - started)
    result = {
        "event": "external_baseline_eval_complete",
        "generation": generation,
        "games": games,
        "seconds": round(elapsed, 3),
        "baseline_model": str(baseline_model),
        "candidate_model": str(candidate_model),
        "exp50_main": str(exp50_main),
        "candidate_vs_exp50_only": bool(args.external_candidate_vs_exp50_only),
        "baseline_wins": baseline_wins,
        "candidate_wins": candidate_wins,
        "exp50_wins": exp50_wins,
        "other_results": other_results,
        "baseline_win_rate": baseline_wins / games if games else 0.0,
        "candidate_win_rate": candidate_wins / games if games else 0.0,
        "candidate_not_worse": candidate_wins >= baseline_wins,
        "candidate_beats_exp50": candidate_wins > exp50_wins,
        "baseline_mean_score_diff": baseline_score_diff / games if games else 0.0,
        "candidate_mean_score_diff": candidate_score_diff / games if games else 0.0,
        "rows": rows,
    }
    print(json.dumps(result), flush=True)
    return result


def load_kaggle_make_for_external_eval(env_src: Path):
    try:
        from kaggle_environments import __file__ as kaggle_init
        from kaggle_environments import make
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "external baseline eval needs kaggle_environments; run in Molab/Kaggle or install it"
        ) from exc

    env_dst = Path(kaggle_init).resolve().parent / "envs" / "orbit_wars"
    if env_src.exists():
        env_dst.mkdir(parents=True, exist_ok=True)
        for item in env_src.iterdir():
            target = env_dst / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    return make


def prepare_external_eval_agent(template_dir: Path, model_bin: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("main.py", "liborbit_wars_agent.so"):
        shutil.copy2(template_dir / name, out_dir / name)
    shutil.copy2(model_bin, out_dir / "model.bin")
    return out_dir / "main.py"


def external_eval_scores(env) -> list[float]:
    return [float(getattr(state, "reward", 0.0) or 0.0) for state in env.steps[-1]]


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
