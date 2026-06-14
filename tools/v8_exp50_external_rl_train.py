#!/usr/bin/env python3
"""External RL against three exp50 agents.

This path is intentionally separate from the GPU-resident self-play loop:
exp50 is a Python Kaggle agent, so training against the real exp50 requires
running the Kaggle environment.  Backprop still runs on CUDA.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.leaderboard_v8_dataset import build_sample
from tools.v8_model import V8ActionSlotTransformer, export_model_bin, load_model_bin_state
from tools.v8_selfplay_train import SelfPlayConfig, weighted_action_loss
from tools.v8_train import collate_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-bin", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--exp50-main", type=Path, required=True)
    parser.add_argument("--submission-template-dir", type=Path, default=Path("kaggle_submission"))
    parser.add_argument("--external-env-src", type=Path, default=Path(".external/kaggle-env-src/kaggle_environments/envs/orbit_wars"))
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=500000)
    parser.add_argument("--progress-games", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    if args.players != 4:
        raise RuntimeError("exp50 external RL currently expects exactly 4 players")
    if not args.exp50_main.exists():
        raise FileNotFoundError(f"exp50 main missing: {args.exp50_main}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    make = load_kaggle_make(args.external_env_src)
    config, state = load_model_bin_state(args.base_model_bin)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    model = V8ActionSlotTransformer(config).to(device)
    model.load_state_dict(state, strict=False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_config = SelfPlayConfig(
        backprop_epochs=args.epochs,
        backprop_batch_size=args.batch_size,
        backprop_lr=args.lr,
    )

    print(json.dumps({
        "event": "exp50_external_rl_start",
        "base_model": str(args.base_model_bin),
        "run_dir": str(args.run_dir),
        "exp50_main": str(args.exp50_main),
        "generations": args.generations,
        "games": args.games,
        "players": args.players,
        "steps": args.steps,
        "reward": "sqrt(score_diff)*sign(score_diff)",
        "device": device,
        "bf16": bool(args.bf16),
    }, ensure_ascii=False), flush=True)

    for generation in range(1, args.generations + 1):
        started = time.perf_counter()
        generation_dir = args.run_dir / f"generation-{generation:04d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        generation_model_bin = generation_dir / "model.bin"
        export_model_bin(model, config, generation_model_bin)
        shutil.copy2(generation_model_bin, args.run_dir / "model.bin")

        samples, game_rows = collect_exp50_rollouts(
            args=args,
            make=make,
            model_bin=generation_model_bin,
            generation=generation,
        )
        metrics = train_on_samples(
            model=model,
            optimizer=optimizer,
            config=loss_config,
            samples=samples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device,
            bf16=args.bf16,
        )

        export_model_bin(model, config, generation_model_bin)
        shutil.copy2(generation_model_bin, args.run_dir / "model.bin")
        checkpoint = generation_dir / f"checkpoint-generation-{generation:04d}.pt"
        torch.save(
            {
                "config": config.__dict__,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "generation": generation,
            },
            checkpoint,
        )

        wins = sum(1 for row in game_rows if row["result"] == "win")
        losses = sum(1 for row in game_rows if row["result"] == "loss")
        draws = sum(1 for row in game_rows if row["result"] == "draw")
        mean_diff = sum(float(row["score_diff"]) for row in game_rows) / max(1, len(game_rows))
        mean_reward = sum(float(row["reward"]) for row in game_rows) / max(1, len(game_rows))
        elapsed = max(1.0e-6, time.perf_counter() - started)
        print(json.dumps({
            "event": "exp50_external_rl_generation_complete",
            "generation": generation,
            "seconds": round(elapsed, 3),
            "games": args.games,
            "games_per_second": round(args.games / elapsed, 3),
            "samples": len(samples),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / max(1, args.games),
            "mean_score_diff": mean_diff,
            "mean_reward": mean_reward,
            "backprop": metrics,
            "checkpoint": str(checkpoint),
            "generation_model_bin": str(generation_model_bin),
            "model_bin": str(args.run_dir / "model.bin"),
        }, ensure_ascii=False), flush=True)


def load_kaggle_make(env_src: Path):
    try:
        from kaggle_environments import __file__ as kaggle_init
        from kaggle_environments import make
    except ModuleNotFoundError as exc:
        raise RuntimeError("kaggle_environments is required for exp50 external RL") from exc

    env_dst = Path(kaggle_init).resolve().parent / "envs" / "orbit_wars"
    env_dst.mkdir(parents=True, exist_ok=True)
    if env_src.exists():
        for item in env_src.iterdir():
            target = env_dst / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    return make


def collect_exp50_rollouts(
    *,
    args: argparse.Namespace,
    make: Any,
    model_bin: Path,
    generation: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    progress_games = max(0, int(args.progress_games))

    with tempfile.TemporaryDirectory(prefix="orbitwars_exp50_rl_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        our_main = prepare_agent(args.submission_template_dir, model_bin, tmp_dir / "candidate")
        exp50_main = args.exp50_main

        for game in range(args.games):
            seat = game % args.players
            agents = [str(exp50_main), str(exp50_main), str(exp50_main), str(exp50_main)]
            agents[seat] = str(our_main)
            seed = args.seed + generation * 100000 + game
            env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": args.steps}, debug=False)
            env.run(agents)
            scores = final_scores(env)
            our_score = scores[seat]
            best_other = max(score for index, score in enumerate(scores) if index != seat)
            score_diff = our_score - best_other
            reward = signed_sqrt(score_diff)
            result = "win" if score_diff > 0 else "loss" if score_diff < 0 else "draw"

            for step_index in range(max(0, len(env.steps) - 1)):
                observation = state_field(env.steps[step_index][seat], "observation")
                action = state_field(env.steps[step_index + 1][seat], "action") or []
                if not observation:
                    continue
                observation = dict(observation)
                observation["player"] = seat
                sample = build_sample(
                    observation,
                    action,
                    f"exp50_rl:g{generation}:game{game}:step{step_index}:seat{seat}",
                    target_traces=None,
                )
                sample["outcome_reward"] = reward
                samples.append(sample)

            rows.append({
                "game": game,
                "seed": seed,
                "seat": seat,
                "scores": scores,
                "our_score": our_score,
                "best_exp50_score": best_other,
                "score_diff": score_diff,
                "reward": reward,
                "result": result,
            })
            if progress_games > 0 and ((game + 1) % progress_games == 0 or game + 1 == args.games):
                print(json.dumps({
                    "event": "exp50_external_rl_game_progress",
                    "generation": generation,
                    "game": game + 1,
                    "games": args.games,
                    "samples": len(samples),
                }, ensure_ascii=False), flush=True)

    return samples, rows


def prepare_agent(template_dir: Path, model_bin: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("main.py", "liborbit_wars_agent.so"):
        src = template_dir / name
        if not src.exists():
            raise FileNotFoundError(f"submission template missing {name}: {template_dir}")
        shutil.copy2(src, out_dir / name)
    shutil.copy2(model_bin, out_dir / "model.bin")
    return out_dir / "main.py"


def final_scores(env: Any) -> list[float]:
    return [float(state_field(state, "reward") or 0.0) for state in env.steps[-1]]


def state_field(state: Any, name: str) -> Any:
    if isinstance(state, dict):
        return state.get(name)
    return getattr(state, name, None)


def signed_sqrt(value: float) -> float:
    if value == 0.0:
        return 0.0
    return math.copysign(math.sqrt(abs(value)), value)


def train_on_samples(
    *,
    model: V8ActionSlotTransformer,
    optimizer: torch.optim.Optimizer,
    config: SelfPlayConfig,
    samples: list[dict[str, Any]],
    batch_size: int,
    epochs: int,
    device: str,
    bf16: bool,
) -> dict[str, float]:
    if not samples:
        return {"samples": 0.0, "steps": 0.0, "loss": 0.0}
    total_loss = 0.0
    steps = 0
    last_parts: dict[str, float] = {}
    model.train()
    for _epoch in range(max(1, epochs)):
        for start in range(0, len(samples), max(1, batch_size)):
            batch_samples = samples[start:start + max(1, batch_size)]
            batch = {key: value.to(device) for key, value in collate_samples(batch_samples).items()}
            rewards = torch.as_tensor(
                [float(sample.get("outcome_reward", 0.0)) for sample in batch_samples],
                dtype=torch.float32,
                device=device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda" and bf16)):
                outputs = model(
                    batch["tokens"],
                    batch["token_type_ids"],
                    batch["owner_ids"],
                    padding_mask=batch["padding_mask"],
                    planet_mask=batch["planet_mask"],
                )
                loss, parts = weighted_action_loss(outputs, batch, config, rewards)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
            last_parts = parts
    metrics = {
        "samples": float(len(samples)),
        "steps": float(steps),
        "loss": total_loss / max(1, steps),
    }
    metrics.update(last_parts)
    return metrics


if __name__ == "__main__":
    main()
