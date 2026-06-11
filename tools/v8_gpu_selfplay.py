#!/usr/bin/env python3
"""GPU inference self-play evaluator for OWV8 action-slot models."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.leaderboard_v8_dataset import ACTION_SLOTS, MAX_PLANETS, build_object_tokens
from tools.v8_model import V8ActionSlotTransformer, load_model_bin_state

BOARD_CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
EPISODE_STEPS = 500
FLEET_SPEED_MAX = 6.0
FLEET_SPEED_REFERENCE_SHIPS = 1000.0
FLEET_SPEED_CURVE_POWER = 1.5
INTERCEPT_ITERATIONS = 32
AMOUNT_PERCENT = {0: 1.0, 1: 0.25, 2: 0.50, 3: 0.67, 4: 0.70, 5: 0.75, 6: 0.90, 7: 0.95}
AMOUNT_COUNT = {8: 10, 9: 20, 10: 50, 11: 100, 12: 200, 13: 500, 14: 1000, 15: 2000}


@dataclass
class GpuSelfPlayStats:
    games: int
    players: int
    seconds: float
    inference_batches: int
    inference_samples: int
    inference_seconds: float
    backprop_samples: int = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dashboard/public/telemetry/latest.json"))
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--batch-games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config, state = load_model_bin_state(args.model_bin)
    model = V8ActionSlotTransformer(config)
    model.load_state_dict(state, strict=False)
    telemetry = run_gpu_selfplay(
        model,
        output_path=args.output,
        games=args.games,
        players=args.players,
        batch_games=args.batch_games,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps({
        "event": "gpu_selfplay_complete",
        "output": str(args.output),
        "games": args.games,
        "players": args.players,
        "turnsPerSecond": telemetry.get("turnsPerSecond"),
        "gpuInferenceBatches": telemetry.get("gpuInferenceBatches"),
        "avgGpuInferenceMs": telemetry.get("avgGpuInferenceMs"),
    }), flush=True)


def run_gpu_selfplay(
    model: V8ActionSlotTransformer,
    *,
    output_path: Path | None,
    games: int,
    players: int,
    batch_games: int,
    seed: int,
    device: str | None = None,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    orbit_wars = load_orbit_wars_reference()
    started = time.perf_counter()
    all_results = []
    inference_batches = 0
    inference_samples = 0
    inference_seconds = 0.0
    game_index = 0
    while game_index < games:
        chunk_size = min(batch_games, games - game_index)
        contexts = [new_game_context(orbit_wars, players, seed + game_index + offset) for offset in range(chunk_size)]
        replay_frames = [[] for _ in contexts]
        while any(not context.env.done for context in contexts):
            observations = []
            owners = []
            for context_index, context in enumerate(contexts):
                if context.env.done:
                    continue
                frame_actions = []
                for player in range(players):
                    observations.append(observation_dict(context.state[player].observation, player))
                    owners.append((context_index, player))
                    frame_actions.append([])
                replay_frames[context_index].append(frame_from_observation(context.state[0].observation, frame_actions))
            if not observations:
                break
            t0 = time.perf_counter()
            action_lists = infer_actions_batch(model, observations, device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - t0
            inference_batches += 1
            inference_samples += len(observations)
            for (context_index, player), actions in zip(owners, action_lists):
                contexts[context_index].state[player].action = actions
                replay_frames[context_index][-1]["actions"][player] = actions
            for context in contexts:
                if context.env.done:
                    continue
                context.state = orbit_wars.interpreter(context.state, context.env)
                step = int(getattr(context.state[0].observation, "step", 0)) + 1
                for player in range(players):
                    context.state[player].observation.step = step
                if all(getattr(row, "status", "ACTIVE") == "DONE" for row in context.state):
                    context.env.done = True
                if step >= EPISODE_STEPS:
                    context.env.done = True
        for context, frames in zip(contexts, replay_frames):
            all_results.append(result_from_context(context, frames, players))
        game_index += chunk_size
    elapsed = max(1.0e-6, time.perf_counter() - started)
    telemetry = telemetry_from_results(
        all_results,
        stats=GpuSelfPlayStats(
            games=games,
            players=players,
            seconds=elapsed,
            inference_batches=inference_batches,
            inference_samples=inference_samples,
            inference_seconds=inference_seconds,
        ),
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(telemetry, ensure_ascii=False), encoding="utf-8")
    return telemetry


def load_orbit_wars_reference():
    candidates = [
        ROOT / ".external/kaggle-env-src/kaggle_environments/envs/orbit_wars/orbit_wars.py",
        Path("C:/tmp/kaggle-env-src/kaggle_environments/envs/orbit_wars/orbit_wars.py"),
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("orbit_wars_reference", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("official orbit_wars.py reference source not found")


def new_game_context(orbit_wars: Any, players: int, seed: int) -> SimpleNamespace:
    state = [
        SimpleNamespace(observation=SimpleNamespace(step=0), action=[], status="ACTIVE", reward=0)
        for _ in range(players)
    ]
    env = SimpleNamespace(
        configuration=SimpleNamespace(seed=seed, shipSpeed=6, episodeSteps=EPISODE_STEPS, cometSpeed=4),
        done=False,
        info={},
    )
    state = orbit_wars.interpreter(state, env)
    for player in range(players):
        state[player].observation.step = 0
    return SimpleNamespace(state=state, env=env)


def observation_dict(obs: Any, player: int) -> dict[str, Any]:
    angular_velocity = float(getattr(obs, "angular_velocity", 0.03))
    return {
        "player": player,
        "step": int(getattr(obs, "step", 0)),
        "angular_velocity": angular_velocity,
        "planets": add_planet_velocity(getattr(obs, "planets", []), angular_velocity, getattr(obs, "comets", [])),
        "initial_planets": add_planet_velocity(getattr(obs, "initial_planets", getattr(obs, "planets", [])), angular_velocity, []),
        "fleets": [list(row) for row in getattr(obs, "fleets", [])],
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", [])),
        "comets": list(getattr(obs, "comets", [])),
    }


def add_planet_velocity(planets: list[Any], angular_velocity: float, comets: list[dict[str, Any]]) -> list[list[float]]:
    comet_velocity = comet_velocity_map(comets)
    rows = []
    for raw in planets:
        row = list(raw)
        planet_id = int(row[0])
        if len(row) >= 9:
            rows.append(row[:9])
            continue
        vx, vy = comet_velocity.get(planet_id, planet_velocity(float(row[2]), float(row[3]), float(row[4]), angular_velocity))
        rows.append(row[:7] + [vx, vy])
    return rows


def comet_velocity_map(comets: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    values = {}
    for group in comets or []:
        path_index = int(group.get("path_index", 0))
        for planet_id, path in zip(group.get("planet_ids", []), group.get("paths", [])):
            if path_index + 1 < len(path):
                current = path[path_index]
                next_point = path[path_index + 1]
                values[int(planet_id)] = (float(next_point[0]) - float(current[0]), float(next_point[1]) - float(current[1]))
    return values


def planet_velocity(x: float, y: float, radius: float, angular_velocity: float) -> tuple[float, float]:
    dx = x - BOARD_CENTER
    dy = y - BOARD_CENTER
    if math.hypot(dx, dy) + radius >= ROTATION_RADIUS_LIMIT:
        return 0.0, 0.0
    return -angular_velocity * dy, angular_velocity * dx


@torch.inference_mode()
def infer_actions_batch(model: V8ActionSlotTransformer, observations: list[dict[str, Any]], device: str) -> list[list[list[float]]]:
    batch = collate_observations(observations)
    tensors = {key: value.to(device) for key, value in batch.items()}
    outputs = model(
        tensors["tokens"],
        tensors["token_type_ids"],
        tensors["owner_ids"],
        padding_mask=tensors["padding_mask"],
        planet_mask=tensors["planet_mask"],
    )
    return [
        decode_actions(observation, {key: value[index].detach().cpu() for key, value in outputs.items()})
        for index, observation in enumerate(observations)
    ]


def collate_observations(observations: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    tokenized = [build_object_tokens(observation) for observation in observations]
    max_len = max(item["tokens"].shape[0] for item in tokenized)
    token_features = tokenized[0]["tokens"].shape[1]
    tokens = np.zeros((len(tokenized), max_len, token_features), dtype=np.float32)
    token_type_ids = np.zeros((len(tokenized), max_len), dtype=np.int64)
    owner_ids = np.zeros((len(tokenized), max_len), dtype=np.int64)
    padding_mask = np.ones((len(tokenized), max_len), dtype=np.bool_)
    planet_mask = np.zeros((len(tokenized), MAX_PLANETS), dtype=np.bool_)
    for index, item in enumerate(tokenized):
        length = item["tokens"].shape[0]
        tokens[index, :length] = item["tokens"]
        token_type_ids[index, :length] = item["token_type_ids"]
        owner_ids[index, :length] = item["owner_ids"]
        padding_mask[index, :length] = False
        planet_mask[index] = item["planet_mask"]
    return {
        "tokens": torch.from_numpy(tokens),
        "token_type_ids": torch.from_numpy(token_type_ids),
        "owner_ids": torch.from_numpy(owner_ids),
        "padding_mask": torch.from_numpy(padding_mask),
        "planet_mask": torch.from_numpy(planet_mask),
    }


def decode_actions(observation: dict[str, Any], output: dict[str, torch.Tensor]) -> list[list[float]]:
    planets = observation["planets"][:MAX_PLANETS]
    initial_planets = observation.get("initial_planets") or planets
    player = int(observation.get("player", 0))
    angular_velocity = float(observation.get("angular_velocity", 0.03))
    fire_logits = output["fire_logits"].float()
    ranked = sorted(range(min(ACTION_SLOTS, fire_logits.numel())), key=lambda slot: float(fire_logits[slot]), reverse=True)
    used_sources = set()
    actions = []
    for slot in ranked:
        if torch.sigmoid(fire_logits[slot]).item() < 0.5:
            continue
        source_row = int(torch.argmax(output["source_logits"][slot]).item())
        if source_row >= len(planets) or source_row in used_sources:
            continue
        source = planets[source_row]
        if int(source[1]) != player or float(source[5]) < 1.0:
            continue
        target_logits = output["target_logits"][slot].clone()
        if source_row < target_logits.numel():
            target_logits[source_row] = -1.0e9
        target_row = int(torch.argmax(target_logits).item())
        if target_row >= len(planets):
            continue
        amount_index = int(torch.argmax(output["amount_logits"][slot]).item())
        ships = amount_ship_count(float(source[5]), amount_index)
        if ships < 1:
            continue
        angle = intercept_angle(source, planets[target_row], initial_planets, ships, angular_velocity)
        used_sources.add(source_row)
        actions.append([int(source[0]), float(angle), int(ships)])
    return actions


def amount_ship_count(source_ships: float, amount_index: int) -> int:
    if amount_index in AMOUNT_PERCENT:
        value = int(math.floor(source_ships * AMOUNT_PERCENT[amount_index]))
    else:
        value = int(min(source_ships, AMOUNT_COUNT.get(amount_index, 0)))
    return max(0, min(int(source_ships), value))


def intercept_angle(source: list[float], target: list[float], initial_planets: list[list[float]], ship_count: int, angular_velocity: float) -> float:
    speed = fleet_speed(float(ship_count))
    predicted_x = float(target[2])
    predicted_y = float(target[3])
    orbiting = target_uses_orbit_prediction(target, initial_planets)
    for _ in range(INTERCEPT_ITERATIONS):
        dx = predicted_x - float(source[2])
        dy = predicted_y - float(source[3])
        travel_time = math.hypot(dx, dy) / speed
        if orbiting:
            predicted_x, predicted_y = predict_orbit_position(target, travel_time, angular_velocity)
        else:
            predicted_x = float(target[2]) + float(target[7]) * travel_time
            predicted_y = float(target[3]) + float(target[8]) * travel_time
    return math.atan2(predicted_y - float(source[3]), predicted_x - float(source[2]))


def fleet_speed(ship_count: float) -> float:
    ships = max(1.0, ship_count)
    speed_ratio = min(1.0, max(0.0, math.log(ships) / math.log(FLEET_SPEED_REFERENCE_SHIPS)))
    return 1.0 + (FLEET_SPEED_MAX - 1.0) * (speed_ratio ** FLEET_SPEED_CURVE_POWER)


def target_uses_orbit_prediction(target: list[float], initial_planets: list[list[float]]) -> bool:
    initial = next((row for row in initial_planets if int(row[0]) == int(target[0])), None)
    if initial is None:
        return False
    return math.hypot(float(initial[2]) - BOARD_CENTER, float(initial[3]) - BOARD_CENTER) + float(target[4]) < ROTATION_RADIUS_LIMIT


def predict_orbit_position(target: list[float], travel_time: float, angular_velocity: float) -> tuple[float, float]:
    dx = float(target[2]) - BOARD_CENTER
    dy = float(target[3]) - BOARD_CENTER
    radius = math.hypot(dx, dy)
    if radius <= 1.0e-6:
        return float(target[2]), float(target[3])
    angle = math.atan2(dy, dx) + angular_velocity * travel_time
    return BOARD_CENTER + radius * math.cos(angle), BOARD_CENTER + radius * math.sin(angle)


def frame_from_observation(obs: Any, actions: list[list[list[float]]]) -> dict[str, Any]:
    return {
        "step": int(getattr(obs, "step", 0)),
        "planets": [
            {"id": int(p[0]), "owner": int(p[1]), "x": float(p[2]), "y": float(p[3]), "radius": float(p[4]), "ships": float(p[5]), "production": float(p[6])}
            for p in getattr(obs, "planets", [])
        ],
        "fleets": [
            {"id": int(f[0]), "owner": int(f[1]), "x": float(f[2]), "y": float(f[3]), "angle": float(f[4]), "from_planet_id": int(f[5]), "ships": float(f[6])}
            for f in getattr(obs, "fleets", [])
        ],
        "comets": list(getattr(obs, "comets", [])),
        "actions": actions,
    }


def result_from_context(context: SimpleNamespace, frames: list[dict[str, Any]], players: int) -> dict[str, Any]:
    rewards = [int(getattr(context.state[player], "reward", 0)) for player in range(players)]
    winners = [player for player, reward in enumerate(rewards) if reward == 1]
    return {"winner": winners[0] if len(winners) == 1 else None, "frames": frames, "rewards": rewards}


def telemetry_from_results(results: list[dict[str, Any]], stats: GpuSelfPlayStats) -> dict[str, Any]:
    winner0 = sum(1 for result in results if result["winner"] == 0)
    draws = sum(1 for result in results if result["winner"] is None)
    total_frames = sum(len(result["frames"]) for result in results)
    models = []
    for player in range(stats.players):
        wins = sum(1 for result in results if result["winner"] == player)
        draws_for_player = draws
        losses = stats.games - wins - draws_for_player
        launch_actions = sum(len(frame["actions"][player]) for result in results for frame in result["frames"] if player < len(frame["actions"]))
        models.append({
            "id": f"P-{player}",
            "parent": "owv8_gpu_inference",
            "wins": wins,
            "draws": draws_for_player,
            "losses": losses,
            "games": stats.games,
            "launchActions": launch_actions,
            "selected": player == 0,
        })
    frames = results[-1]["frames"] if results else []
    win_rate = winner0 / max(1, stats.games)
    avg_gpu_ms = 1000.0 * stats.inference_seconds / max(1, stats.inference_batches)
    return {
        "sourceMessage": "v8 gpu self-play; PyTorch CUDA action-slot inference; official interpreter simulation",
        "runId": "v8-gpu-selfplay",
        "activeGeneration": 1,
        "runProfile": "v8_gpu_selfplay",
        "trainingMode": "v8_action_slots_gpu_selfplay",
        "strictGameRules": True,
        "episodeSteps": EPISODE_STEPS,
        "expectedFramesPerGame": EPISODE_STEPS + 1,
        "playersPerGame": stats.players,
        "mapSource": "official_orbit_wars_reference",
        "turnLoop": "official_orbit_wars_interpreter",
        "fullReplay": True,
        "replayFrameStride": 1,
        "storedReplayGames": 1,
        "gamesPerSecond": stats.games / stats.seconds,
        "turnsPerSecond": total_frames / stats.seconds,
        "gpuInferenceBatches": stats.inference_batches,
        "gpuInferenceSamples": stats.inference_samples,
        "gpuInferenceSeconds": stats.inference_seconds,
        "avgGpuInferenceMs": avg_gpu_ms,
        "metrics": [{
            "generation": 1,
            "winRate": win_rate,
            "evaluatedGames": stats.games,
            "sampledReplayGames": 1,
            "modelActionCalls": stats.inference_samples,
            "inferenceBatchCalls": stats.inference_batches,
            "maxInferenceBatchSize": stats.players * stats.games,
            "avgModelActionMs": avg_gpu_ms,
            "backpropSamples": stats.backprop_samples,
            "generationSeconds": stats.seconds,
        }],
        "generationWinRates": [{
            "validationGeneration": 1,
            "evaluatedGeneration": 1,
            "modelCount": stats.players,
            "games": stats.games,
            "wins": winner0,
            "draws": draws,
            "losses": stats.games - winner0 - draws,
            "winRate": win_rate,
        }],
        "models": models,
        "frames": frames,
        "replayChunks": [],
        "replayGames": [{"generation": 1, "modelId": "OWV8-GPU", "gameIndex": stats.games - 1, "opponentId": "self", "reward": 0, "frames": frames}],
    }


if __name__ == "__main__":
    main()
