#!/usr/bin/env python3
"""Build v8 action-slot imitation shards from Orbit Wars replays.

The replay export stores the action returned for observation N at
``steps[N + 1][player]["action"]``.  The dataset therefore aligns each
training row as ``previous_observation -> current_action``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ACTION_SLOTS = 8
MAX_PLANETS = 64
MAX_FLEETS = 640
TOKEN_FEATURES = 14

TOKEN_GLOBAL = 0
TOKEN_PLANET = 1
TOKEN_FLEET = 2

AMOUNT_CLASSES = [
    "100%",
    "25%",
    "50%",
    "67%",
    "70%",
    "75%",
    "90%",
    "95%",
    "10",
    "20",
    "50",
    "100",
    "200",
    "500",
    "1000",
    "2000",
]
PERCENT_CLASSES = [(0, 1.00), (1, 0.25), (2, 0.50), (3, 0.67), (4, 0.70), (5, 0.75), (6, 0.90), (7, 0.95)]
COUNT_CLASSES = [(8, 10), (9, 20), (10, 50), (11, 100), (12, 200), (13, 500), (14, 1000), (15, 2000)]
TRACE_ANGLE_EPS = 1.0e-3
TRACE_SHIPS_REL_EPS = 0.05
TRACE_MAX_FUTURE_STEPS = 512


@dataclass(frozen=True)
class V8Action:
    source: int
    angle: float
    sent_ships: float


@dataclass(frozen=True)
class FleetHitTrace:
    first_step: int
    fleet_id: int
    owner: int
    source: int
    angle: float
    ships: float
    target_planet_id: int


def nearest_amount_class(source_ships: float, sent_ships: float) -> int:
    if source_ships <= 0:
        return 0
    fraction = sent_ships / source_ships
    if fraction >= 0.985:
        return 0
    candidates: list[tuple[float, int]] = []
    for class_id, percent in PERCENT_CLASSES[1:]:
        candidates.append((abs(source_ships * percent - sent_ships), class_id))
    for class_id, count in COUNT_CLASSES:
        candidates.append((abs(min(float(count), source_ships) - sent_ships), class_id))
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def dedupe_actions(actions: Iterable[Any], action_slots: int = ACTION_SLOTS) -> tuple[list[V8Action], dict[str, int]]:
    by_source: dict[int, V8Action] = {}
    duplicate_actions = 0
    malformed_actions = 0
    for raw in actions:
        try:
            source = int(raw[0])
            angle = float(raw[1])
            sent_ships = float(raw[2])
        except (TypeError, ValueError, IndexError):
            malformed_actions += 1
            continue
        if not (math.isfinite(angle) and math.isfinite(sent_ships)) or sent_ships <= 0:
            malformed_actions += 1
            continue
        candidate = V8Action(source=source, angle=angle, sent_ships=sent_ships)
        existing = by_source.get(source)
        if existing is not None:
            duplicate_actions += 1
        if existing is None or candidate.sent_ships > existing.sent_ships:
            by_source[source] = candidate

    ranked = sorted(by_source.values(), key=lambda action: (-action.sent_ships, action.source))
    kept = ranked[:action_slots]
    stats = {
        "duplicate_actions": duplicate_actions,
        "malformed_actions": malformed_actions,
        "dropped_overflow_actions": max(0, len(ranked) - action_slots),
    }
    return kept, stats


def build_samples_from_replay(
    replay: dict[str, Any],
    source_file: str,
    *,
    max_fleets: int = MAX_FLEETS,
    include_noop: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    winner = _winner_index(replay)
    stats = _empty_stats()
    stats["replays"] = 1
    if winner is None:
        stats["skipped_no_unique_winner"] += 1
        return [], stats

    samples: list[dict[str, Any]] = []
    steps = replay.get("steps") or []
    hit_traces_by_step = build_future_hit_traces(steps, winner)
    previous_observation: dict[str, Any] | None = None
    for step_index, step in enumerate(steps):
        if not isinstance(step, list) or winner >= len(step) or not isinstance(step[winner], dict):
            previous_observation = None
            continue
        agent_row = step[winner]
        current_actions = agent_row.get("action") or []
        if previous_observation is not None and (include_noop or current_actions):
            sample = build_sample(
                previous_observation,
                current_actions,
                source_file,
                max_fleets=max_fleets,
                target_traces=hit_traces_by_step.get(step_index, []),
            )
            samples.append(sample)
            stats["aligned_samples"] += 1
            stats["action_rows"] += int(sample["labels_fire"].sum())
            stats["noop_rows"] += int(sample["labels_fire"].sum() == 0)
            for key, value in sample["action_stats"].items():
                stats[key] += int(value)
        previous_observation = agent_row.get("observation") or None

    return samples, stats


def build_sample(
    observation: dict[str, Any],
    actions: Iterable[Any],
    source_file: str,
    *,
    max_fleets: int = MAX_FLEETS,
    target_traces: list[FleetHitTrace] | None = None,
) -> dict[str, Any]:
    planets = [_planet_from_row(row, row_index) for row_index, row in enumerate(observation.get("planets") or [])]
    planets = planets[:MAX_PLANETS]
    planet_by_id = {planet["id"]: planet for planet in planets}
    kept_actions, action_stats = dedupe_actions(actions)
    action_stats.setdefault("target_trace_hits", 0)
    action_stats.setdefault("target_trace_misses", 0)
    action_stats.setdefault("target_angle_fallbacks", 0)

    labels_fire = np.zeros(ACTION_SLOTS, dtype=np.int64)
    labels_source = np.full(ACTION_SLOTS, -1, dtype=np.int64)
    labels_target = np.full(ACTION_SLOTS, -1, dtype=np.int64)
    labels_amount = np.full(ACTION_SLOTS, -1, dtype=np.int64)
    used_trace_indexes: set[int] = set()
    for slot_index, action in enumerate(kept_actions):
        source = planet_by_id.get(action.source)
        if source is None:
            action_stats["malformed_actions"] += 1
            continue
        target_row = match_future_hit_target_row(action, target_traces or [], planet_by_id, used_trace_indexes)
        if target_row >= 0:
            action_stats["target_trace_hits"] += 1
        else:
            if target_traces is not None:
                action_stats["target_trace_misses"] += 1
            action_stats["target_angle_fallbacks"] += 1
            target_row = infer_target_row(source, planets, action.angle)
        labels_fire[slot_index] = 1
        labels_source[slot_index] = source["row"]
        labels_target[slot_index] = target_row
        labels_amount[slot_index] = nearest_amount_class(source["ships"], action.sent_ships)

    token_rows = build_object_tokens(observation, max_fleets=max_fleets)
    episode_id = _stable_u32(source_file)
    return {
        **token_rows,
        "labels_fire": labels_fire,
        "labels_source": labels_source,
        "labels_target": labels_target,
        "labels_amount": labels_amount,
        "episode_id": episode_id,
        "step": int(observation.get("step", 0)),
        "player": int(observation.get("player", 0)),
        "source_file": source_file,
        "action_stats": action_stats,
    }


def build_object_tokens(observation: dict[str, Any], *, max_fleets: int = MAX_FLEETS) -> dict[str, Any]:
    player = int(observation.get("player", 0))
    step = float(observation.get("step", 0))
    angular_velocity = float(observation.get("angular_velocity", 0.0))
    planets = [_planet_from_row(row, row_index) for row_index, row in enumerate(observation.get("planets") or [])]
    planets = planets[:MAX_PLANETS]
    planet_by_id = {planet["id"]: planet for planet in planets}
    fleets = [_fleet_from_row(row) for row in observation.get("fleets") or []]
    fleets = select_fleets(fleets, planets, player, max_fleets=max_fleets)

    tokens: list[list[float]] = []
    token_type_ids: list[int] = []
    owner_ids: list[int] = []

    tokens.append([player / 3.0, step / 500.0, angular_velocity, len(planets) / MAX_PLANETS, len(fleets) / max(1, max_fleets), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    token_type_ids.append(TOKEN_GLOBAL)
    owner_ids.append(player + 1)

    planet_mask = np.zeros(MAX_PLANETS, dtype=np.bool_)
    for planet in planets:
        planet_mask[planet["row"]] = True
        tokens.append([
            planet["id"] / 128.0,
            planet["owner"] / 3.0,
            planet["x"] / 100.0,
            planet["y"] / 100.0,
            planet["radius"] / 10.0,
            math.log1p(max(0.0, planet["ships"])) / 10.0,
            planet["production"] / 20.0,
            planet["velocity_x"] / 10.0,
            planet["velocity_y"] / 10.0,
            0.0,
            0.0,
            -1.0,
            planet["row"] / MAX_PLANETS,
            0.0,
        ])
        token_type_ids.append(TOKEN_PLANET)
        owner_ids.append(planet["owner"] + 1)

    for fleet in fleets:
        target_row, progress = infer_fleet_target_and_progress(fleet, planets, planet_by_id)
        tokens.append([
            fleet["id"] / 2048.0,
            fleet["owner"] / 3.0,
            fleet["x"] / 100.0,
            fleet["y"] / 100.0,
            0.0,
            math.log1p(max(0.0, fleet["ships"])) / 10.0,
            0.0,
            0.0,
            0.0,
            math.sin(fleet["angle"]),
            math.cos(fleet["angle"]),
            fleet["from_planet_id"] / 128.0,
            target_row / MAX_PLANETS if target_row >= 0 else -1.0,
            progress,
        ])
        token_type_ids.append(TOKEN_FLEET)
        owner_ids.append(fleet["owner"] + 1)

    return {
        "tokens": np.asarray(tokens, dtype=np.float32),
        "token_type_ids": np.asarray(token_type_ids, dtype=np.int64),
        "owner_ids": np.asarray(owner_ids, dtype=np.int64),
        "sample_offsets": np.asarray([0, len(tokens)], dtype=np.int64),
        "planet_mask": planet_mask,
    }


def infer_target_row(source: dict[str, float], planets: list[dict[str, float]], action_angle: float) -> int:
    best_row = -1
    best_score = float("inf")
    for planet in planets:
        if planet["id"] == source["id"]:
            continue
        angle = math.atan2(planet["y"] - source["y"], planet["x"] - source["x"])
        score = abs(_angle_delta(action_angle, angle))
        if score < best_score:
            best_score = score
            best_row = int(planet["row"])
    return best_row


def build_future_hit_traces(steps: list[Any], winner: int, *, max_future_steps: int = TRACE_MAX_FUTURE_STEPS) -> dict[int, list[FleetHitTrace]]:
    active: dict[int, dict[str, Any]] = {}
    traces_by_step: dict[int, list[FleetHitTrace]] = {}
    for step_index, step in enumerate(steps):
        observation = _step_observation(step, winner)
        if observation is None:
            continue
        planets = [_planet_from_row(row, row_index) for row_index, row in enumerate(observation.get("planets") or [])]
        current_fleets = {}
        for row in observation.get("fleets") or []:
            fleet = _fleet_from_row(row)
            current_fleets[fleet["id"]] = fleet

        for fleet_id in sorted(set(active) - set(current_fleets)):
            trace = active.pop(fleet_id)
            if step_index - int(trace["first_step"]) > max_future_steps:
                continue
            target_planet_id = infer_hit_planet_id(trace["last_fleet"], planets)
            if target_planet_id is None:
                continue
            first_fleet = trace["first_fleet"]
            traces_by_step.setdefault(int(trace["first_step"]), []).append(FleetHitTrace(
                first_step=int(trace["first_step"]),
                fleet_id=int(fleet_id),
                owner=int(first_fleet["owner"]),
                source=int(first_fleet["from_planet_id"]),
                angle=float(first_fleet["angle"]),
                ships=float(first_fleet["ships"]),
                target_planet_id=int(target_planet_id),
            ))

        for fleet_id, fleet in current_fleets.items():
            if fleet_id in active:
                active[fleet_id]["last_fleet"] = fleet
                active[fleet_id]["last_step"] = step_index
            else:
                active[fleet_id] = {
                    "first_step": step_index,
                    "first_fleet": fleet,
                    "last_step": step_index,
                    "last_fleet": fleet,
                }
    return traces_by_step


def match_future_hit_target_row(
    action: V8Action,
    target_traces: list[FleetHitTrace],
    planet_by_id: dict[int, dict[str, float]],
    used_trace_indexes: set[int],
) -> int:
    best: tuple[float, int, FleetHitTrace] | None = None
    for trace_index, trace in enumerate(target_traces):
        if trace_index in used_trace_indexes or trace.source != action.source:
            continue
        angle_delta = abs(_angle_delta(trace.angle, action.angle))
        if angle_delta > TRACE_ANGLE_EPS:
            continue
        ships_delta = abs(trace.ships - action.sent_ships)
        if ships_delta > max(1.0e-3, action.sent_ships * TRACE_SHIPS_REL_EPS):
            continue
        if trace.target_planet_id not in planet_by_id:
            continue
        score = angle_delta + ships_delta / max(1.0, action.sent_ships)
        if best is None or score < best[0]:
            best = (score, trace_index, trace)
    if best is None:
        return -1
    used_trace_indexes.add(best[1])
    return int(planet_by_id[best[2].target_planet_id]["row"])


def infer_hit_planet_id(fleet: dict[str, float], planets: list[dict[str, float]]) -> int | None:
    if not planets:
        return None
    fleet_x = float(fleet["x"])
    fleet_y = float(fleet["y"])
    direction_x = math.cos(float(fleet["angle"]))
    direction_y = math.sin(float(fleet["angle"]))
    best: tuple[float, int] | None = None
    for planet in planets:
        dx = planet["x"] - fleet_x
        dy = planet["y"] - fleet_y
        distance = math.hypot(dx, dy)
        projected_ahead = max(0.0, dx * direction_x + dy * direction_y)
        score = distance - min(distance, planet["radius"]) + projected_ahead * 0.001
        if best is None or score < best[0]:
            best = (score, int(planet["id"]))
    return None if best is None else best[1]


def infer_fleet_target_and_progress(
    fleet: dict[str, float],
    planets: list[dict[str, float]],
    planet_by_id: dict[int, dict[str, float]],
) -> tuple[int, float]:
    direction_x = math.cos(fleet["angle"])
    direction_y = math.sin(fleet["angle"])
    best: tuple[float, int] | None = None
    for planet in planets:
        dx = planet["x"] - fleet["x"]
        dy = planet["y"] - fleet["y"]
        projection = dx * direction_x + dy * direction_y
        if projection <= 0:
            continue
        perpendicular = abs(dx * direction_y - dy * direction_x)
        score = perpendicular + projection * 0.001
        if best is None or score < best[0]:
            best = (score, int(planet["row"]))
    source = planet_by_id.get(int(fleet["from_planet_id"]))
    if best is None or source is None:
        return (-1 if best is None else best[1], 0.0)
    target = planets[best[1]]
    total = math.hypot(target["x"] - source["x"], target["y"] - source["y"])
    remaining = math.hypot(target["x"] - fleet["x"], target["y"] - fleet["y"])
    progress = 0.0 if total <= 0 else max(0.0, min(1.0, 1.0 - remaining / total))
    return best[1], progress


def select_fleets(
    fleets: list[dict[str, float]],
    planets: list[dict[str, float]],
    player: int,
    *,
    max_fleets: int = MAX_FLEETS,
) -> list[dict[str, float]]:
    if len(fleets) <= max_fleets:
        return fleets
    owned_planets = [planet for planet in planets if planet["owner"] == player]

    def relevance(fleet: dict[str, float]) -> tuple[float, float]:
        threat = 0.0
        if fleet["owner"] != player and owned_planets:
            nearest = min(math.hypot(fleet["x"] - p["x"], fleet["y"] - p["y"]) for p in owned_planets)
            threat = 1.0 / (1.0 + nearest)
        return (threat, fleet["ships"])

    return sorted(fleets, key=relevance, reverse=True)[:max_fleets]


def write_dataset_shards(samples: list[dict[str, Any]], output_dir: Path, *, shard_size: int = 4096) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard_index, start in enumerate(range(0, len(samples), shard_size)):
        shard_samples = samples[start : start + shard_size]
        shard_file = f"shard-{shard_index:05d}.npz"
        arrays = _pack_samples(shard_samples)
        np.savez_compressed(output_dir / shard_file, **arrays)
        shards.append({"file": shard_file, "samples": len(shard_samples), "max_tokens": int(arrays["tokens"].shape[1])})

    metadata = {
        "version": "v8-action-slots",
        "amount_classes": AMOUNT_CLASSES,
        "action_slots": ACTION_SLOTS,
        "max_planets": MAX_PLANETS,
        "max_fleets": MAX_FLEETS,
        "token_features": TOKEN_FEATURES,
        "samples": len(samples),
        "shards": shards,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def convert_replay(path: Path, *, max_fleets: int = MAX_FLEETS) -> tuple[list[dict[str, Any]], dict[str, int]]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    return build_samples_from_replay(replay, str(path), max_fleets=max_fleets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-fleets", type=int, default=MAX_FLEETS)
    parser.add_argument("--log-every-replays", type=int, default=50)
    args = parser.parse_args()

    all_samples: list[dict[str, Any]] = []
    merged_stats = _empty_stats()
    for replay_index, replay_path in enumerate(args.replays, start=1):
        samples, stats = convert_replay(replay_path, max_fleets=args.max_fleets)
        all_samples.extend(samples)
        for key, value in stats.items():
            merged_stats[key] = merged_stats.get(key, 0) + int(value)
        if args.log_every_replays > 0 and (replay_index == 1 or replay_index % args.log_every_replays == 0):
            print(json.dumps({
                "event": "dataset_progress",
                "replays": replay_index,
                "total_replays": len(args.replays),
                "samples": len(all_samples),
                "target_trace_hits": merged_stats.get("target_trace_hits", 0),
                "target_angle_fallbacks": merged_stats.get("target_angle_fallbacks", 0),
            }, ensure_ascii=False), flush=True)
    metadata = write_dataset_shards(all_samples, args.output, shard_size=args.shard_size)
    metadata["stats"] = merged_stats
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(all_samples), "stats": merged_stats}, ensure_ascii=False))


def _pack_samples(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    max_tokens = max((sample["tokens"].shape[0] for sample in samples), default=1)
    count = len(samples)
    tokens = np.zeros((count, max_tokens, TOKEN_FEATURES), dtype=np.float32)
    token_type_ids = np.zeros((count, max_tokens), dtype=np.int64)
    owner_ids = np.zeros((count, max_tokens), dtype=np.int64)
    sample_offsets = np.zeros((count, 2), dtype=np.int64)
    planet_mask = np.zeros((count, MAX_PLANETS), dtype=np.bool_)
    labels_fire = np.zeros((count, ACTION_SLOTS), dtype=np.int64)
    labels_source = np.full((count, ACTION_SLOTS), -1, dtype=np.int64)
    labels_target = np.full((count, ACTION_SLOTS), -1, dtype=np.int64)
    labels_amount = np.full((count, ACTION_SLOTS), -1, dtype=np.int64)
    episode_id = np.zeros(count, dtype=np.uint32)
    step = np.zeros(count, dtype=np.int64)
    player = np.zeros(count, dtype=np.int64)

    for index, sample in enumerate(samples):
        length = sample["tokens"].shape[0]
        tokens[index, :length, :] = sample["tokens"]
        token_type_ids[index, :length] = sample["token_type_ids"]
        owner_ids[index, :length] = sample["owner_ids"]
        sample_offsets[index] = sample["sample_offsets"]
        planet_mask[index] = sample["planet_mask"]
        labels_fire[index] = sample["labels_fire"]
        labels_source[index] = sample["labels_source"]
        labels_target[index] = sample["labels_target"]
        labels_amount[index] = sample["labels_amount"]
        episode_id[index] = sample["episode_id"]
        step[index] = sample["step"]
        player[index] = sample["player"]

    return {
        "tokens": tokens,
        "token_type_ids": token_type_ids,
        "owner_ids": owner_ids,
        "sample_offsets": sample_offsets,
        "planet_mask": planet_mask,
        "labels_fire": labels_fire,
        "labels_source": labels_source,
        "labels_target": labels_target,
        "labels_amount": labels_amount,
        "episode_id": episode_id,
        "step": step,
        "player": player,
    }


def _winner_index(replay: dict[str, Any]) -> int | None:
    rewards = replay.get("rewards") or []
    if not rewards:
        return None
    best = max(rewards)
    return rewards.index(best) if rewards.count(best) == 1 else None


def _step_observation(step: Any, player: int) -> dict[str, Any] | None:
    if not isinstance(step, list) or player >= len(step) or not isinstance(step[player], dict):
        return None
    observation = step[player].get("observation")
    return observation if isinstance(observation, dict) else None


def _planet_from_row(row: Any, row_index: int) -> dict[str, float]:
    values = list(row)
    return {
        "row": row_index,
        "id": int(values[0]),
        "owner": int(values[1]),
        "x": float(values[2]),
        "y": float(values[3]),
        "radius": float(values[4]),
        "ships": float(values[5]),
        "production": float(values[6]),
        "velocity_x": float(values[7]) if len(values) > 7 else 0.0,
        "velocity_y": float(values[8]) if len(values) > 8 else 0.0,
    }


def _fleet_from_row(row: Any) -> dict[str, float]:
    values = list(row)
    return {
        "id": int(values[0]),
        "owner": int(values[1]),
        "x": float(values[2]),
        "y": float(values[3]),
        "angle": float(values[4]),
        "from_planet_id": int(values[5]),
        "ships": float(values[6]),
    }


def _angle_delta(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


def _stable_u32(value: str) -> int:
    return int.from_bytes(hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest(), "little")


def _empty_stats() -> dict[str, int]:
    return {
        "replays": 0,
        "skipped_no_unique_winner": 0,
        "aligned_samples": 0,
        "action_rows": 0,
        "noop_rows": 0,
        "duplicate_actions": 0,
        "malformed_actions": 0,
        "dropped_overflow_actions": 0,
        "target_trace_hits": 0,
        "target_trace_misses": 0,
        "target_angle_fallbacks": 0,
    }


if __name__ == "__main__":
    main()
