from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "kaggle_submission" / "main.py"
CENTER = 50.0
BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
MAX_STEPS = 500


def load_submission():
    spec = importlib.util.spec_from_file_location("orbit_wars_submission_diag", MAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get(obs: Any, key: str, default: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def row_get(row: Any, index: int, default: Any = 0.0) -> Any:
    if isinstance(row, dict):
        keys = ("id", "owner", "x", "y", "radius", "ships", "production", "velocity_x", "velocity_y")
        return row.get(keys[index], default)
    if isinstance(row, (list, tuple)) and index < len(row):
        return row[index]
    return default


def fleet_speed(ships: int) -> float:
    value = max(1.0, float(ships))
    ratio = min(1.0, max(0.0, math.log(value) / math.log(1000.0)))
    return 1.0 + 5.0 * ratio**1.5


def swept_pair_hit(
    fleet_start_x: float,
    fleet_start_y: float,
    fleet_end_x: float,
    fleet_end_y: float,
    planet_start_x: float,
    planet_start_y: float,
    planet_end_x: float,
    planet_end_y: float,
    radius: float,
) -> bool:
    delta_start_x = fleet_start_x - planet_start_x
    delta_start_y = fleet_start_y - planet_start_y
    delta_velocity_x = (fleet_end_x - fleet_start_x) - (planet_end_x - planet_start_x)
    delta_velocity_y = (fleet_end_y - fleet_start_y) - (planet_end_y - planet_start_y)
    a = delta_velocity_x * delta_velocity_x + delta_velocity_y * delta_velocity_y
    b = 2.0 * (delta_start_x * delta_velocity_x + delta_start_y * delta_velocity_y)
    c = delta_start_x * delta_start_x + delta_start_y * delta_start_y - radius * radius
    if a < 1.0e-12:
        return c <= 0.0
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return False
    root = math.sqrt(discriminant)
    entry = (-b - root) / (2.0 * a)
    exit = (-b + root) / (2.0 * a)
    return exit >= 0.0 and entry <= 1.0


def segment_intersects_sun(start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
    dx = end_x - start_x
    dy = end_y - start_y
    length_squared = dx * dx + dy * dy
    projection = 0.0
    if length_squared > 0.0:
        projection = ((CENTER - start_x) * dx + (CENTER - start_y) * dy) / length_squared
        projection = min(1.0, max(0.0, projection))
    closest_x = start_x + projection * dx
    closest_y = start_y + projection * dy
    return (closest_x - CENTER) ** 2 + (closest_y - CENTER) ** 2 < SUN_RADIUS**2


def comet_paths(obs: Any) -> dict[int, tuple[int, list[tuple[float, float]]]]:
    paths: dict[int, tuple[int, list[tuple[float, float]]]] = {}
    for group in get(obs, "comets", []) or []:
        if not isinstance(group, dict):
            continue
        path_index = int(group.get("path_index", -1))
        for planet_id, path in zip(group.get("planet_ids", []), group.get("paths", [])):
            paths[int(planet_id)] = (path_index, [(float(point[0]), float(point[1])) for point in path])
    return paths


def next_planet_position(
    planet: Any,
    initial_by_id: dict[int, Any],
    comet_path_map: dict[int, tuple[int, list[tuple[float, float]]]],
    old_x: float,
    old_y: float,
    step: int,
    step_offset: int,
    angular_velocity: float,
) -> tuple[float, float, bool]:
    planet_id = int(row_get(planet, 0, -1))
    comet_path = comet_path_map.get(planet_id)
    if comet_path is not None:
        current_index, path = comet_path
        path_index = current_index + step_offset
        if 0 <= path_index < len(path):
            x, y = path[path_index]
            return x, y, x >= 0.0
        return old_x, old_y, True

    initial = initial_by_id.get(planet_id)
    if initial is not None:
        dx = float(row_get(initial, 2, 0.0)) - CENTER
        dy = float(row_get(initial, 3, 0.0)) - CENTER
        orbit_radius = math.hypot(dx, dy)
        radius = float(row_get(planet, 4, 1.0))
        if orbit_radius + radius < 50.0:
            phase_step = max(0, step - 1)
            angle = math.atan2(dy, dx) + angular_velocity * phase_step
            return CENTER + orbit_radius * math.cos(angle), CENTER + orbit_radius * math.sin(angle), True

    return float(row_get(planet, 2, 0.0)), float(row_get(planet, 3, 0.0)), True


def classify_action(obs: Any, action: list[float]) -> dict[str, Any]:
    planets = list(get(obs, "planets", []) or [])
    initial_planets = list(get(obs, "initial_planets", planets) or [])
    comet_ids = {int(value) for value in get(obs, "comet_planet_ids", []) or []}
    comet_path_map = comet_paths(obs)
    current_step = int(get(obs, "step", get(obs, "current_step", 0)))
    angular_velocity = float(get(obs, "angular_velocity", 0.03))

    source_id = int(action[0])
    angle = float(action[1])
    ship_count = int(action[2])
    source = next((planet for planet in planets if int(row_get(planet, 0, -1)) == source_id), None)
    if source is None:
        return {"kind": "bad_source"}

    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    speed = fleet_speed(ship_count)
    fleet_x = float(row_get(source, 2, 0.0)) + direction_x * (float(row_get(source, 4, 1.0)) + 0.1)
    fleet_y = float(row_get(source, 3, 0.0)) + direction_y * (float(row_get(source, 4, 1.0)) + 0.1)
    initial_by_id = {int(row_get(planet, 0, -1)): planet for planet in initial_planets}
    planet_positions = {
        int(row_get(planet, 0, -1)): (float(row_get(planet, 2, 0.0)), float(row_get(planet, 3, 0.0)))
        for planet in planets
    }

    for step_offset in range(1, MAX_STEPS + 1):
        fleet_new_x = fleet_x + direction_x * speed
        fleet_new_y = fleet_y + direction_y * speed
        next_positions: dict[int, tuple[float, float]] = {}
        hits = []
        for planet_index, planet in enumerate(planets):
            planet_id = int(row_get(planet, 0, -1))
            old_x, old_y = planet_positions[planet_id]
            new_x, new_y, check_collision = next_planet_position(
                planet,
                initial_by_id,
                comet_path_map,
                old_x,
                old_y,
                current_step + step_offset,
                step_offset,
                angular_velocity,
            )
            next_positions[planet_id] = (new_x, new_y)
            if check_collision and swept_pair_hit(
                fleet_x,
                fleet_y,
                fleet_new_x,
                fleet_new_y,
                old_x,
                old_y,
                new_x,
                new_y,
                float(row_get(planet, 4, 1.0)),
            ):
                hits.append((planet_index, planet, new_x, new_y))

        if hits:
            planet_index, planet, new_x, new_y = hits[0]
            planet_id = int(row_get(planet, 0, -1))
            return {
                "kind": "comet" if planet_id in comet_ids else "planet",
                "step_offset": step_offset,
                "target_id": planet_id,
                "target_index": planet_index,
                "target_radius": float(row_get(planet, 4, 1.0)),
                "target_owner": int(row_get(planet, 1, -9)),
            }
        if fleet_new_x < 0.0 or fleet_new_y < 0.0 or fleet_new_x > BOARD_SIZE or fleet_new_y > BOARD_SIZE:
            return {
                "kind": "oob",
                "step_offset": step_offset,
                "fleet_x": fleet_new_x,
                "fleet_y": fleet_new_y,
                "source_x": float(row_get(source, 2, 0.0)),
                "source_y": float(row_get(source, 3, 0.0)),
                "source_radius": float(row_get(source, 4, 1.0)),
                "source_ships": float(row_get(source, 5, 0.0)),
                "speed": speed,
            }
        if segment_intersects_sun(fleet_x, fleet_y, fleet_new_x, fleet_new_y):
            return {"kind": "sun", "step_offset": step_offset}

        fleet_x = fleet_new_x
        fleet_y = fleet_new_y
        planet_positions = next_positions

    return {"kind": "none"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--examples", type=int, default=24)
    args = parser.parse_args()

    module = load_submission()
    counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for seed in range(args.seed, args.seed + args.games):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([str(MAIN_PATH), "random", "random", "random"])
        for step_index in range(1, len(env.steps)):
            previous_obs = env.steps[step_index - 1][0].observation
            actions = getattr(env.steps[step_index][0], "action", None) or []
            for action in actions:
                result = classify_action(previous_obs, action)
                kind = str(result["kind"])
                counts[kind] = counts.get(kind, 0) + 1
                if kind != "planet" and len(examples) < args.examples:
                    examples.append({
                        "seed": seed,
                        "step": step_index - 1,
                        "action": action,
                        **result,
                    })
    print(json.dumps({"games": args.games, "seed": args.seed, "counts": counts, "examples": examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
