from __future__ import annotations

import ctypes
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

MAX_ACTIONS = 8
PLANET_STRIDE = 9
FLEET_STRIDE = 7
ACTION_STRIDE = 3
CENTER_X = 50.0
CENTER_Y = 50.0
ROTATION_RADIUS_LIMIT = 50.0
BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
FLEET_SPEED_MAX = 6.0
FLEET_SPEED_REFERENCE_SHIPS = 1000.0
FLEET_SPEED_CURVE_POWER = 1.5
FLEET_SPAWN_OFFSET = 0.1
MAX_COLLISION_STEPS = 500

_MODULE_FILE = globals().get("__file__")
_SEARCH_DIRS: list[Path] = []
if _MODULE_FILE:
    _SEARCH_DIRS.append(Path(_MODULE_FILE).resolve().parent)
_INSPECT_FILE = inspect.getsourcefile(lambda: None)
if _INSPECT_FILE:
    _SEARCH_DIRS.append(Path(_INSPECT_FILE).resolve().parent)
if sys.argv and sys.argv[0]:
    _SEARCH_DIRS.append(Path(sys.argv[0]).resolve().parent)
_SEARCH_DIRS.append(Path("/kaggle_simulations/agent"))
_SEARCH_DIRS.append(Path.cwd())
_SEARCH_DIRS.append(Path.cwd() / "kaggle_submission")


def _artifact_path(name: str) -> Path:
    for directory in _SEARCH_DIRS:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return _SEARCH_DIRS[0] / name


_LIB_PATH = _artifact_path("liborbit_wars_agent.so")
_MODEL_PATH = _artifact_path("model.bin")


class _NativeAgent:
    def __init__(self) -> None:
        if not _LIB_PATH.exists():
            raise RuntimeError(f"native library missing: {_LIB_PATH}")
        if not _MODEL_PATH.exists():
            raise RuntimeError(f"model file missing: {_MODEL_PATH}")

        self.library = ctypes.CDLL(str(_LIB_PATH))
        self.library.agent_create.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]
        self.library.agent_create.restype = ctypes.c_void_p
        self.library.agent_destroy.argtypes = [ctypes.c_void_p]
        self.library.agent_destroy.restype = None
        self.library.agent_last_error_text.argtypes = [ctypes.c_int]
        self.library.agent_last_error_text.restype = ctypes.c_char_p
        self.library.agent_act_v8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        self.library.agent_act_v8.restype = ctypes.c_int

        model_bytes = _MODEL_PATH.read_bytes()
        model_buffer = (ctypes.c_ubyte * len(model_bytes)).from_buffer_copy(model_bytes)
        self.handle = self.library.agent_create(model_buffer, len(model_bytes))
        if not self.handle:
            raise RuntimeError("native model contract rejected")

    def act(self, obs: Any) -> list[list[float]]:
        if isinstance(obs, str):
            obs = json.loads(obs)
        player = int(_get(obs, "player", 0))
        angular_velocity = float(_get(obs, "angular_velocity", 0.03))
        current_step = int(_get(obs, "step", _get(obs, "current_step", 0)))
        planets = _get(obs, "planets", [])
        if not _has_owned_planet(planets, player):
            return []
        initial_planets = _get(obs, "initial_planets", planets)
        fleets = _get(obs, "fleets", [])
        comet_ids = [int(value) for value in _get(obs, "comet_planet_ids", [])]
        comet_velocity = _comet_velocity_map(obs)
        comet_paths = _comet_path_map(obs)

        planet_buffer = _planet_buffer(planets, angular_velocity, comet_velocity)
        initial_buffer = _planet_buffer(initial_planets, angular_velocity, {})
        fleet_buffer = _fleet_buffer(fleets)
        comet_buffer = (ctypes.c_int * max(1, len(comet_ids)))(*comet_ids) if comet_ids else (ctypes.c_int * 1)()
        output_buffer = (ctypes.c_double * (MAX_ACTIONS * ACTION_STRIDE))()

        result = self.library.agent_act_v8(
            self.handle,
            player,
            angular_velocity,
            current_step,
            planet_buffer,
            len(planets),
            initial_buffer,
            len(initial_planets),
            fleet_buffer,
            len(fleets),
            comet_buffer,
            len(comet_ids),
            output_buffer,
            MAX_ACTIONS,
        )
        if result < 0:
            error_text = self.library.agent_last_error_text(result).decode("utf-8")
            raise RuntimeError(f"native agent failed: code={result}; error={error_text}")

        moves: list[list[float]] = []
        for action_index in range(result):
            offset = action_index * ACTION_STRIDE
            move = [
                int(output_buffer[offset]),
                float(output_buffer[offset + 1]),
                int(output_buffer[offset + 2]),
            ]
            if _is_allowed_target(
                planets,
                initial_planets,
                comet_ids,
                comet_paths,
                current_step,
                angular_velocity,
                move[0],
                move[1],
                move[2],
            ):
                moves.append(move)
        return moves

    def __del__(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self.library.agent_destroy(handle)
            self.handle = None


_NATIVE_AGENT: _NativeAgent | None = None


def _get(obs: Any, key: str, default: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _row_get(row: Any, key: str, index: int, default: Any) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    if isinstance(row, (list, tuple)) and index < len(row):
        return row[index]
    return default


def _has_owned_planet(raw_planets: Any, player: int) -> bool:
    return any(int(_row_get(raw_planet, "owner", 1, -1)) == player for raw_planet in raw_planets or [])


def _is_allowed_target(
    raw_planets: Any,
    raw_initial_planets: Any,
    comet_ids: list[int],
    comet_paths: dict[int, tuple[int, list[tuple[float, float]]]],
    current_step: int,
    angular_velocity: float,
    source_id: int,
    angle: float,
    ship_count: int,
) -> bool:
    source = None
    planets = list(raw_planets or [])
    for raw_planet in planets:
        if int(_row_get(raw_planet, "id", 0, -1)) == source_id:
            source = raw_planet
            break
    if source is None:
        return False

    initial_by_id = {
        int(_row_get(raw_planet, "id", 0, -1)): raw_planet
        for raw_planet in raw_initial_planets or []
    }
    comet_id_set = set(comet_ids)
    source_x = float(_row_get(source, "x", 2, 0.0))
    source_y = float(_row_get(source, "y", 3, 0.0))
    source_radius = float(_row_get(source, "radius", 4, 1.0))
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)

    fleet_x = source_x + direction_x * (source_radius + FLEET_SPAWN_OFFSET)
    fleet_y = source_y + direction_y * (source_radius + FLEET_SPAWN_OFFSET)
    speed = _fleet_speed(ship_count)
    planet_positions = {
        int(_row_get(raw_planet, "id", 0, -1)): (
            float(_row_get(raw_planet, "x", 2, 0.0)),
            float(_row_get(raw_planet, "y", 3, 0.0)),
        )
        for raw_planet in planets
    }

    for step_offset in range(1, MAX_COLLISION_STEPS + 1):
        fleet_new_x = fleet_x + direction_x * speed
        fleet_new_y = fleet_y + direction_y * speed
        next_positions: dict[int, tuple[float, float]] = {}

        for raw_planet in planets:
            planet_id = int(_row_get(raw_planet, "id", 0, -1))
            old_x, old_y = planet_positions.get(
                planet_id,
                (
                    float(_row_get(raw_planet, "x", 2, 0.0)),
                    float(_row_get(raw_planet, "y", 3, 0.0)),
                ),
            )
            new_x, new_y, check_collision = _next_planet_position(
                raw_planet,
                initial_by_id.get(planet_id),
                comet_paths.get(planet_id),
                old_x,
                old_y,
                current_step + step_offset,
                step_offset,
                angular_velocity,
            )
            next_positions[planet_id] = (new_x, new_y)
            if not check_collision:
                continue
            if _swept_pair_hit(
                fleet_x,
                fleet_y,
                fleet_new_x,
                fleet_new_y,
                old_x,
                old_y,
                new_x,
                new_y,
                float(_row_get(raw_planet, "radius", 4, 1.0)),
            ):
                return planet_id not in comet_id_set

        if _point_out_of_bounds(fleet_new_x, fleet_new_y):
            return False
        if _segment_intersects_sun(fleet_x, fleet_y, fleet_new_x, fleet_new_y):
            return False

        fleet_x = fleet_new_x
        fleet_y = fleet_new_y
        planet_positions = next_positions

    return False


def _fleet_speed(ship_count: int) -> float:
    ships = max(1.0, float(ship_count))
    speed_ratio = min(1.0, max(0.0, math.log(ships) / math.log(FLEET_SPEED_REFERENCE_SHIPS)))
    return 1.0 + (FLEET_SPEED_MAX - 1.0) * (speed_ratio ** FLEET_SPEED_CURVE_POWER)


def _next_planet_position(
    raw_planet: Any,
    raw_initial_planet: Any,
    comet_path: tuple[int, list[tuple[float, float]]] | None,
    old_x: float,
    old_y: float,
    step: int,
    step_offset: int,
    angular_velocity: float,
) -> tuple[float, float, bool]:
    if comet_path is not None:
        current_path_index, path = comet_path
        path_index = current_path_index + step_offset
        if 0 <= path_index < len(path):
            x, y = path[path_index]
            return x, y, x >= 0.0
        return old_x, old_y, True

    if raw_initial_planet is not None:
        initial_x = float(_row_get(raw_initial_planet, "x", 2, 0.0))
        initial_y = float(_row_get(raw_initial_planet, "y", 3, 0.0))
        radius = float(_row_get(raw_planet, "radius", 4, 1.0))
        dx = initial_x - CENTER_X
        dy = initial_y - CENTER_Y
        orbital_radius = math.hypot(dx, dy)
        if orbital_radius + radius < ROTATION_RADIUS_LIMIT:
            phase_step = max(0, step - 1)
            current_angle = math.atan2(dy, dx) + angular_velocity * phase_step
            return (
                CENTER_X + orbital_radius * math.cos(current_angle),
                CENTER_Y + orbital_radius * math.sin(current_angle),
                True,
            )

    return (
        float(_row_get(raw_planet, "x", 2, 0.0)),
        float(_row_get(raw_planet, "y", 3, 0.0)),
        True,
    )


def _swept_pair_hit(
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
    if a < sys.float_info.epsilon:
        return c <= 0.0
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return False
    root = math.sqrt(discriminant)
    entry = (-b - root) / (2.0 * a)
    exit = (-b + root) / (2.0 * a)
    return exit >= 0.0 and entry <= 1.0


def _segment_intersects_sun(start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
    dx = end_x - start_x
    dy = end_y - start_y
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        closest_x = start_x
        closest_y = start_y
    else:
        projection = ((CENTER_X - start_x) * dx + (CENTER_Y - start_y) * dy) / length_squared
        projection = min(1.0, max(0.0, projection))
        closest_x = start_x + projection * dx
        closest_y = start_y + projection * dy
    closest_dx = closest_x - CENTER_X
    closest_dy = closest_y - CENTER_Y
    return closest_dx * closest_dx + closest_dy * closest_dy < SUN_RADIUS * SUN_RADIUS


def _point_out_of_bounds(x: float, y: float) -> bool:
    return x < 0.0 or y < 0.0 or x > BOARD_SIZE or y > BOARD_SIZE


def _planet_buffer(
    raw_planets: Any,
    angular_velocity: float,
    comet_velocity: dict[int, tuple[float, float]],
) -> ctypes.Array[ctypes.c_double]:
    values: list[float] = []
    for raw_planet in raw_planets or []:
        planet_id = int(_row_get(raw_planet, "id", 0, 0))
        x = float(_row_get(raw_planet, "x", 2, 0.0))
        y = float(_row_get(raw_planet, "y", 3, 0.0))
        radius = float(_row_get(raw_planet, "radius", 4, 1.0))
        velocity_x = _row_get(raw_planet, "velocity_x", 7, None)
        velocity_y = _row_get(raw_planet, "velocity_y", 8, None)
        if velocity_x is None or velocity_y is None:
            velocity_x, velocity_y = comet_velocity.get(
                planet_id,
                _planet_velocity(x, y, radius, angular_velocity),
            )
        values.extend([
            float(planet_id),
            float(_row_get(raw_planet, "owner", 1, -1)),
            x,
            y,
            radius,
            float(_row_get(raw_planet, "ships", 5, 0.0)),
            float(_row_get(raw_planet, "production", 6, 0.0)),
            float(velocity_x),
            float(velocity_y),
        ])
    return (ctypes.c_double * max(1, len(values)))(*values) if values else (ctypes.c_double * 1)()


def _fleet_buffer(raw_fleets: Any) -> ctypes.Array[ctypes.c_double]:
    values: list[float] = []
    for raw_fleet in raw_fleets or []:
        values.extend([
            float(_row_get(raw_fleet, "id", 0, 0)),
            float(_row_get(raw_fleet, "owner", 1, -1)),
            float(_row_get(raw_fleet, "x", 2, 0.0)),
            float(_row_get(raw_fleet, "y", 3, 0.0)),
            float(_row_get(raw_fleet, "angle", 4, 0.0)),
            float(_row_get(raw_fleet, "from_planet_id", 5, -1)),
            float(_row_get(raw_fleet, "ships", 6, 0.0)),
        ])
    return (ctypes.c_double * max(1, len(values)))(*values) if values else (ctypes.c_double * 1)()


def _planet_velocity(
    x: float,
    y: float,
    radius: float,
    angular_velocity: float,
) -> tuple[float, float]:
    dx = x - CENTER_X
    dy = y - CENTER_Y
    orbital_radius = math.hypot(dx, dy)
    if orbital_radius + radius >= ROTATION_RADIUS_LIMIT:
        return 0.0, 0.0
    return -angular_velocity * dy, angular_velocity * dx


def _comet_velocity_map(obs: Any) -> dict[int, tuple[float, float]]:
    velocity_by_planet_id: dict[int, tuple[float, float]] = {}
    for comet_group in _get(obs, "comets", []):
        planet_ids = comet_group.get("planet_ids", []) if isinstance(comet_group, dict) else []
        paths = comet_group.get("paths", []) if isinstance(comet_group, dict) else []
        path_index = int(comet_group.get("path_index", 0)) if isinstance(comet_group, dict) else 0
        for planet_id, path in zip(planet_ids, paths):
            if path_index + 1 < len(path):
                current = path[path_index]
                next_point = path[path_index + 1]
                velocity_by_planet_id[int(planet_id)] = (
                    float(next_point[0]) - float(current[0]),
                    float(next_point[1]) - float(current[1]),
                )
    return velocity_by_planet_id


def _comet_path_map(obs: Any) -> dict[int, tuple[int, list[tuple[float, float]]]]:
    path_by_planet_id: dict[int, tuple[int, list[tuple[float, float]]]] = {}
    for comet_group in _get(obs, "comets", []):
        if not isinstance(comet_group, dict):
            continue
        planet_ids = comet_group.get("planet_ids", [])
        paths = comet_group.get("paths", [])
        path_index = int(comet_group.get("path_index", 0))
        for planet_id, path in zip(planet_ids, paths):
            points: list[tuple[float, float]] = []
            for point in path:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append((float(point[0]), float(point[1])))
            if points:
                path_by_planet_id[int(planet_id)] = (path_index, points)
    return path_by_planet_id


def agent(obs: Any) -> list[list[float]]:
    global _NATIVE_AGENT
    if _NATIVE_AGENT is None:
        _NATIVE_AGENT = _NativeAgent()
    return _NATIVE_AGENT.act(obs)
