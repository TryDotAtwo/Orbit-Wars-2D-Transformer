from __future__ import annotations

import ctypes
import inspect
import math
import sys
from pathlib import Path
from typing import Any

MAX_ROWS = 64
ACTION_TARGETS_PER_SOURCE = 12
MAX_ACTIONS = MAX_ROWS * ACTION_TARGETS_PER_SOURCE
PLANET_STRIDE = 9
ACTION_STRIDE = 3
CENTER_X = 50.0
CENTER_Y = 50.0
ROTATION_RADIUS_LIMIT = 50.0

_MODULE_FILE = globals().get("__file__")
_SEARCH_DIRS = []
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
        self.library.agent_act.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_double,
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
        self.library.agent_act.restype = ctypes.c_int
        self.library.agent_last_error_text.argtypes = [ctypes.c_int]
        self.library.agent_last_error_text.restype = ctypes.c_char_p

        model_bytes = _MODEL_PATH.read_bytes()
        model_buffer = (ctypes.c_ubyte * len(model_bytes)).from_buffer_copy(model_bytes)
        self.handle = self.library.agent_create(model_buffer, len(model_bytes))
        if not self.handle:
            raise RuntimeError("native model contract rejected")

    def act(self, obs: Any) -> list[list[float]]:
        player = int(_get(obs, "player", 0))
        angular_velocity = float(_get(obs, "angular_velocity", 0.0))
        current_step = int(_get(obs, "step", 0))
        planets = _get(obs, "planets", [])
        if not _has_owned_planet(planets, player):
            return []
        initial_planets = _get(obs, "initial_planets", planets)
        comet_ids = [int(value) for value in _get(obs, "comet_planet_ids", [])]
        comet_velocity = _comet_velocity_map(obs)

        planet_buffer = _planet_buffer(planets, angular_velocity, comet_velocity)
        initial_buffer = _planet_buffer(initial_planets, angular_velocity, {})
        comet_buffer = (ctypes.c_int * len(comet_ids))(*comet_ids)
        output_buffer = (ctypes.c_double * (MAX_ACTIONS * ACTION_STRIDE))()

        result = self.library.agent_act(
            self.handle,
            player,
            angular_velocity,
            current_step,
            planet_buffer,
            len(planets),
            initial_buffer,
            len(initial_planets),
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
            moves.append(
                [
                    int(output_buffer[offset]),
                    float(output_buffer[offset + 1]),
                    int(output_buffer[offset + 2]),
                ]
            )
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


def _has_owned_planet(raw_planets: list[list[float]], player: int) -> bool:
    return any(int(raw_planet[1]) == player for raw_planet in raw_planets)


def _planet_buffer(
    raw_planets: list[list[float]],
    angular_velocity: float,
    comet_velocity: dict[int, tuple[float, float]],
) -> ctypes.Array[ctypes.c_double]:
    values: list[float] = []
    for raw_planet in raw_planets:
        planet_id = int(raw_planet[0])
        x = float(raw_planet[2])
        y = float(raw_planet[3])
        radius = float(raw_planet[4])
        velocity_x, velocity_y = comet_velocity.get(
            planet_id, _planet_velocity(x, y, radius, angular_velocity)
        )
        values.extend(
            [
                float(planet_id),
                float(raw_planet[1]),
                x,
                y,
                radius,
                float(raw_planet[5]),
                float(raw_planet[6]),
                velocity_x,
                velocity_y,
            ]
        )
    return (ctypes.c_double * len(values))(*values)


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
        planet_ids = comet_group.get("planet_ids", [])
        paths = comet_group.get("paths", [])
        path_index = int(comet_group.get("path_index", 0))
        for planet_id, path in zip(planet_ids, paths):
            if path_index + 1 < len(path):
                current = path[path_index]
                next_point = path[path_index + 1]
                velocity_by_planet_id[int(planet_id)] = (
                    float(next_point[0]) - float(current[0]),
                    float(next_point[1]) - float(current[1]),
                )
    return velocity_by_planet_id


def agent(obs: Any) -> list[list[float]]:
    global _NATIVE_AGENT
    if _NATIVE_AGENT is None:
        _NATIVE_AGENT = _NativeAgent()
    return _NATIVE_AGENT.act(obs)
