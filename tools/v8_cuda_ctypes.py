"""ctypes bridge for the native OWV8 CUDA runtime."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CudaStatus(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int), ("message", ctypes.c_char_p)]


class CudaTensor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("len", ctypes.c_size_t),
    ]


class CudaShape(ctypes.Structure):
    _fields_ = [
        ("batch_count", ctypes.c_size_t),
        ("token_count", ctypes.c_size_t),
        ("token_features", ctypes.c_size_t),
        ("d_model", ctypes.c_size_t),
        ("head_count", ctypes.c_size_t),
        ("encoder_layer_count", ctypes.c_size_t),
        ("decoder_layer_count", ctypes.c_size_t),
        ("action_slot_count", ctypes.c_size_t),
        ("planet_count", ctypes.c_size_t),
        ("amount_class_count", ctypes.c_size_t),
    ]


class CudaPlanet(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("owner", ctypes.c_int),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("radius", ctypes.c_float),
        ("ships", ctypes.c_float),
        ("production", ctypes.c_float),
        ("velocity_x", ctypes.c_float),
        ("velocity_y", ctypes.c_float),
    ]


class CudaFleet(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("owner", ctypes.c_int),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("angle", ctypes.c_float),
        ("from_planet_id", ctypes.c_int),
        ("ships", ctypes.c_float),
        ("alive", ctypes.c_ubyte),
    ]


class CudaAction(ctypes.Structure):
    _fields_ = [
        ("from_planet_id", ctypes.c_int),
        ("direction_angle", ctypes.c_float),
        ("ship_count", ctypes.c_int),
    ]


class CudaActionLabel(ctypes.Structure):
    _fields_ = [
        ("fire", ctypes.c_int * 8),
        ("source_row", ctypes.c_int * 8),
        ("target_row", ctypes.c_int * 8),
        ("amount_class", ctypes.c_int * 8),
        ("source_planet_id", ctypes.c_int * 8),
        ("target_planet_id", ctypes.c_int * 8),
        ("ship_count", ctypes.c_int * 8),
        ("confidence", ctypes.c_float * 8),
    ]


class CudaSimConfig(ctypes.Structure):
    _fields_ = [
        ("game_count", ctypes.c_size_t),
        ("planet_count", ctypes.c_size_t),
        ("max_fleets_per_game", ctypes.c_size_t),
        ("max_players", ctypes.c_size_t),
        ("max_actions_per_player", ctypes.c_size_t),
        ("step", ctypes.c_int),
        ("episode_steps", ctypes.c_int),
        ("angular_velocity", ctypes.c_float),
        ("board_size", ctypes.c_float),
        ("board_center", ctypes.c_float),
        ("sun_radius", ctypes.c_float),
        ("rotation_radius_limit", ctypes.c_float),
        ("fleet_speed_max", ctypes.c_float),
        ("fleet_speed_reference_ships", ctypes.c_float),
        ("fleet_speed_curve_power", ctypes.c_float),
        ("fleet_spawn_offset", ctypes.c_float),
    ]


class CudaSimStats(ctypes.Structure):
    _fields_ = [
        ("launched_fleet_count", ctypes.c_int),
        ("launched_ship_count", ctypes.c_int),
        ("hit_fleet_count", ctypes.c_int),
        ("hit_ship_count", ctypes.c_int),
        ("sun_destroyed_fleet_count", ctypes.c_int),
        ("sun_destroyed_ship_count", ctypes.c_int),
        ("out_of_bounds_destroyed_fleet_count", ctypes.c_int),
        ("out_of_bounds_destroyed_ship_count", ctypes.c_int),
        ("captured_planet_count", ctypes.c_int),
        ("overflow_fleet_count", ctypes.c_int),
    ]


class CudaGameStatus(ctypes.Structure):
    _fields_ = [("done", ctypes.c_int), ("winner", ctypes.c_int), ("step", ctypes.c_int)]


class DeviceBatchView(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.c_void_p),
        ("token_type_ids", ctypes.c_void_p),
        ("owner_ids", ctypes.c_void_p),
        ("padding_mask", ctypes.c_void_p),
        ("planet_mask", ctypes.c_void_p),
        ("labels", ctypes.c_void_p),
        ("labels_fire", ctypes.c_void_p),
        ("labels_source", ctypes.c_void_p),
        ("labels_target", ctypes.c_void_p),
        ("labels_amount", ctypes.c_void_p),
        ("labels_confidence", ctypes.c_void_p),
        ("request_count", ctypes.c_size_t),
        ("token_count", ctypes.c_size_t),
        ("token_features", ctypes.c_size_t),
        ("planet_count", ctypes.c_size_t),
        ("action_slots", ctypes.c_size_t),
    ]


@dataclass
class NativeModel:
    ptr: ctypes.c_void_p
    tensor_names: list[ctypes.c_char_p]
    host_arrays: list[np.ndarray]


@dataclass
class NativeSim:
    ptr: ctypes.c_void_p
    config: CudaSimConfig
    request_games: np.ndarray
    request_players: np.ndarray


class V8CudaRuntime:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self._bind()

    def _bind(self) -> None:
        self.lib.orbit_wars_cuda_v8_status.restype = CudaStatus
        self.lib.orbit_wars_cuda_v8_model_create.argtypes = [
            ctypes.POINTER(CudaTensor),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.orbit_wars_cuda_v8_model_create.restype = CudaStatus
        self.lib.orbit_wars_cuda_v8_model_destroy.argtypes = [ctypes.c_void_p]
        self.lib.orbit_wars_cuda_sim_create.argtypes = [CudaSimConfig, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.orbit_wars_cuda_sim_create.restype = CudaStatus
        self.lib.orbit_wars_cuda_sim_destroy.argtypes = [ctypes.c_void_p]
        self.lib.orbit_wars_cuda_sim_load_with_angular_velocities.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(CudaPlanet),
            ctypes.POINTER(CudaPlanet),
            ctypes.POINTER(CudaFleet),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.orbit_wars_cuda_sim_load_with_angular_velocities.restype = CudaStatus
        self.lib.orbit_wars_cuda_sim_load_request_plan.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
        ]
        self.lib.orbit_wars_cuda_sim_load_request_plan.restype = CudaStatus
        self.lib.orbit_wars_cuda_sim_clear_actions.argtypes = [ctypes.c_void_p]
        self.lib.orbit_wars_cuda_sim_clear_actions.restype = CudaStatus
        self.lib.orbit_wars_cuda_v8_resident_model_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.orbit_wars_cuda_v8_resident_model_decode.restype = CudaStatus
        self.lib.orbit_wars_cuda_sim_step_device_actions.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.orbit_wars_cuda_sim_step_device_actions.restype = CudaStatus
        self.lib.orbit_wars_cuda_sim_read_status_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(CudaGameStatus),
            ctypes.POINTER(CudaSimStats),
        ]
        self.lib.orbit_wars_cuda_sim_read_status_stats.restype = CudaStatus
        self.lib.orbit_wars_cuda_sim_score_diff_rewards.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.orbit_wars_cuda_sim_score_diff_rewards.restype = CudaStatus
        self.lib.orbit_wars_cuda_sim_read.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(CudaPlanet),
            ctypes.POINTER(CudaFleet),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(CudaSimStats),
        ]
        self.lib.orbit_wars_cuda_sim_read.restype = CudaStatus
        self.lib.orbit_wars_cuda_sim_read_actions.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(CudaAction),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.orbit_wars_cuda_sim_read_actions.restype = CudaStatus
        self.lib.orbit_wars_cuda_v8_last_batch_device_view.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(DeviceBatchView),
        ]
        self.lib.orbit_wars_cuda_v8_last_batch_device_view.restype = CudaStatus

    @staticmethod
    def require(status: CudaStatus, label: str) -> None:
        if status.code != 0:
            message = status.message.decode("utf-8", errors="replace") if status.message else ""
            raise RuntimeError(f"{label} failed: {message}")

    def status(self) -> None:
        self.require(self.lib.orbit_wars_cuda_v8_status(), "cuda_v8_status")

    def create_model_from_state(self, state: dict[str, torch.Tensor]) -> NativeModel:
        names: list[ctypes.c_char_p] = []
        arrays: list[np.ndarray] = []
        tensors = (CudaTensor * len(state))()
        for index, (name, tensor) in enumerate(sorted(state.items())):
            array = tensor.detach().cpu().contiguous().float().numpy()
            arrays.append(array)
            encoded = ctypes.c_char_p(name.encode("utf-8"))
            names.append(encoded)
            tensors[index] = CudaTensor(
                encoded,
                array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                array.size,
            )
        out = ctypes.c_void_p()
        self.require(self.lib.orbit_wars_cuda_v8_model_create(tensors, len(state), ctypes.byref(out)), "model_create")
        return NativeModel(out, names, arrays)

    def destroy_model(self, model: NativeModel) -> None:
        if model.ptr:
            self.lib.orbit_wars_cuda_v8_model_destroy(model.ptr)
            model.ptr = ctypes.c_void_p()

    def create_sim(
        self,
        *,
        games: int,
        players: int,
        planet_count: int = 64,
        max_fleets: int = 4096,
        max_actions: int = 8,
        step_limit: int = 500,
    ) -> NativeSim:
        config = CudaSimConfig(
            games,
            planet_count,
            max_fleets,
            players,
            max_actions,
            0,
            step_limit,
            ctypes.c_float(0.03),
            ctypes.c_float(100.0),
            ctypes.c_float(50.0),
            ctypes.c_float(10.0),
            ctypes.c_float(50.0),
            ctypes.c_float(6.0),
            ctypes.c_float(1000.0),
            ctypes.c_float(1.5),
            ctypes.c_float(0.1),
        )
        out = ctypes.c_void_p()
        self.require(self.lib.orbit_wars_cuda_sim_create(config, ctypes.byref(out)), "sim_create")
        request_games = np.repeat(np.arange(games, dtype=np.int32), players)
        request_players = np.tile(np.arange(players, dtype=np.int32), games)
        self.require(
            self.lib.orbit_wars_cuda_sim_load_request_plan(
                out,
                request_games.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                request_players.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                request_games.size,
            ),
            "sim_load_request_plan",
        )
        return NativeSim(out, config, request_games, request_players)

    def destroy_sim(self, sim: NativeSim) -> None:
        if sim.ptr:
            self.lib.orbit_wars_cuda_sim_destroy(sim.ptr)
            sim.ptr = ctypes.c_void_p()

    def load_simple_games(self, sim: NativeSim) -> None:
        planets, initial_planets, fleets, next_ids, angular = simple_games_host_arrays(
            sim.config.game_count,
            sim.config.max_players,
            sim.config.planet_count,
            sim.config.max_fleets_per_game,
        )
        self.load_host_arrays(sim, planets, initial_planets, fleets, next_ids, angular)

    def load_official_like_games(self, sim: NativeSim, *, generation: int = 1) -> None:
        planets, initial_planets, fleets, next_ids, angular = official_like_games_host_arrays(
            sim.config.game_count,
            sim.config.max_players,
            sim.config.planet_count,
            sim.config.max_fleets_per_game,
            generation=generation,
        )
        self.load_host_arrays(sim, planets, initial_planets, fleets, next_ids, angular)

    def load_host_arrays(
        self,
        sim: NativeSim,
        planets: np.ndarray,
        initial_planets: np.ndarray,
        fleets: np.ndarray,
        next_ids: np.ndarray,
        angular: np.ndarray,
    ) -> None:
        self.require(
            self.lib.orbit_wars_cuda_sim_load_with_angular_velocities(
                sim.ptr,
                planets.ctypes.data_as(ctypes.POINTER(CudaPlanet)),
                initial_planets.ctypes.data_as(ctypes.POINTER(CudaPlanet)),
                fleets.ctypes.data_as(ctypes.POINTER(CudaFleet)),
                next_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                angular.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ),
            "sim_load_with_angular_velocities",
        )

    def decode_model_actions(self, model: NativeModel, sim: NativeSim, step: int, *, clear_actions: bool = True) -> DeviceBatchView:
        if clear_actions:
            self.require(self.lib.orbit_wars_cuda_sim_clear_actions(sim.ptr), "sim_clear_actions")
        return self.decode_model_actions_for_requests(
            model,
            sim,
            step,
            sim.request_games,
            sim.request_players,
        )

    def decode_model_actions_for_requests(
        self,
        model: NativeModel,
        sim: NativeSim,
        step: int,
        request_games: np.ndarray,
        request_players: np.ndarray,
    ) -> DeviceBatchView:
        request_games = np.ascontiguousarray(request_games, dtype=np.int32)
        request_players = np.ascontiguousarray(request_players, dtype=np.int32)
        if request_games.shape != request_players.shape:
            raise ValueError(f"request plan shape mismatch: {request_games.shape} != {request_players.shape}")
        self.require(
            self.lib.orbit_wars_cuda_v8_resident_model_decode(
                model.ptr,
                sim.ptr,
                request_games.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                request_players.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                request_games.size,
                step,
            ),
            "resident_model_decode",
        )
        view = DeviceBatchView()
        self.require(
            self.lib.orbit_wars_cuda_v8_last_batch_device_view(model.ptr, ctypes.byref(view)),
            "last_batch_device_view",
        )
        return view

    def step_device_actions(self, sim: NativeSim, step: int) -> None:
        self.require(self.lib.orbit_wars_cuda_sim_step_device_actions(sim.ptr, step), "sim_step_device_actions")

    def read_status_stats(self, sim: NativeSim) -> tuple[list[CudaGameStatus], list[CudaSimStats]]:
        statuses = (CudaGameStatus * sim.config.game_count)()
        stats = (CudaSimStats * (sim.config.game_count * sim.config.max_players))()
        self.require(
            self.lib.orbit_wars_cuda_sim_read_status_stats(sim.ptr, statuses, stats),
            "sim_read_status_stats",
        )
        return list(statuses), list(stats)

    def score_diff_rewards(self, sim: NativeSim) -> torch.Tensor:
        rewards = torch.empty(
            (sim.config.game_count, sim.config.max_players),
            device="cuda",
            dtype=torch.float32,
        )
        self.require(
            self.lib.orbit_wars_cuda_sim_score_diff_rewards(
                sim.ptr,
                ctypes.c_void_p(rewards.data_ptr()),
            ),
            "sim_score_diff_rewards",
        )
        return rewards

    def read_state(self, sim: NativeSim) -> tuple[list[CudaPlanet], list[CudaFleet], list[int], list[CudaSimStats]]:
        planets = (CudaPlanet * (sim.config.game_count * sim.config.planet_count))()
        fleets = (CudaFleet * (sim.config.game_count * sim.config.max_fleets_per_game))()
        next_ids = (ctypes.c_int * sim.config.game_count)()
        stats = (CudaSimStats * (sim.config.game_count * sim.config.max_players))()
        self.require(
            self.lib.orbit_wars_cuda_sim_read(sim.ptr, planets, fleets, next_ids, stats),
            "sim_read",
        )
        return list(planets), list(fleets), [int(value) for value in next_ids], list(stats)


def simple_games_host_arrays(
    games: int,
    players: int,
    planet_count: int,
    max_fleets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    planet_dtype = np.dtype([
        ("id", np.int32),
        ("owner", np.int32),
        ("x", np.float32),
        ("y", np.float32),
        ("radius", np.float32),
        ("ships", np.float32),
        ("production", np.float32),
        ("velocity_x", np.float32),
        ("velocity_y", np.float32),
    ], align=True)
    fleet_dtype = np.dtype([
        ("id", np.int32),
        ("owner", np.int32),
        ("x", np.float32),
        ("y", np.float32),
        ("angle", np.float32),
        ("from_planet_id", np.int32),
        ("ships", np.float32),
        ("alive", np.uint8),
    ], align=True)
    if planet_dtype.itemsize != ctypes.sizeof(CudaPlanet):
        raise RuntimeError(f"bad planet dtype size {planet_dtype.itemsize} != {ctypes.sizeof(CudaPlanet)}")
    if fleet_dtype.itemsize != ctypes.sizeof(CudaFleet):
        raise RuntimeError(f"bad fleet dtype size {fleet_dtype.itemsize} != {ctypes.sizeof(CudaFleet)}")
    planets = np.zeros(games * planet_count, dtype=planet_dtype)
    initial = np.zeros(games * planet_count, dtype=planet_dtype)
    fleets = np.zeros(games * max_fleets, dtype=fleet_dtype)
    next_ids = np.full(games, 1, dtype=np.int32)
    angular = np.full(games, 0.03, dtype=np.float32)
    home_positions = [(18.0, 18.0), (82.0, 18.0), (82.0, 82.0), (18.0, 82.0)]
    neutral_positions = [(50.0, 20.0), (80.0, 50.0), (50.0, 80.0), (20.0, 50.0), (50.0, 50.0)]
    for game in range(games):
        base = game * planet_count
        row = 0
        for player in range(players):
            x, y = home_positions[player % len(home_positions)]
            planets[base + row] = (row, player, x, y, 3.5, 80.0, 5.0, 0.0, 0.0)
            row += 1
        for index, (x, y) in enumerate(neutral_positions):
            planets[base + row] = (row, -1, x, y, 2.5, 30.0 + 5.0 * index, 2.0, 0.0, 0.0)
            row += 1
        for index in range(row, planet_count):
            planets[base + index] = (-1000 - index, -9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        initial[base:base + planet_count] = planets[base:base + planet_count]
    return planets, initial, fleets, next_ids, angular


MAP_SEED_BASE = 0x4F_57_4D_41_50
MAP_GENERATION_SEED_FACTOR = 1_000_003
MAP_GAME_SEED_FACTOR = 9_176
MAP_RNG_MULTIPLIER = 6_364_136_223_846_793_005
MAP_RNG_INCREMENT = 1_442_695_040_888_963_407
MAP_RNG_FLOAT_SCALE = 16_777_216.0
MAP_GENERATION_ATTEMPT_LIMIT = 5_000
MAP_MIN_PLANET_GROUPS = 5
MAP_MAX_PLANET_GROUPS = 10
MAP_MIN_STATIC_GROUPS = 3
MAP_PLANET_CLEARANCE = 7.0
MAP_HOME_PLANET_SHIPS = 10.0
MAP_STATIC_AXIS_CLEARANCE = 5.0
MAP_ORBITING_MIN_COORDINATE_OFFSET = 15.0
MAP_ORBITING_EDGE_CLEARANCE = 5.0
MAP_SUN_SPAWN_CLEARANCE = 10.0
MAP_STATIC_SHIP_MIN = 5
MAP_STATIC_SHIP_MAX = 99
MAP_ORBITING_SHIP_MIN = 5
MAP_ORBITING_SHIP_MAX = 30
MAP_PRODUCTION_MIN = 1
MAP_PRODUCTION_MAX = 5
MAP_GROUP_SIZE = 4


class MapRng:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFF_FFFF_FFFF_FFFF

    def next_u64(self) -> int:
        self.state = (self.state * MAP_RNG_MULTIPLIER + MAP_RNG_INCREMENT) & 0xFFFF_FFFF_FFFF_FFFF
        return self.state

    def next_f32(self) -> float:
        return float(self.next_u64() >> 40) / MAP_RNG_FLOAT_SCALE

    def range_f32(self, low: float, high: float) -> float:
        return low + (high - low) * self.next_f32()

    def range_usize(self, low: int, high: int) -> int:
        return low + int(self.next_u64() % (high - low + 1))


def official_like_games_host_arrays(
    games: int,
    players: int,
    planet_count: int,
    max_fleets: int,
    *,
    generation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    planet_dtype = np.dtype([
        ("id", np.int32),
        ("owner", np.int32),
        ("x", np.float32),
        ("y", np.float32),
        ("radius", np.float32),
        ("ships", np.float32),
        ("production", np.float32),
        ("velocity_x", np.float32),
        ("velocity_y", np.float32),
    ], align=True)
    fleet_dtype = np.dtype([
        ("id", np.int32),
        ("owner", np.int32),
        ("x", np.float32),
        ("y", np.float32),
        ("angle", np.float32),
        ("from_planet_id", np.int32),
        ("ships", np.float32),
        ("alive", np.uint8),
    ], align=True)
    if planet_dtype.itemsize != ctypes.sizeof(CudaPlanet):
        raise RuntimeError(f"bad planet dtype size {planet_dtype.itemsize} != {ctypes.sizeof(CudaPlanet)}")
    if fleet_dtype.itemsize != ctypes.sizeof(CudaFleet):
        raise RuntimeError(f"bad fleet dtype size {fleet_dtype.itemsize} != {ctypes.sizeof(CudaFleet)}")
    if players not in (2, 4):
        raise ValueError(f"unsupported player count for official-like maps: {players}")
    planets = np.zeros(games * planet_count, dtype=planet_dtype)
    initial = np.zeros(games * planet_count, dtype=planet_dtype)
    fleets = np.zeros(games * max_fleets, dtype=fleet_dtype)
    next_ids = np.full(games, 1, dtype=np.int32)
    angular = np.zeros(games, dtype=np.float32)
    for game in range(games):
        seed = map_seed(generation, game)
        rng = MapRng(seed)
        angular[game] = np.float32(rng.range_f32(0.025, 0.05))
        rows = generate_official_like_planets(rng)
        assign_home_planets(rows, players, rng)
        if len(rows) > planet_count:
            raise RuntimeError(f"official-like map has too many planets: {len(rows)} > {planet_count}")
        base = game * planet_count
        for row, values in enumerate(rows):
            planets[base + row] = values
        for row in range(len(rows), planet_count):
            planets[base + row] = (-1000 - row, -9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        initial[base:base + planet_count] = planets[base:base + planet_count]
    return planets, initial, fleets, next_ids, angular


def map_seed(generation: int, game_index: int) -> int:
    return MAP_SEED_BASE ^ (generation * MAP_GENERATION_SEED_FACTOR) ^ (game_index * MAP_GAME_SEED_FACTOR)


def generate_official_like_planets(rng: MapRng) -> list[tuple[int, int, float, float, float, float, float, float, float]]:
    target_groups = rng.range_usize(MAP_MIN_PLANET_GROUPS, MAP_MAX_PLANET_GROUPS)
    target_planets = target_groups * MAP_GROUP_SIZE
    planets: list[tuple[int, int, float, float, float, float, float, float, float]] = []
    next_id = 0
    static_groups = 0
    for _ in range(MAP_GENERATION_ATTEMPT_LIMIT):
        if static_groups >= MAP_MIN_STATIC_GROUPS:
            break
        group = generate_static_planet_group(next_id, planets, rng)
        if group is not None:
            planets.extend(group)
            next_id += MAP_GROUP_SIZE
            static_groups += 1
    attempts = 0
    has_orbiting = False
    while len(planets) < target_planets or (not has_orbiting and attempts < MAP_GENERATION_ATTEMPT_LIMIT):
        attempts += 1
        if attempts >= MAP_GENERATION_ATTEMPT_LIMIT:
            break
        group = generate_orbiting_or_static_planet_group(next_id, planets, rng)
        if group is None:
            continue
        if any(planet_orbits(row) for row in group):
            has_orbiting = True
        planets.extend(group)
        next_id += MAP_GROUP_SIZE
    if len(planets) < MAP_MIN_PLANET_GROUPS * MAP_GROUP_SIZE:
        raise RuntimeError("official-like map generation produced too few planets")
    return planets


def generate_static_planet_group(next_id: int, existing: list[tuple], rng: MapRng) -> list[tuple] | None:
    production = float(rng.range_usize(MAP_PRODUCTION_MIN, MAP_PRODUCTION_MAX))
    radius = planet_radius_from_production(production)
    angle = rng.range_f32(0.0, np.pi / 2.0)
    min_orbital = 50.0 - radius
    max_orbital = (100.0 - 50.0 - radius) / max(np.cos(angle), np.sin(angle))
    if min_orbital > max_orbital:
        return None
    orbital_radius = rng.range_f32(min_orbital, max_orbital)
    x = 50.0 + orbital_radius * np.cos(angle)
    y = 50.0 + orbital_radius * np.sin(angle)
    if (
        x + radius > 100.0
        or x - radius < 0.0
        or y + radius > 100.0
        or y - radius < 0.0
        or x - 50.0 < radius + MAP_STATIC_AXIS_CLEARANCE
        or y - 50.0 < radius + MAP_STATIC_AXIS_CLEARANCE
    ):
        return None
    ships = float(min(
        rng.range_usize(MAP_STATIC_SHIP_MIN, MAP_STATIC_SHIP_MAX),
        rng.range_usize(MAP_STATIC_SHIP_MIN, MAP_STATIC_SHIP_MAX),
    ))
    group = symmetric_planet_group(next_id, x, y, radius, ships, production)
    return group if planet_group_has_clearance(group, existing) else None


def generate_orbiting_or_static_planet_group(next_id: int, existing: list[tuple], rng: MapRng) -> list[tuple] | None:
    production = float(rng.range_usize(MAP_PRODUCTION_MIN, MAP_PRODUCTION_MAX))
    radius = planet_radius_from_production(production)
    x = rng.range_f32(50.0 + MAP_ORBITING_MIN_COORDINATE_OFFSET, 100.0 - radius - MAP_ORBITING_EDGE_CLEARANCE)
    y = rng.range_f32(50.0 + MAP_ORBITING_MIN_COORDINATE_OFFSET, 100.0 - radius - MAP_ORBITING_EDGE_CLEARANCE)
    orbital_radius = distance_xy(x, y, 50.0, 50.0)
    if orbital_radius < 5.0 + radius + MAP_SUN_SPAWN_CLEARANCE:
        return None
    if orbital_radius + radius >= 50.0 and (
        x + radius > 100.0 or x - radius < 0.0 or y + radius > 100.0 or y - radius < 0.0
    ):
        return None
    ships = float(rng.range_usize(MAP_ORBITING_SHIP_MIN, MAP_ORBITING_SHIP_MAX))
    group = symmetric_planet_group(next_id, x, y, radius, ships, production)
    if planet_group_has_clearance(group, existing) and planet_group_orbit_static_cross_check(group, existing):
        return group
    return None


def symmetric_planet_group(next_id: int, x: float, y: float, radius: float, ships: float, production: float) -> list[tuple]:
    return [
        planet_tuple(next_id, -1, y, x, radius, ships, production),
        planet_tuple(next_id + 1, -1, 100.0 - x, y, radius, ships, production),
        planet_tuple(next_id + 2, -1, x, 100.0 - y, radius, ships, production),
        planet_tuple(next_id + 3, -1, 100.0 - y, 100.0 - x, radius, ships, production),
    ]


def assign_home_planets(planets: list[tuple], players: int, rng: MapRng) -> None:
    if len(planets) < MAP_GROUP_SIZE or len(planets) % MAP_GROUP_SIZE != 0:
        raise RuntimeError("home group unavailable")
    base = rng.range_usize(0, len(planets) // MAP_GROUP_SIZE - 1) * MAP_GROUP_SIZE
    if players == 2:
        planets[base] = with_owner_ships(planets[base], 0, MAP_HOME_PLANET_SHIPS)
        planets[base + 3] = with_owner_ships(planets[base + 3], 1, MAP_HOME_PLANET_SHIPS)
    else:
        for player in range(4):
            planets[base + player] = with_owner_ships(planets[base + player], player, MAP_HOME_PLANET_SHIPS)


def planet_tuple(id_value: int, owner: int, x: float, y: float, radius: float, ships: float, production: float) -> tuple:
    return (id_value, owner, float(x), float(y), float(radius), float(ships), float(production), 0.0, 0.0)


def with_owner_ships(row: tuple, owner: int, ships: float) -> tuple:
    return (row[0], owner, row[2], row[3], row[4], ships, row[6], row[7], row[8])


def planet_radius_from_production(production: float) -> float:
    return 1.0 + float(np.log(production))


def planet_orbits(row: tuple) -> bool:
    return distance_xy(row[2], row[3], 50.0, 50.0) + row[4] < 50.0


def planet_group_has_clearance(group: list[tuple], existing: list[tuple]) -> bool:
    return all(
        distance_xy(candidate[2], candidate[3], planet[2], planet[3])
        >= candidate[4] + planet[4] + MAP_PLANET_CLEARANCE
        for candidate in group
        for planet in existing
    )


def planet_group_orbit_static_cross_check(group: list[tuple], existing: list[tuple]) -> bool:
    for candidate in group:
        for planet in existing:
            if planet_orbits(candidate) == planet_orbits(planet):
                continue
            candidate_orbital = distance_xy(candidate[2], candidate[3], 50.0, 50.0)
            planet_orbital = distance_xy(planet[2], planet[3], 50.0, 50.0)
            if abs(candidate_orbital - planet_orbital) < candidate[4] + planet[4] + MAP_PLANET_CLEARANCE:
                return False
    return True


def distance_xy(left_x: float, left_y: float, right_x: float, right_y: float) -> float:
    return float(np.hypot(left_x - right_x, left_y - right_y))
