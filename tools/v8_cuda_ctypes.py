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
            ctypes.c_float(5.0),
            ctypes.c_float(50.0),
            ctypes.c_float(6.0),
            ctypes.c_float(1000.0),
            ctypes.c_float(1.5),
            ctypes.c_float(0.3),
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

    def decode_model_actions(self, model: NativeModel, sim: NativeSim, step: int) -> DeviceBatchView:
        self.require(self.lib.orbit_wars_cuda_sim_clear_actions(sim.ptr), "sim_clear_actions")
        self.require(
            self.lib.orbit_wars_cuda_v8_resident_model_decode(
                model.ptr,
                sim.ptr,
                sim.request_games.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                sim.request_players.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                sim.request_games.size,
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
