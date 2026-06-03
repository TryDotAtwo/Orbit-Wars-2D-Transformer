import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace


DEFAULT_REFERENCE_PATHS = [
    Path("/kaggle-env-src/kaggle_environments/envs/orbit_wars/orbit_wars.py"),
    Path("C:/tmp/kaggle-env-src/kaggle_environments/envs/orbit_wars/orbit_wars.py"),
]


def load_reference_module():
    configured = os.environ.get("ORBIT_WARS_REFERENCE_PY")
    candidates = [Path(configured)] if configured else DEFAULT_REFERENCE_PATHS
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("orbit_wars_reference", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module, candidate
    raise FileNotFoundError("official orbit_wars.py reference source not found")


def make_state(planets, fleets, step, angular_velocity, player_count, actions):
    first_observation = SimpleNamespace(
        step=step,
        planets=[planet[:] for planet in planets],
        fleets=[fleet[:] for fleet in fleets],
        next_fleet_id=100,
        angular_velocity=angular_velocity,
        initial_planets=[planet[:] for planet in planets],
        comets=[],
        comet_planet_ids=[],
    )
    state = [
        SimpleNamespace(
            observation=first_observation,
            action=actions[0] if actions else [],
            status="ACTIVE",
            reward=0,
        )
    ]
    for player in range(1, player_count):
        state.append(
            SimpleNamespace(
                observation=SimpleNamespace(player=player),
                action=actions[player] if player < len(actions) else [],
                status="ACTIVE",
                reward=0,
            )
        )
    return state


def reference_scenarios():
    return [
        {
            "name": "combat_user_example",
            "player_count": 4,
            "ship_speed": 5,
            "step": 1,
            "angular_velocity": 0.01,
            "planets": [[0, -1, 80.0, 80.0, 5.0, 10, 0]],
            "fleets": [
                [0, 0, 76.0, 80.0, 0.0, 1, 41],
                [1, 1, 76.0, 80.0, 0.0, 2, 20],
                [2, 1, 76.0, 80.0, 0.0, 2, 20],
                [3, 2, 76.0, 80.0, 0.0, 3, 42],
            ],
            "actions": [[], [], [], []],
        },
        {
            "name": "swept_rotating_planet",
            "player_count": 2,
            "ship_speed": 2,
            "step": 1,
            "angular_velocity": math.pi,
            "planets": [[0, -1, 50.0, 52.0, 1.0, 10, 0]],
            "fleets": [[0, 0, 49.0, 50.0, 0.0, 1, 1000]],
            "actions": [[], []],
        },
        {
            "name": "fast_planet_before_bounds",
            "player_count": 2,
            "ship_speed": 6,
            "step": 1,
            "angular_velocity": 0.01,
            "planets": [[0, 1, 98.0, 50.0, 2.0, 50, 1]],
            "fleets": [[0, 0, 95.0, 50.0, 0.0, 99, 1000]],
            "actions": [[], []],
        },
        {
            "name": "fast_planet_before_sun",
            "player_count": 2,
            "ship_speed": 6,
            "step": 1,
            "angular_velocity": 0.0,
            "planets": [[0, 1, 62.0, 50.0, 2.0, 50, 1]],
            "fleets": [[0, 0, 65.0, 50.0, math.pi, 99, 1000]],
            "actions": [[], []],
        },
        {
            "name": "combat_reinforce",
            "player_count": 2,
            "ship_speed": 6,
            "step": 1,
            "angular_velocity": 0.01,
            "planets": [[0, 0, 80.0, 50.0, 3.0, 10, 1]],
            "fleets": [[0, 0, 76.0, 50.0, 0.0, 99, 25]],
            "actions": [[], []],
        },
        {
            "name": "launch_moves_same_turn",
            "player_count": 2,
            "ship_speed": 6,
            "step": 1,
            "angular_velocity": 0.01,
            "planets": [[0, 0, 10.0, 10.0, 1.0, 20, 1]],
            "fleets": [],
            "actions": [[[0, 0.0, 5]], []],
        },
        {
            "name": "sun_tangent_survives",
            "player_count": 2,
            "ship_speed": 20,
            "step": 1,
            "angular_velocity": 0.01,
            "planets": [[0, 0, 10.0, 10.0, 1.0, 20, 1]],
            "fleets": [[0, 0, 40.0, 40.0, 0.0, 0, 1000]],
            "actions": [[], []],
        },
    ]


def run_reference(module):
    outputs = []
    for scenario in reference_scenarios():
        state = make_state(
            scenario["planets"],
            scenario["fleets"],
            scenario["step"],
            scenario["angular_velocity"],
            scenario["player_count"],
            scenario["actions"],
        )
        env = SimpleNamespace(
            configuration=SimpleNamespace(
                shipSpeed=scenario["ship_speed"],
                episodeSteps=500,
                cometSpeed=4,
            ),
            done=False,
        )
        result = module.interpreter(state, env)[0].observation
        outputs.append(
            {
                "name": scenario["name"],
                "planets": result.planets,
                "fleets": result.fleets,
            }
        )
    return outputs


def run_rust_driver():
    completed = subprocess.run(
        ["cargo", "run", "-p", "orbit-wars-trainer", "--", "--reference-scenarios"],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def assert_close(left, right, path):
    if isinstance(left, float) or isinstance(right, float):
        if not math.isclose(float(left), float(right), abs_tol=0.0005):
            raise AssertionError(f"{path}: left={left} right={right}")
        return
    if isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"{path}: left_len={len(left)} right_len={len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_close(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"{path}: left={left} right={right}")


def compare(reference_outputs, rust_outputs):
    reference_by_name = {item["name"]: item for item in reference_outputs}
    rust_by_name = {item["name"]: item for item in rust_outputs}
    if set(reference_by_name) != set(rust_by_name):
        raise AssertionError(
            f"name_mismatch reference={sorted(reference_by_name)} rust={sorted(rust_by_name)}"
        )
    for name in sorted(reference_by_name):
        assert_close(reference_by_name[name]["planets"], rust_by_name[name]["planets"], f"{name}.planets")
        assert_close(reference_by_name[name]["fleets"], rust_by_name[name]["fleets"], f"{name}.fleets")


def main():
    module, reference_path = load_reference_module()
    reference_outputs = run_reference(module)
    rust_outputs = run_rust_driver()
    compare(reference_outputs, rust_outputs)
    print(
        "status=ok; compared_scenarios={}; reference_source={}".format(
            len(reference_outputs), reference_path
        )
    )


if __name__ == "__main__":
    main()
