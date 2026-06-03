from __future__ import annotations

import importlib.util
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "kaggle_submission" / "main.py"


def load_submission_agent():
    spec = importlib.util.spec_from_file_location("orbit_wars_submission", MAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def main() -> None:
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset()
    agent = load_submission_agent()
    obs = env.steps[0][0].observation
    print(f"obs_type={type(obs)}")
    print(f"obs_keys={list(obs.keys()) if isinstance(obs, dict) else dir(obs)}")
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    print(f"planet_count={len(raw_planets)}")
    print(f"first_planet={raw_planets[0] if raw_planets else None}")
    raw_initial = (
        obs.get("initial_planets", []) if isinstance(obs, dict) else getattr(obs, "initial_planets", [])
    )
    player = obs.get("player", None) if isinstance(obs, dict) else getattr(obs, "player", None)
    print(f"player={player}")
    print(f"initial_count={len(raw_initial)}")
    print(f"initial_owned={[p for p in raw_initial if p[1] == player][:4]}")
    print(f"current_owned={[p for p in raw_planets if p[1] == player][:4]}")
    actions = agent(obs)
    print(f"initial_action_count={len(actions)}")
    print(f"initial_actions={actions[:16]}")
    raw_planets[0][5] = 103
    actions = agent(obs)
    print(f"boosted_action_count={len(actions)}")
    print(f"boosted_actions={actions[:16]}")


if __name__ == "__main__":
    main()
