from __future__ import annotations

import importlib.util
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "kaggle_submission" / "main.py"


def load_submission_agent():
    spec = importlib.util.spec_from_file_location("orbit_wars_submission_smoke", MAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def main() -> None:
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run(["/workspace/kaggle_submission/main.py", "random", "random", "random"])
    final = env.steps[-1]
    print(f"steps={len(env.steps)}")
    for step_index, step in enumerate(env.steps):
        action = getattr(step[0], "action", None)
        if action:
            print(f"first_non_empty_action_step={step_index}; action_count={len(action)}; action={action[:8]}")
            break
    else:
        print("first_non_empty_action_step=none")
    agent = load_submission_agent()
    for probe_step in [0, 10, 50, 93]:
        obs = env.steps[min(probe_step, len(env.steps) - 1)][0].observation
        planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
        player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
        owned = [planet for planet in planets if planet[1] == player]
        actions = agent(obs)
        print(
            f"direct_probe_step={probe_step}; owned={[(p[0], p[5]) for p in owned]}; action_count={len(actions)}; actions={actions[:8]}"
        )
    for index, state in enumerate(final):
        print(
            f"player={index}; reward={state.reward}; status={state.status}; "
            f"info={getattr(state, 'info', None)}; logs={getattr(state, 'logs', None)}"
        )


if __name__ == "__main__":
    main()
