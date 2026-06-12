from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = ROOT / "kaggle_submission"
DEFAULT_EXP50 = ROOT / "test_results" / "opponents" / "orbit-wars-exp50" / "main.py"
DEFAULT_ENV_SRC = ROOT / ".external" / "kaggle-env-src" / "kaggle_environments" / "envs" / "orbit_wars"


def ensure_orbit_wars_env(env_src: Path) -> None:
    from kaggle_environments import __file__ as kaggle_init

    env_dst = Path(kaggle_init).resolve().parent / "envs" / "orbit_wars"
    env_dst.mkdir(parents=True, exist_ok=True)
    if env_src.exists():
        for item in env_src.iterdir():
            target = env_dst / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)


def prepare_submission_copy(source_dir: Path, work_dir: Path) -> Path:
    dst = work_dir / "our_submission"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("main.py", "model.bin", "liborbit_wars_agent.so"):
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing submission file: {src}")
        shutil.copy2(src, dst / name)
    return dst / "main.py"


def load_make():
    try:
        from kaggle_environments import make
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "kaggle_environments is not installed. Run this in Kaggle/Molab, "
            "or install kaggle-environments in a short path."
        ) from exc
    return make


def final_scores(env) -> list[float]:
    scores: list[float] = []
    for state in env.steps[-1]:
        reward = getattr(state, "reward", None)
        scores.append(float(reward if reward is not None else 0.0))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--exp50-main", type=Path, default=DEFAULT_EXP50)
    parser.add_argument("--env-src", type=Path, default=DEFAULT_ENV_SRC)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--players", type=int, default=4)
    args = parser.parse_args()

    if args.players != 4:
        raise RuntimeError("this evaluator currently expects 4 players")
    if not args.exp50_main.exists():
        raise FileNotFoundError(f"missing exp50 main.py: {args.exp50_main}")

    make = load_make()
    ensure_orbit_wars_env(args.env_src)

    wins = losses = draws = 0
    score_diff_sum = 0.0
    rows = []
    with tempfile.TemporaryDirectory(prefix="orbitwars_eval_") as tmp:
        our_main = prepare_submission_copy(args.submission_dir, Path(tmp))
        exp50_main = args.exp50_main
        for game in range(args.games):
            seat = game % args.players
            agents = [str(exp50_main), str(exp50_main), str(exp50_main), str(exp50_main)]
            agents[seat] = str(our_main)
            env = make("orbit_wars", configuration={"seed": args.seed + game}, debug=False)
            env.run(agents)
            scores = final_scores(env)
            our_score = scores[seat]
            best_other = max(score for index, score in enumerate(scores) if index != seat)
            diff = our_score - best_other
            if diff > 0:
                wins += 1
                result = "win"
            elif diff < 0:
                losses += 1
                result = "loss"
            else:
                draws += 1
                result = "draw"
            score_diff_sum += diff
            rows.append(
                {
                    "game": game,
                    "seed": args.seed + game,
                    "seat": seat,
                    "scores": scores,
                    "our_score": our_score,
                    "best_exp50_score": best_other,
                    "score_diff": diff,
                    "result": result,
                }
            )

    summary = {
        "event": "eval_submission_vs_exp50",
        "games": args.games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / args.games if args.games else 0.0,
        "mean_score_diff": score_diff_sum / args.games if args.games else 0.0,
        "submission_dir": str(args.submission_dir),
        "exp50_main": str(args.exp50_main),
        "rows": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
