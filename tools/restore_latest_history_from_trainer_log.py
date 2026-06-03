#!/usr/bin/env python3
import glob
import json
import os
import re


LATEST_PATH = "dashboard/public/telemetry/latest.json"
TRAIN_LOG_PATH = "artifacts/full_training_2026-06-01_self_train_current_contract.log"
TELEMETRY_DIR = "dashboard/public/telemetry"
DEFAULT_REPLAY_GAME_COUNT = 16
DEFAULT_ACTUAL_GAME_COUNT = 4
GENERATION_LOG_DIR = "dashboard/public/telemetry/generation_logs"


GEN_DONE_RE = re.compile(
    r"event=generation_done; run_id=(?P<run_id>[^;]+); generation=(?P<generation>\d+); "
    r"seconds=(?P<generationSeconds>[0-9.]+); win_rate=(?P<winRate>[0-9.]+); "
    r"games_per_second=(?P<gamesPerSecond>[0-9.]+); turns_per_second=(?P<turnsPerSecond>[0-9.]+); "
    r"p95_latency_ms=(?P<p95LatencyMs>[0-9.]+); backprop_samples=(?P<backpropSamples>\d+);"
)
PHASE_DONE_RE = re.compile(
    r"event=phase_done; run_id=(?P<run_id>[^;]+); generation=(?P<generation>\d+); "
    r"phase=(?P<phase>[^;]+); seconds=(?P<seconds>[0-9.]+)"
)
REPLAY_WRITE_RE = re.compile(r"replay_write_seconds=(?P<replayWriteSeconds>[0-9.]+)")
EVALUATION_RE = re.compile(
    r"evaluated_games=(?P<evaluatedGames>\d+); model_action_calls=(?P<modelActionCalls>\d+); "
    r"inference_batch_calls=(?P<inferenceBatchCalls>\d+); max_inference_batch_size=(?P<maxInferenceBatchSize>\d+); "
    r"simultaneous_games=(?P<simultaneousGames>\d+)"
)
RUN_ID_RE = re.compile(r"run_id=(?P<run_id>[^;]+)")


def detect_run_id(latest: dict) -> str:
    run_id = latest.get("runId")
    if run_id:
        return str(run_id)
    last_run_id = ""
    if os.path.exists(TRAIN_LOG_PATH):
        with open(TRAIN_LOG_PATH, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = RUN_ID_RE.search(line)
                if match:
                    last_run_id = match.group("run_id")
    if not last_run_id:
        raise RuntimeError("restore_latest_history_run_id_missing")
    return last_run_id


def load_latest() -> dict:
    if not os.path.exists(LATEST_PATH) or os.path.getsize(LATEST_PATH) == 0:
        return {}
    with open(LATEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_template(generation: int) -> dict:
    return {
        "generation": generation,
        "winRate": 0.0,
        "gamesPerSecond": 0.0,
        "turnsPerSecond": 0.0,
        "p95LatencyMs": 0.0,
        "gpuUtilization": 0,
        "evaluatedGames": 0,
        "sampledReplayGames": 40,
        "modelActionCalls": 0,
        "launchActions": 0,
        "launchedShips": 0,
        "captures": 0,
        "fleetHits": 0,
        "hitShips": 0,
        "sunDestroyedFleets": 0,
        "sunDestroyedShips": 0,
        "avgFleetSize": 0.0,
        "avgLaunchActionsPerTurn": 0.0,
        "avgLaunchedShipsPerTurn": 0.0,
        "avgModelActionMs": 0.0,
        "inferenceBatchCalls": 0,
        "maxInferenceBatchSize": 1280,
        "simultaneousGames": 320,
        "modelActionSeconds": 0.0,
        "simulationStepSeconds": 0.0,
        "evaluationSeconds": 0.0,
        "replaySeconds": 0.0,
        "replayWriteSeconds": 0.0,
        "generationValidationGames": 0,
        "generationValidationSeconds": 0.0,
        "backpropSamples": 0,
        "backpropModels": 12,
        "backpropSeconds": 0.0,
        "reproductionSeconds": 0.0,
        "generationSeconds": 0.0,
    }


def parse_log_metrics(run_id: str) -> dict[int, dict]:
    metrics: dict[int, dict] = {}
    if not os.path.exists(TRAIN_LOG_PATH):
        return metrics
    with open(TRAIN_LOG_PATH, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            phase = PHASE_DONE_RE.search(line)
            if phase and phase.group("run_id") == run_id:
                generation = int(phase.group("generation"))
                metric = metrics.setdefault(generation, metric_template(generation))
                seconds = float(phase.group("seconds"))
                phase_name = phase.group("phase")
                if phase_name == "evaluation":
                    metric["evaluationSeconds"] = seconds
                    details = EVALUATION_RE.search(line)
                    if details:
                        for field in (
                            "evaluatedGames",
                            "modelActionCalls",
                            "inferenceBatchCalls",
                            "maxInferenceBatchSize",
                            "simultaneousGames",
                        ):
                            metric[field] = int(details.group(field))
                elif phase_name == "backprop":
                    metric["backpropSeconds"] = seconds
                elif phase_name == "replay_write":
                    match = REPLAY_WRITE_RE.search(line)
                    metric["replayWriteSeconds"] = float(match.group("replayWriteSeconds")) if match else seconds
                elif phase_name == "reproduction":
                    metric["reproductionSeconds"] = seconds
            done = GEN_DONE_RE.search(line)
            if done and done.group("run_id") == run_id:
                generation = int(done.group("generation"))
                metric = metrics.setdefault(generation, metric_template(generation))
                for field in ("winRate", "gamesPerSecond", "turnsPerSecond", "p95LatencyMs", "generationSeconds"):
                    metric[field] = float(done.group(field))
                metric["backpropSamples"] = int(done.group("backpropSamples"))
    return metrics


def generation_log_metrics(run_id: str) -> dict[int, dict]:
    metrics: dict[int, dict] = {}
    pattern = f"{GENERATION_LOG_DIR}/{run_id}/generation_*.json"
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        generation = artifact.get("generation")
        metric = artifact.get("metric")
        if isinstance(generation, int) and isinstance(metric, dict):
            metrics[generation] = metric
    return metrics


def replay_chunks(run_id: str) -> list[dict]:
    generations: set[int] = set()
    patterns = [
        f"{TELEMETRY_DIR}/replays_{run_id}_generation_*.json",
        f"{TELEMETRY_DIR}/live_{run_id}_generation_*.owlive",
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            match = re.search(r"_generation_(\d+)\.(?:json|owlive|owslot)$", path)
            if match:
                generations.add(int(match.group(1)))
    chunks = []
    for generation in sorted(generations):
        chunks.append(
            {
                "generation": generation,
                "path": f"/telemetry/replays_{run_id}_generation_{generation}.json",
                "gameCount": DEFAULT_REPLAY_GAME_COUNT,
                "actualGameCount": DEFAULT_ACTUAL_GAME_COUNT,
            }
        )
    return chunks


def main() -> None:
    latest = load_latest()
    run_id = detect_run_id(latest)
    latest["runId"] = run_id
    by_generation = parse_log_metrics(run_id)
    by_generation.update(generation_log_metrics(run_id))
    completed_generations = set(by_generation)
    for metric in latest.get("metrics", []):
        generation = metric.get("generation")
        if isinstance(generation, int) and generation in completed_generations and generation not in by_generation:
            by_generation[generation] = metric
    latest["metrics"] = [by_generation[key] for key in sorted(by_generation)]
    latest["replayChunks"] = replay_chunks(run_id)
    with open(LATEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(latest, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


if __name__ == "__main__":
    main()
