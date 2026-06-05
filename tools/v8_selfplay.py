#!/usr/bin/env python3
"""Generation-level self-play relaxation scaffolding for v8 models.

The tournament hot loop is intended to live in Rust.  This module keeps Python
at the orchestration boundary: it creates generation manifests, mutation plans,
and dashboard telemetry from arena outputs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeneticRelaxationConfig:
    population: int = 16
    elites: int = 4
    mutations_per_elite: int = 3
    mutation_std: float = 0.02
    opponent_pool: tuple[str, ...] = ("last_best", "current_elites", "imitation_baseline", "noop_baseline", "heuristic_baseline")


def mutate_generation(
    elite_models: list[str],
    *,
    config: GeneticRelaxationConfig | None = None,
    generation: int = 0,
) -> list[dict[str, object]]:
    config = config or GeneticRelaxationConfig()
    if not elite_models:
        elite_models = ["imitation_baseline"]
    children: list[dict[str, object]] = []
    child_index = 0
    for elite in elite_models[: max(1, config.elites)]:
        children.append({"modelId": child_index, "generation": generation, "parent": elite, "mutationStd": 0.0, "role": "elite"})
        child_index += 1
        for mutation_index in range(config.mutations_per_elite):
            if len(children) >= config.population:
                break
            children.append({
                "modelId": child_index,
                "generation": generation,
                "parent": elite,
                "mutationStd": config.mutation_std,
                "mutationSeed": generation * 10_000 + child_index * 97 + mutation_index,
                "role": "mutation",
            })
            child_index += 1
        if len(children) >= config.population:
            break
    while len(children) < config.population:
        parent = elite_models[len(children) % len(elite_models)]
        children.append({
            "modelId": child_index,
            "generation": generation,
            "parent": parent,
            "mutationStd": config.mutation_std,
            "mutationSeed": generation * 10_000 + child_index * 97,
            "role": "fill",
        })
        child_index += 1
    return children[: config.population]


def write_generation_manifest(
    output_dir: Path,
    elite_models: list[str],
    *,
    config: GeneticRelaxationConfig | None = None,
    generation: int = 0,
) -> Path:
    config = config or GeneticRelaxationConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "v8-selfplay-relaxation",
        "generation": generation,
        "config": asdict(config),
        "opponentPool": list(config.opponent_pool),
        "models": mutate_generation(elite_models, config=config, generation=generation),
        "arena": {
            "owner": "rust",
            "binary": "orbit-wars-arena",
            "notes": "Python launches generations only; per-tick arena simulation stays native.",
        },
    }
    path = output_dir / f"generation-{generation:04d}.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/v8-selfplay"))
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--elite", action="append", default=[])
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elites", type=int, default=4)
    parser.add_argument("--mutations-per-elite", type=int, default=3)
    args = parser.parse_args()
    config = GeneticRelaxationConfig(population=args.population, elites=args.elites, mutations_per_elite=args.mutations_per_elite)
    path = write_generation_manifest(args.output_dir, args.elite, config=config, generation=args.generation)
    print(json.dumps({"manifest": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
