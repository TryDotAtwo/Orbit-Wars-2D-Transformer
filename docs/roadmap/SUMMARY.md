# docs/roadmap/SUMMARY.md

timestamp=2026-06-03T16:23:21+03:00
section=roadmap
active_task_id=2026-06-03_simplify_agent_rules

## Current Stage

- The project is in CUDA self-play training and iteration.
- Current target is population 64, 20 participations per model, top-6 weighted reproduction.
- Documentation and agent protocol were compacted so future agents can read only task-relevant context and use `test_results/` for history.

## Next Tasks

- Confirm or perform checkpoint-boundary restart with the rebuilt population=64/top-6 release binary.
- Benchmark one full current-contract generation and compare it against previous timing records.
- Run broad official random-seed trace comparison against Kaggle source.
- Optimize CUDA full-attention forward/backward path after measurement.
- Keep Kaggle submit and timing uploads blocked until explicit user approval.

## Blocked Or Deferred

- Exact official RNG parity: blocked until broad trace comparison is implemented.
- 65,536-game batch target: blocked until VRAM and throughput proof.
- BF16 CUDA backprop: deferred until a separate reference-matched precision contract is approved.
- Decoder Sun guard: deferred because it changes the output behavior contract.

## Acceptance Gate

- Save raw prompt for durable tasks.
- Read only task-relevant root files and summaries.
- Keep external parameters in `project_config.yaml`.
- Update `index.md` and relevant `docs/<section>/SUMMARY.md` for durable changes.
- Store verification in `test_results/` when work changes durable behavior.
- Run the smallest relevant verification; for docs-only changes, run file/encoding/link sanity checks.
