# 2026-06-03 Increase Sun Target Penalty

task_id=2026-06-03_increase_sun_target_penalty
status=implemented_verified_running

## Change

- Raised `TRAINING_SUN_TARGET_PENALTY_SCALE` from `1.0` to `4.0`.
- Updated `project_config.yaml` `training_targets.sun_target_penalty_scale.value` to `4.0`.
- Preserved the existing action-local attribution contract: Sun-destroyed fleet ids trace back to the exact captured sample/source-row/target-slot.
- Preserved global win/draw/loss reward scale and send-head loss semantics.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace` passed: core 34 tests, trainer 16 tests.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.
- Release build warning only: existing unused `legacy_live_replay_public_path`.

## Runtime

- Checkpoint-boundary watcher started at G49 with old PID `78311`.
- Watcher detected checkpoint boundary at G50, stopped old PID `78311`, and launched the rebuilt release trainer.
- New trainer PID is `24390`.
- Latest live telemetry check passed: runId `1780348585`, activeGeneration `50`, phase `evaluation`, live turn `52`.
