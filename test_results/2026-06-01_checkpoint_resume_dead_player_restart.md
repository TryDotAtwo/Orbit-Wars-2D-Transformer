# Checkpoint Resume, Dead Player Skip, And Restart

timestamp=2026-06-01T22:35:00+03:00
task=checkpoint_resume_dead_player_restart

## Changes

- Added binary trainer checkpoint at `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`.
- Checkpoint stores `run_id`, `next_generation`, full population weights, and champion archive.
- Added `--resume-checkpoint <path>` / `--resume-checkpoint=<path>` CLI support.
- Checkpoint is written immediately at run start and after each completed generation.
- Added checkpoint roundtrip test for population and champion archive preservation.
- Dead players are skipped in `collect_action_requests`, so eliminated agents no longer produce actions/training samples.
- Added trainer test for skipping dead players.
- Added trainer test that one alive player is terminal.

## Runtime

- User explicitly approved killing the current run.
- Old run `1780339686` was stopped after generation 3 had completed.
- Old replay chunks G1/G2/G3 were converted to `compact_replay_v2`.
- Final new run started: `runId=1780343126`, PID=`2485`, phase=`generation=1 evaluation`.
- Immediate checkpoint exists, size=`1537360438` bytes.

## Verification

- `cargo fmt --all` passed.
- `cargo test --workspace` passed: 29 core tests and 13 trainer tests.
- `cargo build -p orbit-wars-trainer --release` passed.
- `npm.cmd run build` passed for dashboard.
