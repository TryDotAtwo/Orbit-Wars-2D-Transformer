# 2026-06-02 Cleanup And 4 Replay Restart

status=passed_running
run_id=1780348585

## Changes

- Deleted duplicate production dashboard build `dashboard/dist` after verifying the resolved path was inside the workspace.
- Deleted obsolete temporary submit files `submit_check_tmp` and `submission.tar.gz`.
- Kept durable training state: `dashboard/public/telemetry`, immutable generation logs, completed `.owlive` replays, generation top-4 weights, checkpoint `.bin`, `target`, and `dashboard/node_modules`.
- Changed default durable generation replay count from 10 to 4 actual evaluation games.
- Changed default live replay mirror count from 10 to 4 actual evaluation games.
- Made `tools/restore_latest_history_from_trainer_log.py` tolerate missing/empty `latest.json` and restore run history from trainer logs plus immutable generation logs.
- Made trainer writes to `dashboard/public/telemetry/latest.json` atomic through temp-file write plus rename, so dashboard reads do not see partial JSON during live updates.

## Verification

- `npm.cmd exec tsc -- --noEmit` passed.
- `python -m py_compile tools/restore_latest_history_from_trainer_log.py tools/compact_owslot_replay.py` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test -p orbit-wars-trainer` passed with 14 tests.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test -p orbit-wars-core` passed with 31 tests.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.
- Restored `latest.json` after interrupted write: metrics G1..G24 and replay chunks G1..G24.
- Restarted release trainer from `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin` without deleting completed progress; restarted again after atomic latest-write fix.
- Eight consecutive live reads of `latest.json` parsed successfully after atomic write was applied.

## Runtime Check

- New trainer PID inside `orbit-wars-cuda-dev`: `10878`.
- Active generation after restart: G25.
- `generationReplayGameCount=4`.
- `live_replay_games=16` participant views, equal to 4 actual four-player games.
- New active slot `dashboard/public/telemetry/live_1780348585_generation_25.owslot` is about 263 MB.
- `dashboard/dist` is absent after cleanup.
