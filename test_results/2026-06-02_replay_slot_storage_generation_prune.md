# Replay Slot Storage And Generation Prune

timestamp=2026-06-02T00:00:00+03:00
task_id=2026-06-02_replay_append_slots_generation_prune
status=passed

## Changes

- Replaced live replay frame appends with preallocated `.owslot` storage using fixed per-game frame slots and fixed result slots.
- `latest.json` remains a small manifest; replay chunks can point to `.owslot` through a tiny `binary_replay_alias_v1` JSON.
- Dashboard loader now decodes `orbit_live_replay_slots_v1` and prefers sibling `.owslot` before legacy `.owlive` and JSON.
- Scheduled generation tournament now returns champion scores; when tournament runs, the top 12 tournament models are used as reproduction elites.
- After tournament reproduction, champion archive and replay files are pruned to survivor generations while completed metrics/generation winrate history remain in telemetry.
- `dashboard_live_replay_slot_bytes=131072` was added to `AgentConfig` and `project_config.yaml`.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace` passed: 31 core tests and 14 trainer tests.
- `npm.cmd run build` passed for dashboard TypeScript/Vite.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.

## Runtime Note

- Existing trainer PID from the checkpoint run was not stopped by this task. The new `.owslot` format and tournament-prune behavior apply after restarting/resuming with the rebuilt release trainer.
