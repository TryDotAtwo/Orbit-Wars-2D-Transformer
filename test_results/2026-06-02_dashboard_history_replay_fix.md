# 2026-06-02 Dashboard History Replay Fix

status=implemented_verified_restart_blocked_by_docker_approval_timeout

## Problem

- `latest.json` contained many duplicate G17 metric records after live/progress telemetry rewrites.
- Active G18 progress metrics appeared in charts with zero win/capture values.
- G1 replay was missing from `replayChunks` because recovery looked only for `replays_*.json`, while G1 had only compact binary `.owlive`.
- Active `.owslot` was incorrectly discoverable as a durable replay chunk, causing Replay tab to try loading a preallocated ~627MB file.
- Completed G15 still existed only as `.owslot`, so selecting it would also require a huge slot fetch.
- After G18 completed, live telemetry showed `metrics=19` for active G19 with duplicated G18, because the in-memory completed metric slice itself could contain a duplicate generation.

## Fix

- Rust telemetry now deduplicates existing `metrics` and `replayChunks` by `generation`.
- Rust telemetry also deduplicates the current completed `MetricRecord` slice before serializing `metrics`, so a duplicated completed generation cannot be reintroduced after the old `latest.json` merge.
- Progress/live telemetry no longer appends active partial generation metrics to chart `metrics`; charts show completed generations only.
- Rust replay chunk discovery now restores durable chunks from `replays_*.json` and completed `.owlive`, but not active `.owslot`.
- Python recovery script now mirrors the same durable replay discovery and excludes `.owslot`.
- Dashboard no longer decodes active `orbit_live_replay_slots_v1` from `latest.json`; active view uses preview frames instead of fetching the whole preallocated slot.
- Replay tab no longer auto-loads all generations when `all generations` is selected.
- Added `tools/compact_owslot_replay.py` and compacted G15 `.owslot` into `live_1780348585_generation_15.owlive`, then removed the heavy G15 slot.

## Verification

- `cargo fmt --all` passed.
- `cargo test -p orbit-wars-trainer` passed twice after telemetry fixes: 14 trainer tests.
- `cargo build -p orbit-wars-trainer --release` passed after telemetry fixes.
- `npm.cmd run build` passed after dashboard changes; sandbox build hit `esbuild spawn EPERM`, escalated rerun passed.
- Trainer restarted from checkpoint and is running as pid `98375`.
- `latest.json` now has `metrics=17` with generations `1..17`, `replayChunks=17` with generations `1..17`, and active live path `/telemetry/live_1780348585_generation_18.owslot`.
- Follow-up check while old pid `98375` was still running showed active G19 with duplicated G18 in metrics; code fix and release build are ready, but replacing pid `98375` is blocked because two escalated `docker exec orbit-wars-cuda-dev kill 98375` approval reviews timed out and non-escalated Docker access is denied.
- Later local telemetry file check showed the dashboard-visible state clean again: activeGeneration=23, `metrics=23` with no duplicate generations, `replayChunks=23` with no duplicate generations, and sourceMessage=`compute_timing=tracked`.
- In-app browser partial check before final auto-load removal confirmed G18 zero chart disappeared and generation selector contained G1/G15/G17; final in-app browser check timed out because earlier tabs had been blocked by old huge fetches, so final state was verified through HTTP telemetry/dev-server artifacts and successful dashboard build.
