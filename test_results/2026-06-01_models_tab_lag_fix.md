# Models Tab Lag Fix

timestamp=2026-06-01T21:50:00+03:00
task_id=2026-06-01_models_tab_lag_fix

## User Requirement

- Models tab feels laggy and must stay responsive during live training.

## Root Cause

- Dashboard parsed the growing `.owlive` live replay file on every telemetry poll, even outside the Replay tab.
- Dashboard also preloaded durable replay chunks outside the Replay tab.
- During backprop/progress phases, trainer embedded full replay frames in `latest.json`; observed `latest.json` reached about 13MB.

## Changes

- `loadTelemetry` now accepts `includeLiveReplay`; `.owlive` is parsed only when the active tab is Replay.
- Durable replay chunks are fetched only while the active tab is Replay.
- Trainer progress/final telemetry no longer embeds completed replay frames in `latest.json`; completed games remain available through replay chunk files.
- Removed the now-unused trainer `replay_frames_json` helper.

## Verification

- `npm.cmd run build` passed.
- `cargo fmt --all`, `cargo test --workspace`, and `cargo build -p orbit-wars-trainer --release` passed without warnings in Docker.
- Trainer restarted as runId=1780339686.
- New `latest.json` after restart was about 47KB with one live preview frame.
- Browser resource check on the Models tab for 5 seconds fetched only `/telemetry/latest.json`, about 56-57KB per poll, with no `.owlive` or replay chunk fetches.
