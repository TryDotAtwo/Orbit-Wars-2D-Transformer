# Replay Dropdown Actual Games

timestamp=2026-06-01T21:20:00+03:00
task_id=2026-06-01_replay_dropdown_actual_games

## User Requirement

- Dashboard Replay selector must show 10 different games, not four participant views of the same four-player game.

## Changes

- Dashboard Replay selector now deduplicates participant views by `(generation, gameIndex)` when `all models` is selected.
- Specific model selection still shows model-specific participant replay views.
- Live replay capture default changed from 1 actual game to 10 actual games, producing 40 participant views in four-player mode.
- `project_config.yaml` now documents `dashboard_replays.live_replay_game_count=10`.

## Verification

- `npm.cmd run build` passed after dashboard selector change. Initial sandbox build hit known `esbuild spawn EPERM`; escalated build passed.
- `cargo fmt --all`, `cargo test --workspace`, and `cargo build -p orbit-wars-trainer --release` passed inside `orbit-wars-cuda-dev`.
- Rust tests passed: 29 core tests and 10 trainer tests.
- Browser check on `http://127.0.0.1:5173` passed: Replay selector shows `G1 game 1` through `G1 game 10` for `all models`.
- New trainer run started with runId=1780337978; live telemetry reported `live_replay_games=40`, which is 10 actual four-player games times four participant views.

## Runtime

- Previous runId=1780336658 was stopped to apply the new live replay count.
- Current runId=1780337978 is running with command `target/release/orbit-wars-trainer --cuda --players 4 --generations 64`.
