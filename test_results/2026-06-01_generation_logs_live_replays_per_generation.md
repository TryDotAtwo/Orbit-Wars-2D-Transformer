# Generation Logs And Per-Generation Replay Storage

timestamp=2026-06-01T18:57:07+03:00
task_id=2026-06-01_generation_logs_live_replays_per_generation
status=passed

## Scope

- Added a dashboard `Generation Logs` page with per-generation status, timing, replay-game counts, replay views, throughput, action latency, and direct replay opening.
- Changed normal full-training replay archive cadence to every generation.
- Changed replay capture to store at most `dashboard_generation_replay_game_count=10` real evaluation games per generation, with one participant view per active player, instead of capturing every evaluation game.
- Kept live evaluation replay in `latest.json` and made the Replay page auto-follow the newest live frame when a live feed is active.
- Added `generationReplayGameCount` telemetry and `actualGameCount` replay chunk metadata.

## Verification

- `docker exec -w /workspace orbit-wars-cuda-dev cargo fmt --all` passed.
- `docker exec -w /workspace orbit-wars-cuda-dev cargo test --workspace` passed: 29 core tests, 8 trainer tests.
- `docker exec -w /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.
- `npm.cmd run build` passed after sandbox `spawn EPERM` was rerun with approved elevated execution.
- Browser plugin path was attempted first, but the Node-backed browser bridge failed before code execution with `failed to write kernel assets`.
- Playwright fallback QA passed on `http://127.0.0.1:5179/`: `Generation Logs` rendered, replay button switched to `Replay`, Replay canvas rendered, desktop and mobile checks had zero captured console errors/warnings.

## Evidence

- New trainer unit test `replay_games_limit_actual_games_per_generation` verifies the replay archive cap is actual games times active player views.
- QA screenshots were saved outside the repo:
  - `C:/tmp/orbit-wars-generation-logs.png`
  - `C:/tmp/orbit-wars-replay-final.png`
  - `C:/tmp/orbit-wars-generation-logs-mobile.png`

## Notes

- Current running PID `10576` was not stopped or restarted. It was launched before this change, so it still uses the older replay cadence and older binary until a new run is launched.
- No CUDA trainer smoke was started during this task to avoid interfering with the already-running full training process.
