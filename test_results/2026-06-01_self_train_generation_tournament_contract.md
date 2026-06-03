# Self-Train Generation Tournament Contract Verification

timestamp=2026-06-01T19:38:46+03:00
task_id=2026-06-01_self_train_generation_tournament_contract

## Contract

- `self_train` is the only training loop: evaluation games, reward assignment, top-12 backprop, top-4 champion archive copy, replay chunk writing, reproduction/mutation to 128, then next generation.
- Replay chunks are written from already-played evaluation games; no replay-only simulation batch is launched.
- Generation tournament runs every 32 generations over archived top-4 champion copies from retained generations.
- Tournament participation count is 10 per archived champion model; at the 128-model archive cap this is 320 actual four-player tournament games.
- Pending progress telemetry now reports real game counts: 320 evaluation games for 128 models * 10 participations / 4 players, zero games during `replay_write`.
- Dashboard tournament view shows actual four-player tournament match count from `generationValidationGames`; per-generation table `games` values are labeled as champion participations.

## Verification

- `docker exec -w /workspace orbit-wars-cuda-dev cargo fmt --all` passed.
- `docker exec -w /workspace orbit-wars-cuda-dev cargo test --workspace` passed: 29 core tests and 9 trainer tests.
- `docker exec -w /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.
- `npm.cmd run build` in `dashboard/` passed.
- Playwright MCP render check passed on `http://127.0.0.1:5179`: `Generation Winrate` shows generation tournament sections and `Generation Logs` still renders.

## Notes

- Existing PID 10576 was not stopped or restarted. It predates this contract and remains historical until a fresh trainer process is launched.
- Browser plugin was not needed for this check; Playwright MCP rendered the already-running dashboard.
