# Trajectory Buffer For Backprop Verification

timestamp=2026-06-01T00:22:00+03:00
task_id=2026-06-01_trajectory_buffer_for_backprop

## Change

- Added generation training trajectory capture for every active model/player state.
- Captured metadata fields: `generation`, `game_index`, `player_slot`, `player_id`, `model_index`, `step`.
- Captured input layout: contiguous fp32 packed `64x4` rows per sample.
- Captured output layout: contiguous fp32 raw `64x2` model outputs per sample before decoder actions are applied.
- Final reward assignment: after game terminal scoring, every sample receives the final reward for its `game_index/player_slot`.
- Capture is enabled by `AgentConfig.training_capture_trajectories=true`; generation validation keeps capture disabled.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` -> pass.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace` -> pass; tests=35.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` -> pass.
- `docker exec --workdir /tmp -e ORBIT_WARS_CUDA_LIB_PATH=/workspace/target/liborbit_wars_cuda.so orbit-wars-cuda-dev /workspace/target/release/orbit-wars-trainer --smoke --cuda --generations 1 --players 4` -> pass.

## Smoke Evidence

- smoke evaluated games=4.
- model action calls=1024.
- `training_samples=1024`.
- `training_input_floats=262144`.
- `training_output_floats=131072`.
- Expected counts match `4 games * 4 player_slots * 64 turns`, input stride `64*4`, and output stride `64*2`.

## Limit

- Actual differentiable backprop is still not implemented. This change adds the required trajectory data layer so a future loss/backprop step can consume exact per-step inputs, outputs, and final participant rewards.
