# 2026-06-03 Velocity Input, Phase-Aware Aiming, Restart

## Changes

- Expanded True2D input from `64x5` to `64x7`.
- Added `velocity_x_normalized = velocity_x / fleet_speed_max` and `velocity_y_normalized = velocity_y / fleet_speed_max`, clamped to `[-1, 1]`.
- Added legacy weight migration:
  - `3004114` weights (`64x4`) -> `3004120` by inserting zero production and velocity weights.
  - `3004116` weights (`64x5`) -> `3004120` by inserting zero velocity weights.
- Passed `current_step` into decoder from trainer and Kaggle FFI.
- Made decoder orbit prediction follow official/native phase semantics for future ticks: `tick=0` uses current observed position, later ticks use `max(current_step + tick - 1, 1)`.
- Increased decoder hit-check horizon to board-diagonal flight time plus configured extra ticks.
- Added time-based aim search over future target path samples before falling back to local angle search.
- Added regression test for the `step=1` orbit phase hold.
- Updated restart helper to accept `NEXT_CUDA_LIB` and replace `target/liborbit_wars_cuda.so` only after checkpoint-boundary stop.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace` passed: core 34 tests, trainer 16 tests.
- Built CUDA library without touching the live-loaded default path:
  - `target/liborbit_wars_cuda_velocity.so`.
- C++ CUDA true2d smoke passed against the new library:
  - `max_abs_diff=1.78814e-07`
  - `many_max_abs_diff=1.19209e-07`
- Rust CUDA true2d smoke passed against the new library:
  - `max_abs_diff=0.00000024`
  - `many_max_abs_diff=0.00000024`
  - `resident_max_abs_diff=0.00000024`
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.

## Training Restart

- Old running trainer `pid=3955` finished G41 and checkpoint advanced to G42.
- Watcher detected boundary:
  - `boundary_detected; latest_gen=42; checkpoint_mtime=1780433971`.
- Watcher stopped old trainer and replaced default CUDA library:
  - `cuda_library_replaced; source=target/liborbit_wars_cuda_velocity.so; target=target/liborbit_wars_cuda.so`.
- New trainer launched from checkpoint:
  - `pid=78311`
  - command=`target/release/orbit-wars-trainer --cuda --players 4 --generations 64 --resume-checkpoint artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`
  - active generation restarted at G42 evaluation.
- `cmp` confirmed `target/liborbit_wars_cuda.so` equals `target/liborbit_wars_cuda_velocity.so`.

## Notes

- Trainer smoke was intentionally not run in `/workspace` because the smoke profile writes the default checkpoint path; release build plus CUDA true2d smokes verified the shape without risking checkpoint overwrite.
