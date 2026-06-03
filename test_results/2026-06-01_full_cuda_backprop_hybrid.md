# Full CUDA Backprop Hybrid Verification

timestamp=2026-06-01T15:30:00+03:00
status=passed
task_id=2026-06-01_full_cuda_backprop_hybrid

## Scope

- Implemented GPU-only training path for self-play learning: CUDA inference during training, CUDA full-attention backprop after evaluation, CPU inference reserved for final Kaggle submission path.
- Added `orbit_wars_cuda_true2d_train_full_attention` C ABI for full forward/backward through true2d expand, full `4096x4096` Q/K/V attention, true2d layer update, and output collapse.
- Backprop trains selected elite models from their own captured trajectory samples.
- Removed sample limiter and removed fp32 weight-delta clamp; `backprop_batch_samples` is only CUDA graph/memory chunking and all selected samples are trained.
- Trainer emits flushed `trainer_log;` stdout events including `phase=backprop`; when launched with documented `/proc/1/fd/1` tee these events appear in container logs.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all`
  - result=passed
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace`
  - result=passed
  - core_tests=29
  - trainer_tests=7
- `docker exec --workdir /workspace orbit-wars-cuda-dev nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so`
  - result=passed
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release`
  - result=passed
- `docker exec --workdir /workspace orbit-wars-cuda-dev target/release/orbit-wars-trainer --cuda-true2d-smoke`
  - result=passed
  - max_abs_diff=0.00000018
  - many_max_abs_diff=0.00000018
  - resident_max_abs_diff=0.00000018
- `docker exec --workdir /workspace orbit-wars-cuda-dev target/release/orbit-wars-trainer --smoke --cuda --generations 1 --players 4`
  - result=passed
  - run_id=1780316961
  - evaluation_seconds=0.425779
  - evaluated_games=4
  - model_action_calls=1024
  - training_samples_total=1024
  - backprop_seconds=16.545904
  - backprop_models=2
  - backprop_samples=256
  - generation_seconds=17.346088
  - stdout_logs_include=`trainer_log; event=phase_start; phase=backprop` and `trainer_log; event=phase_done; phase=backprop`
- `npm.cmd run build` in `dashboard/`
  - first sandbox run failed with `spawn EPERM` from esbuild
  - escalated rerun passed
  - final_vite_build_time=1.97s

## Performance Note

- The full-gradient path is correct but intentionally not optimized yet.
- Smoke backprop over 256 selected samples took about 16.55s on the RTX 3070 Laptop container.
- Full population generation target still needs tiled/shared-memory attention, pinned host buffers, CUDA streams, and double-buffering before expecting 10-minute full generations.
