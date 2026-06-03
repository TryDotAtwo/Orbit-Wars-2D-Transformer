# Resident GPU Balanced Generation Verification

timestamp=2026-05-31T23:59:00+03:00
task_id=2026-05-31_resident_gpu_balanced_generation

## Changes Verified

- self-play `games_per_model` now means model participations, not focus-model games.
- full 4-player schedule for population=128 and participations_per_model=10 is 320 games, with exactly 10 player-game participations per model.
- replay chunks now emit one replay view per participant model, so each model can be inspected across its own participations on replay generations.
- CUDA true2d population weights are uploaded once into resident VRAM workspace with `orbit_wars_cuda_true2d_upload_population`.
- trainer turn inference uses `orbit_wars_cuda_true2d_forward_many_resident` after upload; per-turn copies are inputs, model indices, and outputs, not all model weights.
- reward scale now enforces win positive, draw negative but softer than loss: win=2, draw=-1, loss=-2.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so` -> pass.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` -> pass.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace` -> pass; tests=34.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` -> pass.
- `docker exec --workdir /workspace -e ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so orbit-wars-cuda-dev cargo run -p orbit-wars-trainer -- --cuda-true2d-smoke` -> pass; `resident_max_abs_diff=0.00000018`.
- `docker exec --workdir /tmp -e ORBIT_WARS_CUDA_LIB_PATH=/workspace/target/liborbit_wars_cuda.so orbit-wars-cuda-dev /workspace/target/release/orbit-wars-trainer --smoke --cuda --generations 1 --players 4` -> pass; evaluatedGames=4; modelActionCalls=1024; maxInferenceBatchSize=16; evaluationSeconds=1.261832.

## Notes

- Full 128-model generation benchmark was not started because existing long run PID 4883 is still active in the CUDA container and was not stopped.
- Existing PID 4883 was launched before this resident-weight and balanced-schedule patch, so its metrics remain pre-change evidence.
