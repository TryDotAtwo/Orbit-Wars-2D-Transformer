# 2026-06-01 Three Million Mid-Attention Architecture

timestamp=2026-06-01T18:16:23+03:00
status=passed

## Scope

- Changed `True2DTransformer` layout to `64x4 -> 64x64 expand -> 61 learned 64x64 matrix layers with 12 per-cell weights -> one full 4096x4096 Q/K/V self-attention in the middle -> 64x2 output`.
- Current true2d parameter count is `3,002,651`: expand `4,233`, matrix layers `2,998,272`, middle attention `13`, output `133`.
- CUDA forward and backprop were updated to use one middle attention pass and `layers + 2` matrix-history states.
- Trainer elite backprop now compacts selected top models before CUDA training, so the 3M-parameter gradient pass trains the selected elite population slice instead of launching reductions over all 128 models.

## Commands

- `docker exec orbit-wars-cuda-dev bash -lc "cargo fmt --all"`
- `docker exec orbit-wars-cuda-dev bash -lc "nvcc -std=c++17 -I native/cuda -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so"`
- `docker exec orbit-wars-cuda-dev bash -lc "cargo test --workspace"`
- `docker exec orbit-wars-cuda-dev bash -lc "g++ -std=c++17 -I native/cuda native/cuda/orbit_wars_cuda_true2d_smoke.cpp -L target -lorbit_wars_cuda -Wl,-rpath,target -o target/orbit_wars_cuda_true2d_smoke"`
- `docker exec orbit-wars-cuda-dev bash -lc "target/orbit_wars_cuda_true2d_smoke"`
- `docker exec orbit-wars-cuda-dev bash -lc "cargo run -p orbit-wars-trainer -- --cuda-true2d-smoke"`
- `docker exec orbit-wars-cuda-dev bash -lc "cargo run -p orbit-wars-trainer -- --smoke --cuda --generations=1"`
- `docker exec orbit-wars-cuda-dev bash -lc "cargo build -p orbit-wars-trainer --release"`
- `rg -n "atomicAdd" native/cuda/orbit_wars_cuda.cu`

## Results

- Rust workspace tests passed: core `29`, trainer `7`.
- C++ CUDA true2d smoke passed: `max_abs_diff=8.9407e-08`, `many_max_abs_diff=1.19209e-07`.
- Rust FFI CUDA true2d smoke passed: `max_abs_diff=0.00000018`, `many_max_abs_diff=0.00000018`, `resident_max_abs_diff=0.00000018`.
- End-to-end CUDA smoke passed: `runId=1780326926`, `evaluationSeconds=9.465746`, `backpropSeconds=3.460563`, `generationSeconds=14.535768`, `trainingSamples=1024`, `backpropSamples=256`, `trainedModels=2`.
- Release trainer build passed.
- No CUDA atomics remain in the training file; `rg` returned no matches.

## Notes

- The older full 64-generation run `runId=1780323159` was launched before this 3M architecture change; its timings are historical and not measurements of this layout.
- BF16 remains inference-only for CUDA attention operands; CUDA backprop remains fp32.
