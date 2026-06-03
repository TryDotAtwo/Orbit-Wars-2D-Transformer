# BF16 Warp Inference

timestamp=2026-06-01T17:03:00+03:00
task_id=2026-06-01_bf16_warp_inference
status=passed

## User Requirement

- Continue GPU optimization after no-atomics CUDA training.
- CUDA inference may use BF16, while stored/final model weights remain fp32.

## Implementation

- Added warp-level reduction helpers for CUDA attention forward.
- Replaced one-thread-per-query inference attention with warp-per-query kernels for:
  - `orbit_wars_cuda_true2d_forward`
  - `orbit_wars_cuda_true2d_forward_many`
  - `orbit_wars_cuda_true2d_forward_many_resident`
- CUDA inference attention now uses bf16-rounded operands for query/key/value calculations with fp32 accumulation and fp32 output buffers.
- Stored model weights, host population weights, CUDA upload buffers, and CPU submission weights remain fp32.
- CUDA backprop math remains fp32; BF16 was applied only to inference attention, not to gradient computation.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so`: passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev target/release/orbit-wars-trainer --cuda-true2d-smoke`: passed.
- CUDA smoke diff: `max_abs_diff=0.00000024`, `many_max_abs_diff=0.00000024`, `resident_max_abs_diff=0.00000024`, tolerance=`0.0005`.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace`: passed; core tests=29, trainer tests=7.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release`: passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev target/release/orbit-wars-trainer --smoke --cuda --generations 1 --players 4`: passed.

## Smoke Metrics

- run_id=1780322349
- profile=smoke_64
- evaluated_games=4
- model_action_calls=1024
- inference_batch_calls=64
- max_inference_batch_size=16
- training_samples=1024
- backprop_trained_models=2
- backprop_samples=256
- evaluation_seconds=1.193032
- backprop_seconds=1.002741
- generation_seconds=2.253741

## Comparison

- Previous final no-atomics smoke: evaluation `1.723377s`, backprop `1.304315s`, generation `3.072460s`.
- BF16 warp inference smoke: evaluation `1.193032s`, backprop `1.002741s`, generation `2.253741s`.
- Inference tolerance remained well inside the existing fp32 CPU/GPU smoke gate.

## Conclusion

- CUDA inference now uses a warp-per-query attention forward path and BF16-rounded operands while preserving fp32 model storage.
- CPU submission inference remains fp32.
- Next optimization target remains full tiled/tensor-core attention and transfer overlap for larger batches.
