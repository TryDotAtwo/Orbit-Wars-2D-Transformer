# CUDA Training No Atomics

timestamp=2026-06-01T16:42:04+03:00
task_id=2026-06-01_cuda_training_no_atomics
status=passed

## User Requirement

- Use standard transformer training semantics: learn Q/K/V/O, FFN/MLP, embeddings, and output weights.
- Do not make the full `4096x4096` attention matrix an independent learnable parameter table.
- Remove atomics from training CUDA kernels.

## Implementation

- Removed every `atomicAdd` from `native/cuda/orbit_wars_cuda.cu`.
- Replaced shared gradient accumulation with explicit reduction kernels:
  - `true2d_train_output_gradients_kernel`
  - `true2d_train_layer_shared_gradients_kernel`
  - `true2d_train_attention_head_gradients_kernel`
  - `true2d_train_expand_gradients_kernel`
- Kept full end-to-end gradients through output loss, true2d layer apply, full `4096x4096` Q/K/V attention, and learned expansion.
- Attention backward now writes per-sample/per-layer/per-head/per-cell parameter-gradient contributions into a temporary buffer, then reduces them into model weights without atomics.
- `grad_matrices` accumulation is now owner-per-cell and sequential by kernel phase, so it uses direct writes/adds without races.

## Parameter Clarification

- Standard self-attention learns Q/K/V/O projection weights and upstream representation weights.
- The `4096x4096` attention score matrix is dynamic per input/layer/head/sample and is differentiated through, but it is not stored as independent parameters.
- Current true2d model still has a learnable `64x64` pair embedding table, source/target row embeddings, Q/K/V/O scalars per head, shared layer FFN/residual weights, and output weights.
- Adding independent `4096x4096` edge parameters would be a different architecture, not standard transformer training.

## Verification

- `rg -n "atomicAdd" native/cuda/orbit_wars_cuda.cu`: no matches.
- `docker exec --workdir /workspace orbit-wars-cuda-dev nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so`: passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev target/release/orbit-wars-trainer --cuda-true2d-smoke`: passed; `max_abs_diff=0.00000018`, `many_max_abs_diff=0.00000018`, `resident_max_abs_diff=0.00000018`.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace`: passed; core tests=29, trainer tests=7.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release`: passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev target/release/orbit-wars-trainer --smoke --cuda --generations 1 --players 4`: passed; final run after signature cleanup used run_id=1780321687.

## Smoke Metrics

- run_id=1780321687
- profile=smoke_64
- evaluated_games=4
- model_action_calls=1024
- training_samples=1024
- backprop_trained_models=2
- backprop_samples=256
- evaluation_seconds=1.723377
- backprop_seconds=1.304315
- generation_seconds=3.072460

## Conclusion

- Full CUDA training remains end-to-end through true2d full attention.
- Training CUDA code has no atomics.
- The smoke backprop phase improved from previous baseline `16.545904s` to `1.304315s` for 256 selected samples.
