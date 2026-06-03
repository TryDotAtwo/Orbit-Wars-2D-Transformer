# Backprop Atomics And Attention Notes

timestamp=2026-06-01T16:10:00+03:00
status=completed
task_id=2026-06-01_backprop_atomics_attention_notes

## User Question

- Why are there atomics in training if this is a transformer?
- Are the 2D self-attention links trained; is this effectively a 16M transformer?

## Answer Captured

- Current `atomicAdd` usage is a correctness-first custom CUDA scaffold, not the final efficient transformer-training implementation.
- Atomics are used because many CUDA threads concurrently accumulate gradients into the same model weights and into the same previous-layer cell gradients.
- Standard transformer training does not usually expose this as hand-written global atomics on the hot path; dense reductions are expressed as tiled matmul/reduction kernels.
- The 2D self-attention links are trained dynamically: every `4096 x 4096` query-key score participates in forward and backward, and gradients flow into Q/K/V/output attention parameters plus upstream relation-cell representations.
- The current model is not 16M learnable parameters. It has 16,777,216 dynamic attention scores per layer/head/sample, while learnable parameters per model remain small because Q/K/V are scalar projections plus embeddings and output weights.
- Making each attention edge an independent learnable parameter would be a different architecture and would cost roughly 16M fp32 values per layer/head/model before optimizer state.

## Microbenchmarks

- Baseline full CUDA backprop smoke from previous verification:
  - run_id=1780316961
  - backpropSamples=256
  - batchSamples=64
  - backpropSeconds=16.545904
- Increasing `training_backprop_batch_samples` to 512:
  - run_id=1780319077
  - backpropSeconds=21.556026
  - result=slower
- Increasing `training_backprop_batch_samples` to 128:
  - run_id=1780319161
  - backpropSeconds=23.086767
  - result=slower
- Training-only global softmax-stat cache attempt:
  - run_id=1780319489
  - backpropSeconds=22.075998
  - result=slower due to extra global-memory traffic
- Zero-gradient skip attempt:
  - run_id=1780319645
  - backpropSeconds=22.251017
  - result=not retained

## Final State

- `training_backprop_batch_samples` reverted to 64.
- Softmax-stat global cache and zero-gradient skip were reverted.
- Final verification after revert:
  - `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace` passed.
  - `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.
  - CUDA library compile passed after revert.
- `nvidia-smi` after repeated smokes showed no active utilization, memory used 25 MiB, temperature 76C, so no other trainer process was competing for GPU at check time.

## Next Technical Step

- Replace the atomics/scatter attention backward with a tiled reduction kernel:
  - compute query statistics and gradients in blocks;
  - reduce dQ/dK/dV and dInput through shared memory or warp reductions;
  - emit one reduced gradient per block/parameter instead of per edge;
  - avoid global atomic traffic over the `4096 x 4096` score graph.
