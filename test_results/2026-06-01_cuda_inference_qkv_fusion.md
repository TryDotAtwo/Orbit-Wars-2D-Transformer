# 2026-06-01 CUDA Inference QKV Fusion

## Scope

- Stopped the obsolete full self-train runtime after user approval to prioritize inference improvements.
- Optimized CUDA inference forward path for the true2d `4096 x 4096` attention model.
- Restarted full 64-generation training with the optimized CUDA library.

## Implementation

- Fused inference attention output with the residual/FFN layer apply step.
- Added inference-only precompute buffers for attention `Q`, `K`, and `V`.
- Added `true2d_attention_qkv_kernel` and `true2d_attention_qkv_many_kernel`.
- Updated single-model, many-model, and resident-population inference paths to use precomputed `Q/K/V`.
- Left CUDA backprop kernels unchanged; training backward still uses the existing fp32 no-atomics path.
- Tried inference-only `__expf`; it did not improve smoke timing and was reverted.

## Verification

- CUDA library build passed:
  - `nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so`
- CUDA true2d smoke passed:
  - `max_abs_diff=0.00000018`
  - `many_max_abs_diff=0.00000024`
  - `resident_max_abs_diff=0.00000024`
- `cargo fmt --all` passed.
- `cargo test --workspace` passed:
  - 30 core tests
  - 13 trainer tests
- `cargo build -p orbit-wars-trainer --release` passed.

## Timing

- Previous 12-worker full-run live check before CUDA inference optimization:
  - runId=`1780346971`
  - live turn=`107`
  - `turnsPerSecond=187..210` later in evaluation
  - `avgModelActionMs` around `1.0`
- Optimized full-run live check:
  - runId=`1780348585`
  - live turn=`17`: `turnsPerSecond=629.142`, `avgModelActionMs=0.288278`, `maxInferenceBatchSize=1280`
  - live turn=`107`: `turnsPerSecond=536.034`, `avgModelActionMs=0.337223`, `maxInferenceBatchSize=1280`
- Tiny smoke timing is noisy and not representative of full batch:
  - fused-only smoke runId=`1780347996`: `evaluationSeconds=6.076840`
  - QKV-precompute smoke runId=`1780348209`: `evaluationSeconds=4.022330`
  - later smoke runs varied upward while GPU clocks/thermals changed.
- Completed optimized full generation 1:
  - runId=`1780348585`
  - `evaluationSeconds=442.858765`
  - `backpropSeconds=94.084106`
  - `replayWriteSeconds=13.319950`
  - `reproductionSeconds=2.946685`
  - `generationSeconds=554.336731`
  - `evaluatedGames=320`
  - `maxInferenceBatchSize=1280`
  - `modelActionCalls=616067`
  - `avgModelActionMs=0.507978`
  - `turnsPerSecond=288.594`
  - `backpropSamples=58348`
  - target result: full generation stayed under the 600-second target.
- Previous official-like generation 1 comparison:
  - runId=`1780336658`
  - `evaluationSeconds=547.045471`
  - `backpropSeconds=87.647522`
  - `generationSeconds=640.869751`

## Current Run

- New current full run:
  - runId=`1780348585`
  - pid=`4730`
  - command=`target/release/orbit-wars-trainer --cuda --players 4 --generations 64`
  - log=`artifacts/full_training_2026-06-01_self_train_current_contract.log`
  - pid_file=`artifacts/full_training_2026-06-01_self_train_current_contract.pid`
  - current phase=`generation_2_evaluation`
  - live replay=`dashboard/public/telemetry/live_1780348585_generation_2.owlive`
  - completed generation 1 chunk=`dashboard/public/telemetry/replays_1780348585_generation_1.json`

## Residual Issues

- One non-fatal live telemetry write warning occurred during generation 1: `live_replay_telemetry_write_skipped`, `Invalid argument (os error 22)`.
- Training continued, generation 1 completed, checkpoint persisted, and generation 2 live telemetry is active.

## Conclusion

- Yes, inference can be materially accelerated.
- On the real full-training batch size, early evaluation throughput improved from roughly `187..210 turns/sec` to `536..629 turns/sec`.
- End-to-end optimized generation 1 completed in `554.336731s`, under the 10-minute generation target.
- Remaining inference bottlenecks are now matrix-layer global-memory passes, host-to-device input copies, device-to-host output copies, and lack of CUDA stream/double-buffer overlap.
