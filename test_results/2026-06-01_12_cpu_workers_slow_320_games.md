# 2026-06-01 12 CPU Workers And 320-Game Slowdown Check

## Scope

- Investigated why full self-play with 320 actual four-player games is slow.
- Applied the user-approved CPU budget of 12 notebook cores.
- Restarted the current full CUDA self-train run after rebuilding the release trainer.

## Diagnosis

- 320 actual games are not just 320 small steps: every global turn can produce `320 * 4 = 1280` active model/player states.
- A full 500-turn generation therefore reaches up to about `500 * 1280 = 640000` forward states before terminal skips.
- Each state runs one full middle `4096 x 4096` attention pass, which is `16777216` attention scores, plus the learned `64 x 64` matrix layers and output head.
- The rough attention-score scale for one full evaluation is therefore about `640000 * 16777216 ~= 10.7e12` score evaluations before memory traffic and matrix-layer work.
- GPU is a real bottleneck now; current custom CUDA kernels are not tiled flash-attention/tensor-core kernels.
- CPU was also underused in the trainer turn loop before this change.

## Implementation

- `TRAINING_CPU_WORKER_COUNT=12` replaced the old single-worker telemetry/turn-loop contract.
- `collect_action_requests` now chunks active games across up to 12 scoped CPU workers while preserving request order by chunk order.
- CUDA output decoding into per-game actions is parallelized across up to 12 scoped CPU workers.
- Non-live game state application is parallelized across up to 12 scoped CPU workers.
- Live replay games remain sequential during state application because they append to the `.owlive` stream.
- Telemetry now reports `cpuWorkers=12`.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace` passed: 30 core tests and 13 trainer tests.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.
- CUDA smoke after the 12-worker change passed:
  - command: `target/release/orbit-wars-trainer --smoke --cuda --generations 1 --players 4`
  - runId=`1780346918`
  - evaluationSeconds=`6.389905`
  - backpropSeconds=`2.681331`
  - generationSeconds=`9.620644`
  - maxInferenceBatchSize=`16`
  - training_output_floats=`1572864`
- Comparable previous multi-target smoke before the 12-worker change:
  - runId=`1780345777`
  - evaluationSeconds=`11.811246`
  - generationSeconds=`14.940433`

## Current Run

- Stopped old full runId=`1780345954`, pid=`5763`, to apply the rebuilt trainer.
- Started new full 64-generation CUDA run:
  - runId=`1780346971`
  - pid=`6920`
  - command=`target/release/orbit-wars-trainer --cuda --players 4 --generations 64`
  - log=`artifacts/full_training_2026-06-01_self_train_current_contract.log`
  - pid_file=`artifacts/full_training_2026-06-01_self_train_current_contract.pid`
  - checkpoint=`artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`
- Fresh telemetry confirmed:
  - `cpuWorkers=12`
  - `simultaneousGames=320`
  - `maxInferenceBatchSize=1280`
  - early `turnsPerSecond` observed around `210..306`
  - `liveReplayPath=/telemetry/live_1780346971_generation_1.owlive`

## Remaining Bottleneck

- The 12-worker CPU change improves host-side request collection, output decoding, and simulation stepping.
- The next major speedup must come from CUDA-side attention/transfer work: tiled/shared-memory attention, better memory layout, pinned host buffers, CUDA streams, and double buffering.
