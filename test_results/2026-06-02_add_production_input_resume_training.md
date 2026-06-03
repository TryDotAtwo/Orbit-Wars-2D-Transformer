# 2026-06-02 Add Production Input Resume Training

## Changes

- Added `production_normalized = production / 5.0` as the fifth per-row model input feature.
- Expanded true2d input contract from `64x4` to `64x5`.
- Expanded input-pair weights from 8 to 10:
  - source production weight inserted after source x/y weights.
  - target production weight inserted after target x/y weights.
- Added legacy weight migration from 3004114 weights to 3004116 weights by inserting both new production weights as `0.0`, preserving old behavior at load.
- Increased moving-target base intercept iterations from 12 to 128.
- Increased decoder hit-correction angle search samples from 12 to 128 per side.

## Verification

- Docker `cargo fmt --all`: passed.
- Docker `cargo test --workspace`: passed, core 32 tests and trainer 16 tests.
- Docker CUDA shared library build: passed.
- C++ true2d CUDA smoke: passed, `max_abs_diff=1.49012e-07`, `many_max_abs_diff=1.78814e-07`.
- Rust CUDA true2d smoke: passed, `resident_max_abs_diff=0.00000024`.
- Docker release trainer build: passed.
- Release CUDA trainer smoke `--smoke --cuda --generations 1 --players 4`: passed; `training_input_floats=327680`, confirming `1024 * 64 * 5`.

## Runtime

- A smoke run overwrote the default checkpoint path; the smoke checkpoint was preserved as `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.smoke_backup.bin`.
- Recovered the full checkpoint from real runId `1780348585` generation top4 artifacts:
  - population rebuilt to 128 from G35-G37 top4 elites using the trainer crossover/mutation schedule.
  - champion archive restored with the latest 32 champion models.
  - checkpoint restored to `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`.
  - restored checkpoint size: `1922638390` bytes.
- Full CUDA training relaunched from recovered checkpoint:
  - pid=`3955`
  - runId=`1780348585`
  - activeGeneration=`38`
  - phase=`evaluation`
  - telemetry confirmed live turn 79 and `maxInferenceBatchSize=1280`.
