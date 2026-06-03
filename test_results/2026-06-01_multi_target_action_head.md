# Multi Target Action Head

timestamp=2026-06-01T23:09:00+03:00
task=multi_target_action_head

## Changes

- Changed True2D output contract from `64 x 2` to `64 x 24`.
- Each source row now emits 12 target/send pairs: `[target0, send0, ..., target11, send11]`.
- CPU True2D head and CUDA True2D forward kernels normalize the 12 send fractions per source row so their sum is never above `1.0`.
- Decoder can emit up to 12 commands from one source planet, but still floors ship counts and skips commands below one ship.
- Kaggle Python FFI buffer capacity changed to `64 * 12 = 768` commands.
- Trajectory capture now stores `64 x 24` raw outputs per sample.
- True2D parameter count changed to `3004114` because the output head grew to 1596 weights.

## Verification

- `cargo fmt --all` passed in `orbit-wars-cuda-dev`.
- `cargo test --workspace` passed: 30 core tests and 13 trainer tests.
- `nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so` passed.
- `cargo run -p orbit-wars-trainer -- --cuda-true2d-smoke` passed with max diffs `0.00000018`.
- `cargo build -p orbit-wars-trainer --release` passed.
- `target/release/orbit-wars-trainer --smoke --cuda --generations 1 --players 4` passed with `training_output_floats=1572864` and `backprop_samples=256`.

## Runtime

- Old full-run PID `2485`, runId `1780343126`, predating this output-shape change was stopped before launch.
- New full-run started: runId `1780345954`, PID `5763`, generation 1 evaluation in progress.
- New checkpoint exists at `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`, size `1538109494` bytes, for the new `3004114`-parameter model.
