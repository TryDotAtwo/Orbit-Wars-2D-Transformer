# 2026-05-31 GPU Efficiency And Container Logs

status=passed_with_runtime_smoke_timeout
task=inspect_gpu_efficiency_and_add_container_log_output

## Findings

- CUDA trainer path is functional and verified through `orbit_wars_cuda_true2d_forward_many`.
- Current implementation is not fully efficient: each many-forward call still rebuilds/copies all population weights, uses pageable host buffers, has no CUDA streams or double-buffering, and runs scalar full attention over `4096 * 4096` key/query pairs per request per layer.
- Current full run `runId=1780258487` completed generation 1 in `733.652893s`; max request batch was `5120`, below the configured cap `8192`, so the active bottleneck is kernel/transfer cost rather than chunk splitting.
- `docker logs --tail 30 orbit-wars-cuda-dev` showed only the CUDA image banner before this change, confirming the detached trainer did not write runtime progress into container logs.

## Changes

- `crates/orbit-wars-trainer/src/main.rs` now emits flushed stdout `trainer_log;` lines for run start, phase start, phase completion, generation completion, and run completion.
- `docs/deployment/SUMMARY.md` now records a detached launch command that tees trainer stdout/stderr both to an artifact log and to `/proc/1/fd/1`, making the stream visible through `docker logs -f orbit-wars-cuda-dev`.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace` passed with `32` tests.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.
- A debug smoke run from `/tmp` was attempted to observe live stdout lines without overwriting project telemetry, but timed out after `184s`; temporary PID `5076` was stopped.

## Notes

- Current running PID `4883` was started before this logging change and before the container-log launch command. It will not retroactively emit new `trainer_log;` lines unless restarted.
