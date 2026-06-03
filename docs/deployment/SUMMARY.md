# docs/deployment/SUMMARY.md

timestamp=2026-06-03T17:22:00+03:00
section=deployment
active_task_id=2026-06-03_checkpoint_path_config_fix

## Current Environment

- Development image: `cmz-native-dev:2026-05-26`.
- CUDA container: `orbit-wars-cuda-dev`.
- GPU target: RTX 3070 Laptop class, 8 GB VRAM.
- Rust workspace members: `orbit-wars-core`, `orbit-wars-ffi`, `orbit-wars-trainer`.
- CUDA library path: `target/liborbit_wars_cuda.so`.
- Optional CUDA env var: `ORBIT_WARS_CUDA_LIB_PATH`.
- Dashboard preferred dev URL: `http://127.0.0.1:5173`.
- Official reference source: `C:/tmp/kaggle-env-src`.

## Common Commands

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

```powershell
docker run --rm --gpus all -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so
```

```powershell
docker run --rm --gpus all -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --cuda-true2d-smoke"
```

```powershell
docker exec --workdir /workspace orbit-wars-cuda-dev target/release/orbit-wars-trainer --smoke --cuda --generations 1 --players 4
```

```powershell
cd dashboard
npm.cmd run build
```

## Full Training Operations

- Before stopping or replacing a trainer, verify live PID, checkpoint generation, and telemetry state.
- Use checkpoint-boundary restart tooling when preserving progress matters.
- Detached trainer launches should tee stdout/stderr into both artifact log and `/proc/1/fd/1` so `docker logs` remains useful.
- Current checkpoint path is mirrored by `project_config.yaml` and `AgentConfig::trainer_checkpoint_path`; trainer saves to the explicit `--resume-checkpoint` path when supplied, otherwise to the configured path.

## Submission Operations

- Submission package root must contain `main.py`, native library, and `model.bin`.
- `kaggle_submission/main.py` must expose `agent(obs)`.
- No submission or timing package upload is allowed without explicit user approval.
- If Kaggle validation fails, inspect episode logs first and store findings in `test_results/`.

## Cleanup Rules

- Build outputs, dependency caches, telemetry, checkpoints, replays, and generated archives are ignored by Git.
- Do not delete checkpoints, top-model archives, or replay artifacts without checking current runtime and recent test records.
- Prefer existing cleanup/repair scripts in `tools/` over one-off shell deletion.
