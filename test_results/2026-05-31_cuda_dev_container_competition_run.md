timestamp=2026-05-31T19:45:00+03:00
task_id=2026-05-31_cuda_dev_container_competition_run
branch=untracked_workspace
commit=not_available
environment=Docker container orbit-wars-cuda-dev from cmz-native-dev:2026-05-26 with --gpus all
command=CUDA image inspection, local Kaggle environments install, Rust/CUDA/reference checks, four-player CUDA training, Nsight Systems smoke profile
result=pass
artifacts=[artifacts/nsys_orbit_wars_cuda_smoke_2026_05_31.nsys-rep,dashboard/public/telemetry/latest.json,dashboard/public/telemetry/replays_1780245298_generation_1.json,dashboard/public/telemetry/replays_1780245298_generation_2.json,dashboard/public/telemetry/replays_1780245298_generation_3.json,dashboard/public/telemetry/replays_1780245298_generation_4.json]
logs=[]
conclusion=CUDA development image exists, profiler tools are present, dedicated GPU container is running, local Kaggle environments source is installed with an explicit Python-version override, Rust/CUDA/reference checks passed, and four-player CUDA dashboard training completed for 4 generations.

## Environment

- image=`cmz-native-dev:2026-05-26`
- container=`orbit-wars-cuda-dev`
- gpu=`NVIDIA GeForce RTX 3070 Laptop GPU`; memory_total=`8192 MiB`
- tools=`nvcc 12.4`, `nsys 2024.1.1.0`, `ncu`, `rustc 1.95.0`, `cargo 1.95.0`, `g++ 11.4.0`, `python 3.10.12`
- local_reference_mount=`/kaggle-env-src`

## Dependency Install

- `python3 -m pip install --no-deps -e /kaggle-env-src` -> fail; reason=`kaggle-environments requires Python >=3.11`.
- `python3 -m pip install --ignore-requires-python --no-deps -e /kaggle-env-src` -> pass.
- `import kaggle_environments` -> pass.
- `from kaggle_environments.envs.orbit_wars import orbit_wars` -> pass.

## Commands

- `cargo fmt --all` -> pass.
- `cargo test --workspace` -> pass; 27 tests passed.
- `nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so` -> pass.
- `g++ -std=c++17 -O3 native/cuda/orbit_wars_cuda_true2d_smoke.cpp -Inative/cuda -Ltarget -lorbit_wars_cuda -Wl,-rpath,'$ORIGIN' -o target/orbit_wars_cuda_true2d_smoke` -> pass.
- `LD_LIBRARY_PATH=target ./target/orbit_wars_cuda_true2d_smoke` -> pass; max_abs_diff=`1.19209e-07`.
- `ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --cuda-true2d-smoke` -> pass; max_abs_diff=`0.00000018`.
- `python3 tools/orbit_wars_reference_compare.py` -> pass; compared_scenarios=`5`.
- `npm.cmd run build` -> pass; Vite build completed.
- `ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --smoke --cuda --generations 4 --full-replays` -> pass; players_per_game=`4`; population=`8`; best_model=`6`; best_reward=`-2`.
- `ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --dashboard-strict --cuda --generations 4` -> pass; players_per_game=`4`; episode_steps=`500`; population=`4`; best_model=`0`; best_reward=`-1`.
- `nsys profile --trace=cuda,nvtx,osrt --output=/workspace/artifacts/nsys_orbit_wars_cuda_smoke_2026_05_31 ...` -> pass.

## Strict 500 Telemetry

- run_id=`1780245298`
- active_generation=`4`
- training_mode=`dashboard_strict_500_cuda`
- players_per_game=`4`
- simultaneous_games=`8`
- max_inference_batch_size=`8`
- stored_replay_games=`32`
- expected_frames_per_game=`501`
- games_per_second=`0.820`
- turns_per_second=`476.669`
- p95_latency_ms=`4876.249`
- avg_model_action_ms=`0.378947`
- generation_seconds=`12.587350`
- captures=`8`
- fleet_hits=`3764`
- hit_ships=`7652`
- sun_destroyed_fleets=`28`
- sun_destroyed_ships=`132`
- avg_fleet_size=`2.010125`

## Known Limitation

- local package install used `--ignore-requires-python` because the pinned project image has Python 3.10.12 and local Kaggle environments metadata now requires Python >=3.11.
