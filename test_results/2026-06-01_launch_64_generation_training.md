# 64-Generation CUDA Training Launch

timestamp=2026-06-01T17:14:13+03:00
task_id=2026-06-01_launch_64_generation_training
status=running

## User Request

- Launch training for 64 generations to inspect behavior.

## Launch Contract

- command=`target/release/orbit-wars-trainer --cuda --players 4 --generations 64`
- container=`orbit-wars-cuda-dev`
- run_id=`1780323159`
- trainer_pid=`10576`
- profile=`full_500`
- backend=`cuda`
- players_per_game=4
- generations=64
- episode_steps=500
- population=128
- elite=12
- games_per_model=10 participations
- simultaneous_games=1280 participant views / 320 actual four-player games
- replay_interval=32
- validation_interval=32
- cuda_chunk_requests=8192
- precision=`bf16 CUDA inference attention operands, fp32 accumulation/storage/backprop`

## Logging

- artifact_log=`artifacts/full_training_2026-06-01_64gen_bf16_run.log`
- pid_file=`artifacts/full_training_2026-06-01_64gen_bf16_run.pid`
- container_logs=true; stdout/stderr are teed to `/proc/1/fd/1` and visible through `docker logs orbit-wars-cuda-dev`.

## Start Verification

- Existing trainer process check found no active trainer before launch; only defunct old shell/nsys processes remained.
- `docker exec --workdir /workspace orbit-wars-cuda-dev ps -eo pid,ppid,stat,etime,cmd` showed PID `10576` running `target/release/orbit-wars-trainer --cuda --players 4 --generations 64`.
- `docker logs --tail 80 orbit-wars-cuda-dev` showed:
  - `trainer_log; event=start; run_id=1780323159; profile=full_500; backend=cuda; players_per_game=4; generations=64; episode_steps=500; population=128; elite=12; games_per_model=10; replay_interval=32; validation_interval=32; cuda_chunk_requests=8192`
  - `trainer_log; event=phase_start; run_id=1780323159; generation=1; phase=evaluation; profile=full_500; backend=cuda; players_per_game=4; episode_steps=500; population=128; games_per_model=10; cuda_chunk_requests=8192`
- `dashboard/public/telemetry/latest.json` reports `runId=1780323159`, `activeGeneration=1`, `sourceMessage=run_profile=full_500; phase=evaluation; status=in_progress; players_per_game=4; compute_timing=pending`.

## Live Status Check

- After about 6 minutes, PID `10576` was still running with status `Rsl`.
- `nvidia-smi` inside the container reported `NVIDIA GeForce RTX 3070 Laptop GPU, 100% utilization, 562 MiB / 8192 MiB`.
- No `generation_done` event had been emitted yet; generation 1 evaluation was still in progress.

## Notes

- The run is intentionally detached and still running.
- First useful performance conclusion should be taken after generation 1 completes and emits `event=generation_done`.
