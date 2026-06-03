# Restart Self-Train Current Contract

timestamp=2026-06-01T19:52:35+03:00
task_id=2026-06-01_restart_self_train_current_contract

## Actions

- Persisted raw prompt to `prompt_history/2026-06-01_restart_training_clean_old.md`.
- Old runtime container `orbit-wars-cuda-dev` was already exited with status 137; no old trainer process remained.
- Recreated `orbit-wars-cuda-dev` with the same GPU/workspace/kaggle mounts to clear stale container logs.
- Removed old runtime-only files: previous 64-generation log, previous 64-generation pid file, stale current-contract log/pid if present, and stale `dashboard/public/telemetry/latest.json`.
- Kept durable project records: `prompt_history/`, `test_results/`, docs, and registered historical replay chunk files.

## New Run

- command=`target/release/orbit-wars-trainer --cuda --players 4 --generations 64`
- container=`orbit-wars-cuda-dev`
- runId=`1780332620`
- trainer_pid=`39`
- artifact_log=`artifacts/full_training_2026-06-01_self_train_current_contract.log`
- pid_file=`artifacts/full_training_2026-06-01_self_train_current_contract.pid`
- container_logs=true

## Start Verification

- GPU visible: RTX 3070 Laptop, 8192 MiB.
- Release trainer present: `target/release/orbit-wars-trainer`.
- CUDA library present: `target/liborbit_wars_cuda.so`.
- `docker logs --tail 80 orbit-wars-cuda-dev` shows `trainer_log; event=start` and `phase=evaluation` for runId 1780332620.
- Latest telemetry reports `activeGeneration=1`, `phase=evaluation`, `live_replay_turn=83`, `live_replay_games=4`, `simultaneousGames=320`, `maxInferenceBatchSize=1280`, `generationReplayGameCount=10`, `models=128`.
- GPU check during run reported utilization=100%, memory=2026/8192 MiB.
