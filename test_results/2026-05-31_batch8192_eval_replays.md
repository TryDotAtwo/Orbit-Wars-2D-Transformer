timestamp=2026-05-31T23:20:00+03:00
task_id=2026-05-31_increase_batch_collect_replays_from_generation
branch=unknown_untracked_workspace
commit=unknown
environment=cmz-native-dev:2026-05-26; container=orbit-wars-cuda-dev; gpu=RTX_3070_Laptop_8GB
change_1=CUDA true2d many-forward chunk cap increased from 512 to 8192 requests.
change_2=full self-play games per model increased from 8 to 10 so generation games can supply 10 replay games per model when replay interval triggers.
change_3=dashboard replay chunks now use evaluation games on replay generations instead of launching extra replay-only games.
change_4=expensive replay and archived-generation validation cadence remains every 32 generations.
stopped_previous_run=true; stopped_pid=4050; reason=batch_cap_and_replay_source_changed
command_1=docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && cargo fmt --all && cargo test --workspace && cargo build -p orbit-wars-trainer --release"
result_1=pass; tests=32; release_build=pass; warnings=0
command_2=docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && nohup env ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so target/release/orbit-wars-trainer --cuda --players 4 --generations 32 > artifacts/full_training_2026-05-31_batch8192_evalreplay_run.log 2>&1 & ..."
result_2=running; trainer_pid=4883; runId=1780258487
telemetry_state=run_profile=full_500; phase=evaluation; activeGeneration=1; population=128; gamesPerModel=10; replaysPerModel=10; playersPerGame=4; simultaneousGames=1280; maxInferenceBatchSize=8192; storedReplayGames=0
artifacts=[dashboard/public/telemetry/latest.json,artifacts/full_training_2026-05-31_batch8192_evalreplay_run.log,artifacts/full_training_2026-05-31_batch8192_evalreplay_run.pid,artifacts/full_training_2026-05-31_batch8192_evalreplay_run.parent.pid]
conclusion=batch_cap_increased_and_duplicate_replay_simulation_removed; full_training_running
