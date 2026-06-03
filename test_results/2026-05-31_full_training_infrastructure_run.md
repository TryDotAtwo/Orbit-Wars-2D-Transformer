timestamp=2026-05-31T22:40:00+03:00
task_id=2026-05-31_full_training_infrastructure_run
branch=unknown_untracked_workspace
commit=unknown
environment=cmz-native-dev:2026-05-26; container=orbit-wars-cuda-dev; gpu=RTX_3070_Laptop_8GB
command_1=docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && cargo fmt --all"
result_1=pass
command_2=docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && cargo test --workspace"
result_2=pass; tests=32
command_3=docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so"
result_3=pass
command_4=docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && cargo build -p orbit-wars-trainer --release"
result_4=pass
command_5=docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so timeout 1200s target/release/orbit-wars-trainer --cuda --players 4 --generations 1"
result_5=timeout_after_1200s; process_cleanup=pass; conclusion=full_128_model_generation_not_completed_with_naive_full_4096_attention_within_20_minutes
command_6=docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && nohup env ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so target/release/orbit-wars-trainer --cuda --players 4 --generations 1 > artifacts/full_training_2026-05-31_full_training_infrastructure_run.log 2>&1 & ..."
result_6=running; trainer_pid=2944; parent_shell_pid=2941; runId=1780256041
command_7=docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && cargo fmt --all && cargo test --workspace && cargo build -p orbit-wars-trainer --release"
result_7=pass; tests=32; release_build=pass
artifacts=[dashboard/public/telemetry/latest.json,artifacts/full_training_2026-05-31_full_training_infrastructure_run.log,artifacts/full_training_2026-05-31_full_training_infrastructure_run.pid]
telemetry_state=run_profile=full_500; phase=replay_sampling; status=in_progress; population=128; playersPerGame=4; gamesPerModel=8; replaysPerModel=10; simultaneousGames=1280; maxInferenceBatchSize=512
conclusion=training_infrastructure_passed; full_non_smoke_training_started; generation_completion_pending; throughput_bottleneck_unresolved
