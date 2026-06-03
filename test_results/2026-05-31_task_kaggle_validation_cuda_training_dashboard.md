timestamp=2026-05-31T14:34:09+03:00
task_id=task_kaggle_validation_cuda_training_dashboard_2026_05_31
branch=unknown
commit=unknown
environment=Windows_PowerShell + Docker image cmz-native-dev:2026-05-26 + RTX_3070_Laptop_for_cuda_smoke
command_1=docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
result_1=pass; tests=22_passed_0_failed
command_2=docker run --rm -v "<workspace>:/workspace" -v "C:/tmp/kaggle-env-src:/kaggle-env-src" -w /workspace cmz-native-dev:2026-05-26 bash -lc "python3 tools/orbit_wars_reference_compare.py"
result_2=pass; output="status=ok; compared_scenarios=5; reference_source=/kaggle-env-src/kaggle_environments/envs/orbit_wars/orbit_wars.py"
command_3=docker run --rm --gpus all -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "mkdir -p target && nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so && g++ -std=c++17 -O3 native/cuda/orbit_wars_cuda_smoke.cpp -Inative/cuda -Ltarget -lorbit_wars_cuda -Wl,-rpath,'$ORIGIN' -o target/orbit_wars_cuda_smoke && LD_LIBRARY_PATH=target ./target/orbit_wars_cuda_smoke"
result_3=pass; output="status=ok; cuda_trainable_forward_smoke=true; games=4; rows=64; d_model=32"
command_4=docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo run -p orbit-wars-trainer -- --smoke"
result_4=pass; output="status=ok; smoke=true; population=8; elite=2; best_model=7; best_reward=3"; wall_time_seconds=147.7
command_5=npm.cmd run build
result_5=pass; output="vite build complete"
command_6=browser_qa http://127.0.0.1:5173/
result_6=pass; canvas_count=1; replay_select_options=80; stored_games_visible=true
artifacts=["dashboard/public/telemetry/latest.json"]
logs=[]
conclusion=simulator_reference_validation_passed; cuda_forward_runtime_smoke_passed; dashboard_replay_contract_passed; full_cuda_batch_training_not_started
remaining_risk=trainer_cuda_batch_binding_not_implemented; true_pair_matrix_2d_transformer_not_implemented; kaggle_submit_not_run
