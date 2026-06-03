timestamp=2026-05-31T16:46:00+03:00
task_id=2026-05-31_run_4_epochs_full_replays_dashboard
branch=master
commit=none
environment=Windows_host_plus_Docker_cmz-native-dev_2026-05-26
result=pass

## Commands

- prompt_saved=`prompt_history/2026-05-31_run_4_epochs_full_replays_dashboard.md`
- rust_verify=`docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"`
- training_run=`docker run --rm --gpus all -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --smoke --cuda --generations 4 --full-replays"`
- dashboard_build=`npm.cmd run build`
- dashboard_dev_server=`http://127.0.0.1:5173`
- browser_qa=`Playwright snapshot desktop 1521x1025 and mobile 390x844`

## Results

- rust_format=passed
- rust_workspace_tests=passed; tests=26
- trainer_result=`status=ok; smoke=true; cuda=true; full_replays=true; generations=4; population=8; elite=2; best_model=0; best_reward=0`
- telemetry_latest=`dashboard/public/telemetry/latest.json`; bytes=333755
- telemetry_chunks=4
- telemetry_chunk_format=`compact_replay_v1`
- replay_games_total=320
- replay_games_per_generation=80
- replay_frames_per_game=65
- metrics_count=4
- dashboard_build=passed
- dashboard_server=started; url=`http://127.0.0.1:5173`
- browser_console_errors=0 after favicon fix
- desktop_screenshot=`artifacts/orbit-wars-dashboard-4-epochs.png`
- mobile_screenshot=`artifacts/orbit-wars-dashboard-4-epochs-mobile.png`

## Metrics

- generation_1=`winRate=1.000; gamesPerSecond=5.200; p95LatencyMs=192.896; gpuUtilization=0`
- generation_2=`winRate=0.500; gamesPerSecond=4.032; p95LatencyMs=263.974; gpuUtilization=0`
- generation_3=`winRate=0.000; gamesPerSecond=4.341; p95LatencyMs=255.830; gpuUtilization=0`
- generation_4=`winRate=0.000; gamesPerSecond=4.154; p95LatencyMs=272.529; gpuUtilization=0`

## Conclusion

- four_epoch_cuda_training_smoke=completed
- full_replay_dashboard_artifacts=completed
- latest_telemetry_manifest_kept_small=true
- replay_chunks_loaded_by_dashboard_once=true
- dashboard_initial_view_loads_active_generation_chunk=true
- full_128_model_training_not_run_reason=current_cuda_forward_allocates_device_memory_per_call
