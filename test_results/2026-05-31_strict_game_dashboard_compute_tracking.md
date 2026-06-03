timestamp=2026-05-31T17:40:00+03:00
task_id=2026-05-31_strict_game_dashboard_compute_tracking
branch=unknown_untracked_workspace
commit=unknown_untracked_workspace
environment=Windows host + Docker image cmz-native-dev:2026-05-26 + Vite dashboard

## Commands

- command=`cargo fmt --all`; result=blocked; reason=`cargo` not installed on Windows host.
- command=`npm.cmd run build`; result=blocked; reason=`esbuild spawn EPERM` inside sandbox.
- command=`docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"`; result=pass; tests=`26 passed`.
- command=`npm.cmd run build`; result=pass; output=`tsc && vite build`, bundle=`214.68 kB js`, build_time=`963ms vite phase`.
- command=`docker run --rm --gpus all -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --dashboard-strict --cuda --generations 4"`; result=pass; wall_time_seconds=78.0.
- command=`python telemetry validation script`; result=pass; activeGeneration=4; chunks=4; gamesPerChunk=8; framesPerGame=501; lastStep=500.
- command=`docker run --rm -v "<workspace>:/workspace" -v "C:/tmp/kaggle-env-src:/kaggle-env-src" -w /workspace cmz-native-dev:2026-05-26 bash -lc "python3 tools/orbit_wars_reference_compare.py"`; result=pass; output=`status=ok; compared_scenarios=5; reference_source=/kaggle-env-src/kaggle_environments/envs/orbit_wars/orbit_wars.py`.
- command=`browser QA http://127.0.0.1:5173`; result=pass; views=`Live Training`, `Replay`; viewports=`1440x950`, `390x844`.

## Strict Training Result

- run_id=1780237699
- profile=`dashboard_strict_500`
- trainer_result=`status=ok; profile=dashboard_strict_500; smoke=false; dashboard_strict=true; cuda=true; full_replays=false; generations=4; episode_steps=500; population=4; elite=2; best_model=0; best_reward=0`
- telemetry_manifest=`dashboard/public/telemetry/latest.json`
- replay_chunks=`dashboard/public/telemetry/replays_1780237699_generation_1.json` through `dashboard/public/telemetry/replays_1780237699_generation_4.json`
- replay_contract=full game frames; expectedFramesPerGame=501; actual_min_frames=501; actual_max_frames=501; lastStep=500.

## Generation 4 Metrics

- gamesPerSecond=0.695
- turnsPerSecond=309.620
- p95LatencyMs=1479.442
- avgModelActionMs=1.462495
- generationSeconds=19.378584
- evaluationSeconds=5.758322
- replaySeconds=12.586854
- replayWriteSeconds=1.026315
- reproductionSeconds=0.000537
- modelActionSeconds=17.549940
- simulationStepSeconds=0.703694
- launchActions=12000
- launchedShips=24096
- avgLaunchActionsPerTurn=2.000
- avgLaunchedShipsPerTurn=4.016

## Dashboard QA Artifacts

- desktop_live_screenshot=`artifacts/orbit-dashboard-strict-desktop.png`
- desktop_replay_screenshot=`artifacts/orbit-dashboard-strict-replay-desktop-v2.png`
- mobile_replay_screenshot=`artifacts/orbit-dashboard-strict-mobile-v3.png`
- dev_server_log=`artifacts/dashboard-dev-1780237699.log`

## Conclusion

- result=pass
- strict_game_length_fixed=true
- compute_time_tracking=true
- send_pressure_tracking=true
- dashboard_usability_improved=true
- remaining_gap=`official random map distribution is not yet integrated into self-play state generation; current strict run uses native_seeded_reference_map with official 500-turn cap and native official-order turn loop`
