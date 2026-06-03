timestamp=2026-05-31T18:05:00+03:00
task_id=2026-05-31_replay_canvas_and_targeting_question
branch=unknown_untracked_workspace
commit=unknown_untracked_workspace
environment=Windows host + Docker image cmz-native-dev:2026-05-26 + Vite dashboard

## Changes

- replay_canvas_fit=changed; CSS now caps board size with `--replay-canvas-max-size=58vh` while keeping square 100x100 board.
- replay_fleet_visibility=changed; Canvas2D now draws recent fleet trails for the previous 18 frames.
- decoder_targeting=changed; selected non-comet initial-planet targets now use orbital prediction in the decoder after model output.
- targeting_scope=decoder_geometry_only; transformer input remains strict 64x2 and no manual pair features are added before model output.

## Commands

- command=`docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"`; result=pass; tests=`27 passed`.
- command=`npm.cmd run build`; result=pass; output=`tsc && vite build`.
- command=`docker run --rm --gpus all -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --dashboard-strict --cuda --generations 4"`; result=pass; wall_time_seconds=57.0.
- command=`python telemetry validation script`; result=pass; activeGeneration=4; chunks=4; gamesPerChunk=8; framesPerGame=501; lastStep=500.
- command=`browser QA http://127.0.0.1:5173`; result=pass; views=`Replay`; viewports=`1440x950`, `390x844`.

## Strict Replay Refresh

- run_id=1780239301
- profile=`dashboard_strict_500`
- trainer_result=`status=ok; profile=dashboard_strict_500; smoke=false; dashboard_strict=true; cuda=true; full_replays=false; generations=4; episode_steps=500; population=4; elite=2; best_model=0; best_reward=0`
- telemetry_manifest=`dashboard/public/telemetry/latest.json`
- replay_chunks=`dashboard/public/telemetry/replays_1780239301_generation_1.json` through `dashboard/public/telemetry/replays_1780239301_generation_4.json`
- replay_contract=full game frames; expectedFramesPerGame=501; actual_min_frames=501; actual_max_frames=501; lastStep=500.

## Observation

- random_untrained_model_still_sends_sun_blocked_actions=true
- explanation=decoder now aims at selected orbiting target; if selected straight-line route intersects sun, fleet is destroyed by game rules.
- dashboard_trails_make_sun_blocked_paths_visible=true

## Dashboard QA Artifacts

- desktop_initial_screenshot=`artifacts/orbit-dashboard-canvas-targeting-desktop.png`
- desktop_trails_screenshot=`artifacts/orbit-dashboard-canvas-targeting-trails-v2.png`
- mobile_trails_screenshot=`artifacts/orbit-dashboard-canvas-targeting-mobile.png`

## Conclusion

- result=pass
- canvas_fit_fixed=true
- orbit_target_prediction_fixed=true
- remaining_training_issue=`model is still untrained/random and can choose bad sun-blocked targets; training must learn target selection`
