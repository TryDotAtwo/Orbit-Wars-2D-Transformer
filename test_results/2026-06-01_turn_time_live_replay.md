# 2026-06-01 Turn Timing And Live Replay

## Prompt

- `prompt_history/2026-06-01_turn_time_and_live_replay_requirement.md`

## Code Changes

- Added `AgentConfig` live replay controls:
  - `dashboard_live_replay_enabled=true`
  - `dashboard_live_replay_game_count=1`
  - `dashboard_live_replay_frame_stride=1`
  - `dashboard_live_replay_update_interval_turns=1`
- Trainer evaluation now captures a small live replay sample and rewrites `dashboard/public/telemetry/latest.json` at step 0 and after each configured turn interval.
- Live telemetry preserves previous metrics, previous generation validation records, previous replay chunk manifests, and inline `replayGames[]` for the current live sample.
- Existing interval-32 replay chunk archive behavior is unchanged.

## Verification

- `docker exec orbit-wars-cuda-dev bash -lc "cargo fmt --all"` passed.
- `docker exec orbit-wars-cuda-dev bash -lc "cargo check --workspace"` passed.
- `docker exec orbit-wars-cuda-dev bash -lc "cargo test --workspace"` passed: 36 Rust tests passed.
- `docker exec orbit-wars-cuda-dev bash -lc "cargo build --release -p orbit-wars-trainer"` passed.

## Observed Full Generation Timing

- source=`docker logs --tail 80 orbit-wars-cuda-dev`
- runId=`1780323159`
- generation=1
- evaluationSeconds=849.458801
- backpropSeconds=82.097107
- reproductionSeconds=0.003308
- generationSeconds=931.591736
- generation=2
- evaluationSeconds=978.040710
- backpropSeconds=86.911316
- reproductionSeconds=0.003722
- generationSeconds=1064.990845
- evaluatedGames=320
- episodeSteps=500
- modelActionCalls=640000
- inferenceBatchCalls=500
- simultaneousGames=320
- trainingSamples=640000
- eliteBackpropSamples=60000

## Derived Timing

- global_turn_wall_seconds=`849.458801 / 500 = 1.698918`
- per_game_turn_wall_seconds=`849.458801 / (320 * 500) = 0.005309`
- per_model_action_wall_seconds=`849.458801 / 640000 = 0.001327`
- generation_wall_minutes=`931.591736 / 60 = 15.526529`
- evaluation_wall_minutes=`849.458801 / 60 = 14.157647`
- backprop_wall_minutes=`82.097107 / 60 = 1.368285`
- generation_2_global_turn_wall_seconds=`978.040710 / 500 = 1.956081`
- generation_2_per_game_turn_wall_seconds=`978.040710 / (320 * 500) = 0.006113`
- generation_2_per_model_action_wall_seconds=`978.040710 / 640000 = 0.001528`
- generation_2_generation_wall_minutes=`1064.990845 / 60 = 17.749847`
- two_generation_average_wall_minutes=`(931.591736 + 1064.990845) / 2 / 60 = 16.638188`

## Limitations

- Current running PID 10576 was launched before this live replay code was built, so it will not emit live per-turn replay telemetry until the trainer is restarted with the rebuilt release binary.
