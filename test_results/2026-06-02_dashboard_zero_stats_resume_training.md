# 2026-06-02 Dashboard Zero Stats And Resume Training

task_id=2026-06-02_dashboard_zero_stats_resume_training
status=implemented_verified_running

## Changes

- `dashboard/src/Charts.tsx`: `LineChart` now filters zero and non-finite values per metric field before computing scale, path, points, hover target, and latest header value. A chart with no non-zero points returns `null`.
- `dashboard/src/App.tsx`: Live Training metric tiles and compute-time breakdown rows now hide zero/non-finite values.

## Verification

- `npm.cmd exec tsc -- --noEmit`: passed.
- `npm.cmd run build`: sandbox run failed with known `esbuild spawn EPERM`; escalated rerun passed.
- Browser rendered check on `http://127.0.0.1:5173/`: passed with no console errors/warnings.
- Browser evidence: chartCount=9, zeroMetricTiles=[], and charts with historically zero early data (`Action avg`, `Captures`, `Sun losses`, `Fleet hits`, `Avg fleet`) rendered 11 non-zero points instead of connecting through zero generations.

## Docker Resume

- Existing stopped container `orbit-wars-cuda-dev` was started from image `cmz-native-dev:2026-05-26`.
- GPU check inside container passed: `NVIDIA GeForce RTX 3070 Laptop GPU`, 8192 MiB.
- Resume command launched:
  `target/release/orbit-wars-trainer --cuda --players 4 --generations 64 --resume-checkpoint artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`
- Current trainer PID inside container: `82`; `artifacts/full_training_2026-06-01_self_train_current_contract.pid` updated to `82`.
- Initial start check passed: runId=1780348585, activeGeneration=28, phase=evaluation, live_replay_turn=102, liveReplayPath=`/telemetry/live_1780348585_generation_28.owslot`, liveReplayFrameCount=412, turnsPerSecond=590.259, maxInferenceBatchSize=1280.
- Follow-up check passed after replayed G28 completed: checkpoint `.tmp` disappeared, checkpoint `.bin` updated, trainer advanced to activeGeneration=29, phase=evaluation, live_replay_turn=67, liveReplayPath=`/telemetry/live_1780348585_generation_29.owslot`, liveReplayFrameCount=272, turnsPerSecond=720.394, maxInferenceBatchSize=1280.
- Final status check: trainer PID `82` still running at activeGeneration=29, phase=evaluation, live_replay_turn=311, liveReplayFrameCount=1248, turnsPerSecond=544.343, maxInferenceBatchSize=1280, GPU utilization=100%.
- GPU after initial start check: 2301 MiB used, 42% utilization.

## Notes

- The prior run completed G28 in logs but then failed checkpoint persistence with `checkpoint_write_failed=Input/output error (os error 5)`, so resume from checkpoint replayed G28 from the last reliable checkpoint boundary before advancing to G29.
