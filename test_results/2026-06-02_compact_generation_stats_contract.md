# 2026-06-02 compact generation stats contract

status=verified_no_code_change
runId=1780348585

## User Contract

- Per-generation statistics must remain compact aggregate counters for the whole generation.
- Full replay frames are stored only for 4 actual games per generation.

## Inspection

- `crates/orbit-wars-trainer/src/main.rs::write_generation_log_artifact` writes only `runId`, `generation`, aggregate `metric`, replay counts, replay chunk path, top-model path, and top-model metadata.
- The generation log format does not include replay frames, per-turn state, or trajectory input/output samples.
- Existing `dashboard/public/telemetry/generation_logs/1780348585/generation_24.json` is 1408 bytes and contains aggregate counters such as `sunDestroyedFleets`, `sunDestroyedShips`, `captures`, `fleetHits`, `launchActions`, `launchedShips`, throughput, and phase timings.
- G24 has `actualReplayGameCount=10` because it is a historical artifact completed before replay-count reduction.
- Current config and Rust constants set future durable generation replays to 4 actual games and live replay mirroring to 4 actual games.

## Conclusion

- No trainer restart or code change is required for this prompt.
- Current training can continue; future completed generations keep compact aggregate logs plus full replay frames only for 4 actual games.
