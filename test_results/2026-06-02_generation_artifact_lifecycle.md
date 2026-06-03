# 2026-06-02 Generation Artifact Lifecycle

status=implemented_verified_restarted_from_checkpoint

## User Requirement

- Current in-progress generation must use live write-friendly storage so dashboard replay does not lag.
- Completed generations must become immutable compact artifacts: compact logs, compact replays, and top-4 model weights.
- Completed artifacts are dashboard-readable and are not modified after generation completion.
- Every 32 generations, after the generation tournament, heavy replay/model artifacts may be deleted for pruned generations, but compact logs must remain for graphs/history.

## Implementation

- Active generation live replay uses preallocated `.owslot` storage with fixed frame/result slots and lightweight `latest.json`.
- On generation replay write, `.owslot` is compacted into legacy compact `.owlive`; compaction writes through `.tmp`, renames atomically, then removes `.owslot` only after success.
- Completed generation log artifact is written to `dashboard/public/telemetry/generation_logs/{run_id}/generation_{generation}.json`.
- Completed top-4 model weights artifact is written to `dashboard/public/telemetry/generation_models/{run_id}/generation_{generation}_top4.owmodels`.
- Tournament cleanup deletes replay artifacts and top-4 model artifacts for pruned generations, while generation log artifacts are intentionally retained.
- Telemetry progress writes preserve completed metrics, generation tournament records, and replay chunk pointers across checkpoint restarts.
- `tools/restore_latest_history_from_trainer_log.py` restores completed-generation telemetry from trainer logs before restart and now detects `runId` dynamically.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace` passed: 31 core tests, 14 trainer tests.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev python3 -m py_compile tools/restore_latest_history_from_trainer_log.py` passed.
- Training was restarted from `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`.
- New release trainer process started as pid `52681`, runId `1780348585`, activeGeneration `17`, and live telemetry reports `liveReplayFormat=orbit_live_replay_slots_v1`, `liveReplayPath=/telemetry/live_1780348585_generation_17.owslot`.
- Final runtime check before response: G17 evaluation still running, `live_replay_turn=470`, `liveReplayFrameCount=4710`, `turnsPerSecond=244.281`.
- Prior generation 16 compacted successfully before restart: log event `live_replay_compacted`, target `.owlive`, games=10, bytes=251691121.
- Redundant G14 `.owslot` was deleted after confirming G14 compact `.owlive` exists; G15 `.owslot` was preserved because it is currently the only replay storage for G15, and G17 `.owslot` is active live storage.

## Runtime Note

- Generation 17 was not completed before the restart; it is replayed from the checkpoint boundary without trusting partial live state.
