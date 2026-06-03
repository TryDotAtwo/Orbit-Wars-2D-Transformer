# 2026-06-03 Trainer Hot-Loop Allocation Cleanup

task=trainer_hot_loop_allocation_cleanup
code_changes=true

## Change

- Reused `requests_per_game` inside `run_games_batched` instead of allocating a new `Vec<usize>` every turn.
- Replaced per-turn/per-model `Vec<usize>` sample-index allocation with `first_sample_index + request_index` calculation.
- Added `decode_sample_index_offsets_from_first_recorded_sample` to lock the sample-index mapping contract.

## Behavior

- No game logic changed.
- No model input/output shape changed.
- No trainer sample contents changed.
- Action traces still point to the same recorded sample indices.

## Verification

RED command:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer decode_sample_index -- --nocapture"
```

RED result:

- exit_code=1
- compile failed because `decode_sample_index` did not exist yet.

GREEN targeted command:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer decode_sample_index -- --nocapture"
```

GREEN targeted result:

- exit_code=0
- `decode_sample_index_offsets_from_first_recorded_sample`: ok

Full Rust gate:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

Result:

- exit_code=0
- `orbit-wars-core`: 35 passed.
- `orbit-wars-ffi`: 0 tests.
- `orbit-wars-trainer`: 23 passed.
- Known warning remains: `legacy_live_replay_public_path` is unused.

## Notes

- This is a small host-side allocation cleanup, not a full CUDA throughput fix.
- Larger wins still require profiling host packing, H2D/D2H transfer, decode, simulation, and CUDA kernels separately.
