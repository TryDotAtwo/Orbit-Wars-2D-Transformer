# 2026-06-03 Population Forward Scratch

task=trainer_cuda_forward_packing_scratch
code_changes=true

## Change

- Added `PopulationForwardScratch` for CUDA resident population forward packing.
- Reused `input_rows` and `model_indices` buffers across turns/chunks in `run_games_batched`.
- Moved chunk packing into `PopulationForwardScratch::pack_chunk`.
- Kept CUDA forward behavior unchanged: same packed `64 x 7` rows and same model-index order are passed to `forward_population_resident_packed`.

## Memory/Performance Intent

The trainer previously allocated fresh host buffers for packed CUDA forward inputs on each batched population forward call. The scratch keeps capacity and clears buffers before repacking, reducing repeated host allocation churn around CUDA bursts.

## Verification

RED command:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer population_forward_scratch -- --nocapture"
```

RED result:

- exit_code=1
- compile failed because `PopulationForwardScratch` did not exist.

GREEN targeted command:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer population_forward_scratch -- --nocapture"
```

GREEN targeted result:

- exit_code=0
- `population_forward_scratch_reuses_packed_input_capacity`: ok

Full Rust gate:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

Result:

- exit_code=0
- `orbit-wars-core`: 35 passed.
- `orbit-wars-ffi`: 0 tests.
- `orbit-wars-trainer`: 24 passed.
- Known warning remains: `legacy_live_replay_public_path` is unused.

## Notes

- No CUDA ABI changed.
- No model/training/replay logic was cut.
- This reduces allocation churn but does not remove output buffer allocation inside the CUDA binding yet.
