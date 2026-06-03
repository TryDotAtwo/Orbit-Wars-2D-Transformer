# Delete Broken Checkpoint And Repack Replays

timestamp=2026-06-02
status=completed

## Actions

- Deleted stale failed checkpoint temp `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.tmp`, about 1643.15 MB.
- Preserved resume checkpoint `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`.
- Added `tools/trim_owlive_replay_games.py` to trim immutable `OWLIVE1` replay binaries to a target actual-game count.
- Repacked `dashboard/public/telemetry/live_1780348585_generation_*.owlive` for G1..G28 to at most 4 actual games per generation.

## Verification

- `checkpoint.tmp` no longer exists.
- `checkpoint.bin` still exists.
- No `*.tmp` files remain under `dashboard/public/telemetry`.
- All 28 `live_1780348585_generation_*.owlive` files parse as `OWLIVE1` and report 4 actual games.
- Replay binary total is about 2543.12 MB after repack.
- Generation top-4 model archive remains 28 files and about 1283.50 MB.
- Approximate space reclaimed: about 1643.15 MB from checkpoint temp plus about 3011.42 MB from replay trimming.
