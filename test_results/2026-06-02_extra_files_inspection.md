# Extra Files Inspection

timestamp=2026-06-02
status=inspected_no_deletion

## Scope

- Excluded `project_history_full/` per project access rule.
- No files were deleted or modified during inspection, except this registered inspection record.

## Findings

- Clearly removable stale artifact: `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.tmp`, about 1643.15 MB. PID file points to process `10878`, and that process is not alive.
- Required resume checkpoint: `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`, about 2704.52 MB.
- Required generation top-4 model archive: `dashboard/public/telemetry/generation_models/1780348585/`, 28 files, about 1283.50 MB.
- Completed replay binaries: `dashboard/public/telemetry/live_1780348585_generation_*.owlive`, 28 files, about 5554.54 MB total.
- Old replay binaries G1..G24 occupy about 5223.49 MB and were written before the current 4-game replay target; they are candidates for compaction/rewrite to 4 games, not blind deletion.
- New replay binaries G25..G28 occupy about 331.05 MB and already match the smaller replay-count policy better.
- `target/` occupies about 147.78 MB and can be removed when no local Rust build cache is needed.
- `dashboard/node_modules/` occupies about 66.32 MB and is useful for dashboard dev/build.
- `dashboard/dist` and root `submission.tar.gz` are absent.
- No stale 88-byte top-model artifacts were found.
