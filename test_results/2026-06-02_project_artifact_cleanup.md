# 2026-06-02 Project Artifact Cleanup

## Scope

- Cleaned old/generated project artifacts after the workspace grew too large.
- Preserved source code, project docs, current telemetry manifest, current live replay binary files for runId `1780348585`, and the main current checkpoint.

## Removed

- `.git` loose-object database with no tracked files and no commits.
- `target/` Rust build output.
- `dashboard/dist/` production build output and duplicated telemetry payloads.
- Heavy JSON replay chunks in `dashboard/public/telemetry/`.
- Old live replay binaries from previous runIds.
- Old screenshot/profile/log artifacts and root dashboard screenshots.
- Broken checkpoint temp file `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.tmp`.

## Kept

- `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`.
- `dashboard/public/telemetry/latest.json`.
- `dashboard/public/telemetry/live_1780348585_generation_1.owlive`.
- `dashboard/public/telemetry/live_1780348585_generation_2.owlive`.
- Required root files and durable docs.

## Result

- Workspace size reduced from about `21978.91 MB` to about `2088.67 MB`.
- Current largest files after cleanup:
  - checkpoint bin: about `1512.69 MB`.
  - G1 live replay binary: about `260.05 MB`.
  - G2 live replay binary: about `247.95 MB`.

## Runtime Note

- The trainer process `4730` is no longer running.
- Log ended after generation 2 with `checkpoint_write_failed=Input/output error (os error 5)`.
- The main checkpoint file is still present; the failed `.tmp` checkpoint was removed as a broken write artifact.
