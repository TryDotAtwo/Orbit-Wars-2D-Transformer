# 2026-06-02 Expand Champion Archive To Generation Model Artifacts

## Change

- Added `tools/export_champion_archive_top_models.ps1`.
- Exported checkpoint `champion_archive` from `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin` into the external completed-generation model artifact structure under `dashboard/public/telemetry/generation_models/1780348585/`.
- Wrote missing `generation_1_top4.owmodels` through `generation_16_top4.owmodels`.
- Existing `generation_17_top4.owmodels` through `generation_28_top4.owmodels` were preserved; the exporter skips existing files unless `-Overwrite` is passed.

## Verification

- Checkpoint parsed as `OWTRAIN1`, runId `1780348585`, `nextGeneration=28`, population `128`, champion archive `108`.
- Exporter found 27 generations in checkpoint archive and wrote 16 missing generation files, skipped 11 existing files.
- Final external archive check for G1..G28: all 28 files exist.
- Every checked file has `OWTOP4_1`, matching generation id, model count `4`, and first transformer block shape `layers=61`, `rows=64`, `weights=3004114`.
- Final external top-4 archive size: 28 files, 1,345,847,776 bytes total.

## Notes

- Exported G1..G16 artifacts reconstruct `model_index=0..3` and `reward=0` because checkpoint `champion_archive` stores generation and model weights, not the original evaluated model index/reward metadata.
- Tournament correctness depends on generation id and model weights; the recovered G1..G16 files now provide those weights in the same external file format as newer generations.
