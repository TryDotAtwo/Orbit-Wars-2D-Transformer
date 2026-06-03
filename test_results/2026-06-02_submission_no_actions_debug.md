# 2026-06-02 Submission No Actions Debug

status=fixed_locally_not_resubmitted
task_id=2026-06-02_submission_no_actions_debug

## User Report

- Kaggle validation replay showed no visible actions from submitted native true2d baseline.
- User explicitly rejected fallback behavior; diagnosis and fix kept the transformer/decoder path.

## Root Causes

- `encode_state` incorrectly selected home/anchor planet from `initial_planets.owner == player`.
- Official observations keep `initial_planets` as initial geometry and do not mark the current player's home as owned there; ownership is in current `planets`.
- Kaggle local raw runner selects the last callable in `main.py`; `agent()` was defined before helper functions, so the runner could call a helper instead of `agent`.

## Fixes

- `crates/orbit-wars-core/src/encoder.rs` now selects the home/anchor planet from current `planets`.
- Added regression test `official_initial_planets_can_have_neutral_home_owner`.
- `kaggle_submission/main.py` now defines `agent()` as the final callable in the file.
- No fallback policy was added.

## Verification

- `cargo fmt --all` passed.
- `cargo test --workspace` passed: 31 core tests and 13 trainer tests.
- `tools/build_submission.sh` rebuilt `submission.tar.gz`.
- Local Kaggle runner smoke passed and showed first non-empty player action at step 2 with 12 commands.
- Direct probes on fixed agent returned actions at step 10, 50, and 93.
- CPU benchmark after the fix: `mean_ms=106.356704`, `p50_ms=104.469833`, `p95_ms=121.902018`, `max_ms=144.560344`.

## Submit Status

- Fixed `submission.tar.gz` is prepared locally.
- Fixed archive has not been submitted yet; explicit user approval is required before external upload.
