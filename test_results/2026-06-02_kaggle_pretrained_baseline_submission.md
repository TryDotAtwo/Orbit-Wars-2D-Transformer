# 2026-06-02 Kaggle Pretrained Baseline Submission

status=submitted_pending
task_id=2026-06-02_kaggle_submit_pretrained_baseline

## Official Submission Contract Read

- Official `agents.md` confirms `main.py` must be at submission root with `agent(obs)`.
- Multi-file submissions are sent as `submission.tar.gz` with `main.py` at archive root.
- User has entered `orbit-wars`: `kaggle competitions list --group entered` reported `userHasEntered=True`.
- No previous submissions were present: `kaggle competitions submissions orbit-wars` returned `No submissions found`.

## Prepared Artifact

- Built fresh Linux native library with `tools/build_submission.sh`.
- `submission.tar.gz` contents: `main.py`, `liborbit_wars_agent.so`, `model.bin`.
- `model.bin` is the deterministic untrained true2d baseline: `seed=17`, `layers=61`, `mode=true_2d_pair_matrix_transformer`.
- Python remains only a thin `ctypes` wrapper around native Rust FFI.

## Fixes

- `kaggle_submission/main.py` now supports Kaggle local runner execution where `__file__` is absent by searching the module directory, current working directory, and `kaggle_submission/`.
- Added `tools/kaggle_submission_smoke.py` for repeatable local Orbit Wars submission smoke tests.

## Verification

- Archive content check passed: `tar -tzf submission.tar.gz` listed only the three expected root files.
- CPU native benchmark passed: `iterations=2000`, `mean_ms=158.285084`, `p50_ms=151.068509`, `p95_ms=223.867683`, `max_ms=363.377644`.
- Local official Kaggle environment smoke passed against three random agents: `steps=500`, player 0 reward `1`, all statuses `DONE`.

## Submit Status

- Attempted command: `kaggle competitions submit orbit-wars -f submission.tar.gz -m "native true2d seed17 baseline cpu ffi"`.
- The upload was blocked by the approval policy as external disclosure of project code/model artifacts.
- Training process was not interrupted.
- After explicit user approval, submit succeeded.
- Kaggle submission ref=`53283033`.
- Kaggle submission status=`SubmissionStatus.PENDING` at `2026-06-02 08:19:26.163000` Kaggle output time.
