# 2026-06-02 submission validation episode failed

## User Request

- User provided Kaggle validation episode JSON files for `submission.tar.gz` failure and explicitly said not to submit without approval.

## Diagnosis

- Kaggle validation episode `78491291` failed on the first action call.
- Both participant stderr logs report:
  - `RuntimeError: native library missing: /kaggle/working/liborbit_wars_agent.so`
- Root cause: submitted `main.py` was executed from `/kaggle_simulations/agent/main.py`, but when `__file__` was unavailable the shim searched `Path.cwd()` first, which is `/kaggle/working`, not the extracted agent artifact directory.
- This is a packaging/runtime artifact path bug, not a transformer/decoder action bug.

## Local Fix

- `kaggle_submission/main.py` artifact search now also uses:
  - `inspect.getsourcefile(lambda: None)`
  - `sys.argv[0]`
  - explicit `/kaggle_simulations/agent`
- The shim also returns `[]` without calling native inference when the current observation has no owned planet for the player. This is not an action fallback; it is the no-legal-source command contract.

## Verification

- Rebuilt `submission.tar.gz`; archive contents are `main.py`, `liborbit_wars_agent.so`, `model.bin`.
- Verified unpacked archive in a no-`__file__` execution mode from a different cwd:
  - native library and model load successfully.
  - synthetic owned-planet observation returns actions.
  - synthetic no-owned-planet observation returns `[]`.
- No new Kaggle submission was sent after the user instruction requiring explicit approval.

## Training Status

- Existing CUDA self-train resume process was not interrupted.
- Trainer PID check: `69092 Rsl target/release/orbit-wars-trainer --cuda --players 4 --generations 64 --resume-checkpoint artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`.
