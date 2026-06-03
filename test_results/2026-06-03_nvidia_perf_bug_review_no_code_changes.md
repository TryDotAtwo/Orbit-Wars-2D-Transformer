# 2026-06-03 NVIDIA/performance/code review without code changes

task=project_status_perf_bug_review
code_changes=false
used_nvidia_plugin=true

## Scope

- Reviewed current documentation shape, trainer/CUDA hot paths, dashboard build behavior, Kaggle shim/tools syntax, and live training status.
- Used the NVIDIA plugin context where applicable. The available NVIDIA skill was infrastructure/GPU-ops oriented, not a CUDA code-review tool, so the review used read-only GPU checks and CUDA source inspection.
- Did not change Rust/CUDA/Python/TypeScript source code.

## Verification

- Prompt persisted: `prompt_history/20260603_163041_nvidia_perf_bug_review_no_code_changes.md`.
- `cargo test --workspace` through Docker: passed.
  - `orbit-wars-core`: 35 passed.
  - `orbit-wars-ffi`: 0 tests.
  - `orbit-wars-trainer`: 20 passed.
  - Warning: `legacy_live_replay_public_path` is unused.
- Python syntax check: `py -m py_compile` for `kaggle_submission/main.py` and main `tools/*.py`: passed.
- Host `cargo test --workspace`: not available because `cargo` is not on host PATH.
- Dashboard build:
  - Sandbox run failed first with `spawn EPERM` for esbuild.
  - Escalated run reached Vite production build and failed with `ENOSPC`.
  - Root cause: Vite copies `dashboard/public/telemetry` into `dashboard/dist`; current telemetry is about 10.85 GB.
  - Partial generated `dashboard/dist` from the failed build was removed after approval.
- CUDA smoke was not run because a live trainer was already using the active container/GPU.

## Live Runtime Facts

- `orbit-wars-cuda-dev` is running.
- Active process: `target/release/orbit-wars-trainer --cuda --players 4 --generations 64 --resume-checkpoint artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`.
- Latest telemetry at review time:
  - `runId=1780348585`
  - `activeGeneration=61`
  - `profile=full_500`
  - `populationSize=64`
  - `eliteCount=6`
  - `gamesPerModel=20`
  - `playersPerGame=4`
- Recent generation timings after population 64/top 6 restart:
  - generation 58: 1248.763794 s
  - generation 59: 1288.051514 s
  - generation 60: 1397.661743 s
- Short NVIDIA SMI sample inside the container saw bursty GPU utilization: one 100% sample followed by several 0% samples; memory stayed around 1600 MiB.

## Findings

1. The project is workable, but not operationally clean yet.
   - Rust tests pass and training is alive.
   - The repository has no committed baseline: `git status --short` shows the whole workspace as untracked.

2. Dashboard production build is currently broken by runtime artifacts.
   - `dashboard/public/telemetry` is about 10.85 GB.
   - Vite copies public files to `dist`, so `npm run build` can fail with `ENOSPC`.
   - Runtime telemetry should not live in the build public directory, or Vite needs a build/dev split.

3. Current full training speed misses the old 600-second generation target.
   - Current full_500 generations are around 20-23 minutes each.
   - Population was reduced from 128 to 64, but participations stayed 320 actual games per generation, so evaluation work stayed large.

4. CUDA resident weights and persistent workspace are already in place.
   - The biggest remaining speed work is not "copy weights every forward".
   - Current forward still copies request inputs/model indices to device and full outputs back to host each turn.
   - Attention is untiled full relation-cell attention over 4096 cells; it loops through keys twice per query cell.

5. Host-side overhead is likely material.
   - Each turn clones planet vectors into `ActionRequest`.
   - Each turn spawns scoped worker threads in collect/decode/step helpers.
   - Each CUDA forward allocates a fresh host output vector and materializes `ActionOutput` rows.
   - Training sample selection duplicates selected samples into dense host buffers before CUDA backprop.

6. Checkpoint path handling has a contract mismatch.
   - `project_config.yaml` declares the current checkpoint path.
   - Trainer saves to hardcoded `DEFAULT_TRAINER_CHECKPOINT_PATH`.
   - Current default matches, but `--resume-checkpoint <other path>` will still save back to the default path.

7. Documentation is much better than before, but still has small hygiene issues.
   - `AGENTS.md` is lightweight and usable.
   - `PROJECT_MEMORY.md` contains a mojibake workspace path.
   - No root `README.md` exists.

## Recommended Next Changes

1. Fix dashboard telemetry/build separation first.
   - Move runtime telemetry outside `dashboard/public`, or configure a dev-only public telemetry mount.
   - Keep `latest.json` and replay artifacts served in dev without copying them into production `dist`.

2. Fix trainer checkpoint path source.
   - Use the configured checkpoint path for save/resume, or derive the save path from `--resume-checkpoint` when explicitly supplied.

3. Add low-risk hot-path cleanup.
   - Reuse host input/output/model-index buffers.
   - Avoid cloning full planet vectors per request where decoding can borrow/snapshot once per game-turn.
   - Replace per-turn scoped thread spawning with a reusable worker strategy.

4. Profile before CUDA kernel rewrites.
   - Use Nsight Systems/Compute or at least per-phase timers for host packing, H2D, kernel group, D2H, decode, simulation, telemetry.
   - Then decide whether to tile/shared-memory attention, use pinned buffers/streams, or reduce host-side work first.
