# Performance Memory First Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce immediate build/memory pain without cutting model, trainer, replay, dashboard, or submission functionality.

**Architecture:** Keep behavior stable and change implementation boundaries only. First remove the dashboard production-build copy of runtime telemetry, then align checkpoint save path with config/resume intent, then take low-risk hot-loop allocation wins. Deeper CUDA/trajectory memory work needs a profiler-backed follow-up batch.

**Tech Stack:** Rust workspace, CUDA C++ source, Vite/React dashboard, PowerShell/Docker verification, local Git commits.

---

### Task 1: Establish Git Baseline

**Files:**
- Stage: all non-ignored project files.
- Commit: baseline snapshot before performance work.

- [x] **Step 1: Verify ignored heavy artifacts are not staged**

Run:

```powershell
git status --ignored --short
git add -n -A -- .
```

Expected: ignored `artifacts/`, `target/`, `dashboard/node_modules/`, and `dashboard/public/telemetry/`; dry-run does not list those ignored paths.

- [x] **Step 2: Create working branch**

Run:

```powershell
git switch -c codex/perf-memory-first-pass
```

Expected: branch created.

- [x] **Step 3: Commit baseline**

Run:

```powershell
git add -A -- .
git commit -m "chore: establish project baseline"
```

Expected: first local commit contains only non-ignored project files.

### Task 2: Stop Dashboard Build From Copying Runtime Telemetry

**Files:**
- Modify: `dashboard/vite.config.ts`
- Register: `test_results/2026-06-03_dashboard_build_telemetry_copy_fix.md`
- Register: `docs/testing/SUMMARY.md`, `docs/performance/SUMMARY.md`, `index.md`

- [x] **Step 1: Confirm current failure**

Use the existing red evidence from `test_results/2026-06-03_nvidia_perf_bug_review_no_code_changes.md`: `npm.cmd run build` fails with `ENOSPC` while copying `dashboard/public/telemetry` to `dashboard/dist`.

- [x] **Step 2: Change Vite build-only copy behavior**

Set `build.copyPublicDir=false` in `dashboard/vite.config.ts`. Dev server still serves `dashboard/public`; production build no longer copies runtime telemetry into `dist`.

- [x] **Step 3: Verify dashboard build**

Run:

```powershell
cd dashboard
npm.cmd run build
```

Expected: production build exits 0 and `dashboard/dist/telemetry` does not exist.

- [x] **Step 4: Commit**

Run:

```powershell
git add dashboard/vite.config.ts docs/performance/SUMMARY.md docs/testing/SUMMARY.md index.md test_results/2026-06-03_dashboard_build_telemetry_copy_fix.md
git commit -m "fix: keep telemetry out of dashboard build"
```

### Task 3: Align Trainer Checkpoint Save Path With Config And Resume

**Files:**
- Modify: `crates/orbit-wars-core/src/config.rs`
- Modify: `crates/orbit-wars-trainer/src/main.rs`
- Register: `project_config.yaml`
- Register: `test_results/2026-06-03_checkpoint_path_config_fix.md`

- [x] **Step 1: Add a failing Rust test**

Add a trainer test that resumes from a non-default checkpoint path and asserts the selected save path is the resume path. Add a core config assertion that `AgentConfig::default().trainer_checkpoint_path` matches the configured checkpoint path.

- [x] **Step 2: Run targeted test and verify red**

Run:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer checkpoint_save_path -- --nocapture"
```

Expected: fails before implementation because trainer save path always uses the hardcoded default.

- [x] **Step 3: Implement stable path selection**

Add `trainer_checkpoint_path` to `AgentConfig`, mirror `project_config.yaml`, and make `trainer_checkpoint_path(&cli, &agent_config)` return `cli.resume_checkpoint` when supplied, otherwise `agent_config.trainer_checkpoint_path`.

- [x] **Step 4: Verify green**

Run:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer checkpoint_save_path -- --nocapture"
```

Expected: targeted tests pass.

- [x] **Step 5: Commit**

Run:

```powershell
git add crates/orbit-wars-core/src/config.rs crates/orbit-wars-trainer/src/main.rs project_config.yaml test_results/2026-06-03_checkpoint_path_config_fix.md
git commit -m "fix: respect configured trainer checkpoint path"
```

### Task 4: Remove Low-Risk Hot-Loop Allocations

**Files:**
- Modify: `crates/orbit-wars-trainer/src/main.rs`
- Register: `test_results/2026-06-03_hot_loop_allocation_cleanup.md`

- [x] **Step 1: Add targeted tests for unchanged decode sample-index behavior**

Add or extend a trainer unit test that records action traces with sample indices and confirms indices are consecutive when samples are captured.

- [x] **Step 2: Replace per-turn sample-index vectors**

Pass `first_sample_index: Option<usize>` into `decode_outputs_into_actions` and compute `first + request_index` locally instead of allocating a `Vec<usize>` every turn.

- [x] **Step 3: Reuse `requests_per_game` inside `run_games_batched`**

Move `requests_per_game` allocation outside the turn loop and reset it with `fill(0)` each turn.

- [x] **Step 4: Verify Rust workspace**

Run:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

Expected: all Rust tests pass.

- [x] **Step 5: Commit**

Run:

```powershell
git add crates/orbit-wars-trainer/src/main.rs test_results/2026-06-03_hot_loop_allocation_cleanup.md
git commit -m "perf: trim trainer hot-loop allocations"
```

### Task 5: GitHub Publication Gate

**Files:**
- No source changes.

- [x] **Step 1: Check remote**

Run:

```powershell
git remote -v
```

Result: `origin` was missing; user approved creating a private GitHub repository.

- [x] **Step 2: Push when remote exists**

Run:

```powershell
git push -u origin codex/perf-memory-first-pass
```

Result: private repository `TryDotAtwo/Orbit-Wars-2D-Transformer` created and branch `codex/perf-memory-first-pass` pushed.

### Task 6: CUDA Forward Packing Scratch

**Files:**
- Modify: `crates/orbit-wars-trainer/src/main.rs`
- Register: `test_results/2026-06-03_population_forward_scratch.md`

- [x] **Step 1: Add a failing scratch test**

Run:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer population_forward_scratch -- --nocapture"
```

Expected: RED before implementation because `PopulationForwardScratch` does not exist.

- [x] **Step 2: Add reusable scratch**

Implemented `PopulationForwardScratch` with reusable `input_rows` and `model_indices` buffers plus visible overflow handling for capacity calculation.

- [x] **Step 3: Use scratch in trainer loop**

Created scratch once in `run_games_batched` and passed it through `resolve_action_requests` to `batched_population_outputs`.

- [x] **Step 4: Verify**

Run:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

Expected: all Rust tests pass.

### Task 7: Split Live And Completed Replay Storage

**Files:**
- Modify: `crates/orbit-wars-trainer/src/main.rs`
- Modify: `dashboard/src/App.tsx`
- Modify: `dashboard/src/sampleData.ts`
- Register: `test_results/2026-06-03_live_completed_replay_storage_split.md`

- [x] **Step 1: Make live storage append-only**

Replaced new live replay `.owslot` preallocation with append-only `.owlive` records. Kept old `.owslot` cleanup/compaction support for existing artifacts.

- [x] **Step 2: Keep dashboard polling lightweight**

Changed dashboard polling to load `latest.json` only. Active live replay uses lightweight `frames` and metadata there; completed replay chunks remain lazy generation/game loads.

- [x] **Step 3: Verify**

Run:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
cd dashboard
npm.cmd run build
```

Expected: Rust workspace and dashboard production build pass.

### Task 8: Share Completed Replay Frames

**Files:**
- Modify: `crates/orbit-wars-trainer/src/main.rs`
- Register: `test_results/2026-06-03_replay_frame_sharing.md`

- [x] **Step 1: Add a failing sharing test**

Run:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer replay_games_share_frames_for_same_actual_game -- --nocapture"
```

Result: RED before implementation because replay frame buffers were not shared.

- [x] **Step 2: Share immutable frame buffers**

Changed `ReplayGame.frames` to `Arc<Vec<ReplayFrame>>` and made `replay_games_from_evaluation` share one frame vector across all participant views of the same actual game.

- [x] **Step 3: Verify**

Run:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

Result: full Rust workspace passed.

### Task 9: Keep Long-Lived Replay History Lightweight

**Files:**
- Modify: `crates/orbit-wars-trainer/src/main.rs`
- Register: `test_results/2026-06-03_lightweight_replay_history.md`

- [x] **Step 1: Add a failing metadata-history test**

Run:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer replay_chunk_records_keep_counts_without_frame_payloads -- --nocapture"
```

Result: RED before implementation because `ReplayChunkRecord` and the conversion helper did not exist.

- [x] **Step 2: Replace long-lived full replay history**

Changed trainer `replay_history` to store `ReplayChunkRecord` values after `write_replay_chunk`, while completed chunks and generation logs still use full `evaluation.replay_games` before those payloads are dropped.

- [x] **Step 3: Verify**

Run:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

Result: full Rust workspace passed.
