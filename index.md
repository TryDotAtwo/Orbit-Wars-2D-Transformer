# index.md

timestamp=2026-06-03T17:43:00+03:00
index_scope=working_navigation
active_task_id=2026-06-03_population_forward_scratch

## How To Use This Index

- Use this file as the project map when navigation, registration, or unfamiliar files are involved.
- Use `PROJECT_MEMORY.md` for current state and active risks.
- Use `project_config.yaml` for all tunable parameters, limits, paths, and external-service settings.
- Use `docs/<section>/SUMMARY.md` for current section contracts.
- Use `test_results/` for detailed historical verification records.
- Use `prompt_history/` for raw user prompt history.
- Do not read `project_history_full/` without explicit user approval.

## Root Files

| path | purpose |
|---|---|
| `AGENTS.md` | Lightweight agent route, registration rules, coding rules, and communication policy. |
| `PROJECT_MEMORY.md` | Compact current operational memory. |
| `project_config.yaml` | Authoritative external parameters and limits. |
| `index.md` | Compact navigation map and durable registration index. |
| `Cargo.toml` | Rust workspace manifest. |
| `Cargo.lock` | Rust lockfile. |
| `.gitignore` | Bloat guard for build outputs, dependencies, telemetry, checkpoints, replays, archives, and local scratch. |

## Main Directories

| path | purpose | notes |
|---|---|---|
| `crates/orbit-wars-core/` | Native game logic, config, encoder, decoder, simulator, model implementation. | Primary contract source for model and simulator behavior. |
| `crates/orbit-wars-ffi/` | C ABI for Python Kaggle shim. | CPU submission inference path. |
| `crates/orbit-wars-trainer/` | CUDA self-play trainer, backprop, reproduction, checkpoint, telemetry, replay writing. | Large file by design; inspect targeted functions with `rg`. |
| `native/cuda/` | CUDA inference/training library and smoke programs. | Training-only local acceleration. |
| `dashboard/` | Vite/React training dashboard. | Reads telemetry only. |
| `kaggle_submission/` | Submission entrypoint and model scaffold. | `main.py` must expose final callable `agent(obs)`. |
| `tools/` | Operational scripts for build, submission, checkpoint/replay repair, reference compare, and telemetry restore. | Prefer existing scripts over ad hoc rewrites. |
| `docs/` | Current working documentation by section. | Keep concise; do not mirror every old task. |
| `test_results/` | Append-only verification and investigation records. | Detailed history belongs here. |
| `prompt_history/` | Raw prompt persistence. | Required for durable tasks; optional for tiny status/lookups. |
| `artifacts/` | Generated runtime/checkpoint/replay/submission artifacts. | Usually ignored by Git; inspect before deletion. |

## Rust Core Map

| path | public surface |
|---|---|
| `crates/orbit-wars-core/src/lib.rs` | Exports config, encoder, decoder, geometry, simulator, and model types. |
| `crates/orbit-wars-core/src/config.rs` | `AgentConfig`, including `trainer_checkpoint_path`, and named constants mirrored from `project_config.yaml`. |
| `crates/orbit-wars-core/src/types.rs` | `Planet`, `Fleet`, `RowFeature`, `ActionTargetOutput`, `ActionOutput`, `MoveCommand`. |
| `crates/orbit-wars-core/src/encoder.rs` | `encode_state`, row ordering, owner-class mapping, `64x7` input construction. |
| `crates/orbit-wars-core/src/decoder.rs` | `decode_model_outputs`, trace decoder, post-output aiming and command filtering. |
| `crates/orbit-wars-core/src/geometry.rs` | Fleet speed, intercept, orbit prediction, sun/bounds/collision geometry. |
| `crates/orbit-wars-core/src/simulator.rs` | Official-like native turn loop and event traces. |
| `crates/orbit-wars-core/src/true2d_model.rs` | `True2DTransformer`, shape, weights, forward pass, mutation, crossover, migrations. |
| `crates/orbit-wars-core/src/model.rs` | Submission model wrapper and legacy trainable model support. |
| `crates/orbit-wars-core/src/scaler.rs` | Ship-log scaling helper. |

## Trainer And CUDA Map

| path | public surface |
|---|---|
| `crates/orbit-wars-trainer/src/main.rs` | CLI, evaluation, trajectory capture, CUDA backprop, weighted reproduction, checkpoint, tournament, telemetry, replay artifacts. |
| `crates/orbit-wars-trainer/src/cuda_true2d.rs` | Rust dynamic binding for true2d CUDA forward, resident population forward, and training ABI. |
| `native/cuda/orbit_wars_cuda.h` | C ABI declarations. |
| `native/cuda/orbit_wars_cuda.cu` | CUDA kernels and exported ABI. |
| `native/cuda/orbit_wars_cuda_true2d_smoke.cpp` | C++ true2d smoke. |
| `native/cuda/orbit_wars_cuda_smoke.cpp` | Legacy CUDA smoke. |
| `native/cuda/README.md` | CUDA module contract and smoke commands. |

## Dashboard Map

| path | purpose |
|---|---|
| `dashboard/src/App.tsx` | Dashboard views, telemetry loading, replay/chunk interaction. |
| `dashboard/src/Charts.tsx` | Metric chart rendering. |
| `dashboard/src/ReplayCanvas.tsx` | Orbit Wars replay drawing. |
| `dashboard/src/types.ts` | Dashboard telemetry and replay types. |
| `dashboard/src/styles.css` | Dashboard layout and styling. |
| `dashboard/public/telemetry/` | Runtime telemetry files, normally ignored. |

## Tools Map

| path | purpose |
|---|---|
| `tools/build_submission.sh` | Linux submission archive build. |
| `tools/docker_build_submission.ps1` | Windows Docker wrapper for submission build. |
| `tools/benchmark_agent.py` | CPU submission timing benchmark. |
| `tools/kaggle_submission_smoke.py` | Local Kaggle runner smoke. |
| `tools/kaggle_action_probe.py` | Direct action probe for prepared submission. |
| `tools/orbit_wars_reference_compare.py` | Native simulator versus official source comparison. |
| `tools/restart_after_current_generation_owslot.sh` | Checkpoint-boundary trainer restart helper. |
| `tools/restore_latest_history_from_trainer_log.py` | Rebuilds latest telemetry history from logs and generation artifacts. |
| `tools/compact_owslot_replay.py` | Compacts live slot replay storage. |
| `tools/trim_owlive_replay_games.py` | Rewrites completed binary replays to a smaller actual-game count. |
| `tools/export_champion_archive_top_models.ps1` | Exports top-model archive from checkpoint. |
| `tools/submit_staged_kaggle.ps1` | Host-side staged Kaggle submit helper; requires explicit approval. |

## Documentation Sections

| path | purpose |
|---|---|
| `docs/architecture/SUMMARY.md` | Current architecture and boundaries. |
| `docs/contracts/SUMMARY.md` | Active input, output, simulator, training, ABI, and telemetry contracts. |
| `docs/deployment/SUMMARY.md` | Toolchain, runtime, command, and submission gate notes. |
| `docs/domain/SUMMARY.md` | Official Orbit Wars domain rules and source links. |
| `docs/domain/orbit_wars_competition_research.md` | Detailed official competition research snapshot. |
| `docs/models/SUMMARY.md` | Active model and training loop state. |
| `docs/performance/SUMMARY.md` | Current performance facts, bottlenecks, and next measurements. |
| `docs/prompts/SUMMARY.md` | Prompt-history policy and latest prompt registrations. |
| `docs/roadmap/SUMMARY.md` | Next tasks, blocked tasks, and acceptance gate. |
| `docs/testing/SUMMARY.md` | Verification commands, latest test records, and known gaps. |

## Current Durable Registrations

| path | purpose |
|---|---|
| `prompt_history/20260603_161245_docs_cleanup_request.md` | Raw user prompt requesting documentation cleanup and a more agent-friendly project. |
| `prompt_history/20260603_162321_simplify_agent_rules.md` | Raw user prompt requesting simplification of `AGENTS.md` and related workflow docs. |
| `test_results/2026-06-03_docs_cleanup.md` | Documentation cleanup verification record. |
| `test_results/2026-06-03_simplify_agent_rules.md` | Verification record for simplifying the agent protocol and related docs. |
| `test_results/2026-06-03_population64_games20_top6_weighted_repro.md` | Latest code verification for population=64, gamesPerModel=20, top-6 weighted reproduction. |
| `test_results/2026-06-03_increase_sun_target_penalty.md` | Sun target penalty scale verification. |
| `test_results/2026-06-03_velocity_input_phase_aim_restart.md` | `64x7` input and phase-aware aiming verification. |
| `prompt_history/20260603_163041_nvidia_perf_bug_review_no_code_changes.md` | Raw user prompt requesting no-code-change project/code/performance review with NVIDIA plugin use. |
| `test_results/2026-06-03_nvidia_perf_bug_review_no_code_changes.md` | No-code-change code/performance review: Rust tests pass, dashboard build blocked by telemetry copy, current full_500 timing is slow. |
| `prompt_history/20260603_170612_perf_memory_github_plugins_request.md` | Raw user prompt requesting plugin-assisted performance/memory work with incremental GitHub commits. |
| `docs/superpowers/plans/2026-06-03-performance-memory-first-pass.md` | Superpowers implementation plan for the first performance/memory pass. |
| `test_results/2026-06-03_dashboard_build_telemetry_copy_fix.md` | Dashboard production build verification after preventing runtime telemetry copy into `dist`. |
| `test_results/2026-06-03_checkpoint_path_config_fix.md` | Red/green verification for config/resume-aware trainer checkpoint save path. |
| `test_results/2026-06-03_hot_loop_allocation_cleanup.md` | Red/green and full Rust verification for small trainer hot-loop allocation cleanup. |
| `test_results/2026-06-03_population_forward_scratch.md` | Red/green and full Rust verification for reusable CUDA forward host packing scratch. |

## Search Hints

- Use `rg "symbol_or_contract"` before opening large files.
- Use `rg --files -g '!project_history_full/**' -g '!target/**' -g '!dashboard/node_modules/**' -g '!artifacts/**'` for project navigation.
- Use `rg -n "test_name|contract_name" crates tools dashboard docs` for scoped changes.
- Avoid full-directory reads of generated telemetry, build outputs, checkpoints, and replay payloads.
