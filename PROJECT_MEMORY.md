# PROJECT_MEMORY.md

timestamp=2026-06-03T17:12:00+03:00
project_state=performance_memory_first_pass
active_task_id=2026-06-03_dashboard_build_telemetry_copy_fix

## Working Rule

- This file is the current operational memory, not a full changelog.
- Detailed historical evidence stays in `test_results/`.
- Raw user prompts stay in `prompt_history/`.
- Do not read `project_history_full/` without explicit user approval for that specific access.
- Follow `AGENTS.md`: use the lightweight route, read only the context needed for the task, and keep durable state inside project files.

## Project Snapshot

- project=Kaggle Orbit Wars native self-play research workspace.
- workspace_root=`C:/Users/Иван Литвак/Documents/Кораблики`.
- rust_workspace=`Cargo.toml`; crates are `orbit-wars-core`, `orbit-wars-ffi`, `orbit-wars-trainer`.
- cuda_module=`native/cuda/`.
- dashboard_app=`dashboard/`.
- kaggle_submission_scaffold=`kaggle_submission/`.
- official_reference_source=`C:/tmp/kaggle-env-src`, commit=`d754e427f2e1862f99d87fac75f3d6568ab8c31d`.
- official_competition_package=`artifacts/orbit_wars_official_2026_05_31/`.
- current_checkpoint=`artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`.

## Current Architecture

- `crates/orbit-wars-core` owns config, game types, encoder, decoder, geometry, simulator, model wrapper, and `True2DTransformer`.
- `crates/orbit-wars-ffi` exposes the CPU native agent ABI used by the Kaggle Python shim.
- `crates/orbit-wars-trainer` owns CUDA self-play, trajectory capture, elite backprop, weighted reproduction, checkpoints, generation tournament, telemetry, and replay artifact writing.
- `native/cuda` owns true2d CUDA inference and training kernels.
- `dashboard` is a Vite/React telemetry reader; it must not mutate trainer state.
- `kaggle_submission/main.py` loads `model.bin` and native shared library, exposes final callable `agent(obs)`, and returns no actions only when no owned source planet exists.

## Current Contracts

- submission_policy=no Kaggle submit without explicit user approval for that upload.
- training_backend=CUDA for training inference and backprop; CPU inference is reserved for final Kaggle submission.
- model=`True2DTransformer`.
- input_shape=`64 x 7`: owner class, ship log, x, y, production, velocity x, velocity y.
- output_shape=`64 x 24`: 12 target/send pairs per source row.
- forbidden_pair_features=distance, angle, flight time, sun intersection, manual pair score, from-to heuristic.
- decoder_geometry_allowed_after_model_output=true; decoder may refine launch angle for selected target hits.
- decoder_sun_guard=not implemented; sun-blocked selected routes remain visible training failures.
- precision_policy=fp32 stored/final weights; CUDA inference attention may use bf16-rounded operands with fp32 accumulation; CUDA backprop remains fp32.
- reward_scale=win 2, draw -1, loss -2.
- full_training_target=4 players, population 64, 20 participations per model, top 6 selected, 320 actual games per generation.
- weighted_reproduction=selected parent slots weighted by smoothed winrate plus normalized reward, bounded by configured min/max slots.
- generation_archive=top 4 models per generation, at most 32 generations, interval-32 tournament.
- dashboard_replay_policy=4 actual evaluation games per completed generation; live replay uses preallocated slot storage; completed replay payloads are loaded lazily.

## Latest Verified Work

- `test_results/2026-06-03_docs_cleanup.md`: root memory, navigation index, and section summaries compacted for agent work; docs-only sanity checks passed.
- `test_results/2026-06-03_simplify_agent_rules.md`: `AGENTS.md` and related docs changed from mandatory blanket loading/registration to a lightweight, judgment-based route.
- `test_results/2026-06-03_dashboard_build_telemetry_copy_fix.md`: dashboard production build no longer copies runtime telemetry; `npm.cmd run build` passed and `dist/telemetry` was absent.
- `test_results/2026-06-03_nvidia_perf_bug_review_no_code_changes.md`: no-code-change project/code/performance review; Docker Rust tests passed, Python syntax checks passed, dashboard build failed on telemetry copy `ENOSPC`, current full_500 timing is slow.
- `test_results/2026-06-03_population64_games20_top6_weighted_repro.md`: population 64, gamesPerModel 20, top-6 weighted reproduction, checkpoint population shrink support; `cargo fmt --all`, `cargo test --workspace`, and release trainer build passed.
- `test_results/2026-06-03_increase_sun_target_penalty.md`: action-local Sun target penalty scale raised to 4.0 and verified.
- `test_results/2026-06-03_velocity_input_phase_aim_restart.md`: true2d input expanded to `64x7`, decoder phase-aware aiming verified, CUDA smokes passed.
- `test_results/2026-06-02_sun_target_penalty_tournament_dashboard_lag.md`: generation winrate dedup lag fixed and Sun target penalty backprop implemented.
- `test_results/2026-06-02_git_init_bloat_guard.md`: `.gitignore` protects generated and heavy runtime outputs; no commit was created.

## Runtime Status

- last_recorded_run=`runId=1780348585`.
- last_recorded_status=training is running in `orbit-wars-cuda-dev`; active process resumes `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`; latest reviewed telemetry was generation 61 evaluation.
- operational_rule=verify live process, checkpoint generation, and telemetry before killing, restarting, submitting, or judging training quality.

## Known Risks

- Full official random-seed parity is not proven; deterministic reference scenarios passed, but broad seed trace comparison is still needed.
- CUDA full-attention backprop is correct on smoke tests but still throughput-sensitive; tiled/shared-memory/tensor-core attention and transfer overlap remain future optimization.
- GPU utilization telemetry is not a real hardware occupancy metric unless explicitly measured.
- Current full_500 population=64/top-6 generation timing is slow: recent completed generations were about 1249-1398 seconds.
- Runtime telemetry remains large under `dashboard/public/telemetry`, but production build no longer copies it into `dist`.
- Sun-blocked routes are intentionally not hidden by decoder; learning signal comes from action-local penalties.
- Large runtime artifacts may exist under ignored paths; do not stage or delete them blindly.

## Next Actions

- Continue separating dashboard runtime telemetry from source/build lifecycle; production build copy is fixed, artifact retention still needs policy.
- Benchmark/profile a fresh full_500 generation under the current 64-model contract with phase splits.
- Run broad official random-seed trace comparison against Kaggle source before trusting exact map distribution parity.
- Optimize CUDA forward/backward attention and host transfers after measuring the current bottleneck.
- Prepare Kaggle timing/submission packages only after explicit user approval.

## Documentation Policy

- Keep root memory and section summaries short enough for targeted reads.
- Register durable prompts, test records, docs, public entities, and decisions when they affect future work.
- Do not duplicate full test histories in summaries; link to the latest or most relevant `test_results/` files.
