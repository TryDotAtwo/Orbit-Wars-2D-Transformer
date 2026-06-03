# docs/testing/SUMMARY.md

timestamp=2026-06-03T17:12:00+03:00
section=testing
active_task_id=2026-06-03_dashboard_build_telemetry_copy_fix

## Purpose

- Keep current verification commands and latest records.
- Detailed historical check output remains in `test_results/`.

## Standard Verification

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

```powershell
docker run --rm --gpus all -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so cargo run -p orbit-wars-trainer -- --cuda-true2d-smoke"
```

```powershell
cd dashboard
npm.cmd run build
```

## Docs-Only Verification

- Confirm required root files exist.
- Confirm section summaries exist.
- Confirm `PROJECT_MEMORY.md` and `index.md` are readable.
- Confirm durable prompt and verification records are registered when the task changes durable docs.
- Do not run heavy Rust/CUDA/dashboard builds for docs-only rewrites unless docs embed generated snippets from those tools.

## Latest Records

- `test_results/2026-06-03_dashboard_build_telemetry_copy_fix.md`: dashboard production build passes after disabling public-dir copy; `dist/telemetry` is absent and `dist` payload is about 246 KB.
- `test_results/2026-06-03_nvidia_perf_bug_review_no_code_changes.md`: no-code-change review; Docker Rust tests and Python syntax checks passed; dashboard build failed on telemetry copy `ENOSPC`.
- `test_results/2026-06-03_simplify_agent_rules.md`: agent protocol simplification verification.
- `test_results/2026-06-03_docs_cleanup.md`: documentation compaction verification.
- `test_results/2026-06-03_population64_games20_top6_weighted_repro.md`: latest Rust verification for population 64, 20 participations, top-6 weighted reproduction.
- `test_results/2026-06-03_increase_sun_target_penalty.md`: Sun target penalty scale verification.
- `test_results/2026-06-03_velocity_input_phase_aim_restart.md`: velocity input and phase-aware aiming verification.

## Known Gaps

- Broad official random-seed trace comparison is still not complete.
- Current 64-model/20-participation full-generation timing is recorded but slow: generations 58-60 took about 1249-1398 seconds.
- Dashboard production build no longer copies runtime telemetry after `dashboard/vite.config.ts` set `build.copyPublicDir=false`.
- Browser screenshot capture has timed out before; DOM/interaction/canvas checks may be more reliable evidence.
