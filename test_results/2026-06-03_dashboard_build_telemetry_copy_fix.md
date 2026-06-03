# 2026-06-03 Dashboard Build Telemetry Copy Fix

task=dashboard_build_memory_fix
code_changes=true

## Change

- Set `build.copyPublicDir=false` in `dashboard/vite.config.ts`.
- This keeps Vite production build from copying runtime telemetry under `dashboard/public/telemetry` into `dashboard/dist`.
- Dev dashboard behavior is unchanged: Vite dev server still serves files from `dashboard/public`, including `/telemetry/latest.json` and replay artifacts.

## Verification

Command:

```powershell
cd dashboard
npm.cmd run build
```

Result:

- exit_code=0
- TypeScript completed.
- Vite transformed 32 modules.
- Vite built in 1.19 s.
- Output files:
  - `dist/index.html`
  - `dist/assets/index-CozZdZJ3.css`
  - `dist/assets/index-B1Bp75HP.js`

Post-checks:

- `Test-Path dashboard\dist\telemetry` returned `False`.
- `dashboard/dist` file payload was 245,917 bytes.

## Previous Failure Addressed

`test_results/2026-06-03_nvidia_perf_bug_review_no_code_changes.md` recorded `npm.cmd run build` failing with `ENOSPC` while copying about 10.85 GB of runtime telemetry from `dashboard/public/telemetry`.
