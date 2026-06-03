# 2026-06-02 Chart Tooltips And Sun-Loss Metrics

## Change

- `dashboard/src/Charts.tsx`: metric chart header now always shows the latest completed generation value, while the SVG tooltip/callout is rendered only when the pointer selects a chart point.
- Removed the previous implicit `hoveredIndex ?? latestIndex` behavior that drew a chart-local tooltip on every chart even without hover.

## Sun-Loss Data Check

- `dashboard/public/telemetry/latest.json` contains sun-loss metrics for completed generations.
- Current checked values: G17..G28 have non-zero `sunDestroyedFleets`; G1..G16 are zero in the restored history.
- Immutable logs under `dashboard/public/telemetry/generation_logs/1780348585/` exist for G17..G28 and contain nested `metric.sunDestroyedFleets` / `metric.sunDestroyedShips`.
- Example checked value: G25 `sunDestroyedFleets=1194692`, `sunDestroyedShips=1781812`.

## Verification

- `npm.cmd exec tsc -- --noEmit`: passed.
- `npm.cmd run build`: sandbox run failed with `esbuild spawn EPERM`; escalated build passed.
- Cleanup: removed generated `dashboard/dist` after build verification because Vite copied `public/telemetry` into it and expanded it to about 6.5 GB.
- Browser code check after reload showed no `.chart-hover` elements without hover. Full mouse hover confirmation was blocked by in-app browser coordinate translation after page scroll, but the rendered condition is covered by the code path and production build.
