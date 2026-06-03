# 2026-06-02 Completed Replay Binary Chunk Loader

## Scope

- Fixed completed-generation Replay tab behavior after G1 still opened as a partial live replay.
- Kept the current training run intact; no trainer restart was required.

## Root Cause

- When the selected completed generation chunk was not loaded yet, ReplayCanvas fell back to current telemetry frames.
- During active G2/G3 evaluation this meant selected G1 could show partial live frames from the active generation.
- The completed G1 durable JSON chunk for runId `1780348585` was `419544531` bytes, so loading/parsing it in the browser was too slow for interactive replay review.
- The same G1 games were already available in append-only binary live replay storage as `live_1780348585_generation_1.owlive`.

## Implementation

- `dashboard/src/App.tsx` now loads replay chunks only for the selected generation instead of fetching every announced chunk on Replay tab entry.
- `dashboard/src/App.tsx` no longer falls back to active-generation telemetry frames when a different generation is selected.
- `dashboard/src/sampleData.ts` now prefers the sibling binary `.owlive` file for `/telemetry/replays_<run>_generation_<n>.json` chunk paths.
- `dashboard/src/sampleData.ts` can also load `.owlive` paths directly through the existing binary replay decoder.
- JSON chunk loading remains as an explicit fallback when the binary file is unavailable.

## Verification

- Dashboard build passed after the known sandbox `esbuild spawn EPERM` required an escalated rerun:
  - `npm.cmd run build`
- Browser QA used the in-app Browser on `http://127.0.0.1:5173/`.
- Flow under test:
  - Replay tab -> select `G1` -> load completed replay -> inspect game options and frame count.
- Browser DOM result:
  - game options were `G1 game 1` through `G1 game 10`.
  - replay summary reported `frames=501`.
  - console error/warn log list was empty.

## Result

- Completed G1 is now selectable as a full 501-frame replay instead of showing partial live G2 frames.
- Replay tab avoids the worst JSON parse stall by preferring binary replay storage for completed-generation chunks.
