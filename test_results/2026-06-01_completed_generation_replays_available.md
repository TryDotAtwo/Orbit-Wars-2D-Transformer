# Completed Generation Replays Available

timestamp=2026-06-01T21:37:00+03:00
task_id=2026-06-01_completed_generation_replays_available

## User Requirement

- After a generation finishes, its games must remain available in the dashboard Replay selector.

## Root Cause

- Dashboard treated chunk replays and live replays as mutually exclusive.
- On active generation changes, dashboard cleared loaded chunk replay state.
- If a completed generation was selected while live replay was active, the UI could fall back to `sample replay` and current live frames instead of loading the completed generation chunk.

## Changes

- Dashboard now merges durable chunk replay games with current live replay games.
- Loaded replay chunks are cleared only when `runId` changes, not when `activeGeneration` changes.
- Dashboard preloads all announced `replayChunks` for the current run instead of only chunks matching the currently selected generation.

## Verification

- `npm.cmd run build` passed.
- Current runId=1780337978 reached activeGeneration=2 and published `replays_1780337978_generation_1.json`.
- Browser check on `http://127.0.0.1:5173` passed: while active generation was G2, selecting G1 showed `G1 game 1` through `G1 game 10`.
