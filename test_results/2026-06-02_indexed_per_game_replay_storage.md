# 2026-06-02 indexed per-game replay storage

status=implemented_partially_verified

## Scope

- Implement indexed replay storage so the dashboard can list games immediately.
- Load only the selected game's full frames.
- Keep historical replay formats readable.

## Changes

- `crates/orbit-wars-trainer/src/main.rs`
  - `write_replay_chunk` now writes a small `indexed_replay_v1` generation manifest when replay games are present.
  - Each actual game is written to `dashboard/public/telemetry/replay_games/{run_id}/generation_{generation}_game_{game_index}.json`.
  - Per-game payload format is `compact_replay_game_v1`, reusing the compact actual-game tuple.
- `dashboard/src/sampleData.ts`
  - `loadReplayChunk` decodes `indexed_replay_v1` into frame-less replay entries with `gamePath`.
  - `loadReplayGame` fetches one selected per-game payload.
  - Legacy `binary_replay_alias_v1`, `.owlive`, `.owslot`, `compact_replay_v1`, and `compact_replay_v2` remain readable.
- `dashboard/src/App.tsx`
  - Replay tab loads selected generation manifest first.
  - Selecting an indexed game triggers a fetch for only that game payload.
  - Merge logic replaces frame-less indexed entries with loaded full-frame entries.

## Verification

- `npm.cmd exec tsc -- --noEmit` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` passed before the final dead-const cleanup.
- Follow-up Docker `cargo test -p orbit-wars-trainer` could not run because approval review timed out twice.
- `npm.cmd run build` in sandbox failed with known `esbuild spawn EPERM`; escalated retry approval also timed out.
- Browser validation was blocked because the discovered Browser tool surface did not expose the required JS action API.

## Runtime Note

- Existing historical chunks using `binary_replay_alias_v1` still load through the legacy whole-binary fallback.
- New chunks written by the rebuilt trainer use indexed per-game storage and avoid loading the whole generation before game selection.
