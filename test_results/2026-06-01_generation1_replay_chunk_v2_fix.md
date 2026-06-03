# Generation 1 Replay Chunk V2 Fix

timestamp=2026-06-01T22:10:00+03:00
task=generation1_replays_unavailable_again

## Finding

- Current runId=`1780339686` had `dashboard/public/telemetry/replays_1780339686_generation_1.json`.
- The chunk was announced in `latest.json`, but the file was about 258MB.
- Root cause: `compact_replay_v1` stored 40 participant replay entries and duplicated the full 501-frame game state for each participant view.

## Fix

- Trainer replay chunks now write `compact_replay_v2`: one frame list per actual game plus participant metadata.
- Dashboard loader supports both `compact_replay_v1` and `compact_replay_v2`.
- Existing generation 1 chunk for runId=`1780339686` was converted in place from v1 to v2.

## Verification

- Converted chunk: `format=compact_replay_v2`.
- Actual games: 10.
- Participant views after decode: 40.
- First game frames: 501.
- Converted chunk size: 63,682,240 bytes.
- HTTP check through `http://127.0.0.1:5173/telemetry/replays_1780339686_generation_1.json` returned `compact_replay_v2`, 10 actual games, 4 participants in the first game, and 501 frames.
- Dashboard chunk loading effect was fixed so polling `latest.json` does not cancel a large replay chunk load before state update.
- `npm.cmd run build` passed.
- `cargo fmt --all`, `cargo test --workspace`, and `cargo build -p orbit-wars-trainer --release` passed in `orbit-wars-cuda-dev`.

## Runtime Note

- The current trainer process was not restarted, so the already-running process still uses the old binary until a restart.
