# Lightweight Replay History

timestamp=2026-06-03T18:10:50+03:00
task_id=2026-06-03_lightweight_replay_history

## Change

- Long-lived trainer `replay_history` now stores `ReplayChunkRecord` metadata instead of full `ReplayGame` values.
- Completed replay chunks are still written from full `evaluation.replay_games`.
- `latest.json` keeps the same `replayChunks` fields: `generation`, `path`, `gameCount`, and `actualGameCount`.
- Pruning still removes replay artifacts by retained generation, but no longer needs frame payloads in memory.

## Why

- Once a completed replay chunk is written to disk, dashboard telemetry only needs chunk metadata and counts.
- Keeping full `ReplayGame` history held old frame buffers longer than needed.
- This complements immutable frame sharing by releasing completed replay payloads after artifact/log writes.

## Verification

Red:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer replay_chunk_records_keep_counts_without_frame_payloads -- --nocapture"
```

Result: failed before implementation because `ReplayChunkRecord` and `replay_chunk_records_from_replay_games` did not exist.

Green:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer replay_chunk_records_keep_counts_without_frame_payloads -- --nocapture"
```

Result: passed.

Full:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

Result: passed.

- `orbit-wars-core`: 35 passed.
- `orbit-wars-ffi`: 0 tests.
- `orbit-wars-trainer`: 27 passed.
- doc tests: passed.

## Notes

- Dashboard build was not rerun because the dashboard code and telemetry schema did not change.
- Dashboard dev server remained running at `http://127.0.0.1:5174/`.
