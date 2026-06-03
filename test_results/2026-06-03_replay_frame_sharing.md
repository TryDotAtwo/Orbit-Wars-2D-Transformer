# Replay Frame Sharing

timestamp=2026-06-03T18:01:08+03:00
task_id=2026-06-03_replay_frame_sharing

## Change

- `ReplayGame.frames` now stores `Arc<Vec<ReplayFrame>>`.
- `replay_games_from_evaluation` creates one immutable frame buffer per actual game and shares it across all participant views.
- Completed replay JSON and live compaction still serialize the same frame payloads; only in-memory ownership changed.

## Why

- A completed 4-player replay creates one user-facing participant view per player.
- Before this change each view cloned the full frame vector, so one actual game could hold up to four copies of identical frame data before replay artifact writing.
- Live in-progress storage is already append-only `.owlive`; this change targets completed-game replay memory.

## Verification

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer replay_games_share_frames_for_same_actual_game -- --nocapture"
```

Result: targeted test passed after implementation.

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

Result: passed.

- `orbit-wars-core`: 35 passed.
- `orbit-wars-ffi`: 0 tests.
- `orbit-wars-trainer`: 26 passed.
- doc tests: passed.

## Notes

- Dashboard build was not rerun because this is Rust-only and does not change dashboard code or telemetry schema.
- Dashboard dev server remained running for live training observation at `http://127.0.0.1:5174/`.
