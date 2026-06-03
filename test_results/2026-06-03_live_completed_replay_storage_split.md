# 2026-06-03 Live Completed Replay Storage Split

task=live_completed_replay_storage_split
code_changes=true

## Change

- Split live replay and completed replay behavior into separate contracts.
- Trainer now writes active live replay as append-only `.owlive` records instead of preallocating `.owslot` frame slots.
- Legacy `.owslot` cleanup/compaction support remains for existing artifacts.
- Dashboard telemetry polling now fetches `latest.json` only; it does not bulk-load growing live replay files.
- Completed replay chunks remain lazy generation/game artifacts.

## Performance Intent

The previous live `.owslot` path preallocated `frame_slot_bytes * frame_slots_per_game * live_games`. On the observed generation 63 artifact this produced a 262,676,852 byte file for 4 live games. Measured nonzero frame payloads averaged about 35,986 bytes and maxed at 68,912 bytes, so the fixed 131,072 byte slot size created large disk and browser-load pressure.

Append-only `.owlive` keeps writes sequential and proportional to actual replay payload size. Dashboard polling stays lightweight because the growing live artifact is not fetched every telemetry refresh.

## Contract

- Live/in-progress game: lightweight current frames and metadata in `latest.json`; append-only `.owlive` is a growing artifact.
- Completed game: immutable replay chunk or indexed per-game artifact, loaded lazily by generation/game.
- Dashboard must not treat active live artifacts as completed replay chunks during polling.

## Verification

Targeted Rust command:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer live_replay_storage_is_append_only_owlive -- --nocapture"
```

Result:

- exit_code=0
- `live_replay_storage_is_append_only_owlive`: ok

Full Rust gate:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all && cargo test --workspace"
```

Result:

- exit_code=0
- `orbit-wars-core`: 35 passed.
- `orbit-wars-ffi`: 0 tests.
- `orbit-wars-trainer`: 25 passed.
- No Rust warnings were reported.

Dashboard build:

```powershell
cd dashboard
npm.cmd run build
```

Result:

- exit_code=0
- Vite built 32 modules in about 1.01 s.
- JS bundle: 237.07 kB, gzip 73.20 kB.

Browser smoke:

- Opened `http://127.0.0.1:5174/` in Browser.
- Title: `Orbit Wars Trainer`.
- Footer after opening Replay tab: `active_view=Replay; latest_generation=63; data_contract=append_only_artifacts`.
- Console error count: 0.

## Operational Status

- Dashboard dev server is intentionally left running for user observation at `http://127.0.0.1:5174/`.
- No training logic, replay semantics, or completed artifact functionality was removed.
