# Official-Like Map And Decoder Hit Correction

timestamp=2026-06-01T20:58:17+03:00
task_id=2026-06-01_official_like_map_and_decoder_hit_correction
status=passed_and_restarted

## User Report

- Replay showed fleets visually missing selected planets.
- User clarified that the model only selects actions; the decoder owns the launch angle, so target misses are a decoder/geometry issue.
- User also reported too few planets, inconsistent with the official four-player competition map.

## Root Cause

- Trainer self-play used a hand-written 8-planet mini-map, not the official 5-10 symmetric groups of 4 planets.
- Decoder used a continuous center-to-center intercept estimate from the source planet center.
- The simulator collision is discrete swept geometry: fleet starts at source radius plus spawn offset, then each tick checks relative swept fleet/planet collision while planets rotate.

## Changes

- Replaced trainer mini-map with deterministic official-like map generation:
  - 5-10 symmetric planet groups;
  - at least 3 static groups;
  - official clearance, production/radius, ship ranges, and home-group assignment semantics;
  - 4-player games get one home planet for each player in one symmetric group;
  - 2-player compatibility keeps Q1/Q4 homes.
- Added decoder hit correction after model output:
  - model still selects source/target/send only;
  - decoder computes base intercept;
  - decoder then predicts simulator-style swept collision for the selected target;
  - if the base angle misses, decoder searches nearby angles and chooses the closest angle that actually intersects the target radius.
- Added external config entries for official-like map generation and decoder aim search.
- Added trainer unit test `seeded_state_uses_official_like_four_player_map`.

## Verification

- `docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && cargo fmt --all && cargo test --workspace && cargo build -p orbit-wars-trainer --release"` passed.
- Test count: 29 core tests + 10 trainer tests.
- New CUDA self-train restarted cleanly:
  - runId=1780336658;
  - command=`target/release/orbit-wars-trainer --cuda --players 4 --generations 64`;
  - artifact log=`artifacts/full_training_2026-06-01_self_train_current_contract.log`;
  - container logs=true.
- Start telemetry:
  - live turn=27;
  - live frame planet count=28;
  - `liveReplayFrameCount=28`;
  - `latest.json` size=50357 bytes;
  - inline `replayGames=0`;
  - live binary bytes=55285.

## Notes

- Existing screenshots/replays from runId=1780334649 and older runs still show the old mini-map and old decoder angle behavior.
- New runId=1780336658 is the first run with official-like map count and decoder hit correction.
