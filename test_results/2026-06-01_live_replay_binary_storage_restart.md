# Live Replay Binary Storage And Restart

timestamp=2026-06-01T20:35:42+03:00
task_id=2026-06-01_live_replay_binary_storage_restart
status=running_verified

## User Requirement

- Full live match must be preserved; dropping old frames is not acceptable.
- `latest.json` is not suitable as realtime storage for the full match.

## Implementation

- Replaced sliding-window live replay JSON with separate append-only binary storage:
  - manifest path remains `dashboard/public/telemetry/latest.json`;
  - live replay path is exposed as `liveReplayPath`;
  - live replay format is `orbit_live_replay_v1`;
  - full frames are appended to `dashboard/public/telemetry/live_{run_id}_generation_{generation}.owlive`.
- `latest.json` now keeps only telemetry, replay manifest fields, and one current preview frame; inline `replayGames` stays empty during live evaluation.
- Dashboard loader reads `liveReplayPath`, parses the binary replay, and exposes full live `ReplayGame[]` to the existing Replay view.
- Binary records:
  - magic=`OWLIVE1\n`;
  - header record stores generation, captured game ids, participant model ids, and opponent labels;
  - frame records store full planets, fleets, and comet groups;
  - result records store final per-participant rewards when a captured live game finishes.
- Trainer live telemetry writes are retried and remain non-fatal; failures emit visible `trainer_log; event=warning`.

## Verification

- `docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && cargo fmt --all"` passed.
- `docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && cargo test --workspace"` passed with 29 core tests and 9 trainer tests.
- `docker exec orbit-wars-cuda-dev bash -lc "cd /workspace && cargo build -p orbit-wars-trainer --release"` passed.
- `npm.cmd run build` passed after escalated rerun; initial sandbox run failed with `esbuild spawn EPERM`.
- Old runtime telemetry/log/pid files were removed, and `orbit-wars-cuda-dev` was recreated to clear Docker logs.
- Fresh training run started:
  - runId=1780334649;
  - command=`target/release/orbit-wars-trainer --cuda --players 4 --generations 64`;
  - log=`artifacts/full_training_2026-06-01_self_train_current_contract.log`;
  - pid_file=`artifacts/full_training_2026-06-01_self_train_current_contract.pid`;
  - container_logs=true.
- Start checks passed:
  - `latest.json` at live turn 15 was 46629 bytes;
  - live binary was 15497 bytes;
  - inline `replayGames` count was 0;
  - preview `frames` count was 1.
- Long-live check passed past the previous JSON failure point:
  - as of live turn 400, `latest.json` was 51645 bytes;
  - `liveReplayFrameCount=401`;
  - live binary path=`/telemetry/live_1780334649_generation_1.owlive`.
- HTTP artifact check passed through Vite dev server:
  - `/telemetry/latest.json` returned current manifest;
  - `/telemetry/live_1780334649_generation_1.owlive` returned HTTP 200.
- Docker log routing passed: `docker logs orbit-wars-cuda-dev` shows trainer `event=start` and `phase_start` lines.

## Notes

- Browser plugin setup failed before execution with `failed to write kernel assets`; UI verification was limited to TypeScript/Vite build plus real HTTP telemetry and replay artifact reads.
- Generation speed is still below the 10-minute target on this run. The live JSON bloat is fixed, but evaluation remains slow under large fleet growth/CPU simulation pressure and needs separate optimization.
