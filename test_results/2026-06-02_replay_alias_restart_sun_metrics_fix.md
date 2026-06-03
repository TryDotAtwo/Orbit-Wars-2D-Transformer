# 2026-06-02 Replay Alias Restart Sun Metrics Fix

status=implemented_verified_running

## Problem

- Replay selector showed only `sample replay` and `frames=1` for completed G21 even though `.owlive` existed.
- Old `binary_replay_alias_v1` JSON chunks pointed to `.owslot`, but completed generations had already been compacted to `.owlive`.
- `loadReplayChunk` treated an empty decode from stale `.owslot` as success and never tried the sibling `.owlive`.
- Restart history recovery restored G17-G22 timing from trainer log but overwrote rich gameplay metrics from immutable `generation_logs`, so `Sun losses`, launches, captures, and fleet hits appeared as zero.
- Progress telemetry could duplicate the latest completed generation in `metrics[]` by keeping the old `latest.json` item and appending the checkpoint metric history item.

## Fix

- Repointed existing `replays_1780348585_generation_*.json` aliases to completed `.owlive`; recreated missing G1 alias.
- Dashboard replay loader now keeps trying binary candidates when a decoded `.owslot`/`.owlive` returns zero games.
- Trainer now compacts live slot before writing replay alias and prefers completed `.owlive` alias over `.owslot`.
- Trainer progress telemetry excludes all generations already present in metric history before preserving old `latest.json` metrics.
- Dashboard telemetry loader deduplicates `metrics`/`replayChunks` and filters charts to completed generations only.
- Restore script now reads `generation_logs/{run_id}/generation_*.json` and uses those rich metrics over sparse trainer-log metrics.

## Verification

- Restarted trainer from `artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin`; new trainer pid is `61943`, runId remains `1780348585`.
- Current telemetry: activeGeneration=24, metrics=23 for G1..G23, duplicate metric generations absent, replayChunks=23.
- G21 restored metric includes `sunDestroyedFleets=1185260`, `launchActions=9883670`, `captures=65358`.
- Browser direct loader check for G21 returned 40 participant views; first game has 501 frames, step 0..500; load time about 10.3 seconds for a 218 MB `.owlive`.
- Browser UI check for G21 showed 10 game options, `frame_status=complete`, `frames=501`.
- `cargo test -p orbit-wars-trainer` passed: 14 tests.
- `cargo build -p orbit-wars-trainer --release` passed.
- `cargo fmt --all` passed.
- `npm.cmd exec tsc -- --noEmit` passed.
- `npm.cmd run build` passed.
- `python -m py_compile tools/restore_latest_history_from_trainer_log.py tools/compact_owslot_replay.py` passed.

## Remaining Note

- Completed replay chunks are still monolithic `.owlive` files around 130-286 MB. Replay works, but selecting a generation can take about 10 seconds. The next storage improvement should be indexed/per-game replay segments so the UI can show games immediately and fetch only the selected game.
