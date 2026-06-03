# docs/architecture/SUMMARY.md

timestamp=2026-06-03T16:16:09+03:00
section=architecture
active_task_id=2026-06-03_docs_cleanup

## Purpose

- This section describes the current shape of the system.
- Historical design iterations belong in `test_results/`, not in this summary.
- Parameter values must be checked against `project_config.yaml` before edits.

## System Shape

| area | path | role |
|---|---|---|
| Native core | `crates/orbit-wars-core/` | Game config, types, encoder, decoder, geometry, simulator, and model implementation. |
| FFI | `crates/orbit-wars-ffi/` | C ABI used by the Python Kaggle agent. |
| Trainer | `crates/orbit-wars-trainer/` | CUDA self-play, trajectory capture, backprop, reproduction, checkpoints, tournament, telemetry, replay artifacts. |
| CUDA | `native/cuda/` | Local GPU inference/backprop kernels and smokes. |
| Dashboard | `dashboard/` | Read-only training telemetry and replay UI. |
| Submission | `kaggle_submission/` | Kaggle package scaffold with `main.py`, native library, and model file. |
| Operations | `tools/` | Build, submit, compare, restore, restart, and replay maintenance scripts. |

## Runtime Flow

- Kaggle path: `kaggle_submission/main.py` receives `obs`, packs planets/comets, calls Rust FFI, and returns Orbit Wars commands.
- Native CPU path: `orbit-wars-ffi` calls `orbit-wars-core` encoder, `True2DTransformer`, and decoder.
- Training path: `orbit-wars-trainer --cuda` runs self-play, batches active player states, calls resident CUDA true2d inference, records trajectories, trains selected elites on CUDA, then reproduces the next population.
- Dashboard path: trainer writes `dashboard/public/telemetry/latest.json` and replay artifacts; dashboard polls and fetches replay payloads lazily.

## Model Flow

- Encoder emits strict `64 x 7` rows.
- `True2DTransformer` expands rows into a `64 x 64` source-target relation table.
- Learned matrix layers preserve `64 x 64`.
- One middle full self-attention pass attends over 4096 relation cells.
- Output collapse emits `64 x 24`, meaning 12 target/send pairs per source row.
- Decoder converts selected source/target/send outputs into legal command arrays and may refine launch angle after model output so selected moving targets are hit.

## Training Flow

- Full self-play target: 4 players, population 64, 20 participations per model, 320 actual games per generation.
- Selection target: top 6 evaluated models.
- Backprop: selected trajectory samples train through full true2d graph on CUDA.
- Reproduction: next-population slots are weighted by smoothed winrate plus normalized reward and bounded by configured slot caps.
- Generation archive: top 4 models per completed generation are retained for interval tournament.
- Generation tournament: scheduled by config; use `project_config.yaml` for current participations and archive caps.
- Replays: completed generations keep only configured sampled evaluation games, not full training state.

## Boundaries

- CPU inference is for final Kaggle submission.
- CUDA is required for current learning because backprop is enabled.
- Dashboard is read-only and must not alter trainer state.
- Decoder geometry is allowed only after model output; do not add manual pair features before the model.
- Kaggle submit is blocked unless the user explicitly approves that upload.

## Current Incomplete Areas

- Broad official random-seed parity is not proven.
- CUDA attention/backprop is correct on smokes but not fully optimized.
- Pinned host buffers, streams, double-buffering, and tiled/tensor-core attention remain future work.
- Decoder sun-route guard is not implemented; Sun failures remain a training signal.
