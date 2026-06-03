# docs/contracts/SUMMARY.md

timestamp=2026-06-03T16:16:09+03:00
section=contracts
active_task_id=2026-06-03_docs_cleanup

## Purpose

- This file stores active contracts only.
- Values that can change must be mirrored in `project_config.yaml`.
- Historical contract changes are in `test_results/`.

## Native Input

- Shape: `64 x 7`.
- Columns: `owner_class`, `ship_log_percent`, `x_position_normalized`, `y_position_normalized`, `production_normalized`, `velocity_x_normalized`, `velocity_y_normalized`.
- Row layout: home/current-owned anchor first, initial planet slots next, comet slots next, padding last.
- Owner classes: own, enemy1, enemy2, enemy3, neutral, empty, absent_on_map.
- Enemy class mapping: sorted opponent ids excluding current player.
- Empty rows and absent rows have zero numeric state except their owner-class sentinel.
- Invalid owner/player mapping must return visible errors, not fallbacks.

## Native Output

- Shape: `64 x 24`.
- Each source row emits 12 pairs: `target_fraction`, `send_fraction`.
- The model head normalizes each source row so send fractions sum to at most 1.
- Decoder ignores non-owned, empty, absent, and sub-one-ship sends.
- Decoder may refine launch angle for the selected target after model output.
- Decoder must not replace a model-selected target with a different target.
- Sun-blocked routes are not hidden today; the candidate no-command guard requires explicit approval because it changes behavior.

## True2D Model

- Active model: `True2DTransformer`.
- Internal table: `64 x 64`.
- Default architecture: learned expand, learned `64 x 64` matrix layers, one middle full relation-cell attention pass, learned output collapse.
- Attention scores are dynamic differentiable values, not independent `4096 x 4096` parameter tables.
- Trainable weights include expand weights, row/pair embeddings, matrix-layer weights, Q/K/V/O attention weights, shared attention weights, and output weights.
- Forbidden pre-model pair features: distance, angle, flight time, sun intersection, manual pair score, from-to heuristic.
- Raw object coordinates and velocities are allowed input state.

## Simulator

- Native turn order: comet expiration, comet spawn, launches, production, fleet movement, orbit/comet movement, swept collision, combat, step increment.
- Invalid action, comet mismatch, or malformed state must surface as explicit errors.
- Event traces expose launched fleets and Sun-destroyed fleet ids for action-local penalties.
- Official parity is validated by deterministic reference scenarios; broad random-seed parity remains open.

## Training

- Training inference and backprop require CUDA while `training_backprop_enabled` is true.
- CPU inference remains the final submission path.
- `games_per_model` means participations per model, not actual game count.
- Current full target: population 64, players 4, participations 20, actual games 320.
- Current selection target: top 6.
- Reward scale: win 2, draw -1, loss -2.
- Trajectory sample: metadata, final reward, Sun target penalties, packed `64x7` input, raw `64x24` output.
- Backprop uses all selected elite samples; `training_backprop_batch_samples` is a memory chunk size, not a sample limiter.
- CUDA backprop stays fp32; CUDA inference attention may use bf16-rounded operands with fp32 accumulation.
- Weighted reproduction copies selected parents and allocates remaining slots by smoothed winrate plus normalized reward within configured slot caps.

## ABI

- FFI entrypoint: `agent_act(handle, player, angular_velocity, current_step, planets, initial_planets, comet_ids, output)`.
- Python shim must locate native artifacts even when Kaggle execution lacks `__file__`.
- Python shim returns `[]` without native inference only when the current player has no owned source planet.
- CUDA forward ABI supports single model, host-weight many-model, resident many-model, and full-attention training.
- CUDA true2d input layout: `request_count x 64 x 7` contiguous `f32`.
- CUDA true2d output layout: `request_count x 64 x 24` contiguous `f32`.
- CUDA target action reward layout: `sample_count x 64 x 12` contiguous `f32`.

## Dashboard Telemetry

- Producer: trainer.
- Consumer: dashboard.
- Dashboard never mutates trainer state.
- `latest.json` must remain lightweight and must not embed completed full replay frames.
- Active live replay uses live replay path metadata.
- Completed replay chunks are durable generation artifacts and loaded lazily.
- Metrics arrays must describe completed generations only; active generation progress belongs in top-level live fields.
- Generation winrate rows must be deduplicated by `(validationGeneration, evaluatedGeneration)`.
- Compact generation logs store aggregate counters and timings, not frames, per-turn state, or trajectory samples.

## Submission Gate

- No Kaggle submit, timing submit, or host-side tournament submit may run without explicit user approval for that action.
