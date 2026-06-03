# docs/models/SUMMARY.md

timestamp=2026-06-03T16:16:09+03:00
section=models
active_task_id=2026-06-03_docs_cleanup

## Active Model

- Model: `True2DTransformer`.
- Input: `64 x 7`.
- Internal relation table: `64 x 64`.
- Default learned matrix layers: configured in `project_config.yaml`.
- Middle attention: one full Q/K/V self-attention pass over 4096 relation cells.
- Output: `64 x 24`, or 12 target/send pairs for each source row.
- Current parameter count: `true2d_model.actual_parameter_count` in `project_config.yaml`.
- Stored/final weights: fp32.
- CUDA inference attention: bf16-rounded operands with fp32 accumulation.
- CUDA backprop: fp32.

## Training Loop

- Evaluation captures exact model inputs, raw outputs, metadata, and final participant rewards.
- Sun-destroyed launched fleets add a target-slot-local penalty to the responsible source-row/target-slot.
- Elite backprop trains selected models on CUDA after evaluation and before reproduction.
- Reproduction uses weighted parent slot allocation, crossover, and mutation.
- Generation tournament uses archived champions and may replace normal generation elites as reproduction parents on scheduled generations.

## Current Full Target

- Players per game: 4.
- Population: 64.
- Participations per model: 20.
- Actual self-play games per generation: 320.
- Selected elite count: 6.
- Replay archive: configured actual evaluation games per generation.
- Generation archive: top 4 models per generation, capped by config.

## Verified Records

- `test_results/2026-06-03_population64_games20_top6_weighted_repro.md`.
- `test_results/2026-06-03_increase_sun_target_penalty.md`.
- `test_results/2026-06-03_velocity_input_phase_aim_restart.md`.
- Older architecture and CUDA records remain searchable in `test_results/`.

## Open Model Work

- Benchmark current 64-model schedule after checkpoint-boundary restart.
- Optimize CUDA attention and transfers.
- Keep CUDA backprop fp32 until a separate precision contract is approved and reference-matched.
- Decide separately whether to add a decoder no-command guard for Sun-blocked selected routes.
