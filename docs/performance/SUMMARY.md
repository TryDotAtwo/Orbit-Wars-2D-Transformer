# docs/performance/SUMMARY.md

timestamp=2026-06-03T18:10:50+03:00
section=performance
active_task_id=2026-06-03_lightweight_replay_history

## Current Facts

- Dominant inference cost is one full relation-cell attention pass over 4096 cells per request, plus learned `64 x 64` matrix layers.
- Resident population weights and persistent CUDA workspace are implemented for the trainer path.
- CUDA inference precomputes Q/K/V and uses fused attention apply.
- CUDA training no longer uses `atomicAdd`; shared gradients use explicit reductions.
- Live and completed replay storage were changed to avoid giant repeated JSON polling.
- Dashboard replay chunks are loaded lazily; non-Replay tabs should fetch lightweight telemetry only.
- Live replay storage now writes append-only `.owlive` records instead of preallocating `.owslot` frame slots; dashboard polling uses `latest.json` frames and does not bulk-load the growing live file.
- Completed replay participant views now share one immutable `Arc<Vec<ReplayFrame>>` per actual game instead of cloning identical frame vectors per player view.
- Long-lived trainer `replay_history` stores only `ReplayChunkRecord` metadata after chunk writes, so completed frame payloads are not retained for telemetry/pruning.
- Current 64-model/20-participation generations are measured but slow: generation 58 took about 1249 s, generation 59 about 1288 s, generation 60 about 1398 s.
- Dashboard production build no longer copies runtime telemetry into `dist`; `npm.cmd run build` produced a 245,917 byte payload after `build.copyPublicDir=false`.
- Trainer hot-loop cleanup removed one per-turn `requests_per_game` allocation and per-turn/per-model sample-index vector allocation; behavior is covered by `test_results/2026-06-03_hot_loop_allocation_cleanup.md`.
- CUDA resident population forward packing now reuses host `input_rows` and `model_indices` scratch buffers across turns/chunks; behavior is covered by `test_results/2026-06-03_population_forward_scratch.md`.
- Live/completed replay storage split is covered by `test_results/2026-06-03_live_completed_replay_storage_split.md`.
- Completed replay frame sharing is covered by `test_results/2026-06-03_replay_frame_sharing.md`.
- Lightweight replay history is covered by `test_results/2026-06-03_lightweight_replay_history.md`.

## Active Bottlenecks

- Full attention remains expensive and untiled.
- Host transfers still use ordinary buffers; pinned buffers and stream overlap are not implemented.
- CUDA streams and double-buffering are not implemented.
- Backprop is correct on smokes but still needs tiled/shared-memory/tensor-core optimization for stable full-run throughput.
- GPU utilization telemetry is not a real measured occupancy metric unless separately collected.
- Broad 65,536-game batch target is not proven.
- Host-side request packing, decoding, per-turn worker spawning, and completed replay artifact handling remain likely material overheads around CUDA bursts.
- Runtime telemetry still lives under `dashboard/public/telemetry` for dev serving; it remains about 10+ GB and should be managed separately from source/build outputs.

## Measurements To Run Next

- Current full_500 phase timing with host packing, H2D, kernels, D2H, decode, simulation, and telemetry split out.
- CUDA profiler pass on forward and backward attention.
- Host packing and transfer timing after resident weights.
- Memory use of trajectory buffers at full 64-model/20-participation schedule.
- Dashboard replay-load timing for indexed per-game payloads.

## Evidence Locations

- Detailed performance history: `test_results/`.
- CUDA module contract: `native/cuda/README.md`.
- Current config limits: `project_config.yaml`.
