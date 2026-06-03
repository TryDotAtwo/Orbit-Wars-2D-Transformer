# docs/performance/SUMMARY.md

timestamp=2026-06-03T16:45:00+03:00
section=performance
active_task_id=2026-06-03_nvidia_perf_bug_review_no_code_changes

## Current Facts

- Dominant inference cost is one full relation-cell attention pass over 4096 cells per request, plus learned `64 x 64` matrix layers.
- Resident population weights and persistent CUDA workspace are implemented for the trainer path.
- CUDA inference precomputes Q/K/V and uses fused attention apply.
- CUDA training no longer uses `atomicAdd`; shared gradients use explicit reductions.
- Live and completed replay storage were changed to avoid giant repeated JSON polling.
- Dashboard replay chunks are loaded lazily; non-Replay tabs should fetch lightweight telemetry only.
- Current 64-model/20-participation generations are measured but slow: generation 58 took about 1249 s, generation 59 about 1288 s, generation 60 about 1398 s.
- Dashboard production build currently fails if runtime telemetry remains under `dashboard/public/telemetry`; the folder was about 10.85 GB during the 2026-06-03 review.

## Active Bottlenecks

- Full attention remains expensive and untiled.
- Host transfers still use ordinary buffers; pinned buffers and stream overlap are not implemented.
- CUDA streams and double-buffering are not implemented.
- Backprop is correct on smokes but still needs tiled/shared-memory/tensor-core optimization for stable full-run throughput.
- GPU utilization telemetry is not a real measured occupancy metric unless separately collected.
- Broad 65,536-game batch target is not proven.
- Host-side request packing, decoding, per-turn worker spawning, and telemetry/replay file handling are likely material overheads around CUDA bursts.

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
