timestamp=2026-05-31T01:00:00+03:00
task_id=task_orbit_wars_native_transformer_agent_2026_05_31
agent_id=codex
source=user_chat
related_files=[]
intended_action=implement_orbit_wars_native_transformer_agent_plan
result_status=completed
prompt_raw=PLEASE IMPLEMENT THIS PLAN:
# План: Orbit Wars Native Transformer Agent

## Summary
- target=Kaggle `orbit-wars`; submission_runtime=Python `main.py` shim + предсобранный Linux native CPU inference; CUDA используется для локального обучения.
- base_image=`cmz-native-dev:2026-05-26`; verified_toolchain=Rust 1.95, Cargo 1.95, g++ 11.4, CUDA 12.4, Python 3.10.
- model_input=строго `MAX_ROWS x 2`: `owner_sign`, `ship_log_percent`; geometry stays outside transformer and used only by action decoder.
- order=сначала native scaffold + 1..3 Kaggle timing submits; после каждого submit остановка до пользовательского анализа результата.
- dashboard_concept=approved; path=`C:\Users\Иван Литвак\.codex\generated_images\019e7ab9-38c8-7fa3-b296-f972031340a2\ig_0bf52a956f5cec84016a1b5c2845448191b9f74222e172d74e.png`.

## Key Changes
- Add Rust workspace: simulator core, observation parser, row ordering, ship scaler, transformer CPU inference, action decoder, C ABI for Python.
- Add C++/CUDA module: batched transformer inference/training kernels for local self-play after timing phase.
- Add Kaggle package: `main.py`, `liborbit_wars_agent.so`, `model.bin`, minimal config; `main.py` loads library once via `ctypes` and calls native `agent_act(...)`.
- Add config parameters: `max_rows=64`, `initial_planet_slots=40`, `comet_slots=20`, `padding_slots=4`, `ship_log_pivot=1000`, `ship_log_cap=100000`, `act_timeout_seconds=1`.
- Add dashboard: React+Vite+TypeScript, Canvas2D replay board, uPlot metrics, model pool table, submission gate panel, non-blocking file/poll API over training run artifacts.

## Interfaces
- input_row_order: row0=initial_home_planet; rows1..39=initial planets sorted by Euclidean distance from initial home; comet rows=spawn/order/distance slots; missing row=`owner_sign=-1`.
- owner_sign: own=`1.0`; neutral/enemy=`0.0`; missing=`-1.0`.
- ship_log_percent: `ships<=pivot => 0.9*ln(1+ships)/ln(1+pivot)`; `ships>pivot => 0.9+0.1*clamp(ln(ships/pivot)/ln(cap/pivot),0,1)`.
- transformer output per source row: `target_fraction`, `send_fraction`; decoder maps `target_fraction` to active row index; `target==source` means hold.
- decoder: owned source only; `send_ships=floor(source_ships*send_fraction)`; invalid/empty target means no action by explicit contract; moving target angle solved by intercept iteration using fleet speed formula.
- timing submits: submit_1=native parser+decoder overhead; submit_2=tiny transformer `layers=2,d_model=32,heads=4`; submit_3=largest local-bench model under conservative CPU budget after user approval.

## Training Plan
- Build Rust simulator faithful to official turn order: comet expiration/spawn, launches, production, fleet movement, orbit/comet movement, combat.
- Validate simulator against Kaggle environment traces before self-play training.
- Training starts only after Kaggle timing size decision.
- Evolution v1: 128 models; self-play reward win=`+1`, draw=`0`, loss=`-1`; select top 12; mutate/cross to regenerate 128.
- GPU pipeline: CPU workers=12 generate/apply game state; pinned buffers batch states; CUDA inference double-buffered; target batch=65,536 games only after VRAM/throughput proof; infeasible batch triggers explicit stop and report.

## Dashboard Plan
- Route=operational monitoring dashboard + replay viewer.
- Views: Live Training, Replay, Models, Submissions, Artifacts.
- Metrics: win rate by generation, rating distribution, mutation lineage, games/sec, latency histogram, GPU/CPU occupancy, validation errors.
- Replay: 100x100 board, sun, planets, fleets, comet paths, timeline scrubber, selected row/action inspector.
- Non-interference: trainer writes append-only metrics and sampled replay chunks; dashboard reads asynchronously; telemetry failures appear in status/error panel.

## Test Plan
- Unit tests: row ordering, owner encoding, ship scaler, output decoding, intercept solver, sun collision, combat resolution.
- Golden tests: seeded official scenarios versus Kaggle reference for fixed action scripts.
- Bench tests: per-turn native CPU inference p50/p95/max; Python shim overhead; Docker build reproducibility.
- Kaggle validation: maximum 3 timing submissions; download logs/replays after each; no next submit without user approval.
- Dashboard QA: browser desktop/mobile, replay canvas nonblank, charts readable, filters/tabs working, approved concept fidelity check.

## Assumptions
- Kaggle evaluation has reliable Python entrypoint but no assumed GPU; submission inference path is CPU-only.
- Strict two-column model input remains binding until user explicitly changes architecture.
- Neutral planets and enemy planets both encode as `owner_sign=0`.
- CUDA training can use local RTX 3070 Laptop 8GB; final 65,536-game batch is a target, not a guaranteed constant before VRAM proof.
- Plan Mode prevented prompt-history mutation; implementation phase first step must persist current prompt and update project memory/index/test records.
