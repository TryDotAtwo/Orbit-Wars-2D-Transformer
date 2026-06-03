# 2026-06-02 sun target penalty, tournament dashboard lag, boundary stop

status=implemented_verified_training_stopped
runId=1780348585

## Scope

- Stopped the active Docker trainer after it crossed from generation 37 into generation 38.
- Fixed Generation Winrate dashboard lag caused by repeated `generationWinRates` appends.
- Added action-local Sun target penalty backprop without changing global win/draw/loss rewards.
- Verified that generation 32 tournament reproduction used tournament top-12 champions to seed the generation 33 population.

## Implementation

- Trainer now deduplicates `generationWinRates` by `(validationGeneration,evaluatedGeneration)` before writing `latest.json`.
- Dashboard loader also deduplicates legacy bloated `generationWinRates` arrays before rendering.
- Decoder exposes `DecodedMoveCommand` through `decode_model_outputs_with_trace`, preserving source row and target slot for training attribution.
- Simulator events expose launched and Sun-destroyed fleet ids.
- Trainer records `fleet_id -> ActionTrace` and adds `-training_sun_target_penalty_scale` only to the sample/source-row/target-slot that launched a fleet later destroyed by the Sun.
- CUDA training receives a separate `target_action_rewards` buffer. Target logits use `final_reward + local_target_penalty`; send logits keep only the final game reward.

## Verification

- Docker `cargo fmt --all`: passed.
- Docker `cargo test --workspace`: passed, core 31 tests and trainer 16 tests.
- Docker `nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native/cuda/orbit_wars_cuda.cu -o target/liborbit_wars_cuda.so`: passed.
- Docker CUDA true2d smoke with rebuilt library: passed, resident max abs diff `0.00000024`.
- Docker CUDA trainer smoke `--smoke --cuda --generations 1`: passed through evaluation/backprop/reproduction with the new penalty buffer.
- Docker release trainer build: passed.
- Dashboard `npm.cmd exec tsc -- --noEmit`: passed.
- Dashboard `npm.cmd run build`: passed after sandbox `esbuild spawn EPERM` rerun with approval; generated `dashboard/dist` was removed because Vite copied telemetry into a multi-GB output directory.
- Browser QA on `http://localhost:5173/`: passed. `Generation Winrate` clicked in about 808 ms, console error/warn logs were empty, view showed tournament stage G32, 32 tracked generations, 128 champion models, and 320 tournament games.

## Runtime Evidence

- Trainer watcher sent SIGTERM after telemetry reached `activeGeneration=38`; `pgrep -af orbit-wars-trainer` then returned no trainer process.
- Checkpoint `.bin` updated at `2026-06-02 21:37:53`; no `.tmp` checkpoint file remained.
- Current `latest.json` restored to full runId `1780348585`, activeGeneration `38`, stopped status, and compact `generationWinRates=32`.
- Pre-fix `latest.json` had `generationWinRates=80768` and size about `10515472` bytes; post-fix latest has `generationWinRates=32` and size about `67769` bytes.
- Log lines confirm G32 tournament and top-12 reproduction:
  - `generation=32; phase=generation_tournament; tournament_games=320`
  - `generation=32; phase=reproduction; source=generation_tournament_top12`
  - `generation=33; phase=evaluation; population=128`
  - `generation=33; phase=backprop; trained_models=12`

## Notes

- The restored `generationWinRates` for G32 were reconstructed from the previously visible duplicated telemetry rows after a smoke run overwrote `latest.json`.
- Training remains stopped at generation 38 pending explicit restart.
