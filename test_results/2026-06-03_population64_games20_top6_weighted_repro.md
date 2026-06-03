# 2026-06-03 Population 64 / Games 20 / Top-6 Weighted Reproduction

status=implemented_verified_restart_pending

## Changes

- Full self-play defaults changed to `population=64`, `gamesPerModel=20`, `elite=6`; four-player actual evaluation games stay `64*20/4=320`.
- Added weighted reproduction parent slots from top-6 elites using smoothed winrate plus normalized reward.
- Added reproduction guardrails: min 4 slots per elite, max 18 slots per elite in the full profile.
- Generation-tournament champion scores now include win/loss/draw gameplay stats, so tournament top-6 uses the same weighted reproduction path.
- Checkpoint resume can shrink a larger stored population to the configured population and logs `checkpoint_population_resized`.

## Verification

- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo fmt --all` passed.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo test --workspace` passed: core 34/34, trainer 19/19.
- `docker exec --workdir /workspace orbit-wars-cuda-dev cargo build -p orbit-wars-trainer --release` passed.

## Runtime

- Checkpoint-boundary restart is pending after documentation registration.
