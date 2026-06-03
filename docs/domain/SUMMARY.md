# docs/domain/SUMMARY.md

timestamp=2026-06-03T16:16:09+03:00
section=domain
active_task_id=2026-06-03_docs_cleanup

## Purpose

- Store current Orbit Wars domain facts needed for implementation.
- Detailed official research remains in `docs/domain/orbit_wars_competition_research.md`.

## Official Sources

- Competition slug: `orbit-wars`.
- Official URL: `https://www.kaggle.com/competitions/orbit-wars`.
- Downloaded official files: `artifacts/orbit_wars_official_2026_05_31/`.
- Local official source for reference compare: `C:/tmp/kaggle-env-src`.
- Local test dependency: `kaggle-environments>=1.28.0`.

## Game Facts

- Board is a continuous square with a central Sun.
- Supported player counts are 2 and 4; project default training target is 4.
- Planets have id, owner, position, radius, ships, and production.
- Fleets have id, owner, position, angle, source planet, and ships.
- Actions are arrays of `[from_planet_id, direction_angle, num_ships]`.
- Fleets move in straight lines; collisions are swept over movement segments.
- Planets may rotate around the center; comets are temporary moving planets.
- Game ends at step limit or elimination; winner is highest owned planet plus owned fleet ship count.

## Implementation Constraints

- Submission root needs `main.py` with `agent(obs)`.
- Runtime network ingress/egress is forbidden by competition rules.
- Native model input is currently `64 x 7`.
- Native model output is currently `64 x 24`.
- The simulator should stay aligned with official turn ordering and visible rule failures.

## Open Domain Work

- Broad random-seed official trace comparison remains undone.
- Current self-play map is official-like; exact byte-for-byte official RNG parity is not proven.
