# Orbit Wars v8 Pipeline

## Dataset

Build sharded imitation data from replay exports:

```powershell
python tools/leaderboard_v8_dataset.py Реплеи_Лидера\episode-78628518-replay.json --output data\v8_leader --shard-size 4096
```

Replay alignment is `steps[i - 1].observation -> steps[i].action`. The action row has no target id, so the dataset uses simulator hit traces when available and the nearest angle-compatible target fallback in Python.

Shard files include:

```text
tokens, token_type_ids, owner_ids, sample_offsets, planet_mask,
labels_fire, labels_source, labels_target, labels_amount,
episode_id, step, player
```

## Training And Export

Run supervised imitation:

```powershell
python tools/v8_train.py --dataset data\v8_leader --run-dir runs\v8-imitation --epochs 4 --batch-size 64
```

The trainer writes:

```text
runs/v8-imitation/checkpoint.pt
runs/v8-imitation/model.bin
runs/v8-imitation/telemetry/latest.json
```

Copy `model.bin` into `kaggle_submission/model.bin` for the Python FFI wrapper.

## Smoke

Use a tiny CPU smoke before a larger run:

```powershell
python tools/v8_smoke.py Реплеи_Лидера\episode-78628518-replay.json Реплеи_Лидера\episode-78627934-replay.json --output-dir test_results\v8_smoke --device cpu
```

## Self-play Relaxation

Create a generation manifest for the Rust arena:

```powershell
python tools/v8_selfplay.py --output-dir runs\v8-selfplay --generation 1 --elite runs\v8-imitation\model.bin
```

Python only manages generation manifests and telemetry. Per-tick simulation and tournaments should stay in Rust.

## Docker

Validate Compose from this Cyrillic workspace with an explicit project name:

```powershell
docker compose -p orbit-wars config
```

Build or run:

```powershell
docker compose -p orbit-wars build trainer
docker compose -p orbit-wars run --rm trainer python3 -u tools/v8_train.py --help
```

The image tag is `orbit-wars-gpu-trainer:cuda125`; GPU access is requested through Compose device reservations.

## GPU Sweeps

Record sweep rows:

```powershell
python tools/v8_gpu_sweep.py --config-id smoke-b8 --batch 8 --workers 0 --max-fleets 640 --status pass
```

Rows are appended to `test_results/v8_gpu_sweeps.csv` with the required GPU/memory/throughput columns.
