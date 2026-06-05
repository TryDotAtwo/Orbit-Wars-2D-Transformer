#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -q kaggle-environments
ENV_DST="$(python3 -c 'import importlib.util, pathlib; spec=importlib.util.find_spec("kaggle_environments"); print(pathlib.Path(spec.origin).resolve().parent / "envs" / "orbit_wars")')"
mkdir -p "$ENV_DST"
cp -a .external/kaggle-env-src/kaggle_environments/envs/orbit_wars/. "$ENV_DST/"
python3 tools/kaggle_submission_smoke.py
