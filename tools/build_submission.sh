#!/usr/bin/env bash
set -euo pipefail

cargo build --release -p orbit-wars-ffi
cp target/release/liborbit_wars_ffi.so kaggle_submission/liborbit_wars_agent.so
tar -czf submission.tar.gz -C kaggle_submission main.py liborbit_wars_agent.so model.bin
echo "created submission.tar.gz"

