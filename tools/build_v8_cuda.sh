#!/usr/bin/env bash
set -euo pipefail

mkdir -p target
nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC \
  native/cuda/orbit_wars_v8_cuda.cu \
  -o target/liborbit_wars_v8_cuda.so
g++ -std=c++17 -O3 \
  native/cuda/orbit_wars_v8_cuda_smoke.cpp \
  -Inative/cuda \
  -Ltarget \
  -lorbit_wars_v8_cuda \
  -Wl,-rpath,'$ORIGIN' \
  -o target/orbit_wars_v8_cuda_smoke
LD_LIBRARY_PATH=target ./target/orbit_wars_v8_cuda_smoke
