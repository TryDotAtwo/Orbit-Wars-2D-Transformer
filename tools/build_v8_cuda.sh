#!/usr/bin/env bash
set -euo pipefail

mkdir -p target
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
CUTLASS_ROOT="${CUTLASS_PATH:-/opt/cutlass}"
CUTLASS_INCLUDES=()
if [[ -d "${CUTLASS_ROOT}/include" ]]; then
  CUTLASS_INCLUDES+=("-I${CUTLASS_ROOT}/include")
fi
if [[ -d "${CUTLASS_ROOT}/tools/util/include" ]]; then
  CUTLASS_INCLUDES+=("-I${CUTLASS_ROOT}/tools/util/include")
fi
nvcc -std=c++17 -O3 "-arch=${CUDA_ARCH}" -shared -Xcompiler -fPIC \
  "${CUTLASS_INCLUDES[@]}" \
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
