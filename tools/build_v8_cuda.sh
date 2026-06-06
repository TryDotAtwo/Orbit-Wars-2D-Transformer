#!/usr/bin/env bash
set -euo pipefail

mkdir -p target
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python || command -v python3)"
fi
if [[ -z "${CUDA_ARCH:-}" ]]; then
  CUDA_ARCH="$("${PYTHON_BIN}" - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"sm_{major}{minor}")
    else:
        print("sm_86")
except Exception:
    print("sm_86")
PY
)"
fi
NVCC_BIN="${NVCC:-}"
if [[ -z "${NVCC_BIN}" ]]; then
  NVCC_BIN="$(command -v nvcc || true)"
fi
if [[ -z "${NVCC_BIN}" ]]; then
  NVCC_BIN="$("${PYTHON_BIN}" - <<'PY'
import pathlib
import site

for root in site.getsitepackages() + [site.getusersitepackages()]:
    nvidia_root = pathlib.Path(root) / "nvidia"
    if not nvidia_root.exists():
        continue
    candidates = [
        nvidia_root / "cuda_nvcc" / "bin" / "nvcc",
        *nvidia_root.rglob("nvcc"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            try:
                candidate.chmod(candidate.stat().st_mode | 0o111)
            except Exception:
                pass
            print(candidate)
            break
    else:
        continue
    break
PY
)"
fi
if [[ -z "${NVCC_BIN}" || ! -f "${NVCC_BIN}" ]]; then
  echo "nvcc not found; install cuda-toolkit[nvcc] or set NVCC=/path/to/nvcc" >&2
  exit 127
fi
CUTLASS_ROOT="${CUTLASS_PATH:-/opt/cutlass}"
if [[ ! -d "${CUTLASS_ROOT}/include/cutlass" ]]; then
  CUTLASS_ROOT="target/cutlass"
  if [[ ! -d "${CUTLASS_ROOT}/include/cutlass" ]]; then
    git clone --depth 1 https://github.com/NVIDIA/cutlass.git "${CUTLASS_ROOT}"
  fi
fi
CUTLASS_INCLUDES=()
if [[ -d "${CUTLASS_ROOT}/include" ]]; then
  CUTLASS_INCLUDES+=("-I${CUTLASS_ROOT}/include")
fi
if [[ -d "${CUTLASS_ROOT}/tools/util/include" ]]; then
  CUTLASS_INCLUDES+=("-I${CUTLASS_ROOT}/tools/util/include")
fi
CUDA_PIP_FLAGS="$("${PYTHON_BIN}" - <<'PY'
import pathlib
import site

include_flags = []
lib_flags = []
seen_includes = set()
seen_libs = set()
for root in site.getsitepackages() + [site.getusersitepackages()]:
    nvidia_root = pathlib.Path(root) / "nvidia"
    if not nvidia_root.exists():
        continue
    for include in nvidia_root.rglob("include"):
        if (
            (include / "cuda_runtime.h").exists()
            or (include / "nv" / "target").exists()
        ) and include not in seen_includes:
            include_flags.append(f"-I{include}")
            seen_includes.add(include)
    for lib_dir in list(nvidia_root.rglob("lib")) + list(nvidia_root.rglob("lib64")):
        if any(lib_dir.glob("libcudart.so*")) and lib_dir not in seen_libs:
            lib_flags.append(f"-L{lib_dir}")
            seen_libs.add(lib_dir)
print(" ".join(include_flags + lib_flags))
PY
)"
read -r -a CUDA_PIP_ARGS <<< "${CUDA_PIP_FLAGS}"
CUDA_PIP_LIB_PATHS="$("${PYTHON_BIN}" - <<'PY'
import pathlib
import site

paths = []
seen = set()
for root in site.getsitepackages() + [site.getusersitepackages()]:
    nvidia_root = pathlib.Path(root) / "nvidia"
    if not nvidia_root.exists():
        continue
    for lib_dir in list(nvidia_root.rglob("lib")) + list(nvidia_root.rglob("lib64")):
        if any(lib_dir.glob("libcudart.so*")) and lib_dir not in seen:
            paths.append(str(lib_dir))
            seen.add(lib_dir)
print(":".join(paths))
PY
)"
"${NVCC_BIN}" -std=c++17 -O3 "-arch=${CUDA_ARCH}" -shared -Xcompiler -fPIC \
  "${CUTLASS_INCLUDES[@]}" \
  "${CUDA_PIP_ARGS[@]}" \
  native/cuda/orbit_wars_v8_cuda.cu \
  -o target/liborbit_wars_v8_cuda.so
g++ -std=c++17 -O3 \
  native/cuda/orbit_wars_v8_cuda_smoke.cpp \
  -Inative/cuda \
  -Ltarget \
  -lorbit_wars_v8_cuda \
  -Wl,-rpath,'$ORIGIN' \
  -o target/orbit_wars_v8_cuda_smoke
LD_LIBRARY_PATH="target:${CUDA_PIP_LIB_PATHS}:${LD_LIBRARY_PATH:-}" ./target/orbit_wars_v8_cuda_smoke
