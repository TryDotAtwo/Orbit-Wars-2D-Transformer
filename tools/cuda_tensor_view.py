"""Wrap externally-owned CUDA pointers as PyTorch tensors.

The native Orbit Wars CUDA arena owns these buffers.  This module only creates
non-owning tensor views, so the caller must keep the arena/model workspace alive
and must not resize the underlying native buffers while tensors are in use.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import torch
from torch.utils.cpp_extension import load_inline


_CPP_SOURCE = r"""
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <torch/extension.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

c10::ScalarType scalar_type_from_name(const std::string& dtype) {
  if (dtype == "float32") return c10::ScalarType::Float;
  if (dtype == "int64") return c10::ScalarType::Long;
  if (dtype == "int32") return c10::ScalarType::Int;
  if (dtype == "uint8") return c10::ScalarType::Byte;
  if (dtype == "bool") return c10::ScalarType::Bool;
  throw std::runtime_error("unsupported dtype: " + dtype);
}

torch::Tensor wrap_cuda_ptr(
    std::uintptr_t ptr,
    std::vector<int64_t> sizes,
    const std::string& dtype) {
  if (ptr == 0) {
    throw std::runtime_error("cannot wrap null CUDA pointer");
  }
  auto options = torch::TensorOptions()
      .device(torch::kCUDA)
      .dtype(scalar_type_from_name(dtype));
  return torch::from_blob(
      reinterpret_cast<void*>(ptr),
      sizes,
      [](void*) {},
      options);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wrap_cuda_ptr", &wrap_cuda_ptr, "Create a non-owning CUDA tensor view");
}
"""


@lru_cache(maxsize=1)
def _extension():
    return load_inline(
        name="orbit_wars_cuda_tensor_view",
        cpp_sources=[_CPP_SOURCE],
        functions=[],
        with_cuda=False,
        extra_cflags=["-O3"],
        verbose=False,
    )


def cuda_tensor_from_ptr(ptr: int, shape: Sequence[int], dtype: str) -> torch.Tensor:
    """Return a non-owning CUDA tensor view over an external device pointer."""
    return _extension().wrap_cuda_ptr(int(ptr), [int(value) for value in shape], dtype)
