"""Tensor views for OWV8 resident CUDA batches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tools.cuda_tensor_view import cuda_tensor_from_ptr


@dataclass(frozen=True)
class ResidentBatchTensorView:
    tokens: torch.Tensor
    token_type_ids: torch.Tensor
    owner_ids: torch.Tensor
    padding_mask: torch.Tensor
    planet_mask: torch.Tensor
    labels_fire: torch.Tensor
    labels_source: torch.Tensor
    labels_target: torch.Tensor
    labels_amount: torch.Tensor
    labels_confidence: torch.Tensor
    old_logprob: torch.Tensor | None = None

    @property
    def request_count(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def action_slots(self) -> int:
        return int(self.labels_fire.shape[1])


def resident_batch_view_to_tensors(view: Any) -> ResidentBatchTensorView:
    """Wrap a native OrbitWarsV8CudaDeviceBatchView without copying to CPU.

    ``view`` may be a ctypes structure or a Rust/PyO3-like object with matching
    pointer and size attributes.
    """
    requests = int(view.request_count)
    tokens = int(view.token_count)
    features = int(view.token_features)
    planets = int(view.planet_count)
    slots = int(view.action_slots)
    if requests <= 0:
        raise ValueError("resident device batch is empty")
    if tokens <= 0 or features <= 0 or planets <= 0 or slots <= 0:
        raise ValueError(
            f"bad resident device batch shape requests={requests} tokens={tokens} "
            f"features={features} planets={planets} slots={slots}"
        )
    return ResidentBatchTensorView(
        tokens=cuda_tensor_from_ptr(int(view.tokens), (requests, tokens, features), "float32"),
        token_type_ids=cuda_tensor_from_ptr(int(view.token_type_ids), (requests, tokens), "int64"),
        owner_ids=cuda_tensor_from_ptr(int(view.owner_ids), (requests, tokens), "int64"),
        padding_mask=cuda_tensor_from_ptr(int(view.padding_mask), (requests, tokens), "uint8").bool(),
        planet_mask=cuda_tensor_from_ptr(int(view.planet_mask), (requests, planets), "uint8").bool(),
        labels_fire=cuda_tensor_from_ptr(int(view.labels_fire), (requests, slots), "int32"),
        labels_source=cuda_tensor_from_ptr(int(view.labels_source), (requests, slots), "int32"),
        labels_target=cuda_tensor_from_ptr(int(view.labels_target), (requests, slots), "int32"),
        labels_amount=cuda_tensor_from_ptr(int(view.labels_amount), (requests, slots), "int32"),
        labels_confidence=cuda_tensor_from_ptr(int(view.labels_confidence), (requests, slots), "float32"),
    )
