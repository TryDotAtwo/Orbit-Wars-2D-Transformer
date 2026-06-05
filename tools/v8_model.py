"""PyTorch v8 Action-Slot Transformer and model.bin export helpers."""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tools.leaderboard_v8_dataset import ACTION_SLOTS, MAX_PLANETS, TOKEN_FEATURES, AMOUNT_CLASSES

MODEL_SCHEMA = b"OWV8"
MODEL_VERSION = 1


@dataclass(frozen=True)
class V8ModelConfig:
    token_features: int = TOKEN_FEATURES
    d_model: int = 128
    heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 2
    action_slots: int = ACTION_SLOTS
    amount_classes: int = len(AMOUNT_CLASSES)
    max_planets: int = MAX_PLANETS


class V8ActionSlotTransformer(nn.Module):
    def __init__(self, config: V8ModelConfig):
        super().__init__()
        self.config = config
        self.token_projection = nn.Linear(config.token_features, config.d_model)
        self.type_embedding = nn.Embedding(3, config.d_model)
        self.owner_embedding = nn.Embedding(8, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_queries = nn.Parameter(torch.randn(config.action_slots, config.d_model) * 0.02)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.fire_head = nn.Linear(config.d_model, 1)
        self.source_head = nn.Linear(config.d_model, config.max_planets)
        self.target_head = nn.Linear(config.d_model, config.max_planets)
        self.amount_head = nn.Linear(config.d_model, config.amount_classes)

    def forward(
        self,
        tokens: torch.Tensor,
        token_type_ids: torch.Tensor,
        owner_ids: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        planet_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        owner_ids = owner_ids.clamp(min=0, max=self.owner_embedding.num_embeddings - 1)
        hidden = self.token_projection(tokens)
        hidden = hidden + self.type_embedding(token_type_ids.clamp(min=0, max=2))
        hidden = hidden + self.owner_embedding(owner_ids)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        queries = self.slot_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        slots = self.decoder(queries, encoded, memory_key_padding_mask=padding_mask)
        source_logits = self.source_head(slots)
        target_logits = self.target_head(slots)
        if planet_mask is not None:
            invalid = ~planet_mask.bool()
            source_logits = source_logits.masked_fill(invalid[:, None, :], -1.0e9)
            target_logits = target_logits.masked_fill(invalid[:, None, :], -1.0e9)
        return {
            "fire_logits": self.fire_head(slots).squeeze(-1),
            "source_logits": source_logits,
            "target_logits": target_logits,
            "amount_logits": self.amount_head(slots),
        }


def build_tiny_model(config: V8ModelConfig | None = None) -> V8ActionSlotTransformer:
    config = config or V8ModelConfig(d_model=16, heads=4, encoder_layers=1, decoder_layers=1)
    torch.manual_seed(8)
    model = V8ActionSlotTransformer(config)
    model.eval()
    return model


def export_model_bin(model: nn.Module, config: V8ModelConfig, path: Path) -> None:
    tensors = []
    payload = bytearray()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().to(torch.float32)
        offset = len(payload)
        data = array.numpy().astype("<f4", copy=False).tobytes()
        payload.extend(data)
        tensors.append({"name": name, "shape": list(array.shape), "offset": offset, "bytes": len(data)})

    header = {
        "schema": MODEL_SCHEMA.decode("ascii"),
        "version": MODEL_VERSION,
        "config": asdict(config),
        "tensors": tensors,
        "payload_bytes": len(payload),
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    checksum = zlib.crc32(header_bytes)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    with path.open("wb") as handle:
        handle.write(MODEL_SCHEMA)
        handle.write(struct.pack("<III", MODEL_VERSION, len(header_bytes), checksum))
        handle.write(header_bytes)
        handle.write(payload)


def read_model_bin(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 16 or data[:4] != MODEL_SCHEMA:
        raise ValueError("not an OWV8 model")
    version, header_len, checksum = struct.unpack("<III", data[4:16])
    header_start = 16
    header_end = header_start + header_len
    header = json.loads(data[header_start:header_end].decode("utf-8"))
    payload = data[header_end:]
    actual = zlib.crc32(data[header_start:header_end])
    actual = zlib.crc32(payload, actual) & 0xFFFFFFFF
    if actual != checksum:
        raise ValueError("model checksum mismatch")
    tensors = {tensor["name"]: tensor for tensor in header["tensors"]}
    return {"schema": header["schema"], "version": version, "checksum": checksum, "header": header, "tensors": tensors}


def load_model_bin_state(path: Path) -> tuple[V8ModelConfig, dict[str, torch.Tensor]]:
    """Load an exported OWV8 model.bin back into a PyTorch state_dict."""
    data = path.read_bytes()
    if len(data) < 16 or data[:4] != MODEL_SCHEMA:
        raise ValueError("not an OWV8 model")
    version, header_len, checksum = struct.unpack("<III", data[4:16])
    if version != MODEL_VERSION:
        raise ValueError(f"unsupported OWV8 version: {version}")
    header_start = 16
    header_end = header_start + header_len
    header_bytes = data[header_start:header_end]
    header = json.loads(header_bytes.decode("utf-8"))
    payload = data[header_end:]
    actual = zlib.crc32(header_bytes)
    actual = zlib.crc32(payload, actual) & 0xFFFFFFFF
    if actual != checksum:
        raise ValueError("model checksum mismatch")
    config = V8ModelConfig(**header["config"])
    state: dict[str, torch.Tensor] = {}
    for item in header["tensors"]:
        start = int(item["offset"])
        end = start + int(item["bytes"])
        if start < 0 or end > len(payload):
            raise ValueError(f"tensor payload out of bounds: {item['name']}")
        shape = tuple(int(value) for value in item["shape"])
        tensor = torch.frombuffer(bytearray(payload[start:end]), dtype=torch.float32).clone().reshape(shape)
        state[str(item["name"])] = tensor
    return config, state
