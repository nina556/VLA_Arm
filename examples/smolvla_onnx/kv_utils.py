# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flatten / unflatten SmolVLA past_key_values for ONNX I/O."""

from __future__ import annotations

from torch import Tensor


def kv_name(layer_idx: int, kind: str) -> str:
    short = "key" if kind == "key_states" else "value"
    return f"kv_{layer_idx}_{short}"


def sorted_kv_keys(flat: dict[str, Tensor]) -> list[str]:
    return sorted(flat.keys(), key=lambda s: (int(s.split("_")[1]), 0 if s.endswith("_key") else 1))


def flatten_past_key_values(past: dict) -> dict[str, Tensor]:
    flat: dict[str, Tensor] = {}
    for layer_idx in sorted(past.keys()):
        for kind in ("key_states", "value_states"):
            flat[kv_name(int(layer_idx), kind)] = past[layer_idx][kind]
    return flat


def unflatten_past_key_values(flat: dict[str, Tensor]) -> dict:
    past: dict = {}
    for name, tensor in flat.items():
        parts = name.split("_")
        # kv_{idx}_key / kv_{idx}_value
        layer_idx = int(parts[1])
        kind = "key_states" if parts[2] == "key" else "value_states"
        past.setdefault(layer_idx, {})[kind] = tensor
    return past
