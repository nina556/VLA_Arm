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

"""ONNX-friendly wrappers around SmolVLA prefix / denoise steps."""

from __future__ import annotations

import torch
from kv_utils import flatten_past_key_values, sorted_kv_keys, unflatten_past_key_values
from torch import Tensor, nn

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


class PrefixExportWrapper(nn.Module):
    """images + language + state -> prefix_pad_masks + flat KV tensors."""

    def __init__(self, flow_model: nn.Module):
        super().__init__()
        self.model = flow_model

    def forward(
        self,
        image_0: Tensor,
        image_1: Tensor,
        image_2: Tensor,
        img_mask_0: Tensor,
        img_mask_1: Tensor,
        img_mask_2: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, ...]:
        images = [image_0, image_1, image_2]
        img_masks = [img_mask_0, img_mask_1, img_mask_2]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        flat = flatten_past_key_values(past_key_values)
        keys = sorted_kv_keys(flat)
        return (prefix_pad_masks, *[flat[k] for k in keys])


class DenoiseExportWrapper(nn.Module):
    """One flow-matching denoise step with flat KV inputs."""

    def __init__(self, flow_model: nn.Module, kv_keys: list[str]):
        super().__init__()
        self.model = flow_model
        self.kv_keys = list(kv_keys)

    def forward(
        self,
        x_t: Tensor,
        timestep: Tensor,
        prefix_pad_masks: Tensor,
        *kv_tensors: Tensor,
    ) -> Tensor:
        flat = dict(zip(self.kv_keys, kv_tensors, strict=True))
        past = unflatten_past_key_values(flat)
        return self.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past,
            x_t=x_t,
            timestep=timestep,
        )
