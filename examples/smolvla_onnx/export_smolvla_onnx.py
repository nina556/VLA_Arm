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

"""Export SmolVLA checkpoint to prefix + denoise ONNX graphs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import onnx
import torch

# Allow running as `uv run python examples/smolvla_onnx/export_smolvla_onnx.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_wrappers import DenoiseExportWrapper, PrefixExportWrapper  # noqa: E402
from kv_utils import flatten_past_key_values, sorted_kv_keys  # noqa: E402

from lerobot.policies.smolvla import (  # noqa: E402
    SmolVLAConfig,
    SmolVLAPolicy,
    modeling_smolvla as smolvla_modeling,  # noqa: E402
)


def _patch_sinusoidal_pos_embedding_fp32() -> None:
    """ORT CPU lacks Cos/Sin for float64; force FP32 positional encoding for export."""

    def create_sinusoidal_pos_embedding(
        time: torch.Tensor,
        dimension: int,
        min_period: float,
        max_period: float,
        device="cpu",
    ) -> torch.Tensor:
        if dimension % 2 != 0:
            raise ValueError(f"dimension ({dimension}) must be divisible by 2")
        if time.ndim != 1:
            raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
        dtype = torch.float32
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * time[:, None]
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

    smolvla_modeling.create_sinusoidal_pos_embedding = create_sinusoidal_pos_embedding


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/outputs/20000/pretrained_model"),
        help="Directory with config.json + model.safetensors",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/20000/onnx"),
        help="Where to write ONNX + export_meta.json",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--vlm-path",
        type=str,
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="Override config.vlm_model_name when the training path is missing",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset (dynamo exporter may keep >=18 even if lower is requested)",
    )
    return p.parse_args()


def load_policy(checkpoint: Path, device: str, vlm_path: str) -> SmolVLAPolicy:
    config = SmolVLAConfig.from_pretrained(checkpoint)
    config.device = device
    if vlm_path:
        config.vlm_model_name = vlm_path
    # Architecture + processor come from VLM id; trained weights from safetensors.
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config)
    policy.eval()
    return policy


def build_dummy_inputs(policy: SmolVLAPolicy, device: torch.device) -> dict[str, torch.Tensor]:
    cfg = policy.config
    h, w = cfg.resize_imgs_with_padding
    # Values already in SigLIP range [-1, 1] as produced by prepare_images.
    images = {f"image_{i}": torch.zeros(1, 3, h, w, device=device, dtype=torch.float32) for i in range(3)}
    masks = {f"img_mask_{i}": torch.ones(1, device=device, dtype=torch.bool) for i in range(3)}
    return {
        **images,
        **masks,
        "lang_tokens": torch.ones(1, cfg.tokenizer_max_length, device=device, dtype=torch.long),
        "lang_masks": torch.ones(1, cfg.tokenizer_max_length, device=device, dtype=torch.bool),
        "state": torch.zeros(1, cfg.max_state_dim, device=device, dtype=torch.float32),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    _patch_sinusoidal_pos_embedding_fp32()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"Loading policy from {args.checkpoint} (vlm={args.vlm_path}) on {device} ...")
    policy = load_policy(args.checkpoint, str(device), args.vlm_path)
    # VLM loads as bfloat16 by default; ORT/TensorRT Conv paths need float32.
    policy = policy.float()
    flow = policy.model
    flow.to(device)

    dummy = build_dummy_inputs(policy, device)
    prefix_mod = PrefixExportWrapper(flow).to(device).eval()

    prefix_out = prefix_mod(
        dummy["image_0"],
        dummy["image_1"],
        dummy["image_2"],
        dummy["img_mask_0"],
        dummy["img_mask_1"],
        dummy["img_mask_2"],
        dummy["lang_tokens"],
        dummy["lang_masks"],
        dummy["state"],
    )
    prefix_pad_masks = prefix_out[0]
    # Re-derive stable kv key list from a live forward for meta
    images = [dummy["image_0"], dummy["image_1"], dummy["image_2"]]
    img_masks = [dummy["img_mask_0"], dummy["img_mask_1"], dummy["img_mask_2"]]
    prefix_embs, pad_masks, att_masks = flow.embed_prefix(
        images, img_masks, dummy["lang_tokens"], dummy["lang_masks"], state=dummy["state"]
    )
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    att_2d = make_att_2d_masks(pad_masks, att_masks)
    pos_ids = torch.cumsum(pad_masks, dim=1) - 1
    _, past = flow.vlm_with_expert.forward(
        attention_mask=att_2d,
        position_ids=pos_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    flat = flatten_past_key_values(past)
    kv_keys = sorted_kv_keys(flat)
    assert len(prefix_out) == 1 + len(kv_keys)

    prefix_input_names = [
        "image_0",
        "image_1",
        "image_2",
        "img_mask_0",
        "img_mask_1",
        "img_mask_2",
        "lang_tokens",
        "lang_masks",
        "state",
    ]
    prefix_output_names = ["prefix_pad_masks", *kv_keys]
    prefix_args = tuple(dummy[n] for n in prefix_input_names)

    prefix_path = args.output_dir / "smolvla_prefix.onnx"
    print(f"Exporting prefix -> {prefix_path} (dynamo=True)")
    # Legacy TorchScript tracer breaks on SmolVLM vision masks; use dynamo exporter.
    torch.onnx.export(
        prefix_mod,
        prefix_args,
        str(prefix_path),
        input_names=prefix_input_names,
        output_names=prefix_output_names,
        opset_version=args.opset,
        dynamo=True,
        external_data=True,
    )
    onnx.checker.check_model(onnx.load(str(prefix_path), load_external_data=True))

    denoise_mod = DenoiseExportWrapper(flow, kv_keys).to(device).eval()
    x_t = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device=device)
    timestep = torch.tensor([1.0], device=device, dtype=torch.float32)
    kv_tensors = [flat[k].contiguous() for k in kv_keys]
    denoise_args = (x_t, timestep, prefix_pad_masks.contiguous(), *kv_tensors)
    denoise_input_names = ["x_t", "timestep", "prefix_pad_masks", *kv_keys]

    denoise_path = args.output_dir / "smolvla_denoise.onnx"
    print(f"Exporting denoise -> {denoise_path} (dynamo=True)")
    torch.onnx.export(
        denoise_mod,
        denoise_args,
        str(denoise_path),
        input_names=denoise_input_names,
        output_names=["v_t"],
        opset_version=args.opset,
        dynamo=True,
        external_data=True,
    )
    onnx.checker.check_model(onnx.load(str(denoise_path), load_external_data=True))

    meta = {
        "checkpoint": str(args.checkpoint.resolve()),
        "opset": args.opset,
        "num_steps": policy.config.num_steps,
        "chunk_size": policy.config.chunk_size,
        "max_action_dim": policy.config.max_action_dim,
        "action_dim": int(policy.config.action_feature.shape[0]),
        "max_state_dim": policy.config.max_state_dim,
        "tokenizer_max_length": policy.config.tokenizer_max_length,
        "resize_imgs_with_padding": list(policy.config.resize_imgs_with_padding),
        "prefix_len": int(prefix_pad_masks.shape[-1]),
        "kv_keys": kv_keys,
        "kv_shapes": {k: list(flat[k].shape) for k in kv_keys},
        "prefix_onnx": prefix_path.name,
        "denoise_onnx": denoise_path.name,
    }
    meta_path = args.output_dir / "export_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
