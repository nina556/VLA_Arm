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

"""Run SmolVLA dual-ONNX schedule (prefix + 10x denoise) via ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_smolvla_onnx import (  # noqa: E402
    _patch_sinusoidal_pos_embedding_fp32,
    build_dummy_inputs,
    load_policy,
)
from export_wrappers import DenoiseExportWrapper, PrefixExportWrapper  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, default=Path("data/outputs/20000/onnx"))
    p.add_argument("--checkpoint", type=Path, default=Path("data/outputs/20000/pretrained_model"))
    p.add_argument("--device", type=str, default="cuda", help="Device for PyTorch compare path")
    p.add_argument(
        "--vlm-path",
        type=str,
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    )
    p.add_argument("--compare-pytorch", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def run_onnx_schedule(
    onnx_dir: Path,
    meta: dict,
    feeds_prefix: dict[str, np.ndarray],
    noise: np.ndarray,
) -> np.ndarray:
    prefix_sess = ort.InferenceSession(
        str(onnx_dir / meta["prefix_onnx"]), providers=["CPUExecutionProvider"]
    )
    denoise_sess = ort.InferenceSession(
        str(onnx_dir / meta["denoise_onnx"]), providers=["CPUExecutionProvider"]
    )

    prefix_outs = prefix_sess.run(None, feeds_prefix)
    out_names = [o.name for o in prefix_sess.get_outputs()]
    prefix_map = dict(zip(out_names, prefix_outs, strict=True))
    prefix_pad_masks = prefix_map["prefix_pad_masks"]
    kv = {k: prefix_map[k] for k in meta["kv_keys"]}

    num_steps = int(meta["num_steps"])
    dt = -1.0 / num_steps
    x = noise.astype(np.float32)
    for step in range(num_steps):
        time = 1.0 + step * dt
        denoise_feeds = {
            "x_t": x,
            "timestep": np.array([time], dtype=np.float32),
            "prefix_pad_masks": prefix_pad_masks,
            **kv,
        }
        (v_t,) = denoise_sess.run(None, denoise_feeds)
        x = x + dt * v_t

    action_dim = int(meta["action_dim"])
    return x[:, :, :action_dim]


@torch.no_grad()
def run_pytorch_schedule(policy, dummy: dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
    flow = policy.model
    images = [dummy["image_0"], dummy["image_1"], dummy["image_2"]]
    img_masks = [dummy["img_mask_0"], dummy["img_mask_1"], dummy["img_mask_2"]]
    actions = flow.sample_actions(
        images,
        img_masks,
        dummy["lang_tokens"],
        dummy["lang_masks"],
        dummy["state"],
        noise=noise,
    )
    action_dim = int(policy.config.action_feature.shape[0])
    return actions[:, :, :action_dim]


def main() -> None:
    args = parse_args()
    _patch_sinusoidal_pos_embedding_fp32()
    meta = json.loads((args.onnx_dir / "export_meta.json").read_text())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    policy = load_policy(args.checkpoint, str(device), args.vlm_path)
    policy = policy.float()
    dummy = build_dummy_inputs(policy, device)

    noise_t = torch.randn(
        1,
        meta["chunk_size"],
        meta["max_action_dim"],
        device=device,
        dtype=torch.float32,
    )
    feeds_prefix = {
        "image_0": _to_numpy(dummy["image_0"]),
        "image_1": _to_numpy(dummy["image_1"]),
        "image_2": _to_numpy(dummy["image_2"]),
        "img_mask_0": _to_numpy(dummy["img_mask_0"]),
        "img_mask_1": _to_numpy(dummy["img_mask_1"]),
        "img_mask_2": _to_numpy(dummy["img_mask_2"]),
        "lang_tokens": _to_numpy(dummy["lang_tokens"]),
        "lang_masks": _to_numpy(dummy["lang_masks"]),
        "state": _to_numpy(dummy["state"]),
    }

    onnx_actions = run_onnx_schedule(args.onnx_dir, meta, feeds_prefix, _to_numpy(noise_t))
    print("ONNX action shape:", onnx_actions.shape)

    if args.compare_pytorch:
        # Same wrappers path for parity with exported graphs
        prefix_mod = PrefixExportWrapper(policy.model).to(device).eval()
        kv_keys = meta["kv_keys"]
        denoise_mod = DenoiseExportWrapper(policy.model, kv_keys).to(device).eval()
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
        kv_tensors = list(prefix_out[1:])
        x = noise_t.clone()
        num_steps = int(meta["num_steps"])
        dt = -1.0 / num_steps
        for step in range(num_steps):
            time = 1.0 + step * dt
            timestep = torch.tensor([time], device=device, dtype=torch.float32)
            v_t = denoise_mod(x, timestep, prefix_pad_masks, *kv_tensors)
            x = x + dt * v_t
        pt_actions = x[:, :, : int(meta["action_dim"])]
        # Also compare against sample_actions for reference
        ref = run_pytorch_schedule(policy, dummy, noise_t)
        diff_wrap = (pt_actions.detach().cpu().numpy() - onnx_actions).max()
        diff_ref = (ref.detach().cpu().numpy() - onnx_actions).max()
        print(f"max|wrapper_pt - onnx| = {diff_wrap}")
        print(f"max|sample_actions - onnx| = {diff_ref}")


if __name__ == "__main__":
    main()
