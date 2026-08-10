# SmolVLA ONNX (TensorRT dual-engine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the trained SmolVLA checkpoint at `data/outputs/20000/pretrained_model` into two ONNX graphs (`prefix` + `denoise`) plus a Python scheduler that yields action chunks for TensorRT tooling.

**Architecture:** Wrap `VLAFlowMatching.embed_prefix` + VLM forward (fill KV) and `denoise_step` as two `nn.Module`s with flat tensor I/O; export with `torch.onnx.export(dynamo=False, opset=17)`; run 10-step flow matching in Python (or later TensorRT) using ORT for numeric check.

**Tech Stack:** PyTorch 2.11, `onnx`, `onnxruntime` (CPU OK), LeRobot `SmolVLAPolicy`, existing preprocessor/tokenizer.

## Global Constraints

- Checkpoint: `data/outputs/20000/pretrained_model`
- Batch size 1; FP32; no dynamic axes
- Images: 3 cams, each `[1,3,512,512]` after resize-pad (handled outside ONNX or inside Prefix wrapper via calling `prepare_*` before export inputs)
- `state` `[1,32]`, `lang_tokens/masks` `[1,48]`, `x_t`/`v_t` `[1,50,32]`, action crop to 16
- `num_steps=10` fixed; tokenizer + MEAN_STD normalize stay in Python
- Deliver under `examples/smolvla_onnx/` and artifacts under `data/outputs/20000/onnx/`
- Do not modify training code in `src/lerobot/policies/smolvla/` unless export requires a tiny helper (prefer keeping helpers in examples)

## File Structure

| File                                           | Responsibility                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------ |
| `examples/smolvla_onnx/kv_utils.py`            | Flatten / unflatten `past_key_values` dict ↔ named tensors        |
| `examples/smolvla_onnx/export_wrappers.py`     | `PrefixExportWrapper`, `DenoiseExportWrapper`                      |
| `examples/smolvla_onnx/export_smolvla_onnx.py` | CLI: load policy, dry-run shapes, export ONNX + `export_meta.json` |
| `examples/smolvla_onnx/run_smolvla_onnx.py`    | CLI: ORT schedule 10 steps; optional PyTorch compare               |
| `examples/smolvla_onnx/README.md`              | Commands + `trtexec` hints                                         |

---

### Task 1: KV flatten/unflatten helpers

**Files:**

- Create: `examples/smolvla_onnx/kv_utils.py`

**Interfaces:**

- Produces:
  - `flatten_past_key_values(past: dict[int, dict[str, Tensor]]) -> dict[str, Tensor]`
  - `unflatten_past_key_values(flat: dict[str, Tensor]) -> dict[int, dict[str, Tensor]]`
  - `kv_name(layer_idx: int, kind: str) -> str` where `kind` in `{"key_states","value_states"}` → `kv_{layer}_{key|value}`

- [ ] **Step 1: Implement helpers**

```python
# examples/smolvla_onnx/kv_utils.py
from __future__ import annotations

import torch
from torch import Tensor


def kv_name(layer_idx: int, kind: str) -> str:
    short = "key" if kind == "key_states" else "value"
    return f"kv_{layer_idx}_{short}"


def flatten_past_key_values(past: dict) -> dict[str, Tensor]:
    flat: dict[str, Tensor] = {}
    for layer_idx in sorted(past.keys()):
        for kind in ("key_states", "value_states"):
            flat[kv_name(int(layer_idx), kind)] = past[layer_idx][kind]
    return flat


def unflatten_past_key_values(flat: dict[str, Tensor]) -> dict:
    past: dict = {}
    for name, tensor in flat.items():
        # kv_{idx}_key / kv_{idx}_value
        _, idx_s, short = name.split("_", 2)
        layer_idx = int(idx_s)
        kind = "key_states" if short == "key" else "value_states"
        past.setdefault(layer_idx, {})[kind] = tensor
    return past
```

- [ ] **Step 2: Smoke-test in REPL**

Run:

```bash
uv run python -c "
from examples.smolvla_onnx.kv_utils import flatten_past_key_values, unflatten_past_key_values
import torch
past = {0: {'key_states': torch.zeros(1,2,4,8), 'value_states': torch.ones(1,2,4,8)}}
flat = flatten_past_key_values(past)
back = unflatten_past_key_values(flat)
assert torch.equal(back[0]['key_states'], past[0]['key_states'])
print('ok', list(flat))
"
```

Expected: `ok ['kv_0_key', 'kv_0_value']`

- [ ] **Step 3: Commit** (skip if repo has no git / user did not ask)

---

### Task 2: Export wrappers

**Files:**

- Create: `examples/smolvla_onnx/export_wrappers.py`

**Interfaces:**

- Consumes: `VLAFlowMatching` as `policy.model`; `flatten_past_key_values` / `unflatten_past_key_values`
- Produces:
  - `PrefixExportWrapper.forward(image_0, image_1, image_2, img_mask_0, img_mask_1, img_mask_2, lang_tokens, lang_masks, state) -> tuple` leading with `prefix_pad_masks` then sorted KV tensors
  - `DenoiseExportWrapper.forward(x_t, timestep, prefix_pad_masks, *kv_tensors) -> v_t`

- [ ] **Step 1: Implement wrappers**

```python
# examples/smolvla_onnx/export_wrappers.py
from __future__ import annotations

import torch
from torch import Tensor, nn

from examples.smolvla_onnx.kv_utils import flatten_past_key_values, kv_name, unflatten_past_key_values


class PrefixExportWrapper(nn.Module):
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
    ):
        images = [image_0, image_1, image_2]
        img_masks = [img_mask_0, img_mask_1, img_mask_2]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

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
        # Stable order for ONNX
        kv_tensors = [flat[k] for k in sorted(flat.keys(), key=lambda s: (int(s.split("_")[1]), s))]
        return (prefix_pad_masks, *kv_tensors)


class DenoiseExportWrapper(nn.Module):
    def __init__(self, flow_model: nn.Module, kv_keys: list[str]):
        super().__init__()
        self.model = flow_model
        self.kv_keys = list(kv_keys)

    def forward(self, x_t: Tensor, timestep: Tensor, prefix_pad_masks: Tensor, *kv_tensors: Tensor):
        flat = {name: tensor for name, tensor in zip(self.kv_keys, kv_tensors, strict=True)}
        past = unflatten_past_key_values(flat)
        return self.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past,
            x_t=x_t,
            timestep=timestep,
        )
```

- [ ] **Step 2: Instantiation smoke (needs GPU + checkpoint; run in Task 3)**

---

### Task 3: Export CLI + meta

**Files:**

- Create: `examples/smolvla_onnx/export_smolvla_onnx.py`

**Interfaces:**

- CLI args: `--checkpoint`, `--output-dir`, `--device`, `--vlm-path` (optional override)
- Writes: `smolvla_prefix.onnx`, `smolvla_denoise.onnx`, `export_meta.json`

- [ ] **Step 1: Implement export script** that:
  1. Loads `SmolVLAPolicy.from_pretrained(checkpoint)`
  2. If `config.vlm_model_name` missing on disk and `--vlm-path` given, set it before load (or patch config.json copy)
  3. Builds dummy inputs matching config
  4. Runs `PrefixExportWrapper` once to discover KV names/shapes
  5. Exports both ONNX with `input_names`/`output_names`
  6. Writes `export_meta.json` with shapes, `num_steps`, `opset`, `action_dim`, `kv_keys`

- [ ] **Step 2: Run export**

```bash
uv run python examples/smolvla_onnx/export_smolvla_onnx.py \
  --checkpoint data/outputs/20000/pretrained_model \
  --output-dir data/outputs/20000/onnx \
  --device cuda
```

Expected: three files under `data/outputs/20000/onnx/`; `onnx.checker` OK.

---

### Task 4: ORT runner + numeric check

**Files:**

- Create: `examples/smolvla_onnx/run_smolvla_onnx.py`

- [ ] **Step 1: Implement scheduler** reading `export_meta.json`, running prefix once + 10 denoise steps; optional `--compare-pytorch` printing max abs diff.

- [ ] **Step 2: Run**

```bash
uv run python examples/smolvla_onnx/run_smolvla_onnx.py \
  --onnx-dir data/outputs/20000/onnx \
  --checkpoint data/outputs/20000/pretrained_model \
  --compare-pytorch \
  --device cuda
```

Expected: action shape `[1,50,16]`; max abs diff reported (target `< 1e-3` FP32, document if higher).

---

### Task 5: README

**Files:**

- Create: `examples/smolvla_onnx/README.md`

- [ ] **Step 1: Document** export/run commands, I/O tables, `trtexec` example lines for both ONNX files, known limitations (tokenizer outside graph, TRT op support TBD).

---

## Spec coverage checklist

- Dual ONNX prefix/denoise → Tasks 2–3
- Scheduler 10 steps + crop 16 → Task 4
- ORT numeric check → Task 4
- README + trtexec → Task 5
- Fixed shapes / Python preprocess → Tasks 3–4
- No training code change → Global constraint

## Execution

User requested start immediately → **Inline Execution** in this session (executing-plans style checkpoints between tasks).
