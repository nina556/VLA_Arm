# SmolVLA → ONNX（TensorRT）双引擎导出设计

**日期:** 2026-07-24
**状态:** 待用户审阅
**Checkpoint:** `data/outputs/20000/pretrained_model`
**策略:** 方案 B — 逻辑端到端，2 个 ONNX engine

## 1. 背景与目标

用户已在本仓库训练 SmolVLA，权重位于 `data/outputs/20000/pretrained_model`（`model.safetensors` + `config.json` + pre/post processors）。目标是导出可供 **NVIDIA TensorRT** 使用的 ONNX，并实现对外「观测 → 动作」的端到端推理。

SmolVLA 推理（`VLAFlowMatching.sample_actions`）天然分两段：

1. **Prefix**：图像 / 语言 / state → VLM forward，填充 KV cache
2. **Denoise**：固定 `num_steps=10` 的 flow matching 循环，每步调用 `denoise_step`

整模单文件导出（方案 A）因 KV cache、transformers 注意力与 Python 循环，进 TensorRT 风险过高。本设计采用 **2 个 ONNX + 薄调度层**，对外仍表现为端到端。

### 1.1 目标

- 从指定 checkpoint 导出 `smolvla_prefix.onnx` 与 `smolvla_denoise.onnx`
- 调度脚本：预处理 → prefix → 10× denoise → 后处理 → action chunk `[1, 50, 16]`
- 用 ONNX Runtime 做数值对齐（相对 PyTorch）
- 文档说明如何用 `trtexec` 进一步转 TensorRT engine（首版不保证 TRT 一次编过）

### 1.2 非目标（首版不做）

- 单文件整模 ONNX
- 将 tokenizer、图像 resize-pad、MEAN_STD 归一化 / 反归一化塞进 ONNX
- RTC、动态 batch、动态相机数量 / 分辨率
- 在本仓库内完成完整 TensorRT engine 构建与性能调优

## 2. Checkpoint 与模型约定

来自 `config.json`（20000 step）：

| 项                    | 值                                                            |
| --------------------- | ------------------------------------------------------------- |
| Policy                | `smolvla`                                                     |
| 相机                  | `camera1/2/3`，原始 `3×480×640`，推理 resize-pad 到 `512×512` |
| State / Action 原始维 | 16                                                            |
| Pad 维                | `max_state_dim=32`, `max_action_dim=32`                       |
| Action chunk          | `chunk_size=50`, `n_action_steps=50`                          |
| Denoise 步数          | `num_steps=10`                                                |
| Language              | `tokenizer_max_length=48`                                     |
| Attention             | `cross_attn`, `use_cache=true`                                |

导出与校验一律 **batch_size=1**、**float32**（FP16 作为后续优化项，不纳入首版验收）。

## 3. 架构

```
相机帧 + joint state + task 文本
        │
        ▼
┌───────────────────────────────┐
│  Python（现有 LeRobot）        │
│  preprocessor + tokenizer     │
│  resize-pad / normalize       │
└───────────────┬───────────────┘
                │  张量输入（见 §4）
                ▼
┌───────────────────────────────┐
│  smolvla_prefix.onnx          │
│  （可再经 trtexec → .engine） │
└───────────────┬───────────────┘
                │  past_key_values*, prefix_pad_masks
                ▼
┌───────────────────────────────┐
│  调度循环 num_steps=10         │
│  x ← noise                    │
│  for t in 1.0 → 0.0:          │
│    v = denoise.onnx(x, t, kv) │
│    x ← x + dt * v             │
└───────────────┬───────────────┘
                │  x[..., :16]
                ▼
┌───────────────────────────────┐
│  Python postprocessor         │
│  unnormalize → robot action   │
└───────────────────────────────┘
```

`*` KV 以具名张量列表进出 ONNX，避免嵌套 Python tuple。

## 4. ONNX 接口规范

### 4.1 `smolvla_prefix.onnx`

**输入（固定形状）**

| 名称          | Shape              | dtype   | 说明                                |
| ------------- | ------------------ | ------- | ----------------------------------- |
| `image_0`     | `[1, 3, 512, 512]` | float32 | camera1，已 resize-pad              |
| `image_1`     | `[1, 3, 512, 512]` | float32 | camera2                             |
| `image_2`     | `[1, 3, 512, 512]` | float32 | camera3                             |
| `img_mask_0`  | `[1]`              | bool    | 对应相机是否有效                    |
| `img_mask_1`  | `[1]`              | bool    |                                     |
| `img_mask_2`  | `[1]`              | bool    |                                     |
| `lang_tokens` | `[1, 48]`          | int64   | tokenizer 输出                      |
| `lang_masks`  | `[1, 48]`          | bool    | attention mask（与 PyTorch 侧一致） |
| `state`       | `[1, 32]`          | float32 | pad 后的 state                      |

**输出**

| 名称                | Shape                                           | 说明                                                                    |
| ------------------- | ----------------------------------------------- | ----------------------------------------------------------------------- |
| `prefix_pad_masks`  | `[1, prefix_len]`                               | 供 denoise 拼 attention                                                 |
| `kv_{layer}_{k\|v}` | 由 dry-run 写入 `export_meta.json` 的固定 shape | 展平后的 past_key_values；层数与每层 shape 以 meta 为准，导出后不得再变 |

`prefix_len` 与 `kv_*` 具体维数在首次试导出时从一次 PyTorch dry-run 记录到 `export_meta.json`，并写死进第二份 ONNX 的输入声明（两份必须一致）。

实现方式：包装 `nn.Module`，内部调用 `embed_prefix` + `vlm_with_expert.forward(..., fill_kv_cache=True)`，将 `past_key_values` 拆成有序命名输出。

### 4.2 `smolvla_denoise.onnx`

**输入**

| 名称                | Shape              | dtype   |
| ------------------- | ------------------ | ------- |
| `x_t`               | `[1, 50, 32]`      | float32 |
| `timestep`          | `[1]`              | float32 |
| `prefix_pad_masks`  | `[1, prefix_len]`  | bool    |
| `kv_{layer}_{k\|v}` | 与 prefix 输出一致 | float32 |

**输出**

| 名称  | Shape         |
| ----- | ------------- |
| `v_t` | `[1, 50, 32]` |

实现方式：包装 `denoise_step`（`embed_suffix` + expert forward with cache）。

### 4.3 调度约定

```text
dt = -1.0 / 10
x = noise ~ N(0,1)   # shape [1,50,32]；可由外部传入以便复现
for step in 0..9:
    time = 1.0 + step * dt
    v = denoise(x, time, prefix_pad_masks, kv...)
    x = x + dt * v
actions = x[:, :, :16]   # 原始 action_dim
```

## 5. 交付物

| 路径                                           | 内容                                                              |
| ---------------------------------------------- | ----------------------------------------------------------------- |
| `examples/smolvla_onnx/export_smolvla_onnx.py` | 加载 checkpoint、包装模块、导出两份 ONNX、写 meta                 |
| `examples/smolvla_onnx/run_smolvla_onnx.py`    | 调度推理（ORT）；可选与 PyTorch 对比                              |
| `examples/smolvla_onnx/README.md`              | 用法、`trtexec` 示例命令                                          |
| `data/outputs/20000/onnx/`                     | `smolvla_prefix.onnx`, `smolvla_denoise.onnx`, `export_meta.json` |

依赖：现有 `torch` + 已安装的 `onnx`；校验阶段使用 `onnxruntime`（CPU 即可；GPU ORT 非必须）。不修改 SmolVLA 训练代码路径，仅增加导出/推理脚本。

## 6. 导出技术要点

1. **加载**：`SmolVLAPolicy.from_pretrained("data/outputs/20000/pretrained_model")`；处理 config 中 `vlm_model_name` 若指向他机路径，优先依赖 safetensors 已含权重，必要时用环境变量/CLI 覆盖本地 VLM 路径。
2. **包装模块**：独立 `PrefixExportWrapper` / `DenoiseExportWrapper`，`forward` 只接受/返回张量，无 dict、无 Python 控制流侧效应。
3. **导出**：优先 `torch.onnx.export(..., dynamo=False)`，opset=17；禁止动态轴（首版）。若失败再试 dynamo=True，并在 README 记录实际成功路径。
4. **KV 展平**：按层索引稳定命名；`export_meta.json` 记录层数、每张量 shape、dtype、name 列表顺序。
5. **预处理边界**：脚本可复用 `make_pre_post_processors` + 与训练一致的 tokenizer，保证 ORT 校验输入与训练分布一致。

## 7. 测试与验收

1. **导出成功**：两份 ONNX 文件生成，`onnx.checker.check_model` 通过。
2. **数值对齐**：同一组 dummy（或真实一帧）输入，PyTorch `sample_actions` vs ORT 调度；相对误差阈值建议 `max|Δ| < 1e-3`（FP32）或按实测放宽并记入 README。
3. **形状正确**：最终 action `[1, 50, 16]`。
4. **文档**：README 含导出命令、运行命令、示例 `trtexec` 行。

失败时优先排查：KV 命名/顺序不一致、mask dtype、timestep 广播、未裁剪的 32 维 vs 16 维。

## 8. 风险与缓解

| 风险                            | 缓解                                                                         |
| ------------------------------- | ---------------------------------------------------------------------------- |
| transformers 算子 ONNX 导出失败 | 缩小包装范围；必要时对 attention 路径做 export-friendly 分支（仅导出脚本侧） |
| KV cache 结构因版本变化         | meta 文件锁定 shape；导出与推理脚本共用同一 unpack 逻辑                      |
| TensorRT 不支持部分 ONNX 算子   | 首版验收止于 ORT；README 标明已知限制与下一步                                |
| VLM 路径失效                    | CLI `--vlm-path` 覆盖；文档说明                                              |

## 9. 实现顺序

1. Dry-run PyTorch：记录 `prefix_len` 与全部 KV tensor shapes → 写入 meta 草案
2. 实现两个 ExportWrapper + 导出脚本
3. 实现 ORT 调度 + 数值对比
4. 写 README（含 trtexec 示例）
5. 在用户机器上对 `20000` checkpoint 实际导出一次

## 10. 明确决策（消除歧义）

- **拆成 2 个 ONNX**，不是 1 个；调度在 Python（或日后 CUDA graph / TRT loop）。
- **Tokenizer 与 normalize 不进 ONNX**。
- **batch=1、3 相机、512、num_steps=10、FP32** 全部固定。
- **语言输入为 token id**，不是原始字符串。
- **首版验收 = 导出 + ORT 对齐**；TensorRT 编译为文档指引的后续步骤。
