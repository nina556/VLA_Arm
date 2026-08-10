# SmolVLA → ONNX（双引擎，面向 TensorRT）

将训练好的 SmolVLA 导出为两个 ONNX：

1. `smolvla_prefix.onnx`：图像 + language tokens + state → KV cache
2. `smolvla_denoise.onnx`：noisy action + timestep + KV → velocity

Python / TensorRT 宿主循环 10 步 flow matching，得到 action chunk。

设计文档：`docs/superpowers/specs/2026-07-24-smolvla-onnx-tensorrt-design.md`

## 依赖

```bash
uv pip install "onnx>=1.16" onnxscript onnxruntime -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

（导出用 **dynamo** 路径，需要 `onnxscript`。校验用 CPU `onnxruntime` 即可。）

## 导出

```bash
uv run python examples/smolvla_onnx/export_smolvla_onnx.py \
  --checkpoint data/outputs/20000/pretrained_model \
  --output-dir data/outputs/20000/onnx \
  --device cuda \
  --vlm-path HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --opset 18
```

产出：

- `smolvla_prefix.onnx` + `.onnx.data`（大权重外置）
- `smolvla_denoise.onnx` + `.onnx.data`
- `export_meta.json`（`kv_keys`、形状、`num_steps` 等）

说明：训练 config 里的 `vlm_model_name` 可能指向他机路径；用 `--vlm-path` 覆盖为本地可读的 SmolVLM（HF hub id 或目录）。训练权重仍从 checkpoint 的 `model.safetensors` 加载。

导出脚本会把模型转成 **float32**（VLM 默认 bfloat16，ORT Conv 不支持）。

## 运行 / 与 PyTorch 对齐

```bash
uv run python examples/smolvla_onnx/run_smolvla_onnx.py \
  --onnx-dir data/outputs/20000/onnx \
  --checkpoint data/outputs/20000/pretrained_model \
  --compare-pytorch \
  --device cuda
```

期望：action 形状 `[1, 50, 16]`；`max|wrapper_pt - onnx|` 在 FP32 下尽量 `< 1e-3`（以实测为准）。

## 输入约定（ONNX 图内）

| 张量                         | Shape           | 说明                                                               |
| ---------------------------- | --------------- | ------------------------------------------------------------------ |
| `image_*`                    | `[1,3,512,512]` | 已 resize-pad，且已映射到 SigLIP `[-1,1]`（等同 `prepare_images`） |
| `lang_tokens` / `lang_masks` | `[1,48]`        | tokenizer 输出（不在图内）                                         |
| `state`                      | `[1,32]`        | pad 后的 state（MEAN_STD 在 Python preprocessor）                  |
| `x_t`                        | `[1,50,32]`     | denoise 输入噪声 / 中间状态                                        |

Tokenizer、MEAN_STD 归一化 / 反归一化仍在 Python（LeRobot pre/post processors）。

## TensorRT（后续步骤）

首版验收止于 ONNX + ORT。示例（需本机已装 TensorRT）：

```bash
trtexec --onnx=data/outputs/20000/onnx/smolvla_prefix.onnx \
  --saveEngine=data/outputs/20000/onnx/smolvla_prefix.engine \
  --fp16

trtexec --onnx=data/outputs/20000/onnx/smolvla_denoise.onnx \
  --saveEngine=data/outputs/20000/onnx/smolvla_denoise.engine \
  --fp16
```

若 `trtexec` 报不支持的算子，先用 ORT CUDA EP 验证，再针对失败节点做替换或拆分。

## 已知限制

- batch=1、3 相机、`num_steps=10` 固定，无动态轴
- 旧版 TorchScript `dynamo=False` 导出会在 SmolVLM vision mask 处失败，必须用 dynamo
- 大模型权重在旁路 `.onnx.data`，拷贝时请成对带走
- 导出时强制 float32，并把 timestep 正弦位置编码从 float64 改为 float32（ORT CPU 无 double Cos）
