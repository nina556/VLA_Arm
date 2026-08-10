# Unoarm × LeRobot：双臂机器人 VLA 仿真、数据与交互平台

基于 [LeRobot](https://github.com/huggingface/lerobot) 构建的 **Unoarm 双臂机器人**端到端 VLA（Vision-Language-Action）实验平台，覆盖 MuJoCo 仿真、关键帧与 IK 数据生成、数据回放、SmolVLA/ACT 训练评估，以及由 LLM 路由的 Web 交互控制。

Unoarm 的自定义能力集中在 `custom_envs/unoarm/`，以可编辑 Python 包接入 LeRobot，无需侵入其核心实现。目前内置自由空间、抓剑和桌面抓取放置场景，并提供终端与浏览器两套交互入口。

> **English summary:** An end-to-end VLA experimentation platform for the Unoarm dual-arm robot, combining MuJoCo simulation, LeRobot, scripted and IK-based data generation, policy training, and LLM-routed web interaction.

> **环境安装**：见 [REQUIREMENTS.md](./REQUIREMENTS.md)（含完整踩坑记录：bazel、JDK、HF 镜像、相机 rename 等）。

> **VLA 技术综述**：仓库内提供配套资料 [《VLA 各系列综述》](./VLA各系列综述.pdf)，可用于了解 Vision-Language-Action 模型的发展脉络、主要系列和技术背景。

---

## 目录

- [架构概览](#架构概览)
- [VLA 综述](#vla-综述)
- [快速开始](#快速开始)
- [详细工作流](#详细工作流)
  - [1. 环境安装](#1-环境安装)
  - [2. 仿真环境介绍](#2-仿真环境介绍)
  - [3. 数据采集（手动 teleop，可选）](#3-数据采集手动-teleop可选)
  - [4. 脚本化数据生成（推荐）](#4-脚本化数据生成推荐)
  - [5. 数据回放检验](#5-数据回放检验)
  - [6. 开始 SmolVLA 训练](#6-开始-smolvla-训练)
  - [7. 训练后可视化验证](#7-训练后可视化验证)
- [Web 控制台与扩展场景](#web-控制台与扩展场景)
  - [Web 控制台](#web-控制台)
  - [Reach-IK 抓剑数据](#reach-ik-抓剑数据)
  - [桌面抓取放置](#桌面抓取放置)
- [目录结构](#目录结构)
- [常见问题](#常见问题)

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    custom_envs/unoarm/                       │
│                                                              │
│  gym_unoarm/         MuJoCo + Gymnasium 仿真环境             │
│    ├─ env.py         UnoarmEnv（16D 双臂 action/state）      │
│    ├─ ik.py          MuJoCo DLS 逆运动学                     │
│    ├─ *_scene.py     抓剑 / 桌面放置场景                     │
│    └─ *.xml + meshes 机器人、相机与场景资产                  │
│                                                              │
│  lerobot_unoarm/     LeRobot EnvConfig 注册（--env.type=unoarm）│
│                                                              │
│  data_gen/           关键帧 / Reach-IK / Table-Place 数据生成│
│  pose_design/        动作工程、关键帧编排与导出              │
│  webapp/ + static/   FastAPI 后端与浏览器 3D 控制台          │
│  data/               动作设计、设置与生成的数据集            │
│                                                              │
│  scripts/            工作流脚本                              │
│    ├─ 01_convert_urdf_to_mjcf.py   URDF→MJCF 转换            │
│    ├─ 06_generate_scripted_data.py 脚本化数据生成（主入口）  │
│    ├─ 07_replay_dataset.py          数据回放可视化            │
│    ├─ 08_interact.py                终端交互推理              │
│    ├─ 09_web_interact.py            Web 控制台                │
│    └─ 10_generate_reach_ik_data.py  Reach-IK 数据生成         │
└─────────────────────────────────────────────────────────────┘
          │
          │ uv pip install -e custom_envs/unoarm
          ▼
┌─────────────────────────────────────────────────────────────┐
│                       LeRobot 基础设施                        │
│  unoarm-train / eval  +  SmolVLA / ACT policies              │
└─────────────────────────────────────────────────────────────┘
```

**核心设计**：自定义包通过 `uv pip install -e` 装入 `.venv`，注册 `--env.type=unoarm`、`gym_unoarm/UnoarmFreeSpace-v0` 和 `gym_unoarm/UnoarmReachSword-v0`，从而复用 LeRobot 的数据、训练和推理基础设施。桌面放置场景由同一个 `UnoarmEnv` 通过 `scene=table_place` 启用。

---

## VLA 综述

本项目附带 [《VLA 各系列综述》](./VLA各系列综述.pdf)，作为理解项目技术选型的背景资料。建议在运行训练流程前阅读，以建立对 VLA 模型及其在机器人感知、语言理解与动作生成中作用的整体认识。

---

## 快速开始

假设环境已装好（见 [REQUIREMENTS.md](./REQUIREMENTS.md)），下面四步可跑通基础的“数据—训练—推理”闭环：

```bash
# 1. 生成 30 episodes 训练数据（约 50 秒）
uv run python custom_envs/unoarm/scripts/06_generate_scripted_data.py \
  --episodes 30 --segment-steps 20 --hold-steps 5 \
  --midpoint-noise-std 0.02 --hold-noise-std 0.01 --pose-jitter-std 0.05 \
  --output-root custom_envs/unoarm/data/unoarm_prepare_fight_taunt \
  --repo-id doki/unoarm_prepare_fight_taunt --overwrite

# 2. 回放检验数据对不对（开 MuJoCo viewer 看机器人动作）
uv run python custom_envs/unoarm/scripts/07_replay_dataset.py --episode 0

# 3. 训练 SmolVLA（约 8 小时 / 20k 步 / 单卡）
DEVICE=cuda uv run unoarm-train \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}' \
  --env.type=unoarm --env.task=UnoarmFreeSpace-v0 \
  --dataset.repo_id=doki/unoarm_prepare_fight_taunt \
  --dataset.root=custom_envs/unoarm/data/unoarm_prepare_fight_taunt \
  --batch_size=4 --steps=20000 \
  --eval_steps=0 --eval.use_async_envs=false --eval.batch_size=1 \
  --policy.push_to_hub=false \
  --output_dir=data/outputs/smolvla_unoarm

# 4. 训练完，交互式推理看效果（输入语言指令 → 机器人执行）
uv run python custom_envs/unoarm/scripts/08_interact.py
```

也可以启动集成式 Web 控制台，在浏览器中完成对话控制、动作设计、IK 数据生成和数据集回放：

```bash
uv run python custom_envs/unoarm/scripts/09_web_interact.py
```

下面逐节展开说明。

---

## 详细工作流

### 1. 环境安装

完整步骤见 [REQUIREMENTS.md](./REQUIREMENTS.md)。最小路径：

```bash
# 系统依赖
sudo apt install -y build-essential git git-lfs ffmpeg openjdk-17-jdk

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Python 依赖（三个 extra 必须同时带）
uv sync --locked --extra aloha --extra smolvla --extra dataset

# Unoarm 自定义包
uv pip install -e custom_envs/unoarm

# 冒烟测试
uv run python -c "import gym_unoarm, gymnasium as gym; env = gym.make('gym_unoarm/UnoarmFreeSpace-v0'); obs,_ = env.reset(); print('agent_pos:', obs['agent_pos'].shape); env.close()"
```

**国内网络**务必先配 HF 镜像，否则下载 SmolVLA 权重会卡死：

```bash
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com
```

---

### 2. 仿真环境介绍

Unoarm 是一个**双臂上肢机器人**的 MuJoCo 仿真，通过 Gymnasium 接口暴露为 `gym_unoarm/UnoarmFreeSpace-v0`。

**观测 / 动作空间**（16 维，归一化到 [-1, 1]）：

| 维度 | 含义         | 维度 | 含义         |
| ---- | ------------ | ---- | ------------ |
| 0-6  | 左臂关节 1-7 | 8-14 | 右臂关节 1-7 |
| 7    | 左夹爪       | 15   | 右夹爪       |

**相机**（3 个，480×640 RGB）：

- `top`：世界固定俯视相机（外部视角）
- `left_wrist`：挂载在左臂末端连杆 `Left_Link7` 上的手腕相机（随手运动）
- `right_wrist`：挂载在右臂末端连杆 `Right_Link7` 上的手腕相机

**关键特性**：

- `UnoarmEnv.step(action)` 是**运动学执行**：直接把 `action` 写入 MuJoCo qpos，调 `mj_forward` 更新场景，不做物理仿真。这样数据收集稳定可复现。
- 初始位姿 `RAW_ZERO`（全零归一化向量）= 机器人静止姿态。
- `reward` 和 `success` 在 env 里是**硬编码常量**（`0.0` / `False`），因为这是 free-space 行为克隆任务，没有任务成功判定。**eval 的 success rate 数字没有参考价值**，模型好坏要看推理可视化（第 7 节）。

**试用 env**：

```bash
# 开 MuJoCo viewer 拖滑块手动控制
uv run python custom_envs/unoarm/scripts/02_test_sim.py
```

---

### 3. 数据采集（手动 teleop，可选）

如果想用真实遥操数据（而非脚本生成），可以用键盘 teleop 录制：

```bash
uv run python custom_envs/unoarm/scripts/03_record_data.py
```

> 本项目**默认用脚本化生成**（第 4 节），更稳定、可复现、易扩展。teleop 仅作为备选。

---

### 4. 脚本化数据生成（推荐）

这是本项目的主力数据生成方式。核心思路：**在 JSON 文件里定义关键位姿（key pose），脚本自动在相邻 key pose 间线性插值 + 加噪声，生成大量轨迹**。

#### 4.1 JSON 格式

key pose 定义在 `custom_envs/unoarm/data/poses_*.json`，格式：

```json
{
  "task": "语言指令（会写入数据集的 task 字段，模型用它做语言条件）",
  "poses": [
    {
      "Left_Joint1": 1.0,
      "Left_Joint2": 0.41,
      "Left_Joint3": 0.51,
      "Left_Joint4": 1.0,
      "Left_Joint5": -0.08,
      "Left_Joint6": 1.0,
      "Left_Joint7": 0.01,
      "Left_Gripper_Joint": -0.55,
      "Right_Joint1": 0.48,
      "Right_Joint2": -0.44,
      "Right_Joint3": -0.39,
      "Right_Joint4": 1.0,
      "Right_Joint5": -0.17,
      "Right_Joint6": 0.91,
      "Right_Joint7": 0.03,
      "Right_Gripper_Joint": -0.75
    },
    { "...": "更多 key pose..." }
  ]
}
```

**规则**：

- `task`：字符串，是 VLA 模型的语言输入（如 `"Prepare to fight and taunt."`）
- `poses`：按顺序列出机器人要经过的关键位姿，脚本会在相邻 pose 间插值
- 每个 pose 用**关节名映射**（不是数组），16 个关节，名字必须匹配 `CONTROL_JOINTS`：
  - 缺失的关节默认 0.0
  - 写了不存在的关节名会报错
- **脚本自动在序列前后补 `RAW_ZERO`（静止位姿）**，你不用写起止静止
- 想让动作循环，就在 `poses` 里把循环的 pose 多写几遍（如 `[A, B, A, B, A]`）

#### 4.2 获取 key pose 数值

在 MuJoCo viewer 里把滑块拖到目标位置，读出各关节值：

```bash
uv run python custom_envs/unoarm/scripts/02_test_sim.py
# 在 viewer 的 Sliders 面板里拖动关节到目标位姿，记录 Control 值
```

> ⚠️ 关节值是 **MuJoCo 原始弧度值**（不是归一化值），范围见 `gym_unoarm/constants.py` 的 joint range。脚本会自动 clip 到合法范围，超出会打印警告。

#### 4.3 生成命令

```bash
uv run python custom_envs/unoarm/scripts/06_generate_scripted_data.py \
  --poses-json custom_envs/unoarm/data/poses_prepare_fight_taunt.json \
  --episodes 30 \
  --segment-steps 20 \
  --hold-steps 5 \
  --midpoint-noise-std 0.02 \
  --hold-noise-std 0.01 \
  --pose-jitter-std 0.05 \
  --seed 0 \
  --output-root custom_envs/unoarm/data/unoarm_prepare_fight_taunt \
  --repo-id doki/unoarm_prepare_fight_taunt \
  --overwrite
```

**参数说明**：

| 参数                   | 默认                                  | 说明                                                               |
| ---------------------- | ------------------------------------- | ------------------------------------------------------------------ |
| `--poses-json`         | `data/poses_prepare_fight_taunt.json` | key pose 定义文件                                                  |
| `--episodes`           | 30                                    | 生成的 episode 数                                                  |
| `--segment-steps`      | 20                                    | 两个相邻 key pose 间插值多少帧                                     |
| `--hold-steps`         | 5                                     | 每个 key pose 保持多少帧                                           |
| `--midpoint-noise-std` | 0.0                                   | 插值中点加噪声（端点为 0，保持轨迹平滑），增加数据多样性           |
| `--hold-noise-std`     | 0.0                                   | hold 段加微抖动，避免完全静止产生重复帧                            |
| `--pose-jitter-std`    | 0.0                                   | 每个 episode 对 key pose 加固定偏移，让 30 个 episode 风格各有不同 |
| `--seed`               | 0                                     | 基础种子；每个 episode 用 `seed + episode_index`                   |
| `--overwrite`          | -                                     | 输出目录已存在时先删除                                             |

**为什么需要噪声参数**：如果 `midpoint-noise-std`、`hold-noise-std`、`pose-jitter-std` 全是 0，30 个 episode 会生成**完全相同**的轨迹（因为 key pose 是硬编码的）。推荐值（`0.02 / 0.01 / 0.05`）能让轨迹保持主旋律的同时各有差异，提升模型泛化能力。

**生成产物**：标准 LeRobotDataset v3.0 格式（Parquet + 内嵌 PNG 图像），约 7650 帧 / 30 episodes。

---

### 5. 数据回放检验

生成数据后，用 MuJoCo viewer 回放，确认机器人动作符合预期：

```bash
# 回放 episode 0，1 倍速，循环
uv run python custom_envs/unoarm/scripts/07_replay_dataset.py \
  --root custom_envs/unoarm/data/unoarm_prepare_fight_taunt \
  --episode 0 --speed 1.0 --loop
```

**参数**：

- `--episode N`：回放第 N 个 episode
- `--speed 1.0`：播放倍速（`0` = 最快）
- `--loop`：循环播放直到关 viewer
- `--state-units raw|normalized|both`：终端打印的关节单位

脚本会用数据集里的 `action` 驱动 env，同时终端实时打印当前关节状态和与录制 state 的最大误差（`max_state_error`，应接近 0，证明 env 确定性可复现）。

> 这是验证数据质量的关键一步——**回放看到的动作就是你训练要教模型学的动作**。

---

### 6. 开始 SmolVLA 训练

#### 6.1 训练命令

```bash
DEVICE=cuda uv run unoarm-train \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}' \
  --env.type=unoarm \
  --env.task=UnoarmFreeSpace-v0 \
  --dataset.repo_id=doki/unoarm_prepare_fight_taunt \
  --dataset.root=custom_envs/unoarm/data/unoarm_prepare_fight_taunt \
  --batch_size=4 \
  --steps=20000 \
  --log_freq=200 \
  --save_freq=2000 \
  --eval_steps=0 \
  --eval.use_async_envs=false \
  --eval.batch_size=1 \
  --policy.push_to_hub=false \
  --output_dir=data/outputs/smolvla_unoarm
```

**关键参数说明**：

| 参数                                 | 值            | 说明                                             |
| ------------------------------------ | ------------- | ------------------------------------------------ |
| `DEVICE=cuda`                        | 环境变量      | 强制 GPU（SmolVLA 必须用 CUDA）                  |
| `--policy.path=lerobot/smolvla_base` | HF 预训练权重 | **微调模式**，首次下载约 900MB                   |
| `--rename_map`                       | 相机映射      | **必填**，见下方说明                             |
| `--batch_size=4`                     | 小 batch      | 显存大可改 8，OOM 改 2                           |
| `--steps=20000`                      | 训练步数      | 7650 帧 / batch 4 ≈ 10 epoch                     |
| `--eval_steps=0`                     | 不跑 eval     | env 的 reward/success 是常量，eval 无意义        |
| `--eval.use_async_envs=false`        | 单进程        | 避免 NamespaceNotFound（见 REQUIREMENTS.md 3.7） |

#### 6.2 `--rename_map` 为什么必填

`smolvla_base` 预训练时用的相机 key 是 `camera1/2/3`（论文里的 OBS_IMAGE_1/2/3），本数据集用的是 `top/left_wrist/right_wrist`。policy 严格校验 key 一致性，不匹配会立刻报错退出。

按 SmolVLA 论文约定映射：

- `camera1` = top（俯视）
- `camera2` = wrist（手腕）
- `camera3` = side（侧视）

本数据集把 `left_wrist → camera2`、`right_wrist → camera3`（双臂占满 wrist/side 槽位）。

> ⚠️ JSON 必须用**单引号包裹、内部双引号**，直接复制上面的命令即可。详细见 [REQUIREMENTS.md 3.9](./REQUIREMENTS.md#39-smolvla-相机名不匹配)。

#### 6.3 训练输出

```
data/outputs/smolvla_unoarm/
└─ checkpoints/
   ├─ 002000/pretrained_model/   # 每 2000 步存一次
   ├─ 004000/pretrained_model/
   ├─ ...
   ├─ 020000/pretrained_model/
   └─ last/pretrained_model/     # 最后一个的别名（推荐用这个）
```

每个 `pretrained_model/` 含：`config.json`、`model.safetensors`、`policy_preprocessor.json`、`policy_postprocessor.json`。

#### 6.4 关于 eval success rate = 0%

训练结束 LeRobot 会自动跑一次 eval（`env_eval_freq` 默认 = `steps`）。因为 `UnoarmEnv` 的 `reward` / `success` 是硬编码常量（`0.0` / `False`），**eval 报 `pc_success=0.0%` 是必然的，与模型好坏无关**。真正判断模型效果用第 7 节的交互式推理。

---

### 7. 训练后可视化验证

训练完成后，用交互式推理脚本看模型实际表现：

```bash
uv run python custom_envs/unoarm/scripts/08_interact.py
```

**使用流程**：

1. 脚本加载 checkpoint（默认 `data/outputs/smolvla_unoarm/checkpoints/last/pretrained_model`）
2. 打开 MuJoCo viewer 显示机器人
3. 终端提示输入语言指令：

   ```
   === Unoarm SmolVLA interactive inference ===
   Type a language instruction and press Enter to run an episode.
   Type 'quit' / 'exit' (or close the viewer) to leave.

   指令>
   ```

4. 输入训练时用过的指令（如 `Prepare to fight and taunt.`），机器人会从静止位姿开始按模型输出的动作运动 255 步
5. 一个 episode 结束后回到提示符，可输入下一条指令测试泛化
6. 输入 `quit` / `exit` / 空回车，或关闭 viewer 退出

**常用参数**：

| 参数           | 默认                                                            | 说明                                       |
| -------------- | --------------------------------------------------------------- | ------------------------------------------ |
| `--checkpoint` | `data/outputs/smolvla_unoarm/checkpoints/last/pretrained_model` | 指定不同步数的 checkpoint 对比             |
| `--task "..."` | 无                                                              | 单次模式：跑一条指令后退出（不进交互循环） |
| `--max-steps`  | 255                                                             | 每个 episode 步数                          |
| `--speed`      | 1.0                                                             | 播放倍速                                   |

**单次模式示例**：

```bash
uv run python custom_envs/unoarm/scripts/08_interact.py \
  --task "Prepare to fight and taunt." \
  --checkpoint data/outputs/smolvla_unoarm/checkpoints/020000/pretrained_model
```

> 脚本已内置 `HF_HUB_OFFLINE=1`（避免加载 VLM processor 时联网失败，见 [REQUIREMENTS.md 3.10](./REQUIREMENTS.md#310-smolvla-加载-vlm-processor-联网失败)）。

---

## Web 控制台与扩展场景

### Web 控制台

```bash
uv run python custom_envs/unoarm/scripts/09_web_interact.py
```

默认监听 `0.0.0.0:7860` 并尝试打开支持 WebGL 的 Chrome；服务器启动后也可手动访问 `http://127.0.0.1:7860`。页面提供：

- **对话控制**：OpenAI-compatible LLM 将自然语言区分为闲聊或机器人动作请求，再触发已加载策略执行。
- **动作设计**：用 16 个关节滑块设计关键帧、编排播放列表、预览动作，并导出 `.design.json` 与 `.poses.json`。
- **数据生成与回放**：在后台生成关键帧或 IK 数据集，并在浏览器 3D 场景中检查轨迹。
- **运行设置**：配置 checkpoint、设备、LLM API、场景及 rollout 参数，持久化到 `custom_envs/unoarm/data/web_settings.json`。

未加载策略时，动作设计和数据生成仍可使用。Web 端的对话、设计和生成任务采用互斥状态，避免 rollout 与数据生成同时修改仿真状态。更多操作说明见 [`custom_envs/unoarm/README.md`](./custom_envs/unoarm/README.md)。

### Reach-IK 抓剑数据

Reach-IK 使用 MuJoCo DLS 逆运动学，根据剑柄目标点自动生成右臂“接近 → 抓取”轨迹，不需要手工设计关节关键帧。目标点可以来自 JSON，也可以从工作空间 AABB 随机采样：

```bash
uv run python custom_envs/unoarm/scripts/10_generate_reach_ik_data.py \
  --bbox-min -0.45 -0.65 0.95 \
  --bbox-max -0.20 -0.40 1.15 \
  --num-targets 40 --episodes-per-target 1 \
  --output-root custom_envs/unoarm/data/unoarm_reach_ik \
  --overwrite
```

对应 Gym 环境为 `gym_unoarm/UnoarmReachSword-v0`。生成器会记录目标与 IK 元数据，`07_replay_dataset.py` 和 Web 控制台可据此自动恢复场景并回放。

### 桌面抓取放置

`table_place` 场景包含桌面、圆柱和目标圆，支持随机化圆柱起点、IK 生成抓取放置数据、数据集回放和 ACT checkpoint 评估。当前推荐从 Web 控制台进入 **Reach-IK** 页面，将生成模式切换为 `table_place`；生成逻辑位于 `data_gen/table_place_ik.py`。

训练后可用专用脚本评估：

```bash
uv run python custom_envs/unoarm/scripts/05_eval_table_place.py \
  --checkpoint /path/to/pretrained_model \
  --no-from-meta
```

`--no-from-meta` 会在工作空间内随机采样圆柱位置；如需复用训练集目标点，可改为 `--from-meta /path/to/table_place_ik_meta.json`。评估结果会保存视频、指标和逐 episode 信息，便于检查抓取、搬运与放置阶段的实际表现。

---

## 目录结构

```
uno-llm-lerobot-ACT/
├─ README.md                      # 本文档
├─ VLA各系列综述.pdf             # VLA 模型系列与技术背景综述
├─ REQUIREMENTS.md                # 环境安装 + 踩坑记录
├─ AGENTS.md / AGENT_GUIDE.md     # LeRobot 原生 AI agent 指引（保留）
├─ pyproject.toml                 # 依赖定义（含 aloha/smolvla/dataset extras）
├─ uv.lock                        # 锁定的依赖版本
├─ src/lerobot/                   # LeRobot 源码（不改一行）
├─ tests/                         # LeRobot 测试套件
├─ docs/                          # LeRobot 文档
│
├─ custom_envs/unoarm/            # ★ 本项目核心
│  ├─ gym_unoarm/                 # MuJoCo + Gymnasium 仿真
│  │  ├─ env.py                   #   UnoarmEnv（16D 双臂 action/state）
│  │  ├─ constants.py             #   关节定义、相机、控制频率
│  │  ├─ ik.py                    #   MuJoCo DLS 逆运动学
│  │  ├─ reach_scene.py           #   抓剑场景
│  │  ├─ table_place_scene.py     #   桌面抓取放置场景
│  │  ├─ mujoco_unoarm*.xml       #   MJCF 模型与场景
│  │  ├─ unoarm_mujoco.urdf       #   URDF 源文件
│  │  └─ meshes/*.stl             #   机器人与道具网格资产
│  ├─ lerobot_unoarm/             # LeRobot env 注册
│  │  ├─ config.py                #   @register_subclass("unoarm")
│  │  └─ wrapper.py               #   unoarm-train / unoarm-eval 入口
│  ├─ data_gen/                   # 关键帧、Reach-IK、Table-Place 数据生成
│  ├─ pose_design/                # 关键帧工程、预览与 poses 导出
│  ├─ webapp/                     # FastAPI、LLM 路由、模式与仿真 Runner
│  ├─ static/web/                 # 浏览器 3D 控制台
│  ├─ scripts/
│  │  ├─ 01_convert_urdf_to_mjcf.py   # URDF → MJCF 转换
│  │  ├─ 02_test_sim.py               # MuJoCo viewer 手动测试
│  │  ├─ 03_record_data.py            # 键盘 teleop 数据采集（可选）
│  │  ├─ 04_train_smolvla.sh          # 训练命令示例（旧版，参考用）
│  │  ├─ 05_eval.py                   # 通用推理 smoke test
│  │  ├─ 05_eval_table_place.py       # 桌面放置策略评估
│  │  ├─ 06_generate_scripted_data.py # ★ 脚本化数据生成（主入口）
│  │  ├─ 07_replay_dataset.py         # ★ 数据回放可视化
│  │  ├─ 08_interact.py               # 终端交互式推理
│  │  ├─ 09_web_interact.py           # ★ Web 控制台
│  │  └─ 10_generate_reach_ik_data.py # ★ Reach-IK 数据生成
│  ├─ data/
│  │  ├─ poses_prepare_fight_taunt.json  # key pose 定义（入 git）
│  │  ├─ web_settings.json               # Web 设置
│  │  ├─ designs/                        # Web 动作设计工程
│  │  └─ unoarm_*/                       # 生成的数据集（不入 git）
│  ├─ tests/                         # Unoarm 环境、IK、Web 与数据测试
│  └─ pyproject.toml              # unoarm 包定义（unoarm-train 入口）
│
├─ data/outputs/                  # 训练 checkpoint（不入 git，每个 ~1G）
└─ outputs/                       # LeRobot 默认输出（不入 git）
```

**入 git 的**：所有 `.py` / `.md` / `.toml` / `.json` 配置 + `gym_unoarm/` 下的模型资产（`meshes/*.stl`、`mujoco_unoarm.xml`、`unoarm_mujoco.urdf`）。
**不入 git 的**：数据集（`custom_envs/unoarm/data/unoarm_*/`）、训练 checkpoint（`data/outputs/`）、`.venv`、HF cache。详见 `.gitignore`。

---

## 常见问题

<details>
<summary><b>装环境踩坑（bazel / JDK / HF 卡死 / extras 缺失 等）</b></summary>

见 [REQUIREMENTS.md 第 3 节"踩坑记录"](./REQUIREMENTS.md#3-踩坑记录)，包含 10 个实际踩到的坑和完整解决方案。

</details>

<details>
<summary><b>训练时报相机 Feature mismatch</b></summary>

`smolvla_base` 期望 `camera1/2/3`，你的数据是 `top/left_wrist/right_wrist`。训练命令必须带 `--rename_map`，见 [第 6.2 节](#62---rename_map-为什么必填)。

</details>

<details>
<summary><b>eval 的 success rate 一直是 0%</b></summary>

正常现象。`UnoarmEnv` 的 reward/success 是硬编码常量，eval 数字无意义。模型好坏看 [第 7 节](#7-训练后可视化验证) 的交互推理。

</details>

<details>
<summary><b>生成的 30 个 episode 完全相同</b></summary>

噪声参数全是 0 导致的。加 `--midpoint-noise-std 0.02 --hold-noise-std 0.01 --pose-jitter-std 0.05`，见 [第 4.3 节](#43-生成命令)。

</details>

<details>
<summary><b>推理时报 OSError: Can't load processor for SmolVLM2</b></summary>

VLM backbone 加载需要联网，但你的网络访问 HF 不通。`08_interact.py` 已内置 `HF_HUB_OFFLINE=1`；如果训练时遇到，前置该环境变量。见 [REQUIREMENTS.md 3.10](./REQUIREMENTS.md#310-smolvla-加载-vlm-processor-联网失败)。

</details>

<details>
<summary><b>想训练新动作（不是 prepare to fight）</b></summary>

1. 复制 `custom_envs/unoarm/data/poses_prepare_fight_taunt.json` 改名
2. 改 `task` 字段为新指令，改 `poses` 为新动作的 key pose
3. 用新的 `--poses-json` 和 `--output-root` 跑生成命令
4. 用新的 `--dataset.repo_id` / `--dataset.root` 跑训练
</details>
