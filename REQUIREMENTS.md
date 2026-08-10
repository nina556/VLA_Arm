# 环境安装与踩坑记录

本文档记录 Unoarm + LeRobot + SmolVLA 仿真训练 pipeline 的完整安装步骤，以及实际部署中遇到的所有坑和解决方案。**装环境前请先通读第 3 节"踩坑记录"**，能省掉大量排错时间。

---

## 1. 系统要求

| 项       | 要求                                                                                    |
| -------- | --------------------------------------------------------------------------------------- |
| 操作系统 | Ubuntu 24.04（WSL2 亦可，Windows 主机通过 `\\wsl.localhost\Ubuntu-24.04\...` 访问文件） |
| Python   | 3.13（仓库自带 `.python-version`，`uv` 会自动处理）                                     |
| GPU      | NVIDIA GPU + 驱动 ≥ 570.86（SmolVLA 训练必须用 CUDA；纯 CPU 跑不动）                    |
| CUDA     | cu128（pyproject.toml 已锁定 torch 的 cu128 wheel）                                     |
| 磁盘     | ≥ 30 GB（`.venv` 约 8G、HF cache 约 5G、训练 checkpoint 每个 ~1G）                      |
| 内存     | ≥ 32 GB 推荐                                                                            |
| 网络     | 需访问 Hugging Face Hub（国内建议配 hf-mirror，见 3.5）                                 |

---

## 2. 安装步骤

### 2.1 安装系统依赖

```bash
sudo apt update
sudo apt install -y build-essential git git-lfs ffmpeg openjdk-17-jdk
```

说明：

- `ffmpeg`：数据集视频解码 / 推理录像需要
- `openjdk-17-jdk`：部分依赖（labmaze / dwave 等）的构建链需要 JDK，详见 3.3
- `git-lfs`：拉取测试资产用（可选）

### 2.2 安装 uv

LeRobot 用 [`uv`](https://docs.astral.sh/uv/) 管理依赖，不用裸 `pip`。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 重开 shell 或 source ~/.bashrc 让 uv 进 PATH
uv --version  # 验证
```

### 2.3 克隆并安装 Python 依赖

```bash
git clone git@gitee.com:undoki/uno-llm-lerobot.git lerobot-main
cd lerobot-main

# 关键：必须同时带 aloha + smolvla + dataset 三个 extra，缺一不可（见 3.4）
uv sync --locked --extra aloha --extra smolvla --extra dataset
```

这一步会下载约 8 GB 的 wheel 并创建 `.venv`。**国内网络**建议先设置 HF 镜像（见 3.5）再跑。

### 2.4 安装 Unoarm 自定义环境包

```bash
uv pip install -e custom_envs/unoarm
```

这会把 `gym_unoarm`（仿真）和 `lerobot_unoarm`（LeRobot env 注册）以可编辑模式装进 `.venv`，注册 `--env.type=unoarm` 和 `gym_unoarm/UnoarmFreeSpace-v0`。

### 2.5 转换 MJCF 模型

Unoarm 的 MJCF 文件（`gym_unoarm/mujoco_unoarm.xml`）是从 URDF 自动生成的，仓库里已经包含生成好的版本。**只有修改了 URDF / meshes 才需要重新生成**：

```bash
uv run python custom_envs/unoarm/scripts/01_convert_urdf_to_mjcf.py
```

默认从 `/mnt/d/work/code/unoarm_model` 读取资产；用其他路径：

```bash
UNOARM_MODEL_DIR=/path/to/unoarm_model \
  uv run python custom_envs/unoarm/scripts/01_convert_urdf_to_mjcf.py
```

### 2.6 冒烟测试

```bash
# 仿真环境能起来
uv run python -c "import gym_unoarm, gymnasium as gym; env = gym.make('gym_unoarm/UnoarmFreeSpace-v0'); obs, _ = env.reset(); print('agent_pos shape:', obs['agent_pos'].shape); env.close()"

# LeRobot env 注册成功
uv run python -c "import lerobot_unoarm; from lerobot.envs.configs import EnvConfig; print('unoarm registered:', EnvConfig.get_choice_class('unoarm'))"

# 训练入口可用
uv run unoarm-train --help | head -5
```

三条都通过即环境就绪。

---

## 3. 踩坑记录

下面每一条都是实际部署中真实踩到的坑。表格先给速查，后面是详细说明。

### 3.1 速查表

| #   | 问题                                        | 现象                                                                             | 解决                                                         |
| --- | ------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 1   | uv lockfile 不一致                          | `uv sync` 报 lockfile 与 pyproject 不匹配                                        | `uv lock` 重新生成 lockfile                                  |
| 2   | labmaze 需要 bazel 编译                     | 装 deepmind 系包时报 bazel 缺失或版本错                                          | 手动下 bazel 5.4.0 二进制                                    |
| 3   | JDK 缺失                                    | bazel / 某些构建步骤报 `java: command not found`                                 | `apt install openjdk-17-jdk`                                 |
| 4   | aloha extra 没装上                          | `ModuleNotFoundError: gym_aloha` 或 env 列表里没有 aloha                         | `uv sync --extra aloha --extra smolvla --extra dataset`      |
| 5   | HF 下载卡死（xet 401）                      | 下 SmolVLA / dataset 时卡住或报 401 Unauthorized                                 | `HF_HUB_DISABLE_XET=1` + `HF_ENDPOINT=https://hf-mirror.com` |
| 6   | gym_aloha NamespaceNotFound（async worker） | eval 时子进程报命名空间未注册                                                    | `--eval.use_async_envs=false`                                |
| 7   | torchcodec FFmpeg 库版本不匹配              | 加载视频数据集报 ffmpeg 库符号错误                                               | `--dataset.video_backend=pyav`                               |
| 8   | SmolVLA 相机名不匹配                        | `Feature mismatch ... Missing: camera1/2/3, Extra: top/left_wrist/right_wrist`   | 加 `--rename_map`（见 3.8）                                  |
| 9   | SmolVLA 加载 VLM processor 联网失败         | `OSError: Can't load processor for 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'` | `HF_HUB_OFFLINE=1`（见 3.9）                                 |

### 3.2 uv lockfile 不一致

**现象**：`uv sync --locked` 报类似 `lockfile is out of date` 或 `failed to validate lockfile`。

**原因**：`pyproject.toml` 改过但没重新 lock，或拉取的 lockfile 与本地 pyproject 不一致。

**解决**：

```bash
uv lock            # 按 pyproject.toml 重新生成 uv.lock
uv sync --locked --extra aloha --extra smolvla --extra dataset
```

> ⚠️ 仓库根目录的 `uv.lock` 是被追踪的；`custom_envs/unoarm/uv.lock` 已在 `.gitignore` 里排除，不要混淆。

### 3.3 labmaze 需要 bazel 编译

**现象**：安装 deepmind 系依赖（如 `labmaze`，被 `gym_aloha` 链路间接拉入）时构建失败，报 `bazel: command not found` 或 bazel 版本不对。

**原因**：`labmaze` 没有预编译 wheel，需要 bazel 现场编译 C++ 部分。

**解决**：手动下载 bazel 5.4.0 二进制（不要用 apt 的 bazel，版本太旧）：

```bash
# 下载 bazel 5.4.0 官方二进制
wget https://github.com/bazelbuild/bazel/releases/download/5.4.0/bazel-5.4.0-linux-x86_64
chmod +x bazel-5.4.0-linux-x86_64
sudo mv bazel-5.4.0-linux-x86_64 /usr/local/bin/bazel
bazel --version  # 应输出 5.4.0
```

然后再跑 `uv sync`。如果还报 JDK 相关错，先做 3.4。

### 3.4 JDK 缺失

**现象**：bazel 编译时报 `java: command not found` 或 `Could not find a valid JDK`。

**解决**：

```bash
sudo apt install -y openjdk-17-jdk
java -version  # 验证
```

bazel 5.4.0 需要 JDK 17（JDK 21 也行，但部分 plugin 不兼容，17 最稳）。

### 3.5 aloha extra 没装上

**现象**：训练 / eval 时报 `ModuleNotFoundError: No module named 'gym_aloha'`，或 `--env.type=aloha` 找不到。

**原因**：`gym_aloha` 在 `pyproject.toml` 的 `[project.optional-dependencies]` 的 `aloha` extra 里，默认 `uv sync` 不带 extra 不会装。`smolvla`（模型）和 `dataset`（数据集工具）同理。

**解决**：三个 extra 必须同时带上：

```bash
uv sync --locked --extra aloha --extra smolvla --extra dataset
```

> 这是本项目的标准装法，README 和 2.3 节用的也是这条命令。漏掉任何一个都会在后续某步炸。

### 3.6 HF 下载卡死 / xet 401

**现象**：下载 SmolVLA 权重（`lerobot/smolvla_base`，约 900MB）或 dataset 时进度条卡死，或报 `401 Unauthorized` / `xet` 相关错误。

**原因**：Hugging Face 默认走 Xet 协议（`xet.io`），国内访问不稳定；且部分场景鉴权异常。

**解决**：禁用 xet + 用国内镜像：

```bash
# 加到 ~/.bashrc 永久生效
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com
```

临时用：

```bash
HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com \
  uv run python custom_envs/unoarm/scripts/06_generate_scripted_data.py ...
```

> hf-mirror.com 是 Hugging Face 官方国内镜像，dataset 和 model 都能拉。

### 3.7 gym_aloha NamespaceNotFound（async worker）

**现象**：训练结束触发 eval，或单独跑 eval 时，子进程报 `gym_aloha` 或其他 env 命名空间未注册（`NamespaceNotFound`），但主进程明明能 import。

**原因**：LeRobot 的 async eval 用 `multiprocessing` 起子进程，子进程没继承主进程对 `lerobot_unoarm` / `gym_aloha` 的 import，导致 gym 注册表里找不到自定义 env。

**解决**：禁用 async env（单进程 eval）：

```bash
--eval.use_async_envs=false --eval.batch_size=1
```

> 本项目的训练命令（README 第 6 节）已默认带这两个参数。仿真 env 本身 reward/success 是常量，eval 数值仅供参考，真正验证模型看 `08_interact.py` 的可视化。

### 3.8 torchcodec FFmpeg 库版本不匹配

**现象**：加载视频格式数据集时报类似 `undefined symbol` 或 `FFmpeg library version mismatch`。

**原因**：`torchcodec` 对系统 ffmpeg 的 .so 版本有严格要求，Ubuntu apt 装的 ffmpeg 版本经常对不上。

**解决**：改用 `pyav` 后端（纯 Python，不依赖系统 ffmpeg ABI）：

```bash
--dataset.video_backend=pyav
```

> **本项目的数据集用的是 PNG 内嵌图像（`use_videos=False`），不走视频解码**，所以训练命令里其实不需要这个参数。只有当你改用视频格式数据集时才会遇到。

### 3.9 SmolVLA 相机名不匹配

**现象**：训练启动时立刻报：

```
ValueError: Feature mismatch between dataset/environment and policy config.
- Missing features: ['observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3']
- Extra features: ['observation.images.left_wrist', 'observation.images.right_wrist', 'observation.images.top']
```

**原因**：`smolvla_base` 预训练 checkpoint 训练时用的相机 key 是 `camera1/2/3`（对应论文里的 OBS_IMAGE_1/2/3），而本数据集用的是 `top/left_wrist/right_wrist`。policy 严格校验 key 一致性。

**解决**：加 `--rename_map` 把数据集 key 映射到 policy 期望的 key（按论文约定：camera1=top，camera2/3=wrist/side）：

```bash
--rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}'
```

> ⚠️ JSON 必须用单引号包裹、内部双引号。README 第 6 节的训练命令已包含这条。

### 3.10 SmolVLA 加载 VLM processor 联网失败

**现象**：推理 / 加载 checkpoint 时报：

```
OSError: Can't load processor for 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'.
```

**原因**：SmolVLA 的 VLM backbone（SmolVLM2）通过 `transformers.AutoProcessor.from_pretrained` 加载，即使本地已缓存，默认仍会联网检查 / 下载 processor 配置。网络不通就报这个 OSError。

**解决**：强制离线模式，直接用本地缓存：

```bash
export HF_HUB_OFFLINE=1
```

> 本项目的推理脚本 `08_interact.py` 已在脚本开头用 `os.environ.setdefault("HF_HUB_OFFLINE", "1")` 自动设置，无需手动加。训练时如果遇到，也可以前置这个环境变量。

---

## 4. 验证安装是否成功

按 README 的"快速开始"跑一遍最小闭环：

```bash
# 1. 生成少量数据（约 30 秒）
uv run python custom_envs/unoarm/scripts/06_generate_scripted_data.py \
  --episodes 2 --segment-steps 5 --hold-steps 2 \
  --output-root /tmp/smoke_data --repo-id smoke/test --overwrite

# 2. 回放看 viewer 能开（需图形界面）
uv run python custom_envs/unoarm/scripts/07_replay_dataset.py \
  --root /tmp/smoke_data --episode 0

# 3. 启动交互推理（需先有训练 checkpoint，否则跳过）
uv run python custom_envs/unoarm/scripts/08_interact.py
```

第 1 步成功就说明仿真 + 数据生成 pipeline 通了。第 2 步能开 viewer 说明 MuJoCo 渲染 OK。
