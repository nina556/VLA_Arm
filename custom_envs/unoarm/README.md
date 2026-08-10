# Unoarm LeRobot environment

This package adds a MuJoCo/Gymnasium Unoarm simulation without changing the
LeRobot source tree.

## Layout

- `gym_unoarm`: Gymnasium environment registered as
  `gym_unoarm/UnoarmFreeSpace-v0`.
- `lerobot_unoarm`: LeRobot `EnvConfig` registration for `--env.type=unoarm`.
- `scripts/01_convert_urdf_to_mjcf.py`: copies the Unoarm assets and writes
  `gym_unoarm/mujoco_unoarm.xml`.
- `scripts/02_test_sim.py`: interactive MuJoCo smoke test.
- `scripts/03_record_data.py`: keyboard teleoperation recorder for LeRobotDataset.
- `scripts/04_train_smolvla.sh`: SmolVLA finetune from `lerobot/smolvla_base`
  (features are forced from the dataset: 16-D state/action).
- `scripts/05_eval.py`: SmolVLA inference smoke test.
- `scripts/06_generate_scripted_data.py`: CLI for scripted dataset generation (core logic in `data_gen/`).
- `scripts/10_generate_reach_ik_data.py`: multi-target reach-IK grasp + execution-point datasets.
- `scripts/08_interact.py`: terminal chat + SmolVLA interact.
- `scripts/09_web_interact.py`: web console (chat + pose design).
- `webapp/`, `pose_design/`, `static/web/`: modular web backend/frontend.
- `data/designs/`: saved `*.design.json` projects and exported `*.poses.json`.

## Setup

From the LeRobot repository root:

```bash
uv pip install -e custom_envs/unoarm
uv run python custom_envs/unoarm/scripts/01_convert_urdf_to_mjcf.py
```

The converter reads assets from `/mnt/d/work/code/unoarm_model` by default. To
use another source:

```bash
UNOARM_MODEL_DIR=/path/to/unoarm_model \
  uv run python custom_envs/unoarm/scripts/01_convert_urdf_to_mjcf.py
```

## Web console (chat + pose design)

从仓库根目录启动（路径请用 Linux 正斜杠）。**默认无参即可**：

```bash
uv run python custom_envs/unoarm/scripts/09_web_interact.py
```

启动后会自动用 **Windows Chrome**（带 WebGL 参数）打开页面。
不要用 Cursor 内置 Simple Browser / 预览页——那种环境里 `GL_VENDOR=Disabled`，WebGL 无法创建，和剑模型无关。

若需手动打开：

```bash
bash custom_envs/unoarm/scripts/open_web_chrome.sh 7860
```

可选：`--host` / `--port` / `--settings` / `--no-open`。

浏览器打开终端打印的地址后，点顶栏 **设置**：

- Checkpoint（SmolVLA `pretrained_model` 目录）
- VLM 模型名 / 路径、Device
- 对话 LLM：API Base、API Key、模型名
- 任务白名单（一行一条）
- 场景 `free_space` / `reach_sword`、推理参数

保存后写入 `custom_envs/unoarm/data/web_settings.json`，下次启动自动加载。未配置策略时仍可做动作设计；对话执行需先成功加载策略。

顶栏可切换：

- **对话**：自然语言触发 SmolVLA
- **动作设计**：关键帧 / 播放列表 / 生成数据
- **设置**：上述配置

两种业务模式互斥：进入动作设计时会停止正在进行的 rollout。

### Reach sword 场景（固定剑柄接近）

在设置里将 Scene 选为 `reach_sword` 并保存。成功条件：左爪 TCP 距剑柄 < 阈值（默认 5 cm）。也可在设置中勾选「到位即结束」。

- Gym 注册：`gym_unoarm/UnoarmReachSword-v0`
- 剑位姿常量：`gym_unoarm/constants.py`
- 任务白名单请包含：`Reach the sword handle`

### 动作设计模式用法

关节数值为 **MuJoCo 原始关节角（弧度）**，与 `02_test_sim.py` 滑条、`06` 的 poses JSON 一致，**不是** `[-1, 1]` 归一化值。每个滑条范围来自 MJCF 的真实 `jnt_range`。

#### 1. 调姿态

1. 左侧主视图会实时跟随当前 16 维关节。
2. 右侧「当前姿态」拖动滑条调节；点击某一行选中该关节。
3. 键盘（焦点不在输入框时）：
   - `←` / `→`：微调当前关节
   - `[` / `]`：切换上/下一个关节
4. 「归零」：全部关节回到 0。

#### 2. 保存关键帧（关键帧库）

1. 调到想要的姿态后点「保存关键帧」，输入名称（如 `raise_left`）。
2. 关键帧库中可：
   - 点击选中
   - **加载**：把该关键帧写回滑条
   - **覆盖**：用当前滑条姿态覆盖该关键帧
   - **删除**：删除关键帧（播放列表里同名引用也会去掉）

同一关键帧可被播放列表多次引用；改关键帧后，所有引用处都会用新姿态。

#### 3. 编排播放列表

1. 在关键帧库选中一个关键帧，点「添加选中关键帧」（可重复添加）。
2. 用「上移 / 下移 / 移除」调整顺序。
3. 「预览 / 停止预览」：按列表顺序做**线性插值**播放（无噪声，仅可视化）。

#### 4. 绑定任务并保存

1. **任务 (English)**：填写英文指令，例如 `Prepare to fight and then taunt.`（保存时必填）。
2. **名称**：工程名，会写成安全文件名（字母数字、`_`、`-`）。
3. 点「保存动作」，写入：

```text
custom_envs/unoarm/data/designs/<name>.design.json   # 可再编辑的工程（关键帧库 + 播放列表）
custom_envs/unoarm/data/designs/<name>.poses.json    # 给数据生成用的扁平 poses
```

`.poses.json` 格式与 `06_generate_scripted_data.py` 一致：

```json
{
  "task": "Prepare to fight and then taunt.",
  "poses": [
    { "Left_Joint1": 0.5, "...": "..." },
    { "Left_Joint1": 1.0, "...": "..." }
  ]
}
```

首尾的静止零位 `RAW_ZERO` 仍由生成脚本自动补上，无需写进 JSON。

也可在终端直接用导出文件生成数据：

```bash
uv run python custom_envs/unoarm/scripts/06_generate_scripted_data.py \
  --poses-json custom_envs/unoarm/data/designs/<name>.poses.json \
  --episodes 30 --overwrite
```

默认策略是**条间多样性、轨迹内不抖**：

- `--pose-jitter-std=0.05`：每条 episode 给全部关键帧加同一组小偏移（弧度），轨迹仍平滑
- `--midpoint-noise-std=0` / `--hold-noise-std=0`：不要开帧间抖动，否则回放和策略都会发抖
- `--hold-steps=2`、`--segment-steps=20`：减少冗余静止帧

### Reach-IK 自动抓取采数（多目标点）

无需手调关节关键帧：给定剑柄目标点（JSON 列表或工作空间 AABB 随机采样），用 MuJoCo DLS IK 生成 **接近 → 抓取 → 掠过执行点** 的右臂轨迹并落盘。

```bash
# 点列表
uv run python custom_envs/unoarm/scripts/10_generate_reach_ik_data.py \
  --targets-json custom_envs/unoarm/data/targets_reach_ik_example.json \
  --episodes-per-target 1 --overwrite

# 或在盒子内随机采样
uv run python custom_envs/unoarm/scripts/10_generate_reach_ik_data.py \
  --bbox-min -0.45 -0.65 0.95 \
  --bbox-max -0.20 -0.40 1.15 \
  --num-targets 40 --seed 0 --overwrite
```

TCP 姿态固定（夹爪 +X 朝 −Y）；执行点默认 `DEFAULT_EXECUTION_POINT_POS`，可用 `--execution-point` 覆盖。

Web 控制台顶栏也有 **Reach-IK** 页：可切换 JSON / AABB 生成，并在左侧 3D 场景回放已生成数据集。

#### 5. 已保存动作与 Web 内生成数据

「已保存动作」列表中：

- **打开**：加载工程到当前编辑器
- **删除**：删除对应的 `.design.json` / `.poses.json`
- 选中一项后，填写 Episodes / Segment steps / Hold steps / Pose jitter，勾选是否 Overwrite，点「生成数据」

生成在 Web 进程内后台执行（逻辑与 `06` 相同），状态显示在生成区域；生成期间会锁定设计编辑。数据集默认写到：

```text
custom_envs/unoarm/data/unoarm_<name>/
```

### 相关模块

| 路径                         | 作用                                  |
| ---------------------------- | ------------------------------------- |
| `scripts/09_web_interact.py` | 启动入口                              |
| `webapp/`                    | FastAPI、模式切换、仿真 Runner        |
| `pose_design/`               | 关键帧 / 播放列表 / 导出 / 预览       |
| `data_gen/`                  | 插值扰动数据集生成（CLI 与 Web 共用） |
| `static/web/`                | 前端页面                              |

## Smoke tests

```bash
uv run python -c "import gym_unoarm, gymnasium as gym; env = gym.make('gym_unoarm/UnoarmFreeSpace-v0'); obs, _ = env.reset(); print(obs['agent_pos'].shape); env.close()"
uv run python -c "import lerobot_unoarm; from lerobot.envs.configs import EnvConfig; print(EnvConfig.get_choice_class('unoarm'))"
```

## Recording

```bash
uv run python custom_envs/unoarm/scripts/03_record_data.py
```

Keyboard controls:

- `space`: switch left/right arm.
- `1..7`: increase the selected arm joint target.
- `q..u`: decrease the selected arm joint target.
- `g`: toggle selected gripper.
- `enter`: save the current episode.
- `r`: discard and reset the current episode.
- `esc`: finalize and exit.
