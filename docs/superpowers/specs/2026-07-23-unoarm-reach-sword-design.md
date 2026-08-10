# Unoarm Reach Sword（固定剑柄接近）— 设计规格

日期：2026-07-23
状态：已实现（第一版）
范围：`custom_envs/unoarm` 在空场景简单动作之后，增加「固定剑 + 单臂够到剑柄」进阶任务（仍为运动学，无物理抓取）

## 1. 背景与目标

当前 Unoarm Web / Env 为 **运动学驱动**（写 `qpos` + `mj_forward`），已能完成空场景、无避障的简单动作与动作设计采数。下一步不直接上真物理抓取，而是验证策略是否具备 **视觉 → 空间到位** 能力。

**目标**：场景中放置固定的剑模型，左臂末端接近 **剑柄端目标点**；用距离判定成功；沿用现有 Web 动作设计 + 脚本插值采数 + SmolVLA 微调闭环。

**非目标（本阶段不做）**

- 真物理接触、摩擦、夹持、避障不穿模
- 剑位姿随机化 / 多目标指代
- 自动 IK / 运动规划采数
- 仅前端假物体、不进 MuJoCo（会导致训练观测看不见目标）

## 2. 已确认决策

| 项         | 选择                                                |
| ---------- | --------------------------------------------------- |
| 主线       | 仿真里把策略做强，场景逐步变难（真机稍后）          |
| 第一关难度 | 固定目标 + 单臂接近（不做随机、不做多物体）         |
| 目标物     | `data/sword.stl`，目标点为 **剑柄端**，不是整剑质心 |
| 控制       | 保持现有 16 维运动学；不加 `mj_step` 物理           |
| 碰撞       | 桌/剑 geom 可关接触（`contype/conaffinity=0`）      |
| 采数       | Web 动作设计 → poses JSON → 现有 `data_gen`         |
| 任务句     | `"Reach the sword handle"`                          |
| 成功       | `‖left_tcp − sword_handle‖ < 阈值`（默认 0.05 m）   |

## 3. 场景（MJCF）

### 3.1 资源与缩放

- 剑网格：仓库内 `data/sword.stl`
- 原生尺寸约 1 m（Z：0→1，XY 约 0.085 × 0.26 m），桌上 1:1 过大
- 实现时使用统一缩放常量（默认先按 `0.35`，摆位时用 `02_test_sim` / Web 目视微调）
- 桌面：简单 box；剑：`body` 固定（无 `freejoint`）

### 3.2 Sites

- `left_tcp`：挂在左夹爪合适位置，用于量末端
- `sword_handle`：挂在剑 `body` 局部坐标下的 **柄端中心**
  - 网格 Z≈0 端更像柄侧；若目视反了，只改 site 局部坐标，不改成功逻辑

### 3.3 与空场景共存

- 保留 `UnoarmFreeSpace-v0` / 现有空场景行为
- Reach 使用独立任务名或 `task`/`scene` 配置加载带桌+剑的 XML（或 base + include）
- 不得破坏现有 free-space 数据与设计工程的默认路径

## 4. Env API

- 新入口示例：`gym_unoarm/UnoarmReachSword-v0`，或 `UnoarmEnv(..., task="reach_sword")` / `scene="reach_sword"`
- Action / observation 维度与 free-space **不变**（16 维 + 现有相机键）
- 每步在 `mj_forward` 后计算 `reach_distance`
- `info` 字段：
  - `success` / `is_success`：bool
  - `reach_distance`：float（米）
- 配置项：
  - `reach_success_threshold`（默认 `0.05`）
  - `terminate_on_success`（评测建议 `True`；采数可 `False`，避免 episode 过短）
- 本关 **不要求** 夹爪开合

## 5. 采数与训练

1. Web「动作设计」录制关键帧：home → 伸向剑柄 →（可选）短暂停住
2. 导出 `*.poses.json`，`task` 必须为 `"Reach the sword handle"`
3. 经现有 `06` / Web 生成脚本调用 `data_gen` 写出 LeRobot 数据集
4. 成功标签以 Env 距离为准，不人工标 success
5. 建议先少量高质量轨迹（约 20–50 episodes）再微调 SmolVLA

## 6. Web 与评测

### 6.1 后端

- `09_web_interact.py` / `WebConfig` 增加场景开关，例如 `--scene reach_sword`（默认 `free_space`）
- Reach 模式下加载带剑 XML；WebSocket snapshot 增加 `reach_distance`、`success`
- `--allowed-task "Reach the sword handle"`

### 6.2 前端（Three.js）

- 策略观测仍来自 MuJoCo 相机渲染
- Reach 场景下用现有 `STLLoader` 加载剑模型，**位姿/缩放与 MJCF 共用同一套常量**
- 可选：在剑柄处画调试小球/坐标轴（仅人看，不进观测）

### 6.3 评测

- Web 对话 rollout 或现有 `05_eval` 路径
- 指标：episode `is_success` 率；不以「动作像不像」为主观标准

## 7. 模块边界（建议改动面）

```text
custom_envs/unoarm/
├── gym_unoarm/
│   ├── mujoco_unoarm.xml              # 或 reach include / 新 XML
│   ├── env.py                         # reach 距离与 info
│   └── constants.py                   # 场景路径、阈值、剑位姿/缩放常量
├── webapp/
│   ├── runner.py                      # scene 切换、snapshot 字段
│   └── ...
├── static/web/js/scene.js             # 加载剑 STL（与 MJCF 对齐）
├── scripts/09_web_interact.py         # --scene / allowed-task
└── data/sword.stl                     # 已有资源
```

- 常量单源：剑的 `pos/quat/scale` 与 `sword_handle` 局部偏移只定义一处，供 MJCF 生成/手写 XML 与前端引用（若前端无法 import Python，则复制到前端配置并在 README 注明必须同步）

## 8. 成功标准（本规格完成时）

1. Reach 场景下 MuJoCo 相机能看见剑；左臂够到柄端时 `info["success"]==True`
2. 可用动作设计导出 poses 并生成数据集，`task` 为约定英文句
3. Web `--scene reach_sword` 下可对话触发策略，并看到距离/成功状态
4. Free-space 原有流程不被破坏

## 9. 后续进阶（本规格范围外）

1. 剑位姿小范围随机（仍无障碍）
2. 接近 + 夹爪开合时序
3. 伪抓取（闭合附着）
4. 简化物理 → 真接触抓取

## 10. 开放实现细节（实现计划阶段再钉死）

- 剑最终缩放值与世界坐标（以目视标定结果为准）
- `sword_handle` 在 mesh 局部的精确偏移（柄端确认后写入常量）
- 新 Gym 注册名 vs `task`/`scene` 字符串的最终命名
- 前端剑资源是静态拷贝还是由后端挂载 `data/sword.stl`
