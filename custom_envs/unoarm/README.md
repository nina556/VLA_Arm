# 双臂机器人自定义环境

本目录是项目的机器人业务核心，包含 MuJoCo/Gymnasium 环境、LeRobot 注册、数据生成、动作设计、Web 服务、三维前端和测试。本文侧重具体开发与使用；总体技术背景、训练方法和系统架构见仓库根目录 README。

> 文档约定：以下命令均在本目录执行，因此不需要写出本目录的实际名称。

---

## 1. 模块结构

```text
.
├── gym_*/                 # 环境实现、常量、IK、场景和模型资产
├── lerobot_*/             # LeRobot 环境配置与训练/评估包装
├── data_gen/
│   ├── scripted.py        # 关键帧插值数据
│   ├── reach_ik.py        # 多目标 Reach-IK 数据
│   └── table_place_ik.py  # 桌面抓取放置数据
├── pose_design/           # 动作工程、存储、导出和预览
├── webapp/
│   ├── app.py             # FastAPI 路由和静态资源
│   ├── runner.py          # 仿真、策略、生成与回放调度
│   ├── modes.py           # 业务模式互斥
│   ├── settings_store.py  # 设置持久化
│   ├── llm_router.py      # 对话任务路由
│   ├── vla_bridge_client.py # 可选远端执行桥
│   └── ws.py              # WebSocket 快照
├── static/web/            # HTML、CSS、JavaScript 和前端依赖
├── scripts/               # 命令行工作流
├── tests/                 # 单元与集成测试
├── data/                  # 本地设置、动作工程和生成数据
└── pyproject.toml
```

## 2. 安装与启动

从当前目录安装：

```bash
uv pip install -e .
```

仿真检查：

```bash
uv run python scripts/02_test_sim.py
```

启动网页：

```bash
uv run python scripts/09_web_interact.py
```

常用启动参数可通过帮助查看：

```bash
uv run python scripts/09_web_interact.py --help
```

浏览器需支持 WebGL。若自动打开失败，可手动访问终端打印的地址。

---

## 3. 环境规格

- 控制频率：20 Hz
- 时间步长：0.05 秒
- 动作维度：16
- 状态维度：16
- 相机：顶视、左腕、右腕
- 图像：RGB，480×640
- 机械臂：左右各 7 个关节
- 末端：左右各 1 个夹爪控制量

动作索引：

```text
[左臂 J1..J7, 左夹爪, 右臂 J1..J7, 右夹爪]
```

动作设计页面显示 MuJoCo 原始弧度值；策略特征可使用归一化表示。修改数据或接口时必须明确单位。

---

## 4. 网页控制台

顶部功能包括：

- 对话
- 动作设计
- 目标位姿
- Reach-IK
- 验证
- 设置

左侧可显示三路观测相机，中间显示 Three.js 场景，右侧根据模式显示控制面板。

### 4.1 对话模式

LLM 路由器将用户输入分类为闲聊或动作请求。动作请求必须匹配允许任务列表，随后由 Runner 调用策略。未加载 checkpoint 时仍可使用动作设计和数据工具，但不能执行策略 rollout。

### 4.2 动作设计模式

1. 用 16 个滑块调整姿态。
2. 保存关键帧并命名。
3. 添加关键帧到播放列表。
4. 调整顺序，预览插值。
5. 填写任务文本和工程名。
6. 保存动作工程。
7. 选择 episode 和随机化参数生成数据。

关键帧可以被播放列表重复引用；修改关键帧后，所有引用使用最新姿态。

### 4.3 目标位姿

目标位姿页面用于调整场景道具、抓取点和可选执行点。设置通过后端写入场景并广播快照，Three.js 展示应与 MuJoCo 状态一致。

### 4.4 Reach-IK

支持目标列表和工作空间随机采样。生成任务在后台执行，期间进入独占生成模式，避免设计或 rollout 同时修改环境。

### 4.5 验证

验证页用于加载 checkpoint、选择场景和运行 episode。桌面任务应关注抓取、抬升、搬运、释放和最终落点等阶段指标。

### 4.6 设置

设置包括模型、设备、LLM、任务白名单、场景、目标位姿、rollout、显示和桥接参数。API 密钥优先通过指定环境变量读取。

---

## 5. 数据生成

### 5.1 手动录制

```bash
uv run python scripts/03_record_data.py
```

典型键盘操作：

- 空格：切换左右臂
- 数字键：增加所选关节目标
- 字母键：减少所选关节目标
- `g`：切换夹爪
- 回车：保存 episode
- `r`：丢弃并重置
- `Esc`：结束录制

具体键位以脚本当前帮助为准。

### 5.2 关键帧生成

```bash
uv run python scripts/06_generate_scripted_data.py --help
```

关键参数：

| 参数 | 含义 |
| --- | --- |
| `--poses-json` | 任务和关键帧文件 |
| `--episodes` | episode 数 |
| `--segment-steps` | 相邻姿态间插值帧数 |
| `--hold-steps` | 关键姿态保持帧数 |
| `--pose-jitter-std` | episode 级姿态扰动 |
| `--midpoint-noise-std` | 插值路径扰动 |
| `--hold-noise-std` | 保持段扰动 |
| `--seed` | 随机种子 |
| `--overwrite` | 覆盖已有输出 |

推荐以 episode 级扰动为主，避免逐帧噪声导致轨迹抖动。

### 5.3 Reach-IK

```bash
uv run python scripts/10_generate_reach_ik_data.py --help
```

生成器可使用目标 JSON 或三维 AABB 采样。每个目标会经过 IK 求解、关节限位检查、关键阶段构造、插值和数据写入。

### 5.4 桌面放置

桌面生成逻辑位于 `data_gen/table_place_ik.py`，主要处理物体位置采样、净空约束、TCP 偏移、抓取与放置关键路径，以及生成元数据。

---

## 6. 数据回放

```bash
uv run python scripts/07_replay_dataset.py --help
```

回放检查重点：

- 模型动作是否符合任务
- 相机是否完整
- 物体初始状态是否从元数据恢复
- 夹爪时序是否正确
- 状态重现误差是否接近零
- episode 间是否存在合理多样性

不要跳过回放直接训练。数据中的错误通常会被策略稳定地学会。

---

## 7. 推理与评估

终端交互：

```bash
uv run python scripts/08_interact.py
```

通用验证：

```bash
uv run python scripts/05_eval.py --help
```

桌面放置评估：

```bash
uv run python scripts/05_eval_table_place.py --help
```

推理链路包括观测采集、特征预处理、模型动作块预测、后处理、EMA、变化量限制和逐步执行。checkpoint 必须与训练时的相机映射、状态维度、动作维度和处理器配置一致。

---

## 8. 模式与并发

后端主要业务状态包括对话、设计和生成。生成任务为独占状态：

- 已在生成时不能再次启动生成。
- 生成期间不能切换到会修改环境的模式。
- 退出生成后恢复到设计模式。
- 切换业务模式时应停止冲突的 rollout 或预览。

该约束用于避免后台线程和用户操作同时写入环境状态。

---

## 9. 实时状态

WebSocket 向前端发送环境快照，包括：

- 关节位置
- 当前模式
- rollout 状态
- 任务和步数
- 场景参数
- 目标距离
- 成功状态
- 相机图像
- 生成进度
- 系统消息

前端根据快照更新三维模型，不把浏览器中的视觉状态视为后端真值。

---

## 10. 远端执行桥接

桥接客户端允许把策略动作发送给独立执行服务。配置包括：

- 是否启用
- 服务地址
- 执行手臂
- 是否真正执行
- 结果等待超时
- HTTP 超时

推荐先关闭真正执行，只验证请求结构、关节顺序、单位、响应和错误处理。实体执行必须额外配置硬件限位、速度限制、碰撞检测和急停。

---

## 11. 测试

运行全部自定义测试：

```bash
uv run pytest tests -v
```

重点测试：

```bash
uv run pytest tests/test_ik.py -v
uv run pytest tests/test_pose_design.py -v
uv run pytest tests/test_reach_ik_datagen.py -v
uv run pytest tests/test_table_place_env.py -v
uv run pytest tests/test_table_place_ik_datagen.py -v
uv run pytest tests/test_webapp_modes.py -v
uv run pytest tests/test_settings_store.py -v
uv run pytest tests/test_vla_bridge_client.py -v
```

涉及前端变更时，除自动测试外还需手动检查：

1. 页面加载无控制台错误。
2. URDF 和网格资源正常。
3. 三维关节与后端状态一致。
4. 三路相机可显示。
5. 模式切换和生成锁正确。
6. WebSocket 断线后能恢复。
7. 日间和夜间主题均可用。

---

## 12. 开发约定

- 场景常量集中维护，避免前后端各自复制不同数值。
- 生成器和 Web 应复用同一核心函数。
- 新场景必须提供可复现的 reset 参数。
- 元数据应足以恢复回放场景。
- 对公开 API 增加测试。
- 不提交数据集、checkpoint、缓存、密钥或本机绝对路径。
- 修改动作顺序、单位、相机键时视为破坏性接口变更。
- 真实执行相关改动必须先在仿真和仅验证模式中测试。

---

## 13. 故障排查

### WebGL 创建失败

使用支持硬件加速的现代浏览器，检查 GPU 是否被禁用。某些编辑器内置预览器不提供完整 WebGL。

### 模型资源加载失败

检查后端静态挂载、资源路径和文件权限；通过浏览器网络面板确认 URDF 与网格请求状态。

### 设置保存后不生效

确认保存请求成功、设置文件可写，并检查后端是否对字符串、布尔值和三维向量进行了规范化。

### IK 不收敛

检查目标是否在工作空间内、TCP 偏移和目标姿态是否合理、初始姿态是否接近奇异位形，并适当调整阻尼和迭代参数。

### 生成任务卡住

查看后台日志和生成状态，确认没有旧任务占用生成模式；检查输出目录、磁盘空间和视频编码依赖。

### 策略动作维度错误

确认 checkpoint、数据集和当前环境均使用相同的 16 维关节顺序，检查预处理与后处理文件是否来自同一 checkpoint。
