# Unoarm Web 动作设计模式 — 设计规格

日期：2026-07-23  
状态：已确认（实现中/已落地）  
范围：`custom_envs/unoarm` Web 控制台扩展「动作设计」能力，并与插值扰动数据生成打通

## 1. 背景与目标

现有 `09_web_interact.py` 提供对话驱动的 SmolVLA 交互与 Three.js 可视化。脚本化训练数据依赖人工编写符合 `06_generate_scripted_data.py` 格式的 poses JSON。`02_test_sim.py` 虽可用 MuJoCo 滑条调 16 维关节，但无法在 Web 上完成「关键帧 → 顺序 → 绑定英文任务 → 生成数据」闭环。

**目标**：在现有 Web 中增加互斥的「动作设计模式」，支持关键帧库 + 可重复引用的播放列表、实时可视化、简单预览、双文件保存，以及在 Web 内触发生成 LeRobot 数据集。

**成功标准**

1. 设计模式下拖滑条/键盘，主视图机械臂实时跟随  
2. 关键帧库 + 可重复引用的播放列表可编辑并预览  
3. 保存产出可被现有 `06` CLI 直接读取的 `.poses.json`  
4. Web 上可选参数触发生成，进度可见  
5. 代码按约定目录拆分，`09` 仅作启动入口  

## 2. 已确认决策

| 项 | 选择 |
|----|------|
| 关键帧模型 | 关键帧库 + 播放列表（同一关键帧可多次引用） |
| 入口 | 扩展现有 Web，顶栏切换「对话 / 动作设计」 |
| 数据生成 | Web 进程内调用从 `06` 抽出的模块 |
| 模式共存 | 互斥；切设计时停止 rollout |
| 键盘 | 选中关节后方向键微调，`[` / `]` 切换关节 |
| 落盘 | `*.design.json`（工程）+ `*.poses.json`（导出给 06） |
| 预览 | 关键帧间线性插值简单预览（无噪声） |
| 架构 | 后端包 + `static/web` 静态前端（方案 1） |

## 3. 目录与模块边界

```text
custom_envs/unoarm/
├── scripts/
│   ├── 06_generate_scripted_data.py   # CLI 包装，调用 data_gen
│   └── 09_web_interact.py             # 仅 argparse + 启动 uvicorn
├── webapp/
│   ├── __init__.py
│   ├── app.py                         # FastAPI：路由、静态资源、生命周期
│   ├── runner.py                      # 仿真 / 策略 / 对话（自现有 Runner 抽出）
│   ├── modes.py                       # chat | design | generating 状态机
│   └── ws.py                          # WebSocket 状态推送与设计姿态上行
├── pose_design/
│   ├── __init__.py
│   ├── models.py                      # Keyframe / Playlist / DesignProject
│   ├── store.py                       # 读写 design / poses JSON
│   ├── preview.py                     # 线性插值预览轨迹
│   └── export.py                      # playlist 展开 → 扁平 poses
├── data_gen/
│   ├── __init__.py
│   └── scripted.py                    # 自 06 抽出的 load / interpolate / generate
├── static/web/
│   ├── index.html
│   ├── css/app.css
│   └── js/
│       ├── main.js
│       ├── scene.js
│       ├── chat.js
│       └── design.js
└── data/
    └── designs/
        ├── <name>.design.json
        └── <name>.poses.json
```

**职责**

- `webapp`：HTTP/WS、模式互斥、请求转发；不内嵌领域算法细节  
- `pose_design`：动作设计领域逻辑，不依赖 FastAPI  
- `data_gen`：训练数据生成；CLI 与 Web 共用  
- `static/web`：UI；对话与设计侧栏按模式切换  

## 4. 数据格式

### 4.1 工程文件 `data/designs/<name>.design.json`

```json
{
  "version": 1,
  "name": "prepare_fight_taunt",
  "task": "Prepare to fight and then taunt.",
  "keyframes": {
    "raise_left": {
      "Left_Joint1": 1.0,
      "Left_Joint2": 0.41
    }
  },
  "playlist": ["raise_left", "open_grip", "raise_left"]
}
```

- `task`：英文任务指令；保存时校验非空  
- `keyframes`：命名关键帧；值为 **原始关节角**（与 `06` / MuJoCo qpos 一致，非归一化）  
- 关节名必须属于 `CONTROL_JOINTS`；缺失关节视为 `0.0`；未知名报错  
- `playlist`：有序引用；同一 key 可重复；引用必须存在于 `keyframes`  
- 编辑某一关键帧后，所有引用处使用新姿态  

### 4.2 导出文件 `data/designs/<name>.poses.json`

```json
{
  "task": "Prepare to fight and then taunt.",
  "poses": [
    { "Left_Joint1": 1.0 }
  ]
}
```

- 由 `playlist` 按序展开 `keyframes` 得到  
- **兼容** `06_generate_scripted_data.load_poses_from_json`  
- **不**写入 `RAW_ZERO` 首尾（仍由 `build_pose_sequence` 自动添加）  

### 4.3 文件名

用户提供的 `name` 清洗为安全 slug（字母、数字、`_`、`-`）。成对写入同名 `.design.json` / `.poses.json`。

## 5. UI 与交互

### 5.1 顶栏

- 模式切换：`对话` | `动作设计`  
- 切到设计：若正在 rollout → 先 Stop，再进入；对话输入锁定  
- 切回对话：停止预览；若有未保存改动，提示丢弃或保留内存草稿  

### 5.2 主视图

两种模式共用 Three.js / URDF 场景；设计模式下关节变化实时驱动模型。

### 5.3 对话模式侧栏

保持现有：消息区可滚动、输入栏固定、日志区。

### 5.4 动作设计侧栏

1. **当前姿态**：16 维滑条（左右臂分组）；点选后 `←/→` 微调、`[` / `]` 切换关节；归零；从关键帧加载  
2. **关键帧库**：列表；加载到滑条 / 覆盖保存 / 删除；「保存为关键帧」命名  
3. **播放列表**：增删、上下移动、可重复加入同一关键帧；预览（播/停）；英文 `task`；「保存动作」  
4. **已保存动作**：列出 `data/designs/`；打开 / 删除 / 生成数据（参数弹窗对齐 `06` CLI）  

## 6. 后端行为

### 6.1 模式状态机

| 状态 | 允许 | 禁止 |
|------|------|------|
| `chat` | 对话、reset/stop、rollout | 设计姿态写入、预览 |
| `design` | 改姿态、关键帧/播放列表、预览、保存、触发生成 | 对话触发 action、新 rollout |
| `generating` | 查看进度/日志；请求取消（若可打断） | 设计编辑、新生成、rollout |

### 6.2 关节实时同步

- 设计模式：前端经 WebSocket 上行当前 16 维 raw 姿态（拖动高频）  
- HTTP 用于离散操作（关键帧 CRUD、保存、生成等）  
- 服务端写入 MuJoCo qpos（含 mimic）→ `mj_forward` → 快照 `joint_state` → 现有前端刷新路径  
- 设计模式不跑 SmolVLA 推理  

### 6.3 HTTP API

| 方法 | 路径 | 作用 |
|------|------|------|
| `POST` | `/api/mode` | 切换 `chat` / `design` |
| `GET`/`POST` | `/api/design/pose` | 读/写当前 16 维姿态 |
| `GET`/`POST` | `/api/design/project` | 读/写内存中当前工程 |
| `POST` | `/api/design/keyframes` | 新建/更新/删除关键帧 |
| `POST` | `/api/design/playlist` | 更新播放列表 |
| `POST` | `/api/design/preview/start` | 开始预览 |
| `POST` | `/api/design/preview/stop` | 停止预览 |
| `POST` | `/api/design/save` | 落盘 design + poses |
| `GET` | `/api/design/list` | 已保存工程列表 |
| `POST` | `/api/design/load` | 打开工程 |
| `DELETE` | `/api/design/delete` | 删除工程文件 |
| `POST` | `/api/design/generate` | 后台生成数据集 |
| `GET` | `/api/design/generate/status` | 进度/日志 |

现有 `/api/message`、`/api/stop`、`/api/reset`、`/ws` 保留。在 `design` / `generating` 下，`/api/message` 返回明确错误（如 409）。

### 6.4 `06` 改造

`data_gen/scripted.py` 承载核心逻辑；`06_generate_scripted_data.py` 仅作 CLI 包装，默认行为与现网一致。

## 7. 错误处理

- 模式冲突：设计中发对话动作 → `409` + 中文提示  
- 保存：空 task / 空 playlist / 非法名 → `400` + 字段级错误  
- 关节超限：滑条钳制；API 超限值 clip 并写日志警告  
- 生成：输出目录已存在且未 overwrite → 失败提示；异常写入生成日志，状态 `failed`  
- 预览被切模式打断：自动 stop，不写盘  
- 同时只允许一个生成任务  

## 8. 非目标（第一版不做）

- 贴近 `06` 的噪声完整预览  
- 多用户 / 鉴权  
- 数据集自动上传 Hugging Face  
- 修改 SmolVLA 训练脚本  
- 完整撤销栈（可用重新加载关键帧代替）  
- 第一版可不做 `localStorage` 草稿持久化（内存草稿 + 切模式提示即可）  

## 9. 测试要点

- `pose_design.export`：playlist 展开顺序与重复引用正确  
- `pose_design.store`：读写往返与非法 JSON 拒绝  
- `data_gen.scripted.load_poses_from_json`：仍能读导出的 `.poses.json`  
- 模式切换：design 下 message 被拒；chat 下 design pose 写入被拒  
- 手动：滑条/键盘驱动 URDF；预览播放；保存后用 `06` CLI 读文件成功  

## 10. 实现顺序（供后续计划拆分）

1. 抽出 `data_gen/scripted.py`，改 `06` 为薄 CLI  
2. 实现 `pose_design`（models / store / export / preview）与单测  
3. 拆分 `webapp`（runner / modes / app / ws），`09` 变启动器；静态前端迁出  
4. 设计模式 UI + 姿态 WS 同步  
5. 保存/列表/加载/删除  
6. Web 触发生成 + 进度  
7. 端到端手测与文档补充（如 `AGENT_GUIDE` 或 unoarm README 一小段）  
