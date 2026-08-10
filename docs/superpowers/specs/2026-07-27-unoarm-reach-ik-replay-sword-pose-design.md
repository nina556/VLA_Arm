# Reach-IK：单点抓取生成 + 顺序回放（剑位同步）

日期：2026-07-27
状态：在原「回放显示抓取点」基础上扩展；待用户确认后进入实现计划

## 目标

1. **生成**：只做「接近 → 抓取」单个定点，**不要执行点段**（先测单点抓取）。
2. **回放**：点播放后，按 episode **一个接一个顺序播完**；用户可随时 **Stop**。
3. **回放场景**：每个 episode 开始时，左侧 3D 把**剑 + 红点**放到该集 `handle`。
4. **训练**：确认落盘格式可直接用于现有 Unoarm SmolVLA 微调流程。

## 非目标

- 执行点 IK / 掠过判定 / 回放绿点
- 选中 episode 未播放前的预览
- 改 AABB 采样算法本身（仍采样 handle XYZ）
- 停止回放后自动恢复设置页默认剑位

---

## 1. 生成：仅单点抓取

### 轨迹关键帧（改后）

`build_reach_ik_poses` 只返回 3 个关键帧：

1. approach（handle + Y 方向 offset，夹爪开）
2. grasp_open（handle，夹爪开）
3. grasp_close（handle，夹爪关）

**删除** execution IK 关键帧。不再要求 `execution_point_passed`。

### Env / meta

- 生成时 `enable_execution_point=False`（或不调用执行点逻辑）。
- `reach_ik_meta.json` 仍写 `episodes[].handle`；`execution_point_pos` 可省略或写 `null`（回放忽略）。
- 默认 task 文案改为抓取向，例如：`Grasp the sword handle`（Web/CLI 仍可覆盖）。

### Web / CLI

- AABB / JSON 仍只决定 **handle 采样点**；执行点输入本轮可隐藏或忽略。
- 验收：生成的 episode 在抓取闭合后结束，无「移向执行点」段。

---

## 2. 回放：顺序连播 + 可停 + 剑位

### 行为

1. 用户选数据集，可选起始 `episode`（默认 0），点「播放」。
2. 后端从该 index 起：`ep, ep+1, …, n-1` **自动连播**。
3. 每个 episode 开始前：
   - 从 meta 取 `handle` → `place_sword_handle`
   - 关闭执行点显示/逻辑
4. 用户点 **Stop** → 中断当前及后续 episode；**Pause** 仍只暂停当前帧推进。
5. 全部播完 → `state=done`。

### 前端

- 状态文案显示当前 `episode i/n` 与帧进度。
- Three.js：位姿变化时立刻钉住剑+红点（修 `applySceneConfig` 早退不同步问题）。

---

## 3. 与 SmolVLA 训练格式（结论）

**可以直接用于当前 Unoarm SmolVLA 微调**，前提是沿用现有 `dataset_features()` / `add_frame`（与 scripted 采数相同）。

| 字段                     | 形状 / 类型           | 与训练脚本                    |
| ------------------------ | --------------------- | ----------------------------- |
| `action`                 | float32 `(16,)`       | Unoarm 16 关节                |
| `observation.state`      | float32 `(16,)`       | 同上                          |
| `observation.images.top` | image `(480,640,3)`   | 需 `--rename_map` → `camera1` |
| `observation.pointmap`   | float32 `(480,640,3)` | `--policy.use_pointmap=true`  |
| `task`                   | 字符串                | 语言条件                      |

训练示例（把 `dataset.root` / `repo_id` 换成 Reach-IK 输出即可）：

```bash
uv run unoarm-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<reach_ik_repo_id> \
  --dataset.root=custom_envs/unoarm/data/<reach_ik_dir> \
  --rename_map='{"observation.images.top":"observation.images.camera1"}' \
  --policy.use_pointmap=true \
  --policy.pointmap_feature=observation.pointmap \
  --policy.pointmap_norm_radius=0.7 \
  ...
```

**注意（不是格式问题）：**

- 单点抓取数据只教「伸手抓住」，不含执行点运动；策略任务语义应与 `task` 文案一致。
- 数据量 / 多样性仍取决于 AABB 采样点数与 `episodes_per_target`。
- 需与现网一致：`use_videos=False` 的 image 数据集；`video_backend` 仍可按现脚本配置。

---

## 验收

- 生成：轨迹止于抓取闭合；meta 有 `handle`；无执行点依赖。
- 回放：从所选 episode 顺序播到末尾；Stop 可停；每集剑+红点在对应 `handle`。
- 训练：特征与 `04_train_smolvla.sh` 所用 demo 集同构，可直接改 `dataset.root` 开训。
