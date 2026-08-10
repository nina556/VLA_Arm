# Unoarm 桌面放置场景（table_place）

日期：2026-07-29  
状态：Phase 1 已实现（场景 + 设置切换 + Three.js）；Phase 1.5 Reach-IK 抓放待做

## 目标

在现有 Web + MuJoCo 管线上新增 **桌面抓取放置** 场景，作为高精度 pick-and-place 的简化版：

1. **场景**：桌子 + 小圆柱 + 桌面上的圆圈目标标记（无剑、无盾、无空心筒）。
2. **设置**：设置页提供「场景替换」能力，一键加载桌面场景。
3. **运动约定**：抓取与放置统一 **垂直抓取**（TCP 竖直向下）。
4. **后续数据**：Reach-IK 在该场景下生成「先抓后放到标记」轨迹（紧随场景落地后实现）。

成功判定（评估期）采用平面拟合，而非真插入物理：放下后小圆柱中心相对目标圆圆心的距离 / 拟合度。

## 非目标（本轮 / 第一期）

- 空心圆柱真接触插入、碰撞卡住、掉落动力学（完整 peg-in-hole 物理）
- 左臂参与、盾牌、执行点（execution point）
- 第一期不改 SmolVLA 训练入口格式（仍兼容 16D + top + pointmap）
- 第一期不要求完整评估脚本；拟合度指标在场景与 meta 就绪后另开任务
- 不迁移到 LIBERO / Meta-World 等外部仿真

## 已确认选择

| 项 | 选择 |
|----|------|
| 架构 | **新 scene 类型** `table_place`（方案 A），不塞进 `reach_sword` |
| 分期 | **第一期**：设置切换 + 场景可见 + 垂直抓取约定；**紧随**：Reach-IK 生成/回放适配 |
| 采样 | **目标圆圆心固定**；数据生成时 **只随机小圆柱** 在桌面可达区域内的 XY |

---

## 1. Scene 与资源

### 1.1 常量

- 新增 `SCENE_TABLE_PLACE = "table_place"`（与 `free_space` / `reach_sword` 并列）。
- 新增 MJCF（或由 base XML include）：例如 `mujoco_unoarm_table_place.xml`。
- 默认 task 文案（生成用）：`Pick up the cylinder and place it on the target circle`。

### 1.2 场景几何（建议初值，实现时可微调）

| 物体 | 说明 |
|------|------|
| 桌子 | 机器人前方固定桌面；高度与臂可达匹配 |
| 小圆柱 `peg` | 直立于桌面；半径/高度适合右夹爪竖直抓取；kinematic attach（沿用剑吸附思路） |
| 目标圆 `target_circle` | 桌面平面上的可视化圆环 + 世界系 `(cx, cy, cz)` 与半径 `r`；**默认固定** |
| 剑 / 盾 | 本 scene **不存在**（不要依赖 `remove_sword` 停车技巧作为主路径） |

### 1.3 控制与抓取

- 动作/状态仍为 16D 归一化关节（左+右）。
- 右夹爪：靠近 peg 抓取点 + 闭合 → `_peg_attached`；张开 → 释放（物体落在当前 XY，Z 贴桌或保持简化放置）。
- **垂直抓取**：IK / 关键帧生成时约束右 TCP 朝下（与桌面法向对齐）；禁止依赖侧向抓姿态。

### 1.4 观测

与现网一致，避免训练链路分叉：

- `pixels.top` / `observation.images.top`
- `pointmap`（若开启）
- `agent_pos` 16D

场景信息进 Web snapshot（供 Three.js），不强制写入每帧 parquet（meta 另存）。

---

## 2. Web 设置「场景替换」

### 2.1 UI

在设置页增加（或强化现有 Scene 下拉）区块，标题：**场景替换**。

选项至少：

- `reach_sword` — 当前剑盾场景（默认保持兼容）
- `table_place` — 桌面放置场景

文案示例：「加载桌面场景」说明：去掉剑/盾，加载桌子、小圆柱与圆圈标记。

保存并加载 → 走现有 `保存并加载` 流程：写 `web_settings.json` 的 `scene` 字段并 **重建 env**。

### 2.2 切换后的 UI 行为

当 `scene=table_place`：

| 模块 | 行为 |
|------|------|
| 剑/盾/执行点设置 | 隐藏或禁用 |
| 验证抓取（剑） | 禁用或提示「当前场景不支持」 |
| Reach-IK 页 | **第一期可先提示「生成逻辑即将切换」**；1.5 期改为桌面抓放生成 UI |
| 主 3D | 渲染桌、柱、圆标记；无剑盾 mesh |
| 观测相机 | 仍可用 |

当切回 `reach_sword`：恢复原行为。

### 2.3 设置存储

- `settings.scene = "table_place" | "reach_sword" | "free_space"`
- 可选：`table_peg_pos`、`table_circle_center`、`table_circle_radius`（第一期可用常量默认值，后续再暴露到设置）

---

## 3. Env / Runner 行为

### 3.1 `UnoarmEnv`

- `scene=table_place` 时加载 table_place XML。
- `_reach_info` / 剑盾 attach 分支不跑；改为 peg attach +（可选）放置距离信息：
  - `peg_attached`
  - `place_distance`（peg 底面中心到圆心水平距离）
  - `place_fit`（如 `max(0, 1 - dist/r)`）
  - `success`：第一期可不用于 terminate；约定评估时 `dist < r` 或 `fit ≥ 阈值`
- `terminate_on_success`：第一期默认 False。

### 3.2 Runner snapshot

- `_scene_payload_locked` 在 `table_place` 下返回桌子/peg/circle 字段，供 `scene.js` 绘制。
- 重建 env 时清理剑相关 override。

### 3.3 Three.js

- 新增桌面场景可视化（简单 box + cylinder + ring 即可，不必精模）。
- `applySceneConfig` 按 `scene.name === "table_place"` 分支，避免误用剑逻辑。

---

## 4. Reach-IK 适配（第一期之后紧随，标为 Phase 1.5）

仅当 `scene=table_place`（或生成配置显式 `mode=table_place`）时启用。

### 4.1 轨迹关键帧（垂直抓取）

建议顺序：

1. approach_peg（柱上方偏移，爪开，TCP 朝下）
2. grasp_open（柱抓取位，爪开）
3. grasp_close（爪关 → attach）
4. lift（抬高）
5. above_circle（圆心上方，爪关）
6. place_down（降至放置高度，爪关）
7. release（爪开 → 脱附，柱留在圆附近）

### 4.2 采样

- **圆**：固定默认圆心 + 半径（与场景一致）。
- **柱**：在桌面 AABB 内随机 XY（避开圆过近 overlap，可设最小间距）。
- 失败 IK：跳过并重采样（同现有 grasp-only 逻辑）。

### 4.3 Meta

`table_place_meta.json`（或复用 `reach_ik_meta.json` 并加 `kind: table_place`）：

- `mode: "pick_place_circle"`
- `task`: 英文指令
- `circle`: `{center, radius}`
- `episodes[].peg` / `episodes[].circle`（circle 可冗余固定值）
- `success_targets` / `skipped_targets`

### 4.4 回放

- 每集开始：放置 peg 到该集初始位；圆固定显示。
- 顺序播 action；Stop/Pause 行为与现 Reach-IK 回放一致。

---

## 5. 分期验收

### Phase 1 — 场景 + 设置

1. 设置页可选 `table_place`，保存并加载后主视图出现桌、柱、圆，无剑盾。
2. 切回 `reach_sword` 恢复原场景。
3. 关节/观测预览在新场景下仍可用。
4. 右夹爪在柱附近闭合可吸附；张开可释放（手工或设计模式可测）。

### Phase 1.5 — Reach-IK 抓放

1. 在 `table_place` 下生成数据集：轨迹含抓→放，TCP 垂直。
2. 回放可见柱从采样点移到圆附近。
3. 落盘格式仍可被现有 `unoarm-train` + rename_map 训练。

### 后续（不在本文实现范围）

- 评估脚本：成功率 = 最终 `dist < r` + 平均 fit。
- 可选：圆也随机、公差扫参、真插入物理。

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 桌高/柱尺寸导致 IK 大量失败 | 先用保守几何 + 垂直约束；AABB 限制在已知可达区 |
| Three.js 与 MuJoCo 不同步 | snapshot 显式下发 peg/circle 位姿，对齐剑场景做法 |
| 与剑场景设置字段互相污染 | scene 分支严格隔离；table 不用 sword_* 字段 |
| 用户期望「插进去」 | 文档与 UI 写明：本阶段为 **平面圆标记放置** |

---

## 7. 实现顺序建议

1. MJCF + `SCENE_TABLE_PLACE` + env 分支（attach/release/payload）
2. settings / runner 重建 + 设置页「场景替换」
3. Three.js 桌面可视化
4. 冒烟测试（切场景、吸附）
5. Phase 1.5：Reach-IK 生成/回放/meta
6.（可选）评估脚本与拟合度报表
