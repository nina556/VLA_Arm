# Unoarm VLA 外部桥接 — 完整改动总结（供迁移到最新 Web）

> 来源：在 `40eed41` 上开发的桥接功能。对端接口见  
> `/home/doki/project/edge-robot-agent/docs/mujoco_vla_bridge_api.md`  
> 本文件用于丢弃旧工作树后，在最新 Web（settings_store 版）原样恢复。

## 1. 目标行为（方案 A）

- 本地 MuJoCo **继续** `env.step`（镜像闭环，保证相机/关节观测更新）
- 同时把动作发到外部 HTTP：`POST {base}/api/mujoco/vla_joint_chunk`
- **Payload 的 `actions` 必须是绝对关节目标，单位 rad**
- **不要**对桥接发送路径做 `[-1,1] → rad` 反归一化
- 发送内容：本地 `env.step` 之后的 `env._current_control_qpos()`（原始弧度）
- 按 `n_action_steps` 攒 chunk；rollout 结束 flush 剩余
- HTTP / `ok=false` 只打日志，不中断本地镜像

## 2. 默认配置

| 项 | 值 |
|---|---|
| Base URL | `http://192.168.10.38:8765` |
| Path | `/api/mujoco/vla_joint_chunk` |
| `command_type` | `joint_chunk` |
| `unit` | `rad` |
| `arms` | `both`（可选 `right` / `left`） |
| `fps` | `20`（与 `FPS` 一致） |
| `execute` | `true`（可关做 dry-run） |
| `result_timeout_sec` | `5` |
| `http_timeout_sec` | `8` |
| chunk 大小 | `n_action_steps` |

### 请求体示例

```json
{
  "command_type": "joint_chunk",
  "unit": "rad",
  "arms": "both",
  "fps": 20,
  "execute": true,
  "result_timeout_sec": 5,
  "actions": [[/* 16 floats rad */, "..."]],
  "meta": { "instruction": "<task string>" }
}
```

`actions`：非空 `N x 16`，下标 0–7 左臂+左爪，8–15 右臂+右爪。夹爪字段对端当前可能不执行，但仍应发送。

## 3. 文件清单

### 新建（可直接拷回最新树）

| 路径 | 职责 |
|---|---|
| `custom_envs/unoarm/webapp/vla_bridge_client.py` | `BridgeConfig` / `build_joint_chunk_body` / `VlaBridgeClient`（urllib JSON POST）；保留可选 `denormalize_actions` 仅供烟测 `--from-normalized`，**rollout 不用** |
| `custom_envs/unoarm/scripts/test_vla_bridge.py` | 独立烟测 CLI |
| `custom_envs/unoarm/tests/test_vla_bridge_client.py` | 单测（body / config / mock POST / denorm helper） |
| `docs/superpowers/specs/2026-07-24-unoarm-vla-bridge-design.md` | 短设计备忘 |

### 旧版（40eed41 CLI 配置）曾改动的文件

| 路径 | 改动要点 |
|---|---|
| `webapp/runner.py` | `WebConfig` 增加 bridge_*；初始化 `VlaBridgeClient`；`get/set_bridge_config`；`_flush_bridge_chunk`；`_rollout_loop` 镜像后取 raw qpos 攒包发送 |
| `webapp/app.py` | `BridgeConfigRequest`；`GET/POST /api/bridge` |
| `scripts/09_web_interact.py` | `--bridge` / `--bridge-base-url` / `--bridge-arms` / `--bridge-execute` 传入 WebConfig |
| `static/web/index.html` | 对话区顶部 bridge 面板：enabled / URL / arms / execute / 应用按钮 |
| `static/web/js/main.js` | load/save `/api/bridge` |
| `static/web/css/app.css` | `.bridge-panel` 等样式 |

### 非桥接、同会话其它改动（迁移时**不要**当作桥接需求）

- `08_interact.py`：API key、`DEFAULT_ALLOWED_TASKS` 三条太极任务 —— **不属于桥接**，切回最新后按最新仓库为准。

## 4. Runner 核心逻辑（必须复现）

```text
for each policy step:
  action_norm = select_action → postprocess → clip → EMA/rate-limit
  env.step(action_norm)                    # 本地镜像
  if bridge.enabled:
    raw_rad = env._current_control_qpos()  # 原始弧度，禁止再 denorm
    chunk.append(raw_rad)
    if len(chunk) >= n_action_steps:
      POST joint_chunk(actions=chunk, meta.instruction=task)
      chunk.clear()
finally:
  flush remaining chunk
```

错误处理：`try/except` 包住 POST，日志 `bridge error: ...` / `bridge rejected chunk`。

## 5. API 契约

- `GET /api/bridge` → `BridgeConfig.to_dict()`（含 `endpoint`）
- `POST /api/bridge` body 可选字段：`enabled`, `base_url`, `arms`, `fps`, `execute`, `result_timeout_sec`, `http_timeout_sec`
- 非法 `arms` → 400

## 6. Web UI 契约

- 开关文案：`外部桥接测试（本地镜像 + 发送到对端）`
- 默认 URL：`http://192.168.10.38:8765`
- arms select：`both|right|left`
- execute checkbox（默认勾选）
- 「应用桥接设置」→ POST `/api/bridge`
- 状态：`桥接：关闭` / `桥接：开启 · {arms} · execute|dry-run`

## 7. CLI / 烟测

```bash
# 独立烟测（默认 dry-run）
uv run python custom_envs/unoarm/scripts/test_vla_bridge.py
uv run python custom_envs/unoarm/scripts/test_vla_bridge.py --execute --arms right

# 旧版 Web 启动预开桥接（最新版可改为 settings 字段 + 可选 CLI）
uv run python custom_envs/unoarm/scripts/09_web_interact.py \
  --checkpoint <pretrained_model> --bridge
```

## 8. 迁移到「最新 Web」（settings_store）时的落地方式

**已在 `b7799d4`（最新本地 tip）落地。** 具体：

1. 保留三个新建文件（client / smoke / tests）
2. `default_settings()` 已增加 `bridge_*` 字段
3. `WebConfig` / `webconfig_from_settings` / `settings_from_webconfig` 已接入
4. `runner._rollout_loop` 镜像后取 raw qpos 攒包发送
5. `GET/POST /api/bridge` + settings 面板「外部桥接测试」
6. 保存设置会写入 `web_settings.json` 并热更新 bridge client

## 9. 验收

- [x] `pytest custom_envs/unoarm/tests/test_vla_bridge_client.py` 通过
- [ ] `test_vla_bridge.py` dry-run 能打到 `192.168.10.38:8765`
- [ ] Web「设置」勾选桥接 → 保存 → 发任务 → 日志 `bridge sent chunk`
- [ ] 对端确认 `actions` 为 rad 量级（非 `[-1,1]`）
