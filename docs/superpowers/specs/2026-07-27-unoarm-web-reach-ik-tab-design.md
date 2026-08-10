# Unoarm Web Reach-IK Tab — Design

**Date:** 2026-07-27
**Status:** Approved for planning (pending user review of this written spec)
**Scope:** New Web console tab for multi-target reach-IK dataset generation + 3D episode replay

## Goal

Add a top-bar mode **「Reach-IK」** so users can:

1. Configure targets (JSON list **or** AABB sampling) and generate grasp→execution-point training data without CLI
2. List generated datasets and **replay episodes in the existing left Three.js scene** (joint-driven)

## Decisions (locked)

| Topic        | Choice                                                                                                            |
| ------------ | ----------------------------------------------------------------------------------------------------------------- |
| UI placement | New independent tab (approach **1**)                                                                              |
| Target input | Both: JSON textarea **and** AABB form, toggleable (**C**)                                                         |
| Replay       | Left 3D scene only (**A**); no camera-video player in v1                                                          |
| Backend      | Reuse `data_gen.reach_ik.generate_reach_ik_dataset` in a background thread (same pattern as design-mode generate) |

## Non-goals (v1)

- Clicking targets in the 3D view
- Camera / top-RGB video replay strip
- Editing IK orientation conventions in the UI
- Merging this into the pose-design generate panel

## Architecture

```
[Reach-IK tab]
   │  generate form (JSON | AABB)
   │  POST /api/reach-ik/generate
   ▼
UnoarmWebRunner.start_reach_ik_generate()  → thread → generate_reach_ik_dataset
   │
   │  GET /api/reach-ik/generate/status
   ▼
status + logs in panel

[Reach-IK tab]
   │  list datasets / pick episode
   │  POST /api/reach-ik/replay/start|stop
   ▼
Runner loads LeRobot episode frames → steps env with stored actions
   │
   ▼
existing WS snapshot → Three.js (same as live sim)
```

### New / modified pieces

| Path                        | Role                                                       |
| --------------------------- | ---------------------------------------------------------- |
| `static/web/index.html`     | Tab button + `#reachIkPanel` markup                        |
| `static/web/js/reach_ik.js` | Form collect, generate poll, dataset list, replay controls |
| `static/web/js/main.js`     | Mode switch wiring for `reach_ik`                          |
| `webapp/app.py`             | API routes                                                 |
| `webapp/runner.py`          | Generate thread + replay loop; mode exclusivity            |
| `webapp/modes.py`           | Register `reach_ik` mode if needed                         |

Reuse without rewrite: `data_gen/reach_ik.py`, scene snapshot path, `ModeController` stop-other-jobs pattern.

## UI layout (Reach-IK panel)

### Section A — Generate

- Mode toggle: **点列表 (JSON)** | **范围采样 (AABB)**
- JSON mode: textarea (pre-fill example or last used); optional 「加载示例」
- AABB mode: `bbox_min` XYZ, `bbox_max` XYZ, `num_targets`, `seed`
- Shared: `episodes_per_target`, `segment_steps`, `hold_steps`, `execution_point` XYZ (optional), `output` name/subdir, `overwrite` checkbox
- Buttons: **开始生成** / (optional) cancel if easy; otherwise wait for finish
- `<pre>` status + rolling logs (poll every ~0.5–1s while running)

### Section B — Datasets & replay

- Dataset list: scan `custom_envs/unoarm/data/` for dirs that look like LeRobot roots (prefer names matching `unoarm_reach_ik*`, also show path returned by last generate)
- Select dataset → load episode count → select episode index
- Controls: **播放** / **暂停** / **停止**；speed multiplier (reuse global speed or local)
- Status line: frame i/N, attached/exec flags if available from live env info

## API sketch

| Method | Path                                   | Body / notes                           |
| ------ | -------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| POST   | `/api/reach-ik/generate`               | `{mode: "json"                         | "bbox", targets_json?: str, bbox_min?, bbox_max?, num_targets?, seed?, episodes_per_target?, segment_steps?, hold_steps?, execution_point?, output_name?, overwrite?}` |
| GET    | `/api/reach-ik/generate/status`        | `{state, message, logs, output_root}`  |
| GET    | `/api/reach-ik/datasets`               | `[{root, repo_id?, n_episodes?}, ...]` |
| GET    | `/api/reach-ik/datasets/info?root=...` | episode count / task                   |
| POST   | `/api/reach-ik/replay/start`           | `{root, episode, speed?}`              |
| POST   | `/api/reach-ik/replay/stop`            | —                                      |
| GET    | `/api/reach-ik/replay/status`          | `{state, episode, frame, n_frames}`    |

Generation and replay are mutually exclusive with each other and with chat rollout / validate / design preview (enter tab or start job → stop others).

## Replay behavior

1. Load `LeRobotDataset(root=..., episodes=[episode])`
2. For each frame in order: take `action` (normalized 16-D, matching training), `env.step(action)` (or apply + forward), sleep according to FPS/speed
3. Existing snapshot stream continues to update Three.js joints / sword attach visuals
4. Pause freezes the loop; stop resets replay state and clears thread
5. Sword placement for replay: prefer whatever is encoded in the episode observation path; if handle pose is not stored, keep current env sword pose (document limitation). **Preferred v1:** before replay, if dataset meta or first-frame side info is unavailable, still replay arm motion; sword may not match original target unless we store target in episode metadata later.

**v1 mitigation:** When generating via Web, write a small `reach_ik_meta.json` next to the dataset (`targets`, `execution_point_pos`). On replay start, place sword to that episode’s target if meta lists per-episode handles; else place to first target / skip.

## Mode exclusivity

Entering Reach-IK or starting generate/replay:

- Stop chat rollout, validate, design preview, and the other reach-ik job
- Align with existing `ModeController` / `handleModeSwitch` patterns in `main.js`

## Error handling

- Invalid JSON / empty AABB → 400 with message, no thread
- Generate already running → 409
- Replay missing root / bad episode → 400
- Zero episodes written → failed status with log (same as CLI)

## Success criteria

1. User can generate a reach-IK dataset from the Web without CLI
2. User can select that dataset and watch the arm replay in the left 3D view
3. Existing tabs (对话 / 动作设计 / 目标位姿 / 设置 / 验证) keep working
4. CLI `10_generate_reach_ik_data.py` unchanged in behavior

## Testing

- API smoke: generate tiny 1-target job → status done → dataset appears in list
- Replay start/stop does not crash WS snapshot
- Mode switch away from Reach-IK stops replay thread
