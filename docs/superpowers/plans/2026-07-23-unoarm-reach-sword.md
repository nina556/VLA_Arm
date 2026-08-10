# Unoarm Reach Sword Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) or subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a kinematic `reach_sword` scene: fixed sword mesh, success when left TCP is within 5 cm of the sword handle; wire Env, Web CLI, snapshot, and Three.js STL.

**Architecture:** Keep free-space XML as source of truth. A small builder injects table/sword/`left_tcp`/`sword_handle` into a derived MJCF. `UnoarmEnv` gains `scene` + success metrics. Web `--scene reach_sword` loads it; snapshot exposes scene constants for Three.js.

**Tech Stack:** Python 3.12, MuJoCo, Gymnasium, FastAPI, Three.js STLLoader

## Global Constraints

- Kinematic control only (no `mj_step` physics / contact response)
- Task string: `Reach the sword handle`
- Sword asset: repo `data/sword.stl` (copied/linked under `gym_unoarm/meshes/` for MuJoCo)
- Do not commit unless the user asks
- Prefer `uv run` for Python commands
- TDD for Env success logic

---

### Task 1: Scene constants + reach XML builder + Env success

**Files:**
- Create: `custom_envs/unoarm/gym_unoarm/reach_scene.py`
- Modify: `custom_envs/unoarm/gym_unoarm/constants.py`
- Modify: `custom_envs/unoarm/gym_unoarm/env.py`
- Modify: `custom_envs/unoarm/gym_unoarm/__init__.py`
- Create: `custom_envs/unoarm/tests/test_reach_sword_env.py`
- Copy/link: `data/sword.stl` → `custom_envs/unoarm/gym_unoarm/meshes/sword.stl`

**Interfaces:**
- `SCENE_FREE_SPACE = "free_space"`, `SCENE_REACH_SWORD = "reach_sword"`
- `TASK_REACH_SWORD = "Reach the sword handle"`
- `REACH_SUCCESS_THRESHOLD = 0.05`
- `SWORD_MESH_SCALE`, `SWORD_BODY_POS`, `SWORD_BODY_QUAT`, `SWORD_HANDLE_LOCAL`, `LEFT_TCP_LOCAL`, `TABLE_*`
- `ensure_reach_sword_xml() -> Path`
- `UnoarmEnv(..., scene: str = "free_space", reach_success_threshold: float = 0.05, terminate_on_success: bool = False)`
- `info`: `success`, `is_success`, `reach_distance` (reach scene only; free_space keeps False/None-compatible)

- [ ] **Step 1:** Write failing tests for reach env success when TCP teleported near handle / fail when far.
- [ ] **Step 2:** Implement constants, XML builder, copy mesh, Env scene + distance.
- [ ] **Step 3:** Register `gym_unoarm/UnoarmReachSword-v0`.
- [ ] **Step 4:** Run `uv run pytest custom_envs/unoarm/tests/test_reach_sword_env.py -v`

---

### Task 2: Web backend scene switch + snapshot fields

**Files:**
- Modify: `custom_envs/unoarm/webapp/runner.py`
- Modify: `custom_envs/unoarm/scripts/09_web_interact.py`
- Modify: `custom_envs/unoarm/webapp/app.py` (mount `/assets` for `data/`)

**Interfaces:**
- `WebConfig.scene: str`
- Env constructed with `scene=cfg.scene`
- `snapshot()["scene"]` dict with name + sword pose/scale/url + handle_local
- `snapshot()["reach_distance"]`, `snapshot()["success"]`

- [ ] **Step 1:** Add `--scene` CLI; pass into WebConfig/runner.
- [ ] **Step 2:** Mount sword static path; extend snapshot.
- [ ] **Step 3:** Smoke: import runner config path / unit-test snapshot keys if cheap.

---

### Task 3: Three.js sword + optional handle marker

**Files:**
- Modify: `custom_envs/unoarm/static/web/js/scene.js`
- Modify: `custom_envs/unoarm/static/web/js/main.js`
- Modify: `custom_envs/unoarm/README.md` (short reach_sword usage)

- [ ] **Step 1:** `applySceneConfig(sceneCfg)` loads STL when `reach_sword`, MuJoCo Z-up group aligned with robot.
- [ ] **Step 2:** main.js applies config from websocket snapshot.
- [ ] **Step 3:** Document `--scene reach_sword` in README.

---

### Task 4: Verify

- [ ] Run reach env tests + existing unoarm tests.
- [ ] Optional: one-shot script print `reach_distance` at START_POSE.
