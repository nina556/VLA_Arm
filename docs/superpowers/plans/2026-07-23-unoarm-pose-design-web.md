# Unoarm Web Pose Design Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Unoarm web console with an exclusive Action Design mode: edit 16D poses, manage a keyframe library + playlist, preview, save `.design.json`/`.poses.json`, and generate LeRobot datasets in-process.

**Architecture:** Split `09_web_interact.py` into `webapp/` (FastAPI + runner + modes), `pose_design/` (domain), `data_gen/` (scripted generation from `06`), and `static/web/` (frontend). CLI scripts become thin wrappers.

**Tech Stack:** Python 3.12+, FastAPI, WebSocket, MuJoCo/UnoarmEnv, Three.js (existing), vanilla JS

## Global Constraints

- Pose values are **raw qpos** (not normalized), joint names = `CONTROL_JOINTS`
- Modes are **mutually exclusive**: `chat` | `design` | `generating`
- Dual save: `<name>.design.json` + `<name>.poses.json` under `custom_envs/unoarm/data/designs/`
- Do not commit unless the user asks
- Prefer `uv run` for Python commands
- Keep modules focused; do not dump all logic into one script

---

### Task 1: Extract `data_gen.scripted` from `06`

**Files:**
- Create: `custom_envs/unoarm/data_gen/__init__.py`
- Create: `custom_envs/unoarm/data_gen/scripted.py`
- Modify: `custom_envs/unoarm/scripts/06_generate_scripted_data.py` (thin CLI)
- Test: `custom_envs/unoarm/tests/test_data_gen_scripted.py`

**Interfaces:**
- Produces: `load_poses_from_json(path) -> tuple[str, list[np.ndarray]]`, `build_pose_sequence`, `build_episode_actions`, `interpolate`, `clip_raw_pose`, `dataset_features`, `add_frame`, `generate_scripted_dataset(cfg, log=...) -> Path`, `ScriptedGenConfig` dataclass, `RAW_ZERO`, `EXPECTED_ACTION_ORDER`

- [ ] **Step 1:** Create `data_gen/scripted.py` by moving logic from `06` (keep behavior identical). Add `ScriptedGenConfig` and `generate_scripted_dataset(config, *, log=print) -> Path`.
- [ ] **Step 2:** Rewrite `06` to parse args and call `generate_scripted_dataset`.
- [ ] **Step 3:** Add unit tests for `load_poses_from_json` and `build_pose_sequence` / playlist-independent helpers.
- [ ] **Step 4:** Run `uv run pytest custom_envs/unoarm/tests/test_data_gen_scripted.py -v`

---

### Task 2: `pose_design` models, export, store, preview

**Files:**
- Create: `custom_envs/unoarm/pose_design/__init__.py`
- Create: `custom_envs/unoarm/pose_design/models.py`
- Create: `custom_envs/unoarm/pose_design/export.py`
- Create: `custom_envs/unoarm/pose_design/store.py`
- Create: `custom_envs/unoarm/pose_design/preview.py`
- Test: `custom_envs/unoarm/tests/test_pose_design.py`

**Interfaces:**
- Produces:
  - `DesignProject(version, name, task, keyframes: dict[str, dict[str, float]], playlist: list[str])`
  - `slugify_name(name: str) -> str`
  - `validate_project(project) -> None` (raises ValueError)
  - `expand_playlist(project) -> list[dict[str, float]]`
  - `to_poses_payload(project) -> dict`  # `{task, poses}`
  - `DesignStore(root: Path)` with `list()`, `save(project)`, `load(name)`, `delete(name)`
  - `build_preview_trajectory(project, segment_steps: int) -> list[np.ndarray]`  # linear, no noise, no RAW_ZERO wrap

- [ ] **Step 1:** Write failing tests for slugify, validate, expand (incl. repeats), store roundtrip, preview length.
- [ ] **Step 2:** Implement modules to pass tests.
- [ ] **Step 3:** Run `uv run pytest custom_envs/unoarm/tests/test_pose_design.py -v`

---

### Task 3: Split webapp core (modes + runner + app shell)

**Files:**
- Create: `custom_envs/unoarm/webapp/__init__.py`
- Create: `custom_envs/unoarm/webapp/modes.py`
- Create: `custom_envs/unoarm/webapp/runner.py` (move `UnoarmWebRunner` from `09`)
- Create: `custom_envs/unoarm/webapp/app.py`
- Create: `custom_envs/unoarm/webapp/ws.py`
- Create: `custom_envs/unoarm/static/web/` (move HTML/CSS/JS from INDEX_HTML)
- Modify: `custom_envs/unoarm/scripts/09_web_interact.py` → thin launcher
- Test: `custom_envs/unoarm/tests/test_webapp_modes.py`

**Interfaces:**
- Produces: `AppMode` enum / Literal; `ModeController` with `mode`, `set_mode(mode)`, `require_mode(*allowed)` raising HTTP-friendly errors; runner API unchanged from current chat behavior

- [ ] **Step 1:** Implement `modes.py` + tests (chat→design stops conceptually; design blocks chat actions).
- [ ] **Step 2:** Move runner + build FastAPI app; serve static `index.html`; keep existing chat APIs working.
- [ ] **Step 3:** Smoke: import `build_app` and hit `/` returns 200 (TestClient).

---

### Task 4: Design APIs + live pose sync

**Files:**
- Modify: `webapp/app.py`, `webapp/runner.py`, `webapp/ws.py`
- Modify: `static/web/js/design.js`, `main.js`, `css/app.css`, `index.html`
- Test: `custom_envs/unoarm/tests/test_design_api.py` (mode gates + save/list with tmp dir)

**Behavior:**
- WS client may send `{type:"design_pose", pose:[16 floats raw]}` when mode=design
- Runner applies raw pose via env denormalize path inverted: set qpos from raw + mimics + mj_forward; update obs/joint_state for stream
- HTTP endpoints per spec section 6.3
- UI: mode toggle; design sidebar with 16 sliders, keyframe library, playlist, preview controls, save form

- [ ] **Step 1:** Backend design session object on runner + APIs + WS pose apply
- [ ] **Step 2:** Frontend design mode UI + keyboard (select + arrows + `[`/`]`)
- [ ] **Step 3:** Preview start/stop drives runner through `build_preview_trajectory`
- [ ] **Step 4:** API tests with TestClient + tmp designs dir

---

### Task 5: Generate-from-web + wire `06` CLI path

**Files:**
- Modify: `webapp/app.py`, `static/web/js/design.js`
- Reuse: `data_gen.scripted.generate_scripted_dataset`
- Test: mock/short generate status transitions (`idle→running→done` / fail on existing dir)

- [ ] **Step 1:** `POST /api/design/generate` starts background thread; status endpoint reports progress logs
- [ ] **Step 2:** UI dialog for episodes/segment-steps/hold-steps/noise/overwrite/output-root/repo-id
- [ ] **Step 3:** Mode enters `generating` until complete; design edits locked

---

### Task 6: Docs touch-up

**Files:**
- Modify: `custom_envs/unoarm/README.md` (short section on design mode + JSON layout)

- [ ] **Step 1:** Document how to launch, design, save, generate
- [ ] **Step 2:** Point to `data/designs/` dual-file format

---

## Spec coverage check

| Spec requirement | Task |
|------------------|------|
| Keyframe library + playlist | 2, 4 |
| Mode switch exclusive | 3, 4 |
| Live 16D + keyboard | 4 |
| Simple preview | 2, 4 |
| Dual JSON save | 2, 4 |
| List / load / delete / generate | 4, 5 |
| Extract `06` module | 1, 5 |
| Modular layout, thin `09` | 3 |
| Scrollable chat composer fix already done | — |
