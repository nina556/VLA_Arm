# Table-Place Scene (Phase 1) Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `table_place` MuJoCo scene (table + peg + circle) and Web settings scene switch so the UI can load it without sword/shield.

**Architecture:** New scene constant + generated/static MJCF from free-space XML; env branches for peg kinematic attach and place metrics; runner/settings/Three.js payload for visualization. Reach-IK pick-place is Phase 1.5 (out of this plan).

**Tech Stack:** MuJoCo MJCF, Gymnasium `UnoarmEnv`, FastAPI web runner, static JS settings + Three.js scene.

**Spec:** `docs/superpowers/specs/2026-07-29-unoarm-table-place-scene-design.md`

## Global Constraints

- Scene id: `table_place` (not bolted onto `reach_sword`)
- No sword/shield bodies in this scene
- Vertical grasp convention documented; IK datagen deferred to Phase 1.5
- Keep obs schema: top RGB + pointmap + 16D state
- Circle fixed; peg position configurable/default
- Do not commit unless user asks

---

### Task 1: Constants + MJCF table_place

**Files:**
- Modify: `custom_envs/unoarm/gym_unoarm/constants.py`
- Create: `custom_envs/unoarm/gym_unoarm/table_place_scene.py`
- Create/generate: `custom_envs/unoarm/gym_unoarm/mujoco_unoarm_table_place.xml` (via ensure helper)

**Produces:** `SCENE_TABLE_PLACE`, table/peg/circle defaults, `ensure_table_place_xml()`, `TABLE_PLACE_XML_PATH`

- [ ] Add scene constants and default poses (table top Z, peg size, circle center/radius)
- [ ] Implement XML builder: include robot from free-space, add table box, peg cylinder, target circle sites
- [ ] Unit test: XML exists and loads in MuJoCo

---

### Task 2: Env support for table_place

**Files:**
- Modify: `custom_envs/unoarm/gym_unoarm/env.py`
- Create/Modify: `custom_envs/unoarm/tests/test_table_place_env.py`

**Produces:** peg attach/release, `place_distance` / `place_fit` in info, scene payload helpers

- [ ] Load table_place XML when `scene=table_place`
- [ ] Peg attach when near grasp site + right gripper closed
- [ ] Release places peg on table plane at current XY
- [ ] Tests for attach and place metrics

---

### Task 3: Runner + settings + Web UI

**Files:**
- Modify: `custom_envs/unoarm/webapp/runner.py`, `settings_store.py`, `app.py` (if needed)
- Modify: `custom_envs/unoarm/static/web/index.html`, settings JS, `scene.js`, `main.js`/`reach_ik.js` hints

**Produces:** settings can select `table_place`; rebuild env; Three.js draws table/peg/circle; sword UI gated

- [ ] Accept `table_place` in scene validation
- [ ] `_scene_payload_locked` for table_place
- [ ] Settings dropdown +「场景替换」copy; hide sword options when table_place
- [ ] Three.js applySceneConfig branch
- [ ] Reach-IK tab: short notice that pick-place datagen comes next

---

### Task 4: Smoke

- [ ] `pytest` table_place tests pass
- [ ] Manual: load scene via settings (document command)

Phase 1.5 (Reach-IK pick-place) is a separate plan after Phase 1 works.
