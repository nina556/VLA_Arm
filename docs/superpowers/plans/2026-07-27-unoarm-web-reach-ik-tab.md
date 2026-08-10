# Web Reach-IK Tab Implementation Plan

> **For agentic workers:** Inline execution in this session (user: 开始吧).

**Goal:** Add Web tab for reach-IK generate (JSON|AABB) + 3D episode replay.

**Architecture:** Overlay panel like validate; runner threads for generate/replay; reuse `generate_reach_ik_dataset`; write `reach_ik_meta.json` for sword placement on replay.

**Tech Stack:** FastAPI, existing WS snapshot, Three.js scene, LeRobotDataset.

## Global Constraints

- Overlay tab (not new AppMode); mutually exclusive with validate/generate/rollout
- Target input: JSON | AABB toggle
- Replay: left 3D only via env.step(action)
- Skip git commits unless user asks

### Tasks

1. Meta write in `data_gen/reach_ik.py` + list/info helpers
2. Runner: generate + replay APIs
3. `app.py` routes
4. Frontend: panel + `reach_ik.js` + `main.js` wiring

Execute immediately.
