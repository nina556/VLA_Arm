# Switch Unoarm Web to Multi-Task Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the Unoarm web console (`09_web_interact.py`) to load the newly trained multi-task checkpoint and recognize both "Prepare to fight and taunt." and "Raise your arms and stretch them wide.".

**Architecture:** The web script imports its defaults (`DEFAULT_CHECKPOINT`, `DEFAULT_ALLOWED_TASKS`) and the LLM chat router from `08_interact.py`. We therefore update the shared defaults and system prompt in `08_interact.py`; `09_web_interact.py` will inherit the new behavior automatically.

**Tech Stack:** Python 3.11+, LeRobot SmolVLA, FastAPI, OpenAI-compatible SiliconFlow router.

## Global Constraints

- Target file paths are on WSL2 (`\\wsl.localhost\Ubuntu-24.04\home\doki\project\lerobot-main\...`).
- The new checkpoint directory is `data/outputs/mutiTask/10000步权重/pretrained_model` (verified to contain `pretrained_model/` and `training_state/`).
- Allowed task strings must match the exact instructions used during SmolVLA training.
- Keep the existing trailing-period convention for task strings.
- Do not break `08_interact.py` CLI usage; both scripts share the updated defaults.

---

## File Structure

| File                                            | Responsibility                    | Change                                                                           |
| ----------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------- |
| `custom_envs/unoarm/scripts/08_interact.py`     | Shared defaults + LLM chat router | Update `DEFAULT_CHECKPOINT`, `DEFAULT_ALLOWED_TASKS`, and system-prompt examples |
| `custom_envs/unoarm/scripts/09_web_interact.py` | Web console                       | No code changes required; inherits new defaults from `08_interact.py`            |

---

### Task 1: Update Default Checkpoint Path

**Files:**

- Modify: `custom_envs/unoarm/scripts/08_interact.py:69-71`

**Interfaces:**

- Consumes: `ROOT` resolves to `lerobot-main/custom_envs/unoarm`
- Produces: `DEFAULT_CHECKPOINT` now points to `lerobot-main/data/outputs/mutiTask/10000步权重/pretrained_model`

- [ ] **Step 1: Change the default checkpoint constant**

```python
DEFAULT_CHECKPOINT = (
    ROOT.parent.parent / "data" / "outputs" / "mutiTask" / "10000步权重" / "pretrained_model"
)
```

- [ ] **Step 2: Verify the path exists**

Run:

```bash
wsl -d Ubuntu-24.04 test -d /home/doki/project/lerobot-main/data/outputs/mutiTask/10000步权重/pretrained_model && echo "OK"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add custom_envs/unoarm/scripts/08_interact.py
git commit -m "chore: switch default checkpoint to mutiTask 10000-step weights"
```

---

### Task 2: Add the New Allowed Task

**Files:**

- Modify: `custom_envs/unoarm/scripts/08_interact.py:100`

**Interfaces:**

- Consumes: `DEFAULT_ALLOWED_TASKS` tuple
- Produces: tuple now contains both trained task instructions

- [ ] **Step 1: Extend the allowed-tasks tuple**

```python
DEFAULT_ALLOWED_TASKS = (
    "Prepare to fight and taunt.",
    "Raise your arms and stretch them wide.",
)
```

- [ ] **Step 2: Validate the tuple syntax**

Run:

```bash
wsl -d Ubuntu-24.04 cd /home/doki/project/lerobot-main && python -c "from custom_envs.unoarm.scripts import 08_interact as m; print(m.DEFAULT_ALLOWED_TASKS)"
```

Expected: `('Prepare to fight and taunt.', 'Raise your arms and stretch them wide.')`

- [ ] **Step 3: Commit**

```bash
git add custom_envs/unoarm/scripts/08_interact.py
git commit -m "feat: add raise-arms task to default allowed tasks"
```

---

### Task 3: Update Router System Prompt Examples

**Files:**

- Modify: `custom_envs/unoarm/scripts/08_interact.py:144-165`

**Interfaces:**

- Consumes: `_build_system_prompt(allowed_tasks)` receives the updated `allowed_tasks`
- Produces: system prompt includes an example for the new task

- [ ] **Step 1: Add a routing example for the new task**

Inside `_build_system_prompt`, update the `示例:` block to:

```python
示例：
- 用户说 "hello"：{{"route":"chat","reply":"你好！我可以陪你聊天，也可以执行已经训练过的动作。"}}
- 用户说 "做出格斗姿势并挑衅一下"：{{"route":"action","instruction":"Prepare to fight and taunt."}}
- 用户说 "张开双臂伸展"：{{"route":"action","instruction":"Raise your arms and stretch them wide."}}
- 用户说 "挥挥手"：{{"route":"chat","reply":"这个动作我现在还没训练过，先不让机器人执行。"}}
"""
```

- [ ] **Step 2: Smoke-test the router with the new task**

Run:

```bash
wsl -d Ubuntu-24.04 cd /home/doki/project/lerobot-main && python custom_envs/unoarm/scripts/08_interact.py --llm-smoke-test --llm-smoke-text "张开双臂伸展"
```

Expected output contains: `ACTION -> Raise your arms and stretch them wide.`

- [ ] **Step 3: Commit**

```bash
git add custom_envs/unoarm/scripts/08_interact.py
git commit -m "docs(router): add example for raise-arms task in system prompt"
```

---

### Task 4: Validate Web Console Inherits Changes

**Files:**

- Read-only: `custom_envs/unoarm/scripts/09_web_interact.py:1275-1327`

**Interfaces:**

- Consumes: `interact.DEFAULT_CHECKPOINT` and `interact.DEFAULT_ALLOWED_TASKS`
- Produces: confirmation that `09_web_interact.py` will use the new defaults

- [ ] **Step 1: Verify argparse defaults reference the shared constants**

Run:

```bash
wsl -d Ubuntu-24.04 cd /home/doki/project/lerobot-main && python -c "from custom_envs.unoarm.scripts import 08_interact as m, 09_web_interact as w; print('checkpoint default:', w.parse_args.__defaults__); print('allowed defaults:', m.DEFAULT_ALLOWED_TASKS)"
```

Expected: checkpoint default resolves to the `mutiTask/10000步权重/pretrained_model` path and allowed tasks include both instructions.

- [ ] **Step 2: Confirm the web script can load**

Run:

```bash
wsl -d Ubuntu-24.04 cd /home/doki/project/lerobot-main && python custom_envs/unoarm/scripts/09_web_interact.py --help
```

Expected: help text prints without errors and `--checkpoint` default shows the new path.

- [ ] **Step 3: Final commit (if any doc/validation artifacts were added)**

No additional commit needed for read-only validation.

---

## Self-Review

**1. Spec coverage:**

- Use new model checkpoint → Task 1 updates `DEFAULT_CHECKPOINT`.
- Support both tasks → Task 2 updates `DEFAULT_ALLOWED_TASKS`.
- Web console uses the changes → Task 4 confirms `09_web_interact.py` inherits them.

**2. Placeholder scan:**

- No TBD/TODO placeholders.
- All code blocks contain the exact strings to use.
- All commands include expected output.

**3. Type consistency:**

- `DEFAULT_CHECKPOINT` remains a `pathlib.Path`.
- `DEFAULT_ALLOWED_TASKS` remains a `tuple[str, ...]`.
- `_build_system_prompt` signature is unchanged.
