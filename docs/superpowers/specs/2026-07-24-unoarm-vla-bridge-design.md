# Unoarm VLA → MuJoCo Bridge (外部关节 chunk)

## Goal

Web 对话触发 VLA 后，在本地 MuJoCo 镜像执行的同时，把**原始关节弧度**按 chunk POST 到外部桥接服务。

详见完整迁移说明：`2026-07-24-unoarm-vla-bridge-port.md`

## Defaults

- Base URL: `http://192.168.10.38:8765`
- Endpoint: `POST /api/mujoco/vla_joint_chunk`
- Mode A: local `env.step` + remote send
- Chunk size: `n_action_steps`
- Payload: raw `env._current_control_qpos()` radians（不做 [-1,1] 反归一化）
- Settings UI + `web_settings.json` + `/api/bridge`
