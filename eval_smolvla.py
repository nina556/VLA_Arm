"""
带语言指令的 SmolVLA Aloha 评估脚本
绕过 Aloha env 不提供 task_description 的问题（手动注入）

用法：
    uv run python eval_smolvla.py

如需改 checkpoint / 语言指令 / 是否开画面，改下方 CONFIG 区即可。
"""

import numpy as np
import torch

# ====== CONFIG 区（按需修改）======
CHECKPOINT = "outputs/train/2026-07-21/11-06-27_aloha_smolvla/checkpoints/last/pretrained_model"
TASK_DESC = "Pick up the cube with the right arm and transfer it to the left arm."
N_EPISODES = 10
RENDER_MODE = "human"  # "human" 实时画面 / "rgb_array" 不开窗口（只看成功率）
SEED_BASE = 1000
# ==================================


def main():
    print("=" * 60)
    print(f"加载模型 : {CHECKPOINT}")
    print(f'语言指令 : "{TASK_DESC}"')
    print(f"episodes : {N_EPISODES}")
    print(f"render   : {RENDER_MODE}")
    print("=" * 60)
    print()

    # 1. 加载 policy + processor
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT)
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    print(f"policy 加载完成，device={device}")

    # processor 加载：优先用 checkpoint 里训练时保存的 processor
    # 第二个参数是 pretrained_path（不是 model_id），传 None 会从零构造
    try:
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            pretrained_path=CHECKPOINT,  # 用 checkpoint 里训练时保存的 processor
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        print("processor 加载完成（来自 checkpoint）")
    except Exception as e:
        print(f"从 checkpoint 加载 processor 失败：{e}")
        print("退而从零构造 processor...")
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            pretrained_path=None,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        print("processor 从零构造完成")

    # 2. Aloha env 的 feature 定义（14 维 action/state + 顶视相机 480x640）
    # 格式必须符合 LeRobot 标准（见 hw_to_dataset_features）：
    #   - names 是 list（不是 dict）
    #   - state/action: dtype="float32", names=[各 motor 名]
    #   - 图像: dtype="video", names=["height", "width", "channels"]
    motor_names = [f"motor_{i}" for i in range(14)]
    dataset_features = {
        "action": {
            "shape": (14,),
            "dtype": "float32",
            "names": motor_names,
        },
        "observation.state": {
            "shape": (14,),
            "dtype": "float32",
            "names": motor_names,
        },
        "observation.images.top": {
            "shape": (480, 640, 3),
            "dtype": "video",
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
    }
    robot_type = "aloha"

    # 3. 创建环境
    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        render_mode=RENDER_MODE,
        obs_type="pixels_agent_pos",
    )
    print(f"env 创建完成: {env.spec.id}")
    print()

    # 4. 跑 N 个 episode
    success_count = 0
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=SEED_BASE + ep)
        terminated = truncated = False
        step = 0
        ep_reward = 0.0

        print(f"--- Episode {ep + 1}/{N_EPISODES} 开始 ---")

        while not (terminated or truncated):
            # gym obs -> LeRobot 格式
            # build_dataset_frame 会按 motor 名从 values 字典里取值，
            # 所以 observation.state 要传 {"motor_0": v0, "motor_1": v1, ...}
            state_np = np.asarray(obs["agent_pos"], dtype=np.float32)
            lerobot_obs = {f"motor_{i}": float(state_np[i]) for i in range(14)}
            # 图像在 build_dataset_frame 里通过 values[key.removeprefix("observation.images.")] 取
            # 即 values["top"]，所以放在 "top" key 下
            lerobot_obs["top"] = np.asarray(obs["pixels"]["top"])

            # 关键：build_inference_frame 时手动注入 task=TASK_DESC
            obs_frame = build_inference_frame(
                observation=lerobot_obs,
                ds_features=dataset_features,
                device=device,
                task=TASK_DESC,
                robot_type=robot_type,
            )

            # 前处理 -> 推理 -> 后处理
            processed = preprocess(obs_frame)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = postprocess(action)

            # tensor -> numpy 给 env
            if isinstance(action, torch.Tensor):
                action_np = action.squeeze(0).cpu().numpy()
            else:
                action_np = np.asarray(action).squeeze(0)

            # 兼容 action chunk：只取第一个 action（当前步）
            if action_np.ndim >= 1 and action_np.shape[0] > 14:
                action_np = action_np[:14]

            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += float(reward)
            step += 1

        success = bool(info.get("success", False)) or ep_reward > 0
        print(f"--- Episode {ep + 1}: steps={step}, reward={ep_reward:.3f}, success={success} ---")
        print()
        if success:
            success_count += 1

    # 5. 总结
    print("=" * 60)
    print("=== 总结 ===")
    print(f"成功率: {success_count}/{N_EPISODES} = {success_count / max(N_EPISODES, 1) * 100:.1f}%")
    print("=" * 60)

    env.close()


# 延迟 import：让 CONFIG 区的 print 先输出，import 报错也更清楚
import gym_aloha  # noqa: F401  触发 gym_aloha namespace 注册
import gymnasium as gym

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame

if __name__ == "__main__":
    main()
