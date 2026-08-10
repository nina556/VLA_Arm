from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry

from gym_unoarm.constants import SCENE_REACH_SWORD

ENV_ID = "gym_unoarm/UnoarmFreeSpace-v0"
REACH_SWORD_ENV_ID = "gym_unoarm/UnoarmReachSword-v0"


def register_envs() -> None:
    if ENV_ID not in registry:
        gym.register(
            id=ENV_ID,
            entry_point="gym_unoarm.env:UnoarmEnv",
            max_episode_steps=300,
            nondeterministic=True,
        )
    if REACH_SWORD_ENV_ID not in registry:
        gym.register(
            id=REACH_SWORD_ENV_ID,
            entry_point="gym_unoarm.env:UnoarmEnv",
            max_episode_steps=300,
            nondeterministic=True,
            kwargs={"scene": SCENE_REACH_SWORD, "task": "UnoarmReachSword-v0"},
        )


register_envs()
