from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

UNOARM_IMAGE_SHAPE = (480, 640, 3)
UNOARM_STATE_SHAPE = (16,)


@EnvConfig.register_subclass("unoarm")
@dataclass
class UnoarmEnv(EnvConfig):
    task: str | None = "UnoarmFreeSpace-v0"
    fps: int = 20
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    observation_height: int = 480
    observation_width: int = 640
    render_mode: str = "rgb_array"
    disable_env_checker: bool = True
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=UNOARM_STATE_SHAPE),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "pixels/top": f"{OBS_IMAGES}.top",
            "pixels/left_wrist": f"{OBS_IMAGES}.left_wrist",
            "pixels/right_wrist": f"{OBS_IMAGES}.right_wrist",
        }
    )

    def __post_init__(self) -> None:
        image_shape = (self.observation_height, self.observation_width, 3)
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=UNOARM_STATE_SHAPE)
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
            self.features["pixels/left_wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
            self.features["pixels/right_wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
        elif self.obs_type == "pixels":
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
            self.features["pixels/left_wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
            self.features["pixels/right_wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "task": self.task,
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "observation_width": self.observation_width,
            "observation_height": self.observation_height,
            "max_episode_steps": self.episode_length,
        }
