"""Simulation / policy / chat / design runner for the Unoarm web console."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent.parent
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gym_unoarm.constants import (  # noqa: E402
    CAMERAS,
    CONTROL_JOINTS,
    DEFAULT_EXECUTION_POINT_POS,
    FPS,
    HELMET_BODY_POS,
    HELMET_BODY_QUAT,
    HELMET_HANDLE_LOCAL,
    HELMET_MESH_SCALE,
    HELMET_THREE_COLOR,
    LEFT_TCP_LOCAL,
    PEG_HALF_HEIGHT,
    PEG_RADIUS,
    PEG_SAMPLE_EDGE_MARGIN,
    PEG_THREE_COLOR,
    PEG_WORKSPACE_X_RANGE,
    PEG_WORKSPACE_Y_RANGE,
    RIGHT_TCP_LOCAL,
    SCENE_FREE_SPACE,
    SCENE_REACH_SWORD,
    SCENE_TABLE_PLACE,
    START_POSE,
    SWORD_HANDLE_LOCAL,
    SWORD_MESH_SCALE,
    SWORD_THREE_COLOR,
    TABLE_CIRCLE_RADIUS,
    TABLE_CIRCLE_THREE_COLOR,
    TABLE_PLACE_HALF,
    TABLE_PLACE_POS,
    TABLE_PLACE_THREE_COLOR,
    TASK_REACH_SWORD,
    TASK_TABLE_PLACE,
    sample_peg_xy,
)
from gym_unoarm.env import UnoarmEnv  # noqa: E402
from gym_unoarm.sword_pose import (  # noqa: E402
    default_shield_euler_deg,
    default_shield_handle_pos,
    default_sword_euler_deg,
    default_sword_handle_pos,
    resolve_shield_pose,
    resolve_sword_pose,
)
from pose_design.export import to_poses_payload  # noqa: E402
from pose_design.models import (  # noqa: E402
    DesignProject,
    normalize_pose_dict,
    pose_dict_from_vector,
    slugify_name,
    validate_project,
    vector_from_pose_dict,
)
from pose_design.preview import build_preview_trajectory  # noqa: E402
from pose_design.store import DesignStore  # noqa: E402
from data_gen.scripted import ScriptedGenConfig, generate_scripted_dataset  # noqa: E402
from data_gen.reach_ik import (  # noqa: E402
    DATA_ROOT as UNOARM_DATA_ROOT,
    ReachIkGenConfig,
    dataset_episode_count,
    generate_reach_ik_dataset,
    list_reach_ik_datasets,
    load_reach_ik_meta,
    place_sword_handle,
)
from data_gen.table_place_ik import (  # noqa: E402
    episode_peg_xy_from_meta,
    generate_table_place_ik_dataset,
    load_table_place_ik_meta,
    table_place_ik_config_from_params,
)
from webapp.modes import AppMode, ModeController, ModeError  # noqa: E402
from webapp.vla_bridge_client import BridgeConfig, VlaBridgeClient  # noqa: E402
from webapp.llm_router import ActionDecision, ChatRouter  # noqa: E402
from webapp.settings_store import (  # noqa: E402
    DEFAULT_SETTINGS_PATH,
    load_settings,
    merge_settings_patch,
    public_settings,
    save_settings,
)


def load_interact_module():
    interact_path = SCRIPTS / "08_interact.py"
    spec = importlib.util.spec_from_file_location("unoarm_interact_cli", interact_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {interact_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


interact = load_interact_module()

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.act import ACTPolicy  # noqa: E402

# Map policy config "type" -> policy class. Used to auto-dispatch at load time
# so the web UI works with any trained checkpoint without hardcoding.
_POLICY_REGISTRY = {
    "act": ACTPolicy,
}


DEFAULT_VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 7860
    checkpoint: Path | None = None
    max_steps: int = 150
    speed: float = 1.0
    device: str | None = None
    stream_fps: float = 12.0
    display_width: int = 1280
    display_height: int = 720
    jpeg_quality: int = 90
    llm_model: str = "THUDM/GLM-4-9B-0414"
    llm_timeout: float = 20.0
    api_key: str | None = None
    api_key_env: str = "SILICONFLOW_API_KEY"
    api_base_url: str = "https://api.siliconflow.cn/v1"
    allowed_tasks: list[str] = field(default_factory=list)
    quiet_router_logs: bool = False
    designs_dir: Path = field(default_factory=lambda: ROOT / "data" / "designs")
    vlm_model_name: str | None = None
    n_action_steps: int = 15
    action_ema: float = 0.0
    max_action_delta: float = 0.0
    scene: str = SCENE_REACH_SWORD
    remove_sword: bool = False
    remove_shield: bool = False
    enable_execution_point: bool = False
    execution_point_pos: tuple[float, float, float] = field(
        default_factory=lambda: tuple(DEFAULT_EXECUTION_POINT_POS)  # type: ignore[arg-type]
    )
    terminate_on_success: bool = False
    reach_success_threshold: float = 0.05
    sword_handle_pos: tuple[float, float, float] = field(
        default_factory=lambda: tuple(default_sword_handle_pos())  # type: ignore[arg-type]
    )
    sword_euler_deg: tuple[float, float, float] = field(
        default_factory=lambda: tuple(default_sword_euler_deg())  # type: ignore[arg-type]
    )
    shield_handle_pos: tuple[float, float, float] = field(
        default_factory=lambda: tuple(default_shield_handle_pos())  # type: ignore[arg-type]
    )
    shield_euler_deg: tuple[float, float, float] = field(
        default_factory=lambda: tuple(default_shield_euler_deg())  # type: ignore[arg-type]
    )
    settings_path: Path = field(default_factory=lambda: DEFAULT_SETTINGS_PATH)
    bridge_enabled: bool = False
    bridge_base_url: str = "http://192.168.10.38:8765"
    bridge_arms: str = "both"
    bridge_execute: bool = True
    bridge_result_timeout_sec: float = 5.0
    bridge_http_timeout_sec: float = 8.0

    def resolved_sword_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float]]:
        return resolve_sword_pose(
            handle_pos=self.sword_handle_pos,
            euler_deg=self.sword_euler_deg,
        )

    def resolved_shield_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float]]:
        return resolve_shield_pose(
            handle_pos=self.shield_handle_pos,
            euler_deg=self.shield_euler_deg,
        )


def webconfig_from_settings(
    settings: dict[str, Any],
    *,
    host: str = "0.0.0.0",
    port: int = 7860,
    settings_path: Path | None = None,
) -> WebConfig:
    ckpt = str(settings.get("checkpoint") or "").strip()
    device = str(settings.get("device") or "").strip() or None
    vlm = str(settings.get("vlm_model_name") or "").strip() or None
    api_key = str(settings.get("api_key") or "").strip() or None
    designs = Path(str(settings.get("designs_dir") or (ROOT / "data" / "designs")))
    tasks = list(settings.get("allowed_tasks") or [])
    handle = tuple(float(x) for x in (settings.get("sword_handle_pos") or default_sword_handle_pos()))
    if len(handle) != 3:
        handle = tuple(default_sword_handle_pos())
    euler = tuple(float(x) for x in (settings.get("sword_euler_deg") or default_sword_euler_deg()))
    if len(euler) != 3:
        euler = tuple(default_sword_euler_deg())
    shield_handle = tuple(
        float(x) for x in (settings.get("shield_handle_pos") or default_shield_handle_pos())
    )
    if len(shield_handle) != 3:
        shield_handle = tuple(default_shield_handle_pos())
    shield_euler = tuple(
        float(x) for x in (settings.get("shield_euler_deg") or default_shield_euler_deg())
    )
    if len(shield_euler) != 3:
        shield_euler = tuple(default_shield_euler_deg())
    exec_pos = tuple(
        float(x) for x in (settings.get("execution_point_pos") or DEFAULT_EXECUTION_POINT_POS)
    )
    if len(exec_pos) != 3:
        exec_pos = tuple(DEFAULT_EXECUTION_POINT_POS)
    return WebConfig(
        host=host,
        port=port,
        checkpoint=Path(ckpt) if ckpt else None,
        max_steps=int(settings.get("max_steps", 150)),
        speed=float(settings.get("speed", 1.0)),
        device=device,
        stream_fps=float(settings.get("stream_fps", 12.0)),
        display_width=int(settings.get("display_width", 1280)),
        display_height=int(settings.get("display_height", 720)),
        jpeg_quality=int(settings.get("jpeg_quality", 90)),
        llm_model=str(settings.get("llm_model") or "THUDM/GLM-4-9B-0414"),
        llm_timeout=float(settings.get("llm_timeout", 20.0)),
        api_key=api_key,
        api_key_env=str(settings.get("api_key_env") or "SILICONFLOW_API_KEY"),
        api_base_url=str(settings.get("api_base_url") or "https://api.siliconflow.cn/v1"),
        allowed_tasks=tasks,
        quiet_router_logs=bool(settings.get("quiet_router_logs", False)),
        designs_dir=designs,
        vlm_model_name=vlm,
        n_action_steps=int(settings.get("n_action_steps", 15)),
        action_ema=float(settings.get("action_ema", 0.4)),
        max_action_delta=float(settings.get("max_action_delta", 0.08)),
        scene=str(settings.get("scene") or SCENE_FREE_SPACE),
        remove_sword=bool(settings.get("remove_sword", False)),
        remove_shield=bool(settings.get("remove_shield", False)),
        enable_execution_point=bool(settings.get("enable_execution_point", False)),
        execution_point_pos=exec_pos,  # type: ignore[arg-type]
        terminate_on_success=bool(settings.get("terminate_on_success", False)),
        reach_success_threshold=float(settings.get("reach_success_threshold", 0.05)),
        sword_handle_pos=handle,  # type: ignore[arg-type]
        sword_euler_deg=euler,  # type: ignore[arg-type]
        shield_handle_pos=shield_handle,  # type: ignore[arg-type]
        shield_euler_deg=shield_euler,  # type: ignore[arg-type]
        settings_path=Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS_PATH,
        bridge_enabled=bool(settings.get("bridge_enabled", False)),
        bridge_base_url=str(settings.get("bridge_base_url") or "http://192.168.10.38:8765").strip(),
        bridge_arms=str(settings.get("bridge_arms") or "both").strip().lower(),
        bridge_execute=bool(settings.get("bridge_execute", True)),
        bridge_result_timeout_sec=float(settings.get("bridge_result_timeout_sec", 5.0)),
        bridge_http_timeout_sec=float(settings.get("bridge_http_timeout_sec", 8.0)),
    )


def settings_from_webconfig(cfg: WebConfig) -> dict[str, Any]:
    return {
        "checkpoint": str(cfg.checkpoint) if cfg.checkpoint else "",
        "vlm_model_name": cfg.vlm_model_name or "",
        "device": cfg.device or "",
        "llm_model": cfg.llm_model,
        "api_key": cfg.api_key or "",
        "api_key_env": cfg.api_key_env,
        "api_base_url": cfg.api_base_url,
        "llm_timeout": cfg.llm_timeout,
        "allowed_tasks": list(cfg.allowed_tasks),
        "scene": cfg.scene,
        "remove_sword": cfg.remove_sword,
        "remove_shield": cfg.remove_shield,
        "enable_execution_point": cfg.enable_execution_point,
        "execution_point_pos": list(cfg.execution_point_pos),
        "max_steps": cfg.max_steps,
        "n_action_steps": cfg.n_action_steps,
        "action_ema": cfg.action_ema,
        "max_action_delta": cfg.max_action_delta,
        "terminate_on_success": cfg.terminate_on_success,
        "reach_success_threshold": cfg.reach_success_threshold,
        "sword_handle_pos": list(cfg.sword_handle_pos),
        "sword_euler_deg": list(cfg.sword_euler_deg),
        "shield_handle_pos": list(cfg.shield_handle_pos),
        "shield_euler_deg": list(cfg.shield_euler_deg),
        "speed": cfg.speed,
        "stream_fps": cfg.stream_fps,
        "display_width": cfg.display_width,
        "display_height": cfg.display_height,
        "jpeg_quality": cfg.jpeg_quality,
        "quiet_router_logs": cfg.quiet_router_logs,
        "designs_dir": str(cfg.designs_dir),
        "bridge_enabled": cfg.bridge_enabled,
        "bridge_base_url": cfg.bridge_base_url,
        "bridge_arms": cfg.bridge_arms,
        "bridge_execute": cfg.bridge_execute,
        "bridge_result_timeout_sec": cfg.bridge_result_timeout_sec,
        "bridge_http_timeout_sec": cfg.bridge_http_timeout_sec,
    }


@dataclass
class CameraRequest:
    delta_azimuth: float = 0.0
    delta_elevation: float = 0.0
    delta_distance: float = 0.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    reset: bool = False


@dataclass
class GenerateStatus:
    state: str = "idle"  # idle | running | done | failed
    message: str = ""
    logs: list[str] = field(default_factory=list)
    output_root: str = ""


@dataclass
class ReplayStatus:
    state: str = "idle"  # idle | playing | paused | done | error
    message: str = ""
    root: str = ""
    episode: int = 0
    n_episodes: int = 0
    frame: int = 0
    n_frames: int = 0
    paused: bool = False
    handle: list[float] | None = None
    peg_xy: list[float] | None = None


@dataclass
class ValidateStatus:
    state: str = "idle"  # idle | running | done | error
    message: str = ""
    episode: int = 0
    episodes: int = 0
    successes: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    current_handle: list[float] | None = None
    current_shield_handle: list[float] | None = None
    task: str = ""
    seed: int = 0


class UnoarmWebRunner:
    def __init__(self, cfg: WebConfig) -> None:
        self.cfg = cfg
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.preview_stop = threading.Event()
        self.validate_stop = threading.Event()
        self.rollout_thread: threading.Thread | None = None
        self.preview_thread: threading.Thread | None = None
        self.generate_thread: threading.Thread | None = None
        self.validate_thread: threading.Thread | None = None
        self.reach_ik_gen_thread: threading.Thread | None = None
        self.reach_ik_replay_thread: threading.Thread | None = None
        self.reach_ik_replay_stop = threading.Event()
        self.reach_ik_replay_pause = threading.Event()
        self.status = "loading"
        self.current_task = ""
        self.step = 0
        self.logs: list[dict[str, str]] = []
        self.obs: dict[str, Any] | None = None
        self.camera_lookat = np.array([0.0, 0.0, 1.2], dtype=np.float32)
        self.camera_azimuth = 120.0
        self.camera_elevation = -20.0
        self.camera_distance = 3.5
        self.display_renderer: mujoco.Renderer | None = None
        self.obs_preview_renderer: mujoco.Renderer | None = None
        self._obs_preview_width = 320
        self._obs_preview_height = 240
        self._obs_preview_jpeg_quality = 72
        self.obs_preview_enabled = True
        self._eval_handle_override: list[float] | None = None
        self._eval_euler_override: list[float] | None = None
        self._eval_shield_handle_override: list[float] | None = None
        self._eval_shield_euler_override: list[float] | None = None

        self.modes = ModeController()
        self.store = DesignStore(cfg.designs_dir)
        self.project = DesignProject(name="untitled", task="")
        self.dirty = False
        self.generate_status = GenerateStatus()
        self.reach_ik_gen_status = GenerateStatus()
        self.reach_ik_replay_status = ReplayStatus()
        self.validate_status = ValidateStatus()
        self._joint_limits = None  # filled after env init
        self.bridge = VlaBridgeClient(
            BridgeConfig(
                enabled=cfg.bridge_enabled,
                base_url=cfg.bridge_base_url,
                arms=cfg.bridge_arms,
                fps=float(FPS),
                execute=cfg.bridge_execute,
                result_timeout_sec=cfg.bridge_result_timeout_sec,
                http_timeout_sec=cfg.bridge_http_timeout_sec,
            )
        )

        self.policy = None
        self.preprocess = None
        self.postprocess = None
        self.policy_ready = False
        self.policy_error = ""
        self.router = None
        self.router_error = ""
        self.device = torch.device(
            cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._build_env()
        self._build_router()
        self._try_load_policy()
        self.reset()
        if self.bridge.cfg.enabled:
            self._log(
                f"vla bridge ON -> {self.bridge.endpoint} "
                f"(arms={self.bridge.cfg.arms}, execute={self.bridge.cfg.execute})"
            )
        else:
            self._log("vla bridge OFF (enable in 设置)")
        self._log("web runner ready (configure settings in the Web UI if needed)")

    def _build_env(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            try:
                if self.display_renderer is not None:
                    self.display_renderer.close()
                    self.display_renderer = None
                if self.obs_preview_renderer is not None:
                    self.obs_preview_renderer.close()
                    self.obs_preview_renderer = None
                self.env.close()
            except Exception as exc:
                self._log(f"env close warning: {exc}")
        body_pos = None
        body_quat = None
        shield_pos = None
        shield_quat = None
        if self.cfg.scene == SCENE_REACH_SWORD:
            body_pos, body_quat, handle = self.cfg.resolved_sword_pose()
            shield_pos, shield_quat, shield_handle = self.cfg.resolved_shield_pose()
            self._log(
                f"sword handle={handle} euler_deg={self.cfg.sword_euler_deg} "
                f"body_pos={body_pos}"
            )
            self._log(
                f"shield handle={shield_handle} euler_deg={self.cfg.shield_euler_deg} "
                f"body_pos={shield_pos}"
            )
        self.env = UnoarmEnv(
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            max_episode_steps=self.cfg.max_steps,
            scene=self.cfg.scene,
            reach_success_threshold=self.cfg.reach_success_threshold,
            terminate_on_success=self.cfg.terminate_on_success,
            sword_body_pos=body_pos,
            sword_body_quat=body_quat,
            shield_body_pos=shield_pos,
            shield_body_quat=shield_quat,
            remove_sword=self.cfg.remove_sword,
            remove_shield=self.cfg.remove_shield,
            enable_execution_point=self.cfg.enable_execution_point,
            execution_point_pos=self.cfg.execution_point_pos,
        )
        self._log(f"scene={self.cfg.scene}")
        if self.cfg.scene == SCENE_REACH_SWORD:
            self._log(f"reach task hint: {TASK_REACH_SWORD!r}")
        elif self.cfg.scene == SCENE_TABLE_PLACE:
            self._log(f"table-place task hint: {TASK_TABLE_PLACE!r}")
        self._joint_limits = self.env._joint_limits.copy()
        self.env.model.vis.global_.offwidth = max(
            int(self.env.model.vis.global_.offwidth),
            self.cfg.display_width,
            self._obs_preview_width,
        )
        self.env.model.vis.global_.offheight = max(
            int(self.env.model.vis.global_.offheight),
            self.cfg.display_height,
            self._obs_preview_height,
        )
        self._display_camera = mujoco.MjvCamera()

    def _build_router(self) -> None:
        """(Re)create the LLM intent router from current settings."""
        api_key = (
            self.cfg.api_key
            or os.environ.get(self.cfg.api_key_env)
            or ""
        ).strip()
        try:
            self.router = ChatRouter(
                model=self.cfg.llm_model,
                api_key=api_key,
                base_url=self.cfg.api_base_url,
                timeout_s=self.cfg.llm_timeout,
                verbose=not self.cfg.quiet_router_logs,
            )
            self.router_error = ""
            self._log(f"LLM router ready (model={self.cfg.llm_model})")
        except Exception as exc:
            self.router = None
            self.router_error = f"LLM 路由器未就绪：{exc}"
            self._log(self.router_error)

    def _try_load_policy(self) -> None:
        self.policy = None
        self.preprocess = None
        self.postprocess = None
        self.policy_ready = False
        self.policy_error = ""
        ckpt = self.cfg.checkpoint
        if ckpt is None or not str(ckpt).strip():
            self.policy_error = "未配置 checkpoint，请在「设置」中填写策略路径并保存。"
            self._log(self.policy_error)
            return
        ckpt = Path(ckpt)
        if not ckpt.exists():
            self.policy_error = f"Checkpoint 不存在: {ckpt}"
            self._log(self.policy_error)
            return
        try:
            self.device = torch.device(
                self.cfg.device
                if self.cfg.device
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._log(f"loading policy from {ckpt} on {self.device}")
            policy_config = PreTrainedConfig.from_pretrained(ckpt)
            policy_type = getattr(policy_config, "type", None) or getattr(
                policy_config, "name", ""
            ).lower()
            policy_cls = _POLICY_REGISTRY.get(policy_type)
            if policy_cls is None:
                raise ValueError(
                    f"Unsupported policy type {policy_type!r} in checkpoint. "
                    f"Supported: {list(_POLICY_REGISTRY.keys())}"
                )
            self._log(f"policy type: {policy_type} -> {policy_cls.__name__}")

            if self.cfg.n_action_steps < 1:
                raise ValueError(f"n_action_steps must be >= 1, got {self.cfg.n_action_steps}")
            chunk = int(getattr(policy_config, "chunk_size", self.cfg.n_action_steps))
            if self.cfg.n_action_steps > chunk:
                raise ValueError(
                    f"n_action_steps ({self.cfg.n_action_steps}) cannot exceed chunk_size ({chunk})"
                )
            policy_config.n_action_steps = self.cfg.n_action_steps
            self.policy = policy_cls.from_pretrained(ckpt, config=policy_config)
            self.policy.config.n_action_steps = self.cfg.n_action_steps
            self.policy.reset()
            self.policy.eval().to(self.device)
            processor_device_override = {"device_processor": {"device": str(self.device)}}
            self.preprocess, self.postprocess = make_pre_post_processors(
                self.policy.config,
                pretrained_path=str(ckpt),
                preprocessor_overrides=processor_device_override,
                postprocessor_overrides=processor_device_override,
            )
            self.policy_ready = True
            self.policy_error = ""
            self._log(
                f"inference: max_steps={self.cfg.max_steps}, "
                f"n_action_steps={self.policy.config.n_action_steps}, "
                f"action_ema={self.cfg.action_ema}, max_action_delta={self.cfg.max_action_delta}"
            )
        except Exception as exc:
            self.policy = None
            self.preprocess = None
            self.postprocess = None
            self.policy_ready = False
            self.policy_error = f"{type(exc).__name__}: {exc}"
            self._log(f"policy load failed: {self.policy_error}")

    def get_settings(self) -> dict[str, Any]:
        with self.lock:
            data = settings_from_webconfig(self.cfg)
            pub = public_settings(data)
            pub["policy_ready"] = self.policy_ready
            pub["policy_error"] = self.policy_error
            return pub

    def apply_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Merge UI settings, persist, and hot-reload router / env / policy as needed."""
        if self.is_running:
            raise RuntimeError("执行中无法保存设置，请先 Stop。")
        if self.is_validating:
            raise RuntimeError("验证中无法保存设置，请先停止验证。")
        if self.modes.mode == AppMode.GENERATING:
            raise RuntimeError("数据生成中无法保存设置。")
        self.stop_preview()

        with self.lock:
            current = settings_from_webconfig(self.cfg)
            merged = merge_settings_patch(current, patch)
            # Validate light fields
            if int(merged.get("max_steps", 1)) < 1:
                raise ValueError("max_steps must be >= 1")
            if int(merged.get("n_action_steps", 1)) < 1:
                raise ValueError("n_action_steps must be >= 1")
            if not (0.0 <= float(merged.get("action_ema", 0.0)) <= 1.0):
                raise ValueError("action_ema must be in [0, 1]")
            if float(merged.get("max_action_delta", 0.0)) < 0.0:
                raise ValueError("max_action_delta must be >= 0")
            if float(merged.get("reach_success_threshold", 0.05)) <= 0.0:
                raise ValueError("reach_success_threshold must be > 0")
            scene = str(merged.get("scene") or SCENE_FREE_SPACE)
            if scene not in (SCENE_FREE_SPACE, SCENE_REACH_SWORD, SCENE_TABLE_PLACE):
                raise ValueError(f"Unknown scene: {scene}")
            tasks = list(merged.get("allowed_tasks") or [])
            if not tasks:
                raise ValueError("allowed_tasks 至少保留一条任务指令")
            arms = str(merged.get("bridge_arms") or "both").strip().lower()
            if arms not in ("right", "left", "both"):
                raise ValueError("bridge_arms must be right|left|both")
            if float(merged.get("bridge_result_timeout_sec", 5.0)) <= 0:
                raise ValueError("bridge_result_timeout_sec must be > 0")
            if float(merged.get("bridge_http_timeout_sec", 8.0)) <= 0:
                raise ValueError("bridge_http_timeout_sec must be > 0")

            old_scene = self.cfg.scene
            old_ckpt = str(self.cfg.checkpoint) if self.cfg.checkpoint else ""
            old_vlm = self.cfg.vlm_model_name or ""
            old_device = self.cfg.device or ""
            old_n = self.cfg.n_action_steps
            old_reach_thr = float(current.get("reach_success_threshold", 0.05))
            old_term = bool(current.get("terminate_on_success"))
            old_max_steps = int(current.get("max_steps", 150))
            old_remove_sword = bool(current.get("remove_sword", False))
            old_remove_shield = bool(current.get("remove_shield", False))
            old_enable_exec = bool(current.get("enable_execution_point", False))
            old_exec_pos = tuple(
                float(x)
                for x in (current.get("execution_point_pos") or DEFAULT_EXECUTION_POINT_POS)
            )
            old_dw = int(current.get("display_width", 1280))
            old_dh = int(current.get("display_height", 720))
            old_handle = tuple(
                float(x) for x in (current.get("sword_handle_pos") or default_sword_handle_pos())
            )
            old_euler = tuple(
                float(x) for x in (current.get("sword_euler_deg") or default_sword_euler_deg())
            )
            old_shield_handle = tuple(
                float(x) for x in (current.get("shield_handle_pos") or default_shield_handle_pos())
            )
            old_shield_euler = tuple(
                float(x) for x in (current.get("shield_euler_deg") or default_shield_euler_deg())
            )

            save_settings(merged, self.cfg.settings_path)
            new_cfg = webconfig_from_settings(
                merged,
                host=self.cfg.host,
                port=self.cfg.port,
                settings_path=self.cfg.settings_path,
            )
            self.cfg = new_cfg
            self.store = DesignStore(self.cfg.designs_dir)

            pose_changed = any(
                abs(a - b) > 1e-9
                for a, b in zip(old_handle, self.cfg.sword_handle_pos, strict=True)
            ) or any(
                abs(a - b) > 1e-9
                for a, b in zip(old_euler, self.cfg.sword_euler_deg, strict=True)
            ) or any(
                abs(a - b) > 1e-9
                for a, b in zip(old_shield_handle, self.cfg.shield_handle_pos, strict=True)
            ) or any(
                abs(a - b) > 1e-9
                for a, b in zip(old_shield_euler, self.cfg.shield_euler_deg, strict=True)
            )
            rebuilt_env = False
            if (
                old_scene != self.cfg.scene
                or pose_changed
                or abs(old_reach_thr - float(self.cfg.reach_success_threshold)) > 1e-12
                or old_term != self.cfg.terminate_on_success
                or old_max_steps != self.cfg.max_steps
                or old_dw != self.cfg.display_width
                or old_dh != self.cfg.display_height
            ):
                self._build_env()
                self.obs, _ = self.env.reset()
                rebuilt_env = True
            elif (
                old_remove_sword != self.cfg.remove_sword
                or old_remove_shield != self.cfg.remove_shield
            ):
                self.env.set_prop_removed(
                    remove_sword=self.cfg.remove_sword,
                    remove_shield=self.cfg.remove_shield,
                )
                self.obs = self.env._get_obs()
                self._log(
                    f"props: remove_sword={self.cfg.remove_sword} "
                    f"remove_shield={self.cfg.remove_shield}"
                )

            exec_pos_changed = any(
                abs(a - b) > 1e-9
                for a, b in zip(old_exec_pos, self.cfg.execution_point_pos, strict=True)
            )
            if not rebuilt_env and (
                old_enable_exec != self.cfg.enable_execution_point or exec_pos_changed
            ):
                self.env.set_execution_point(
                    enable=self.cfg.enable_execution_point,
                    position=self.cfg.execution_point_pos,
                )
                self._log(
                    f"execution_point: enable={self.cfg.enable_execution_point} "
                    f"pos={list(self.cfg.execution_point_pos)}"
                )

            self._build_router()

            new_ckpt = str(self.cfg.checkpoint) if self.cfg.checkpoint else ""
            need_policy = (
                new_ckpt != old_ckpt
                or (self.cfg.vlm_model_name or "") != old_vlm
                or (self.cfg.device or "") != old_device
                or self.cfg.n_action_steps != old_n
                or not self.policy_ready
            )
            if need_policy:
                self._try_load_policy()

            self._sync_bridge_from_cfg()
            self._log(f"settings saved -> {self.cfg.settings_path}")
            return self.get_settings()

    def _sync_bridge_from_cfg(self) -> None:
        self.bridge.update(
            enabled=self.cfg.bridge_enabled,
            base_url=self.cfg.bridge_base_url,
            arms=self.cfg.bridge_arms,
            fps=float(FPS),
            execute=self.cfg.bridge_execute,
            result_timeout_sec=self.cfg.bridge_result_timeout_sec,
            http_timeout_sec=self.cfg.bridge_http_timeout_sec,
        )
        state = "ON" if self.bridge.cfg.enabled else "OFF"
        self._log(
            f"vla bridge {state} -> {self.bridge.endpoint} "
            f"(arms={self.bridge.cfg.arms}, execute={self.bridge.cfg.execute})"
        )

    def get_bridge_config(self) -> dict[str, Any]:
        return self.bridge.cfg.to_dict()

    def set_bridge_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Hot-update bridge settings and persist into web_settings.json."""
        if self.is_running:
            raise RuntimeError("执行中无法修改桥接设置，请先 Stop。")
        allowed = {
            "enabled": "bridge_enabled",
            "base_url": "bridge_base_url",
            "arms": "bridge_arms",
            "execute": "bridge_execute",
            "result_timeout_sec": "bridge_result_timeout_sec",
            "http_timeout_sec": "bridge_http_timeout_sec",
            "fps": None,  # always follow sim FPS; ignore client override for persistence
        }
        settings_patch: dict[str, Any] = {}
        for key, settings_key in allowed.items():
            if key not in patch or settings_key is None:
                continue
            settings_patch[settings_key] = patch[key]
        # Also accept already-prefixed keys from /api/settings.
        for settings_key in (
            "bridge_enabled",
            "bridge_base_url",
            "bridge_arms",
            "bridge_execute",
            "bridge_result_timeout_sec",
            "bridge_http_timeout_sec",
        ):
            if settings_key in patch:
                settings_patch[settings_key] = patch[settings_key]

        with self.lock:
            current = settings_from_webconfig(self.cfg)
            merged = merge_settings_patch(current, settings_patch)
            # Validate via BridgeConfig
            probe = BridgeConfig(
                enabled=bool(merged.get("bridge_enabled", False)),
                base_url=str(merged.get("bridge_base_url") or "http://192.168.10.38:8765"),
                arms=str(merged.get("bridge_arms") or "both"),
                fps=float(FPS),
                execute=bool(merged.get("bridge_execute", True)),
                result_timeout_sec=float(merged.get("bridge_result_timeout_sec", 5.0)),
                http_timeout_sec=float(merged.get("bridge_http_timeout_sec", 8.0)),
            ).normalized()
            merged["bridge_enabled"] = probe.enabled
            merged["bridge_base_url"] = probe.base_url
            merged["bridge_arms"] = probe.arms
            merged["bridge_execute"] = probe.execute
            merged["bridge_result_timeout_sec"] = probe.result_timeout_sec
            merged["bridge_http_timeout_sec"] = probe.http_timeout_sec
            save_settings(merged, self.cfg.settings_path)
            self.cfg = webconfig_from_settings(
                merged,
                host=self.cfg.host,
                port=self.cfg.port,
                settings_path=self.cfg.settings_path,
            )
            self._sync_bridge_from_cfg()
            return self.bridge.cfg.to_dict()

    def close(self) -> None:
        self.stop()
        self.stop_preview()
        self.stop_reach_ik_replay()
        for thread in (
            self.rollout_thread,
            self.preview_thread,
            self.generate_thread,
            self.validate_thread,
            self.reach_ik_gen_thread,
            self.reach_ik_replay_thread,
        ):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        if self.display_renderer is not None:
            self.display_renderer.close()
            self.display_renderer = None
        if self.obs_preview_renderer is not None:
            self.obs_preview_renderer.close()
            self.obs_preview_renderer = None
        self.env.close()

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.logs.append({"text": f"[{stamp}] {text}"})
            self.logs = self.logs[-240:]
        print(f"[web] {text}", flush=True)

    def _copy_obs(self) -> dict[str, Any]:
        if self.obs is None:
            raise RuntimeError("Environment has not been reset.")
        out: dict[str, Any] = {
            "agent_pos": np.asarray(self.obs["agent_pos"], dtype=np.float32).copy(),
            "pixels": {name: image.copy() for name, image in self.obs["pixels"].items()},
        }
        return out

    def _close_policy_renderer_locked(self) -> None:
        renderer = getattr(self.env, "_renderer", None)
        if renderer is None:
            return
        try:
            renderer.close()
        except Exception as exc:
            self._log(f"policy renderer close warning: {type(exc).__name__}: {exc}")
        finally:
            self.env._renderer = None
            self.env._render_failed = False

    def _ensure_display_renderer(self) -> mujoco.Renderer:
        if self.display_renderer is None:
            self.display_renderer = mujoco.Renderer(
                self.env.model,
                height=self.cfg.display_height,
                width=self.cfg.display_width,
            )
        return self.display_renderer

    def _ensure_obs_preview_renderer(self) -> mujoco.Renderer:
        if self.obs_preview_renderer is None:
            self.obs_preview_renderer = mujoco.Renderer(
                self.env.model,
                height=self._obs_preview_height,
                width=self._obs_preview_width,
            )
        return self.obs_preview_renderer

    def _encode_jpeg_b64(self, image: np.ndarray, *, quality: int | None = None) -> str:
        buffer = io.BytesIO()
        Image.fromarray(image).save(
            buffer,
            format="JPEG",
            quality=int(self.cfg.jpeg_quality if quality is None else quality),
            optimize=False,
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _render_obs_cameras_b64_locked(self) -> dict[str, str]:
        """Render the same named MuJoCo cameras used as policy observations."""
        renderer = self._ensure_obs_preview_renderer()
        out: dict[str, str] = {}
        for name in CAMERAS:
            try:
                renderer.update_scene(self.env.data, camera=name)
                image = np.asarray(renderer.render(), dtype=np.uint8).copy()
                out[name] = self._encode_jpeg_b64(image, quality=self._obs_preview_jpeg_quality)
            except Exception as exc:
                self._log(f"obs camera {name} render failed: {type(exc).__name__}: {exc}")
                out[name] = ""
        return out

    def _apply_display_camera_locked(self, camera: CameraRequest | None = None) -> None:
        if camera is not None and camera.reset:
            self.camera_lookat[:] = [0.0, 0.0, 1.2]
            self.camera_azimuth = 120.0
            self.camera_elevation = -20.0
            self.camera_distance = 3.5
            return
        if camera is None:
            return
        self.camera_azimuth += camera.delta_azimuth
        self.camera_elevation = float(np.clip(self.camera_elevation + camera.delta_elevation, -85.0, 5.0))
        if camera.delta_distance:
            self.camera_distance = float(
                np.clip(self.camera_distance * (1.0 + camera.delta_distance), 1.2, 12.0)
            )
        pan_scale = 0.0012 * self.camera_distance
        az = np.deg2rad(self.camera_azimuth)
        right = np.array([np.cos(az), np.sin(az), 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.camera_lookat -= right * (camera.pan_x * pan_scale)
        self.camera_lookat += up * (camera.pan_y * pan_scale)
        self.camera_lookat[2] = float(np.clip(self.camera_lookat[2], 0.4, 2.0))

    def _set_display_camera_locked(self) -> None:
        self._display_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._display_camera.lookat[:] = self.camera_lookat
        self._display_camera.distance = float(self.camera_distance)
        self._display_camera.azimuth = float(self.camera_azimuth)
        self._display_camera.elevation = float(self.camera_elevation)

    def _render_display_b64_locked(self) -> str:
        renderer = self._ensure_display_renderer()
        self._set_display_camera_locked()
        renderer.update_scene(self.env.data, camera=self._display_camera)
        image = np.asarray(renderer.render(), dtype=np.uint8).copy()
        return self._encode_jpeg_b64(image)

    def _state_text_locked(self) -> str:
        if self.obs is None:
            return ""
        state = np.asarray(self.obs["agent_pos"], dtype=np.float32).reshape(16)
        return interact.format_state(state)

    def _refresh_obs_locked(self) -> None:
        self.obs = self.env._get_obs()

    def reset(self) -> None:
        if self.is_running:
            self.stop()
            thread = self.rollout_thread
            if thread is not None:
                thread.join(timeout=2.0)
        self.stop_preview()
        with self.lock:
            # Drop a broken policy renderer *before* reset so RGB pixels are
            # captured with a fresh EGL context (not a zeroed failure buffer).
            self._close_policy_renderer_locked()
            # Match offline eval / default env reset: normalized START_POSE (all
            # zeros). Do NOT use normalize(raw zeros) — that is a different pose
            # and leaves ACT nearly frozen.
            self.obs, _info = self.env.reset(
                options={"state": np.asarray(START_POSE, dtype=np.float32)}
            )
            self.step = 0
            self.current_task = ""
            self.status = "idle"
            self._apply_display_camera_locked(CameraRequest(reset=True))
        self._log("reset")

    def stop(self) -> None:
        self.stop_event.set()
        self.validate_stop.set()
        with self.lock:
            if self.status == "running":
                self.status = "stopping"
        self._log("stop requested")

    @property
    def is_running(self) -> bool:
        thread = self.rollout_thread
        return thread is not None and thread.is_alive()

    @property
    def is_validating(self) -> bool:
        thread = self.validate_thread
        return thread is not None and thread.is_alive()

    def frame_jpeg_b64(self) -> str:
        with self.lock:
            if self.obs is None:
                return ""
            return self._render_display_b64_locked()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            joint_state = None
            if self.obs is not None:
                try:
                    joint_state = {
                        "name": list(CONTROL_JOINTS),
                        "position": self.env._current_control_qpos().tolist(),
                        "velocity": [0.0] * len(CONTROL_JOINTS),
                    }
                except Exception as exc:
                    self._log(f"joint_state read warning: {exc}")
            gen = {
                "state": self.generate_status.state,
                "message": self.generate_status.message,
                "logs": list(self.generate_status.logs[-40:]),
                "output_root": self.generate_status.output_root,
            }
            return {
                "status": self.status,
                "mode": self.modes.mode.value,
                "current_task": self.current_task,
                "step": self.step,
                "max_steps": self.cfg.max_steps,
                "state": self._state_text_locked(),
                "camera": {
                    "azimuth": self.camera_azimuth,
                    "elevation": self.camera_elevation,
                    "distance": self.camera_distance,
                    "lookat": self.camera_lookat.tolist(),
                },
                "input_locked": self.status == "routing" or self.modes.mode != AppMode.CHAT,
                "logs": list(self.logs),
                # Obs cameras = same MuJoCo views as training/inference pixels.
                "cameras": (
                    self._render_obs_cameras_b64_locked() if self.obs_preview_enabled else {}
                ),
                "obs_preview": bool(self.obs_preview_enabled),
                "frame": "",
                "joint_state": joint_state,
                "joint_limits": self._joint_limits.tolist() if self._joint_limits is not None else None,
                "generate": gen,
                "validate": self._validate_payload_locked(),
                "dirty": self.dirty,
                "scene": self._scene_payload_locked(),
                **self._reach_payload_locked(),
                "policy_ready": self.policy_ready,
                "policy_error": self.policy_error,
            }

    def _validate_payload_locked(self) -> dict[str, Any]:
        vs = self.validate_status
        return {
            "state": vs.state,
            "message": vs.message,
            "episode": vs.episode,
            "episodes": vs.episodes,
            "successes": vs.successes,
            "results": list(vs.results[-40:]),
            "current_handle": list(vs.current_handle) if vs.current_handle is not None else None,
            "current_shield_handle": (
                list(vs.current_shield_handle) if vs.current_shield_handle is not None else None
            ),
            "task": vs.task,
            "seed": vs.seed,
            "success_rate": (vs.successes / vs.episode) if vs.episode > 0 else None,
        }

    def _scene_payload_locked(self) -> dict[str, Any]:
        if self.cfg.scene == SCENE_TABLE_PLACE:
            return self._table_place_scene_payload_locked()
        if self.cfg.scene != SCENE_REACH_SWORD:
            return {"name": self.cfg.scene}
        if self._eval_handle_override is not None:
            handle = tuple(float(x) for x in self._eval_handle_override)
            euler = tuple(
                float(x)
                for x in (
                    self._eval_euler_override
                    if self._eval_euler_override is not None
                    else self.cfg.sword_euler_deg
                )
            )
            body_pos, body_quat, handle = resolve_sword_pose(handle_pos=handle, euler_deg=euler)
        else:
            body_pos, body_quat, handle = self.cfg.resolved_sword_pose()

        # Prefer live MuJoCo body so Three.js follows kinematic attach/carry.
        right_attached = bool(getattr(self.env, "_sword_attached", False)) and not bool(
            self.cfg.remove_sword
        )
        left_attached = bool(getattr(self.env, "_shield_attached", False)) and not bool(
            self.cfg.remove_shield
        )
        sword_id = int(getattr(self.env, "_sword_body_id", -1))
        handle_id = int(getattr(self.env, "_handle_site_id", -1))
        shield_id = int(getattr(self.env, "_shield_body_id", -1))
        shield_handle_id = int(getattr(self.env, "_shield_handle_site_id", -1))
        if sword_id >= 0:
            body_pos = tuple(float(x) for x in self.env.model.body_pos[sword_id])
            body_quat = tuple(float(x) for x in self.env.model.body_quat[sword_id])
        if handle_id >= 0:
            handle = tuple(float(x) for x in self.env.data.site_xpos[handle_id])

        shield_pos = tuple(float(x) for x in self.env.model.body_pos[shield_id]) if shield_id >= 0 else None
        shield_quat = (
            tuple(float(x) for x in self.env.model.body_quat[shield_id]) if shield_id >= 0 else None
        )
        shield_handle = (
            tuple(float(x) for x in self.env.data.site_xpos[shield_handle_id])
            if shield_handle_id >= 0
            else None
        )

        return {
            "name": SCENE_REACH_SWORD,
            "task": TASK_REACH_SWORD,
            "attached": right_attached,
            "left_attached": left_attached,
            "right_attached": right_attached,
            "both_attached": left_attached and right_attached,
            "sword": {
                "url": "/urdf/meshes/sword.stl",
                "present": not bool(self.cfg.remove_sword),
                "scale": float(SWORD_MESH_SCALE),
                "position": list(body_pos),
                "quaternion_wxyz": list(body_quat),
                "handle_local": list(SWORD_HANDLE_LOCAL),
                "handle_world": list(handle),
                "euler_deg": list(
                    self._eval_euler_override
                    if self._eval_euler_override is not None
                    else self.cfg.sword_euler_deg
                ),
                "color": int(SWORD_THREE_COLOR),
                "attached": right_attached and not bool(self.cfg.remove_sword),
            },
            "shield": {
                "url": "/urdf/meshes/helmet.stl",
                "present": not bool(self.cfg.remove_shield),
                "scale": float(HELMET_MESH_SCALE),
                "position": list(shield_pos) if shield_pos is not None else list(HELMET_BODY_POS),
                "quaternion_wxyz": list(shield_quat)
                if shield_quat is not None
                else list(HELMET_BODY_QUAT),
                "handle_local": list(HELMET_HANDLE_LOCAL),
                "handle_world": list(shield_handle) if shield_handle is not None else None,
                "color": int(HELMET_THREE_COLOR),
                "attached": left_attached and not bool(self.cfg.remove_shield),
            },
            "left_tcp_local": list(LEFT_TCP_LOCAL),
            "right_tcp_local": list(RIGHT_TCP_LOCAL),
            "reach_tcp": "right_tcp",
            "execution_point": {
                "enabled": (
                    bool(self.cfg.enable_execution_point)
                    and not bool(self.cfg.remove_sword)
                    and self.status != "replay"
                ),
                "position": list(self.cfg.execution_point_pos),
                "passed": bool(getattr(self.env, "_execution_point_passed", False)),
                "color": 0x2ecc71,
            },
        }

    def _table_place_scene_payload_locked(self) -> dict[str, Any]:
        peg_id = int(getattr(self.env, "_peg_body_id", -1))
        grasp_id = int(getattr(self.env, "_peg_grasp_site_id", -1))
        circle_id = int(getattr(self.env, "_target_circle_site_id", -1))
        peg_attached = bool(getattr(self.env, "_peg_attached", False))
        peg_pos = (
            tuple(float(x) for x in self.env.model.body_pos[peg_id])
            if peg_id >= 0
            else tuple(float(x) for x in TABLE_PLACE_POS)
        )
        peg_quat = (
            tuple(float(x) for x in self.env.model.body_quat[peg_id])
            if peg_id >= 0
            else (1.0, 0.0, 0.0, 0.0)
        )
        grasp_world = (
            tuple(float(x) for x in self.env.data.site_xpos[grasp_id])
            if grasp_id >= 0
            else list(peg_pos)
        )
        circle_pos = (
            tuple(float(x) for x in self.env.data.site_xpos[circle_id])
            if circle_id >= 0
            else (
                float(TABLE_PLACE_POS[0]),
                float(TABLE_PLACE_POS[1]),
                float(TABLE_PLACE_POS[2] + TABLE_PLACE_HALF[2]),
            )
        )
        info = {}
        try:
            info = self.env._reach_info()
        except Exception:
            info = {}
        return {
            "name": SCENE_TABLE_PLACE,
            "task": TASK_TABLE_PLACE,
            "attached": peg_attached,
            "left_attached": False,
            "right_attached": peg_attached,
            "both_attached": False,
            "table": {
                "present": True,
                "position": list(TABLE_PLACE_POS),
                "half_size": list(TABLE_PLACE_HALF),
                "color": int(TABLE_PLACE_THREE_COLOR),
            },
            "peg": {
                "present": True,
                "position": list(peg_pos),
                "quaternion_wxyz": list(peg_quat),
                "radius": float(PEG_RADIUS),
                "half_height": float(PEG_HALF_HEIGHT),
                "grasp_local": [0.0, 0.0, float(PEG_HALF_HEIGHT)],
                "grasp_world": list(grasp_world),
                "color": int(PEG_THREE_COLOR),
                "attached": peg_attached,
            },
            "circle": {
                "present": True,
                "center": list(circle_pos),
                "radius": float(info.get("circle_radius") or TABLE_CIRCLE_RADIUS),
                "color": int(TABLE_CIRCLE_THREE_COLOR),
            },
            "peg_workspace": {
                "x_range": list(PEG_WORKSPACE_X_RANGE),
                "y_range": list(PEG_WORKSPACE_Y_RANGE),
                "edge_margin": float(PEG_SAMPLE_EDGE_MARGIN),
            },
            "place_distance": info.get("place_distance"),
            "place_fit": info.get("place_fit"),
            "sword": {"present": False},
            "shield": {"present": False},
            "left_tcp_local": list(LEFT_TCP_LOCAL),
            "right_tcp_local": list(RIGHT_TCP_LOCAL),
            "reach_tcp": "right_tcp",
            "execution_point": {"enabled": False, "position": [0, 0, 0], "passed": False},
        }

    def _reach_payload_locked(self) -> dict[str, Any]:
        if self.cfg.scene == SCENE_TABLE_PLACE:
            try:
                info = self.env._reach_info()
                return {
                    "reach_distance": info.get("reach_distance"),
                    "left_reach_distance": None,
                    "right_reach_distance": info.get("right_reach_distance"),
                    "success": bool(info.get("success", False)),
                    "left_success": False,
                    "right_success": bool(info.get("right_success", False)),
                    "attached": bool(info.get("peg_attached", False)),
                    "left_attached": False,
                    "right_attached": bool(info.get("peg_attached", False)),
                    "both_attached": False,
                    "execution_point_enabled": False,
                    "execution_point_passed": False,
                    "execution_point_distance": None,
                    "place_distance": info.get("place_distance"),
                    "place_fit": info.get("place_fit"),
                    "peg_attached": bool(info.get("peg_attached", False)),
                }
            except Exception:
                return {
                    "reach_distance": None,
                    "left_reach_distance": None,
                    "right_reach_distance": None,
                    "success": False,
                    "left_success": False,
                    "right_success": False,
                    "attached": False,
                    "left_attached": False,
                    "right_attached": False,
                    "both_attached": False,
                    "execution_point_enabled": False,
                    "execution_point_passed": False,
                    "execution_point_distance": None,
                    "place_distance": None,
                    "place_fit": None,
                    "peg_attached": False,
                }
        if self.cfg.scene != SCENE_REACH_SWORD:
            return {
                "reach_distance": None,
                "left_reach_distance": None,
                "right_reach_distance": None,
                "success": False,
                "left_success": False,
                "right_success": False,
                "attached": False,
                "left_attached": False,
                "right_attached": False,
                "both_attached": False,
                "execution_point_enabled": False,
                "execution_point_passed": False,
                "execution_point_distance": None,
            }
        try:
            info = self.env._reach_info()
            return {
                "reach_distance": info.get("reach_distance"),
                "left_reach_distance": info.get("left_reach_distance"),
                "right_reach_distance": info.get("right_reach_distance"),
                "success": bool(info.get("success", False)),
                "left_success": bool(info.get("left_success", False)),
                "right_success": bool(info.get("right_success", False)),
                "attached": bool(info.get("attached", False)),
                "left_attached": bool(info.get("left_attached", False)),
                "right_attached": bool(info.get("right_attached", False)),
                "both_attached": bool(info.get("both_attached", False)),
                "execution_point_enabled": bool(info.get("execution_point_enabled", False)),
                "execution_point_passed": bool(info.get("execution_point_passed", False)),
                "execution_point_distance": info.get("execution_point_distance"),
            }
        except Exception:
            return {
                "reach_distance": None,
                "left_reach_distance": None,
                "right_reach_distance": None,
                "success": False,
                "left_success": False,
                "right_success": False,
                "attached": False,
                "left_attached": False,
                "right_attached": False,
                "both_attached": False,
                "execution_point_enabled": False,
                "execution_point_passed": False,
                "execution_point_distance": None,
            }

    def _reach_distance_locked(self) -> float | None:
        return self._reach_payload_locked().get("reach_distance")

    def _success_locked(self) -> bool:
        return bool(self._reach_payload_locked().get("success", False))

    def _attached_locked(self) -> bool:
        return bool(self._reach_payload_locked().get("attached", False))

    def set_mode(self, mode: str) -> dict[str, Any]:
        target = AppMode(mode)
        if target == AppMode.DESIGN:
            if self.is_running:
                self.stop()
                thread = self.rollout_thread
                if thread is not None:
                    thread.join(timeout=2.0)
            self.stop_preview()
            self.stop_reach_ik_replay()
        elif target == AppMode.CHAT:
            self.stop_preview()
            self.stop_reach_ik_replay()
        self.modes.set_mode(target)
        self._log(f"mode -> {target.value}")
        return {"mode": self.modes.mode.value, "dirty": self.dirty}

    def handle_message(self, text: str) -> dict[str, str]:
        """CHAT mode entry: LLM intent gate, then ACT rollout on action."""
        self.modes.require_mode(AppMode.CHAT)
        text = text.strip()
        if not text:
            raise ValueError("empty message")

        self._log(f"user: {text}")
        with self.lock:
            if not self.policy_ready:
                reply = self.policy_error or "策略未加载，请先在「设置」中配置 checkpoint 并保存。"
                self._log(f"assistant: {reply}")
                return {"route": "chat", "reply": reply}
            if self.router is None:
                reply = self.router_error or "LLM 路由器未就绪，请在「设置」中填写 API key。"
                self._log(f"assistant: {reply}")
                return {"route": "chat", "reply": reply}
            if self.status == "running":
                reply = "我还在执行上一个动作，等它完成或先停止。"
                self._log(f"assistant: {reply}")
                return {"route": "chat", "reply": reply}
            self.status = "routing"

        try:
            decision = self.router.route(text)
        except Exception as exc:
            with self.lock:
                self.status = "idle"
            self._log(f"LLM 调用失败：{type(exc).__name__}: {exc}")
            return {"route": "chat", "reply": "LLM 调用失败，先不执行机器人动作。"}

        if isinstance(decision, ActionDecision):
            try:
                self.start_rollout(decision.instruction)
            except Exception as exc:
                with self.lock:
                    self.status = "idle"
                self._log(f"rollout 启动失败：{type(exc).__name__}: {exc}")
                return {"route": "chat", "reply": f"启动失败：{exc}"}
            reply = f"好的，开始执行：{decision.instruction}"
            self.router.history.append({"role": "assistant", "content": reply})
            self._log(f"assistant: {reply}")
            return {"route": "action", "reply": reply, "instruction": decision.instruction}

        with self.lock:
            self.status = "idle"
        reply = decision.text or "..."
        self._log(f"assistant: {reply}")
        return {"route": "chat", "reply": reply}

    def adjust_camera(self, patch: CameraRequest) -> None:
        with self.lock:
            self._apply_display_camera_locked(patch)
        self._log(
            "camera updated: "
            f"az={self.camera_azimuth:.1f}, el={self.camera_elevation:.1f}, dist={self.camera_distance:.2f}"
        )

    def set_obs_preview(self, enabled: bool) -> dict[str, Any]:
        with self.lock:
            self.obs_preview_enabled = bool(enabled)
            enabled_now = bool(self.obs_preview_enabled)
        return {"ok": True, "obs_preview": enabled_now}

    def start_rollout(self, task: str) -> None:
        self.modes.require_mode(AppMode.CHAT)
        if self.is_validating:
            raise RuntimeError("验证进行中，请先停止验证。")
        if not self.policy_ready or self.policy is None or self.preprocess is None or self.postprocess is None:
            raise RuntimeError(self.policy_error or "策略未加载，请先在设置中配置 checkpoint。")
        if self.is_running:
            raise RuntimeError("Rollout is already running.")
        self.stop_event.clear()
        with self.lock:
            self.status = "running"
            self.current_task = task
            self.step = 0
            reset_opts: dict[str, Any] = {
                "state": np.asarray(START_POSE, dtype=np.float32)
            }
            # Match offline eval: randomize peg XY each rollout. The fixed
            # default peg sits in a hard-to-grasp pocket where ACT often fails.
            if self.cfg.scene == SCENE_TABLE_PLACE:
                peg_xy = sample_peg_xy(np.random.default_rng())
                reset_opts["peg_xy"] = peg_xy
                self._log(
                    f"table_place peg_xy=[{peg_xy[0]:.3f},{peg_xy[1]:.3f}]"
                )
            self.obs, _info = self.env.reset(options=reset_opts)
            self._close_policy_renderer_locked()
            self.policy.reset()
        thread = threading.Thread(target=self._rollout_loop, args=(task,), daemon=True)
        self.rollout_thread = thread
        thread.start()

    def _smooth_action(self, action: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
        """EMA blend then per-step rate limit in normalized action space."""
        action = np.asarray(action, dtype=np.float32).reshape(16)
        ema = float(self.cfg.action_ema)
        if prev is None or ema <= 0.0:
            smoothed = action.copy()
        elif ema >= 1.0:
            smoothed = prev.copy()
        else:
            smoothed = (ema * prev + (1.0 - ema) * action).astype(np.float32)

        max_delta = float(self.cfg.max_action_delta)
        if prev is not None and max_delta > 0.0:
            delta = smoothed - prev
            delta = np.clip(delta, -max_delta, max_delta)
            smoothed = (prev + delta).astype(np.float32)

        return np.clip(smoothed, -1.0, 1.0).astype(np.float32)

    def _flush_bridge_chunk(self, chunk: list[np.ndarray], task: str) -> None:
        if not chunk or not self.bridge.cfg.enabled:
            return
        # Chunk rows are absolute joint targets in radians (raw MuJoCo qpos).
        actions_rad = np.stack(chunk, axis=0).astype(np.float64)
        try:
            result = self.bridge.post_joint_chunk(actions_rad, instruction=task)
            ok = result.get("ok")
            points = result.get("points", len(chunk))
            self._log(f"bridge sent chunk points={points} ok={ok}")
            if ok is False:
                self._log(f"bridge rejected chunk: {result}")
        except Exception as exc:
            self._log(f"bridge error: {type(exc).__name__}: {exc}")

    def _rollout_loop(self, task: str) -> None:
        self._log(f"rollout start: {task}")
        step_delay = 0.0 if self.cfg.speed <= 0.0 else 1.0 / (FPS * self.cfg.speed)
        prev_action: np.ndarray | None = None
        bridge_chunk: list[np.ndarray] = []
        chunk_size = max(1, int(self.cfg.n_action_steps))
        try:
            for step in range(self.cfg.max_steps):
                if self.stop_event.is_set():
                    break
                with self.lock:
                    obs_snapshot = self._copy_obs()
                    if prev_action is None:
                        prev_action = np.asarray(obs_snapshot["agent_pos"], dtype=np.float32).reshape(16)

                frame = interact.obs_to_frame(obs_snapshot, self.device)
                processed = self.preprocess(frame)
                with torch.inference_mode():
                    action = self.policy.select_action(processed)
                action = self.postprocess(action)
                if isinstance(action, torch.Tensor):
                    action_np = action.squeeze(0).detach().cpu().numpy()
                else:
                    action_np = np.asarray(action).squeeze(0)
                action_np = np.asarray(action_np, dtype=np.float32).reshape(-1)
                if action_np.shape[0] > 16:
                    action_np = action_np[:16]
                action_np = np.clip(action_np, -1.0, 1.0)
                action_np = self._smooth_action(action_np, prev_action)
                prev_action = action_np

                with self.lock:
                    self.obs, _reward, terminated, truncated, info = self.env.step(action_np)
                    self.step = step + 1
                    reach_dist = info.get("reach_distance")
                    success = bool(info.get("success"))
                    raw_rad = None
                    if self.bridge.cfg.enabled:
                        raw_rad = self.env._current_control_qpos().astype(np.float32).copy()

                if raw_rad is not None:
                    bridge_chunk.append(raw_rad)
                    if len(bridge_chunk) >= chunk_size:
                        self._flush_bridge_chunk(bridge_chunk, task)
                        bridge_chunk = []

                if success:
                    self._log(
                        f"reach success at step {step + 1}"
                        + (f" (dist={reach_dist:.4f}m)" if reach_dist is not None else "")
                    )
                if terminated or truncated:
                    break
                if step_delay > 0.0:
                    time.sleep(step_delay)
        except Exception as exc:
            self._log(f"rollout error: {type(exc).__name__}: {exc}")
        finally:
            if bridge_chunk:
                self._flush_bridge_chunk(bridge_chunk, task)
            with self.lock:
                self._close_policy_renderer_locked()
                stopped = self.stop_event.is_set()
                self.status = "idle"
                self.current_task = ""
                reach_info = ""
                if self.cfg.scene == SCENE_REACH_SWORD:
                    try:
                        info = self.env._reach_info()
                        reach_info = (
                            f" success={info['success']} dist={info['reach_distance']:.4f}m"
                        )
                    except Exception:
                        reach_info = ""
            self.stop_event.clear()
            self._log(
                ("rollout stopped" if stopped else "rollout complete") + reach_info
            )

    # ---- validate (post-training reach eval) ------------------------------

    def _apply_sword_handle_locked(
        self,
        handle: tuple[float, float, float] | list[float],
        euler: tuple[float, float, float] | list[float],
    ) -> tuple[float, float, float]:
        body, quat, hw = resolve_sword_pose(handle_pos=handle, euler_deg=euler)
        if getattr(self.env, "_sword_body_id", -1) < 0:
            raise RuntimeError("当前环境没有剑目标，请先切换到 reach_sword 场景。")
        body_arr = np.asarray(body, dtype=np.float64)
        quat_arr = np.asarray(quat, dtype=np.float64)
        self.env.model.body_pos[self.env._sword_body_id] = body_arr
        self.env.model.body_quat[self.env._sword_body_id] = quat_arr
        # Keep env.reset() from restoring the pre-eval home pose.
        self.env._sword_home_pos = body_arr.copy()
        self.env._sword_home_quat = quat_arr.copy()
        mujoco.mj_forward(self.env.model, self.env.data)
        self._eval_handle_override = [float(x) for x in hw]
        self._eval_euler_override = [float(x) for x in euler]
        return hw

    def _apply_shield_handle_locked(
        self,
        handle: tuple[float, float, float] | list[float],
        euler: tuple[float, float, float] | list[float],
    ) -> tuple[float, float, float]:
        body, quat, hw = resolve_shield_pose(handle_pos=handle, euler_deg=euler)
        if getattr(self.env, "_shield_body_id", -1) < 0:
            raise RuntimeError("当前环境没有盾牌目标，请先切换到 reach_sword 场景。")
        body_arr = np.asarray(body, dtype=np.float64)
        quat_arr = np.asarray(quat, dtype=np.float64)
        self.env.model.body_pos[self.env._shield_body_id] = body_arr
        self.env.model.body_quat[self.env._shield_body_id] = quat_arr
        self.env._shield_home_pos = body_arr.copy()
        self.env._shield_home_quat = quat_arr.copy()
        mujoco.mj_forward(self.env.model, self.env.data)
        self._eval_shield_handle_override = [float(x) for x in hw]
        self._eval_shield_euler_override = [float(x) for x in euler]
        return hw

    def _clear_eval_overrides_locked(self) -> None:
        self._eval_handle_override = None
        self._eval_euler_override = None
        self._eval_shield_handle_override = None
        self._eval_shield_euler_override = None
        if self.cfg.scene != SCENE_REACH_SWORD:
            return
        if getattr(self.env, "_sword_body_id", -1) >= 0:
            body, quat, _ = self.cfg.resolved_sword_pose()
            body_arr = np.asarray(body, dtype=np.float64)
            quat_arr = np.asarray(quat, dtype=np.float64)
            self.env.model.body_pos[self.env._sword_body_id] = body_arr
            self.env.model.body_quat[self.env._sword_body_id] = quat_arr
            self.env._sword_home_pos = body_arr.copy()
            self.env._sword_home_quat = quat_arr.copy()
        if getattr(self.env, "_shield_body_id", -1) >= 0:
            body, quat, _ = self.cfg.resolved_shield_pose()
            body_arr = np.asarray(body, dtype=np.float64)
            quat_arr = np.asarray(quat, dtype=np.float64)
            self.env.model.body_pos[self.env._shield_body_id] = body_arr
            self.env.model.body_quat[self.env._shield_body_id] = quat_arr
            self.env._shield_home_pos = body_arr.copy()
            self.env._shield_home_quat = quat_arr.copy()
        if getattr(self.env, "_sword_body_id", -1) >= 0 or getattr(self.env, "_shield_body_id", -1) >= 0:
            mujoco.mj_forward(self.env.model, self.env.data)

    def _clear_sword_override_locked(self) -> None:
        """Backward-compatible alias: clear sword + shield eval overrides. """
        self._clear_eval_overrides_locked()

    def get_validate_status(self) -> dict[str, Any]:
        with self.lock:
            return self._validate_payload_locked()

    def start_validate(
        self,
        *,
        episodes: int = 10,
        max_steps: int | None = None,
        seed: int = 0,
        task: str | None = None,
        handle_jitter: tuple[float, float, float] | list[float] | None = None,
        randomize_euler: bool = False,
        shield_handle_jitter: tuple[float, float, float] | list[float] | None = None,
        randomize_shield_euler: bool = False,
    ) -> dict[str, Any]:
        if self.cfg.scene != SCENE_REACH_SWORD:
            raise RuntimeError("验证抓取需要 scene=reach_sword。请在设置中切换场景。")
        if not self.policy_ready or self.policy is None or self.preprocess is None or self.postprocess is None:
            raise RuntimeError(self.policy_error or "策略未加载，请先在「设置」中配置 checkpoint 并保存。")
        if self.is_running:
            raise RuntimeError("对话执行中，请先 Stop。")
        if self.is_validating:
            raise RuntimeError("验证已在进行中。")
        if self.modes.mode == AppMode.GENERATING:
            raise RuntimeError("数据生成中无法验证。")
        episodes = int(episodes)
        if episodes < 1 or episodes > 200:
            raise ValueError("episodes 需在 1–200")
        steps = int(max_steps) if max_steps is not None else int(self.cfg.max_steps)
        if steps < 1:
            raise ValueError("max_steps must be >= 1")
        jitter = (
            (0.15, 0.12, 0.08)
            if handle_jitter is None
            else tuple(float(x) for x in handle_jitter)
        )
        if len(jitter) != 3 or any(x < 0 for x in jitter):
            raise ValueError("handle_jitter 需为非负三元组 [dx, dy, dz]")
        shield_jitter = (
            (0.15, 0.12, 0.08)
            if shield_handle_jitter is None
            else tuple(float(x) for x in shield_handle_jitter)
        )
        if len(shield_jitter) != 3 or any(x < 0 for x in shield_jitter):
            raise ValueError("shield_handle_jitter 需为非负三元组 [dx, dy, dz]")
        task_text = (task or TASK_REACH_SWORD).strip() or TASK_REACH_SWORD

        self.stop_preview()
        self.validate_stop.clear()
        with self.lock:
            self.validate_status = ValidateStatus(
                state="running",
                message="starting",
                episode=0,
                episodes=episodes,
                successes=0,
                results=[],
                current_handle=None,
                current_shield_handle=None,
                task=task_text,
                seed=int(seed),
            )
            self.status = "running"
            self.current_task = f"validate 0/{episodes}"
            self.step = 0

        thread = threading.Thread(
            target=self._validate_loop,
            kwargs={
                "episodes": episodes,
                "max_steps": steps,
                "seed": int(seed),
                "task": task_text,
                "handle_jitter": jitter,
                "randomize_euler": bool(randomize_euler),
                "shield_handle_jitter": shield_jitter,
                "randomize_shield_euler": bool(randomize_shield_euler),
            },
            daemon=True,
        )
        self.validate_thread = thread
        thread.start()
        self._log(f"validate start: episodes={episodes} seed={seed} task={task_text!r}")
        return self.get_validate_status()

    def stop_validate(self) -> dict[str, Any]:
        self.validate_stop.set()
        with self.lock:
            if self.validate_status.state == "running":
                self.validate_status.message = "stopping"
                if self.status == "running":
                    self.status = "stopping"
        self._log("validate stop requested")
        return self.get_validate_status()

    def _validate_loop(
        self,
        *,
        episodes: int,
        max_steps: int,
        seed: int,
        task: str,
        handle_jitter: tuple[float, float, float],
        randomize_euler: bool,
        shield_handle_jitter: tuple[float, float, float],
        randomize_shield_euler: bool,
    ) -> None:
        rng = np.random.default_rng(seed)
        base_handle = np.asarray(self.cfg.sword_handle_pos, dtype=np.float64)
        base_euler = np.asarray(self.cfg.sword_euler_deg, dtype=np.float64)
        base_shield_handle = np.asarray(self.cfg.shield_handle_pos, dtype=np.float64)
        base_shield_euler = np.asarray(self.cfg.shield_euler_deg, dtype=np.float64)
        step_delay = 0.0 if self.cfg.speed <= 0.0 else 1.0 / (FPS * self.cfg.speed)
        old_term = bool(self.env.terminate_on_success)
        stopped = False
        try:
            self.env.terminate_on_success = True
            for ep in range(1, episodes + 1):
                if self.validate_stop.is_set():
                    stopped = True
                    break
                offset = rng.uniform(-1.0, 1.0, size=3) * np.asarray(handle_jitter, dtype=np.float64)
                handle = (base_handle + offset).tolist()
                if randomize_euler:
                    euler = (base_euler + rng.uniform(-15.0, 15.0, size=3)).tolist()
                else:
                    euler = base_euler.tolist()

                shield_offset = rng.uniform(-1.0, 1.0, size=3) * np.asarray(
                    shield_handle_jitter, dtype=np.float64
                )
                shield_handle = (base_shield_handle + shield_offset).tolist()
                if randomize_shield_euler:
                    shield_euler = (
                        base_shield_euler + rng.uniform(-15.0, 15.0, size=3)
                    ).tolist()
                else:
                    shield_euler = base_shield_euler.tolist()

                with self.lock:
                    self.validate_status.episode = ep
                    self.validate_status.current_handle = [float(x) for x in handle]
                    self.validate_status.current_shield_handle = [
                        float(x) for x in shield_handle
                    ]
                    self.validate_status.message = f"episode {ep}/{episodes}"
                    self.current_task = f"validate {ep}/{episodes}"
                    self.step = 0
                    self._apply_sword_handle_locked(handle, euler)
                    self._apply_shield_handle_locked(shield_handle, shield_euler)
                    self.obs, info0 = self.env.reset(
                        options={"state": np.asarray(START_POSE, dtype=np.float32)}
                    )
                    self._close_policy_renderer_locked()
                    if self.policy is not None:
                        self.policy.reset()
                    start_dist = info0.get("reach_distance")

                self._log(
                    f"validate ep {ep}/{episodes} sword="
                    f"[{handle[0]:.3f},{handle[1]:.3f},{handle[2]:.3f}] "
                    f"shield=[{shield_handle[0]:.3f},{shield_handle[1]:.3f},{shield_handle[2]:.3f}]"
                    + (f" start_dist={start_dist:.3f}m" if start_dist is not None else "")
                )

                prev_action: np.ndarray | None = None
                success = False
                final_dist = None
                used_steps = 0
                try:
                    for step in range(max_steps):
                        if self.validate_stop.is_set():
                            stopped = True
                            break
                        with self.lock:
                            obs_snapshot = self._copy_obs()
                            if prev_action is None:
                                prev_action = np.asarray(
                                    obs_snapshot["agent_pos"], dtype=np.float32
                                ).reshape(16)

                        frame = interact.obs_to_frame(obs_snapshot, self.device)
                        processed = self.preprocess(frame)
                        with torch.inference_mode():
                            action = self.policy.select_action(processed)
                        action = self.postprocess(action)
                        if isinstance(action, torch.Tensor):
                            action_np = action.squeeze(0).detach().cpu().numpy()
                        else:
                            action_np = np.asarray(action).squeeze(0)
                        action_np = np.asarray(action_np, dtype=np.float32).reshape(-1)
                        if action_np.shape[0] > 16:
                            action_np = action_np[:16]
                        action_np = np.clip(action_np, -1.0, 1.0)
                        action_np = self._smooth_action(action_np, prev_action)
                        prev_action = action_np

                        with self.lock:
                            self.obs, _reward, terminated, truncated, info = self.env.step(action_np)
                            self.step = step + 1
                            used_steps = step + 1
                            final_dist = info.get("reach_distance")
                            success = bool(info.get("success"))
                        if success or terminated or truncated:
                            break
                        if step_delay > 0.0:
                            time.sleep(step_delay)
                except Exception as exc:
                    with self.lock:
                        self.validate_status.state = "error"
                        self.validate_status.message = f"{type(exc).__name__}: {exc}"
                    self._log(f"validate error: {type(exc).__name__}: {exc}")
                    break

                row = {
                    "episode": ep,
                    "success": bool(success),
                    "steps": used_steps,
                    "reach_distance": float(final_dist) if final_dist is not None else None,
                    "handle": [float(x) for x in handle],
                    "shield_handle": [float(x) for x in shield_handle],
                }
                with self.lock:
                    self.validate_status.results.append(row)
                    if success:
                        self.validate_status.successes += 1
                    self.validate_status.message = (
                        f"ep {ep}/{episodes} "
                        f"{'OK' if success else 'FAIL'}"
                        + (f" dist={final_dist:.3f}m" if final_dist is not None else "")
                    )
                self._log(
                    f"validate ep {ep} -> {'success' if success else 'fail'}"
                    + (f" dist={final_dist:.4f}m" if final_dist is not None else "")
                )
                if stopped:
                    break

            with self.lock:
                if self.validate_status.state != "error":
                    rate = (
                        self.validate_status.successes / self.validate_status.episode
                        if self.validate_status.episode > 0
                        else 0.0
                    )
                    self.validate_status.state = "done" if not stopped else "done"
                    self.validate_status.message = (
                        ("stopped · " if stopped else "done · ")
                        + f"{self.validate_status.successes}/{self.validate_status.episode} "
                        + f"({100.0 * rate:.0f}%)"
                    )
        finally:
            with self.lock:
                self.env.terminate_on_success = old_term
                self._clear_eval_overrides_locked()
                self._close_policy_renderer_locked()
                self.status = "idle"
                self.current_task = ""
                self.step = 0
            self.validate_stop.clear()
            self._log(f"validate finished: {self.validate_status.message}")

    # ---- design mode -----------------------------------------------------

    def _fill_reach_fields(self, out: dict[str, Any]) -> None:
        if self.cfg.scene != SCENE_REACH_SWORD:
            return
        try:
            info = self.env._reach_info()
            out["reach_distance"] = float(info["reach_distance"])
            out["left_reach_distance"] = float(info["left_reach_distance"])
            out["right_reach_distance"] = float(info["right_reach_distance"])
            out["success"] = bool(info["success"])
            out["left_success"] = bool(info["left_success"])
            out["right_success"] = bool(info["right_success"])
            out["attached"] = bool(info.get("attached", False))
            out["left_attached"] = bool(info.get("left_attached", False))
            out["right_attached"] = bool(info.get("right_attached", False))
            out["both_attached"] = bool(info.get("both_attached", False))
        except Exception:
            out["reach_distance"] = None
            out["left_reach_distance"] = None
            out["right_reach_distance"] = None
            out["success"] = False
            out["left_success"] = False
            out["right_success"] = False
            out["attached"] = False
            out["left_attached"] = False
            out["right_attached"] = False
            out["both_attached"] = False

    def get_pose(self) -> dict[str, Any]:
        with self.lock:
            raw = self.env._current_control_qpos().tolist()
            out: dict[str, Any] = {
                "pose": raw,
                "pose_dict": pose_dict_from_vector(raw),
                "limits": self._joint_limits.tolist() if self._joint_limits is not None else None,
            }
            self._fill_reach_fields(out)
            return out

    def set_pose(self, pose: list[float] | dict[str, float]) -> dict[str, Any]:
        self.modes.require_mode(AppMode.DESIGN)
        if isinstance(pose, dict):
            values = vector_from_pose_dict(pose)
        else:
            values = [float(v) for v in pose]
        if len(values) != 16:
            raise ValueError(f"pose must have 16 values, got {len(values)}")
        raw = np.asarray(values, dtype=np.float32)
        with self.lock:
            before = raw.copy()
            self.env.apply_raw_qpos(raw)
            # Avoid full multi-camera render on every slider tick; joint_state/frame
            # in snapshots read directly from MuJoCo data.
            if self.obs is not None:
                self.obs["agent_pos"] = self.env._normalize(self.env._current_control_qpos())
            applied = self.env._current_control_qpos()
            if np.any(np.abs(applied - before) > 1e-5):
                self._log("pose clipped to joint limits")
            applied_list = applied.tolist()
            out: dict[str, Any] = {
                "pose": applied_list,
                "pose_dict": pose_dict_from_vector(applied_list),
                "limits": self._joint_limits.tolist() if self._joint_limits is not None else None,
            }
            self._fill_reach_fields(out)
            return out

    def get_project(self) -> dict[str, Any]:
        with self.lock:
            return {**self.project.to_dict(), "dirty": self.dirty}

    def set_project(self, data: dict[str, Any]) -> dict[str, Any]:
        self.modes.require_mode(AppMode.DESIGN)
        project = DesignProject.from_dict(data)
        with self.lock:
            self.project = project
            self.dirty = True
        return self.get_project()

    def upsert_keyframe(self, name: str, pose: list[float] | dict[str, float] | None = None) -> dict[str, Any]:
        self.modes.require_mode(AppMode.DESIGN)
        key = str(name).strip()
        if not key:
            raise ValueError("keyframe name must be non-empty")
        if pose is None:
            with self.lock:
                pose_dict = pose_dict_from_vector(self.env._current_control_qpos().tolist())
        elif isinstance(pose, dict):
            pose_dict = normalize_pose_dict(pose)
        else:
            pose_dict = pose_dict_from_vector(pose)
        with self.lock:
            self.project.keyframes[key] = pose_dict
            self.dirty = True
        self._log(f"keyframe upsert: {key}")
        return self.get_project()

    def delete_keyframe(self, name: str) -> dict[str, Any]:
        self.modes.require_mode(AppMode.DESIGN)
        key = str(name).strip()
        with self.lock:
            if key not in self.project.keyframes:
                raise KeyError(f"keyframe not found: {key}")
            del self.project.keyframes[key]
            self.project.playlist = [item for item in self.project.playlist if item != key]
            self.dirty = True
        self._log(f"keyframe delete: {key}")
        return self.get_project()

    def set_playlist(self, playlist: list[str]) -> dict[str, Any]:
        self.modes.require_mode(AppMode.DESIGN)
        with self.lock:
            self.project.playlist = [str(item) for item in playlist]
            self.dirty = True
        return self.get_project()

    def save_project(self, name: str | None = None, task: str | None = None) -> dict[str, str]:
        self.modes.require_mode(AppMode.DESIGN, AppMode.GENERATING)
        with self.lock:
            if name:
                self.project.name = slugify_name(name)
            if task is not None:
                self.project.task = task
            validate_project(self.project)
            paths = self.store.save(self.project)
            self.dirty = False
        self._log(f"saved design {paths['design_path']}")
        return paths

    def list_designs(self) -> list[dict[str, Any]]:
        items = self.store.list()
        return [
            {
                "name": item.name,
                "task": item.task,
                "design_path": item.design_path,
                "poses_path": item.poses_path,
                "keyframe_count": item.keyframe_count,
                "playlist_length": item.playlist_length,
            }
            for item in items
        ]

    def load_design(self, name: str) -> dict[str, Any]:
        self.modes.require_mode(AppMode.DESIGN)
        project = self.store.load(name)
        with self.lock:
            self.project = project
            self.dirty = False
            if project.playlist:
                first = project.keyframes[project.playlist[0]]
                self.env.apply_raw_qpos(np.asarray(vector_from_pose_dict(first), dtype=np.float32))
                if self.obs is not None:
                    self.obs["agent_pos"] = self.env._normalize(self.env._current_control_qpos())
                else:
                    self._refresh_obs_locked()
        self._log(f"loaded design {project.name}")
        return self.get_project()

    def delete_design(self, name: str) -> None:
        self.store.delete(name)
        self._log(f"deleted design {name}")

    def start_preview(self, segment_steps: int = 20) -> dict[str, Any]:
        self.modes.require_mode(AppMode.DESIGN)
        with self.lock:
            project = DesignProject.from_dict(self.project.to_dict())
        # Preview only needs keyframes + playlist; task/name required for save/generate.
        from pose_design.models import validate_playlist_structure

        try:
            validate_playlist_structure(project)
        except ValueError as exc:
            msg = str(exc)
            if "playlist" in msg and "at least one" in msg:
                raise ValueError("播放列表为空：请先把关键帧加入播放列表再预览") from exc
            if "keyframes" in msg and "at least one" in msg:
                raise ValueError("还没有关键帧：请先保存关键帧") from exc
            raise
        traj = build_preview_trajectory(project, segment_steps=segment_steps)
        self.stop_preview()
        self.preview_stop.clear()
        thread = threading.Thread(target=self._preview_loop, args=(traj,), daemon=True)
        self.preview_thread = thread
        with self.lock:
            self.status = "preview"
        thread.start()
        self._log(f"preview start ({len(traj)} frames)")
        return {"frames": len(traj)}

    def stop_preview(self) -> None:
        self.preview_stop.set()
        thread = self.preview_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self.lock:
            if self.status == "preview":
                self.status = "idle"
        self.preview_stop.clear()

    def _preview_loop(self, trajectory: list[np.ndarray]) -> None:
        delay = 1.0 / FPS
        try:
            for pose in trajectory:
                if self.preview_stop.is_set():
                    break
                with self.lock:
                    self.env.apply_raw_qpos(pose)
                    if self.obs is not None:
                        self.obs["agent_pos"] = self.env._normalize(self.env._current_control_qpos())
                time.sleep(delay)
        finally:
            with self.lock:
                if self.status == "preview":
                    self.status = "idle"
            self._log("preview stopped")

    def start_generate(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.generate_thread is not None and self.generate_thread.is_alive():
            raise ModeError("A generate job is already running.")
        self.modes.require_mode(AppMode.DESIGN)
        name = slugify_name(str(params.get("name") or self.project.name))
        # Ensure poses file exists (save current project if matching name / dirty).
        with self.lock:
            if self.dirty or self.project.name != name:
                self.project.name = name
                if params.get("task"):
                    self.project.task = str(params["task"])
                validate_project(self.project)
                self.store.save(self.project)
                self.dirty = False
        poses_path = self.store.poses_path(name)
        if not poses_path.exists():
            raise FileNotFoundError(f"poses file not found: {poses_path}. Save the design first.")

        output_root = Path(params["output_root"]) if params.get("output_root") else (
            ROOT / "data" / f"unoarm_{name}"
        )
        cfg = ScriptedGenConfig(
            poses_json=poses_path,
            episodes=int(params.get("episodes", 30)),
            segment_steps=int(params.get("segment_steps", 20)),
            hold_steps=int(params.get("hold_steps", 2)),
            midpoint_noise_std=float(params.get("midpoint_noise_std", 0.0)),
            hold_noise_std=float(params.get("hold_noise_std", 0.0)),
            pose_jitter_std=float(params.get("pose_jitter_std", 0.05)),
            seed=int(params.get("seed", 0)),
            output_root=output_root,
            repo_id=str(params.get("repo_id", f"doki/unoarm_{name}")),
            overwrite=bool(params.get("overwrite", False)),
        )
        self.modes.enter_generating()
        self.generate_status = GenerateStatus(state="running", message="starting", logs=[])
        thread = threading.Thread(target=self._generate_loop, args=(cfg,), daemon=True)
        self.generate_thread = thread
        thread.start()
        return {"ok": True, "state": "running"}

    def _generate_loop(self, cfg: ScriptedGenConfig) -> None:
        def log(msg: str) -> None:
            with self.lock:
                self.generate_status.logs.append(msg)
                self.generate_status.message = msg
            self._log(msg)

        try:
            out = generate_scripted_dataset(cfg, log=log)
            with self.lock:
                self.generate_status.state = "done"
                self.generate_status.output_root = str(out)
                self.generate_status.message = f"done: {out}"
        except Exception as exc:
            with self.lock:
                self.generate_status.state = "failed"
                self.generate_status.message = f"{type(exc).__name__}: {exc}"
                self.generate_status.logs.append(self.generate_status.message)
            self._log(f"generate failed: {exc}")
        finally:
            self.modes.leave_generating(resume=AppMode.DESIGN)

    def get_generate_status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.generate_status.state,
                "message": self.generate_status.message,
                "logs": list(self.generate_status.logs),
                "output_root": self.generate_status.output_root,
                "mode": self.modes.mode.value,
            }

    # --- Reach-IK generate + replay (Web tab) ---

    def start_reach_ik_generate(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.reach_ik_gen_thread is not None and self.reach_ik_gen_thread.is_alive():
            raise ModeError("Reach-IK 生成任务已在运行。")
        if self.modes.mode == AppMode.GENERATING:
            raise ModeError("动作设计数据生成中，请稍后再试。")
        if self.is_running:
            raise RuntimeError("对话执行中，请先 Stop。")
        if self.is_validating:
            raise RuntimeError("验证进行中，请先停止验证。")
        self.stop_reach_ik_replay()
        self.stop_preview()

        mode = str(params.get("mode") or "json").strip().lower()
        scene = str(getattr(self.cfg, "scene", "") or "")
        default_out = (
            "table_place_ik_web"
            if (scene == SCENE_TABLE_PLACE or mode == "table_place")
            else "reach_ik_web"
        )
        output_name = slugify_name(str(params.get("output_name") or default_out))
        output_root = UNOARM_DATA_ROOT / f"unoarm_{output_name}"

        # table_place: pick-and-place IK generator (multi-point peg + fixed circle).
        if scene == SCENE_TABLE_PLACE or mode == "table_place":
            tp_cfg = table_place_ik_config_from_params(
                params,
                output_root=output_root,
                repo_id=str(params.get("repo_id") or f"doki/unoarm_{output_name}"),
            )
            if not str(tp_cfg.task or "").strip():
                tp_cfg.task = TASK_TABLE_PLACE
            self.reach_ik_gen_status = GenerateStatus(
                state="running", message="starting table_place IK", logs=[]
            )
            thread = threading.Thread(
                target=self._table_place_ik_generate_loop, args=(tp_cfg,), daemon=True
            )
            self.reach_ik_gen_thread = thread
            thread.start()
            return {
                "ok": True,
                "state": "running",
                "output_root": str(output_root),
                "kind": "table_place_ik",
            }

        targets_json_path: Path | None = None
        bbox_min = None
        bbox_max = None
        num_targets = 0
        default_task = "Grasp the sword handle"

        if mode == "json":
            raw = params.get("targets_json_text") or params.get("targets_json")
            if not raw or not str(raw).strip():
                raise ValueError("JSON 模式需要 targets 内容")
            # Accept either a path or inline JSON text
            text = str(raw).strip()
            maybe_path = Path(text)
            if maybe_path.is_file() and text.endswith(".json") and len(text) < 512:
                targets_json_path = maybe_path
            else:
                staging = UNOARM_DATA_ROOT / "_web_reach_ik_targets.json"
                staging.parent.mkdir(parents=True, exist_ok=True)
                staging.write_text(text if text.startswith("{") else text, encoding="utf-8")
                # If user pasted only the targets array, wrap it
                try:
                    parsed = json.loads(staging.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"targets JSON 无效: {exc}") from exc
                if isinstance(parsed, list):
                    wrapped = {
                        "task": str(params.get("task") or default_task),
                        "targets": parsed,
                    }
                    staging.write_text(
                        json.dumps(wrapped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
                targets_json_path = staging
        elif mode == "bbox":
            if params.get("bbox_min") is None or params.get("bbox_max") is None:
                raise ValueError("AABB 模式需要 bbox_min 与 bbox_max")
            bbox_min = tuple(float(x) for x in params["bbox_min"])
            bbox_max = tuple(float(x) for x in params["bbox_max"])
            if len(bbox_min) != 3 or len(bbox_max) != 3:
                raise ValueError("bbox_min/max must have 3 values")
            num_targets = int(params.get("num_targets") or 0)
            if num_targets < 1:
                raise ValueError("num_targets must be >= 1")
        else:
            raise ValueError("mode must be json|bbox")

        cfg = ReachIkGenConfig(
            targets_json=targets_json_path,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            num_targets=num_targets,
            episodes_per_target=int(params.get("episodes_per_target", 1)),
            segment_steps=int(params.get("segment_steps", 20)),
            hold_steps=int(params.get("hold_steps", 2)),
            pose_jitter_std=float(params.get("pose_jitter_std", 0.0)),
            seed=int(params.get("seed", 0)),
            output_root=output_root,
            repo_id=str(params.get("repo_id") or f"doki/unoarm_{output_name}"),
            overwrite=bool(params.get("overwrite", False)),
            approach_offset_m=float(params.get("approach_offset", 0.06)),
            task=str(params.get("task") or default_task),
        )

        self.reach_ik_gen_status = GenerateStatus(state="running", message="starting", logs=[])
        thread = threading.Thread(target=self._reach_ik_generate_loop, args=(cfg,), daemon=True)
        self.reach_ik_gen_thread = thread
        thread.start()
        return {"ok": True, "state": "running", "output_root": str(output_root)}

    def _table_place_ik_generate_loop(self, cfg) -> None:
        def log(msg: str) -> None:
            with self.lock:
                self.reach_ik_gen_status.logs.append(msg)
                self.reach_ik_gen_status.message = msg
            self._log(msg)

        try:
            out = generate_table_place_ik_dataset(cfg, log=log)
            with self.lock:
                self.reach_ik_gen_status.state = "done"
                self.reach_ik_gen_status.output_root = str(out)
                self.reach_ik_gen_status.message = f"done: {out}"
        except Exception as exc:
            with self.lock:
                self.reach_ik_gen_status.state = "failed"
                self.reach_ik_gen_status.message = f"{type(exc).__name__}: {exc}"
                self.reach_ik_gen_status.logs.append(self.reach_ik_gen_status.message)
            self._log(self.reach_ik_gen_status.message)

    def _reach_ik_generate_loop(self, cfg: ReachIkGenConfig) -> None:
        def log(msg: str) -> None:
            with self.lock:
                self.reach_ik_gen_status.logs.append(msg)
                self.reach_ik_gen_status.message = msg
            self._log(msg)

        try:
            out = generate_reach_ik_dataset(cfg, log=log)
            with self.lock:
                self.reach_ik_gen_status.state = "done"
                self.reach_ik_gen_status.output_root = str(out)
                self.reach_ik_gen_status.message = f"done: {out}"
                # Concurrent EGL env can poison the main RGB renderer; recover.
                self._close_policy_renderer_locked()
                self._refresh_obs_locked()
        except Exception as exc:
            with self.lock:
                self.reach_ik_gen_status.state = "failed"
                self.reach_ik_gen_status.message = f"{type(exc).__name__}: {exc}"
                self.reach_ik_gen_status.logs.append(self.reach_ik_gen_status.message)
                self._close_policy_renderer_locked()
            self._log(f"reach-ik generate failed: {exc}")

    def get_reach_ik_generate_status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.reach_ik_gen_status.state,
                "message": self.reach_ik_gen_status.message,
                "logs": list(self.reach_ik_gen_status.logs[-80:]),
                "output_root": self.reach_ik_gen_status.output_root,
            }

    def list_reach_ik_datasets(self) -> list[dict[str, Any]]:
        return list_reach_ik_datasets(UNOARM_DATA_ROOT)

    def get_reach_ik_dataset_info(self, root: str | Path) -> dict[str, Any]:
        path = Path(root)
        if not path.is_dir():
            raise FileNotFoundError(f"dataset root not found: {path}")
        meta = load_table_place_ik_meta(path) or load_reach_ik_meta(path)
        n = dataset_episode_count(path, repo_id=(meta or {}).get("repo_id") if meta else None)
        return {
            "root": str(path),
            "n_episodes": n,
            "meta": meta,
            "repo_id": (meta or {}).get("repo_id") or f"local/{path.name}",
        }

    def start_reach_ik_replay(
        self,
        *,
        root: str | Path,
        episode: int = 0,
        speed: float = 1.0,
    ) -> dict[str, Any]:
        if self.reach_ik_replay_thread is not None and self.reach_ik_replay_thread.is_alive():
            raise ModeError("回放已在进行中。")
        if self.reach_ik_gen_thread is not None and self.reach_ik_gen_thread.is_alive():
            raise ModeError("生成进行中，无法回放。")
        if self.is_running:
            raise RuntimeError("对话执行中，请先 Stop。")
        if self.is_validating:
            raise RuntimeError("验证进行中，请先停止。")
        if self.modes.mode == AppMode.GENERATING:
            raise ModeError("数据生成中无法回放。")
        self.stop_preview()

        path = Path(root)
        info = self.get_reach_ik_dataset_info(path)
        ep = int(episode)
        n_eps = int(info["n_episodes"] or 0)
        if n_eps < 1:
            raise ValueError("数据集没有 episode")
        if ep < 0 or ep >= n_eps:
            raise ValueError(f"episode 需在 0..{n_eps - 1}")

        self.reach_ik_replay_stop.clear()
        self.reach_ik_replay_pause.clear()
        with self.lock:
            self.reach_ik_replay_status = ReplayStatus(
                state="playing",
                message="starting",
                root=str(path),
                episode=ep,
                n_episodes=n_eps,
                frame=0,
                n_frames=0,
                paused=False,
                handle=None,
            )
            self.status = "replay"

        thread = threading.Thread(
            target=self._reach_ik_replay_loop,
            kwargs={
                "root": path,
                "start_episode": ep,
                "n_episodes": n_eps,
                "speed": float(speed),
                "repo_id": str(info["repo_id"]),
                "meta": info.get("meta"),
            },
            daemon=True,
        )
        self.reach_ik_replay_thread = thread
        thread.start()
        return {"ok": True, "state": "playing"}

    def pause_reach_ik_replay(self, paused: bool = True) -> dict[str, Any]:
        if paused:
            self.reach_ik_replay_pause.set()
        else:
            self.reach_ik_replay_pause.clear()
        with self.lock:
            self.reach_ik_replay_status.paused = bool(paused)
            if self.reach_ik_replay_status.state == "playing" and paused:
                self.reach_ik_replay_status.state = "paused"
            elif self.reach_ik_replay_status.state == "paused" and not paused:
                self.reach_ik_replay_status.state = "playing"
        return self.get_reach_ik_replay_status()

    def stop_reach_ik_replay(self) -> dict[str, Any]:
        self.reach_ik_replay_stop.set()
        self.reach_ik_replay_pause.clear()
        thread = self.reach_ik_replay_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self.lock:
            if self.reach_ik_replay_status.state in ("playing", "paused"):
                self.reach_ik_replay_status.state = "idle"
                self.reach_ik_replay_status.message = "stopped"
            if self.status == "replay":
                self.status = "idle"
        return self.get_reach_ik_replay_status()

    def get_reach_ik_replay_status(self) -> dict[str, Any]:
        with self.lock:
            s = self.reach_ik_replay_status
            return {
                "state": s.state,
                "message": s.message,
                "root": s.root,
                "episode": s.episode,
                "n_episodes": s.n_episodes,
                "frame": s.frame,
                "n_frames": s.n_frames,
                "paused": s.paused,
                "handle": list(s.handle) if s.handle is not None else None,
                "peg_xy": list(s.peg_xy) if s.peg_xy is not None else None,
            }

    def _reach_ik_replay_loop(
        self,
        *,
        root: Path,
        start_episode: int,
        n_episodes: int,
        speed: float,
        repo_id: str,
        meta: dict[str, Any] | None,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        def handle_for_episode(ep_idx: int) -> list[float] | None:
            if not isinstance(meta, dict):
                return None
            for row in meta.get("episodes") or []:
                if isinstance(row, dict) and int(row.get("episode_index", -1)) == int(ep_idx):
                    h = row.get("handle")
                    if h is None:
                        return None
                    return [float(x) for x in h]
            return None

        try:
            dt = 1.0 / max(1e-6, float(FPS) * max(0.05, float(speed)))
            stopped = False
            is_table_place = (
                getattr(self.env, "scene", None) == SCENE_TABLE_PLACE
                or (
                    isinstance(meta, dict)
                    and meta.get("mode") == "table_place_pick_place"
                )
            )
            for ep_idx in range(int(start_episode), int(n_episodes)):
                if self.reach_ik_replay_stop.is_set():
                    stopped = True
                    break

                handle = None if is_table_place else handle_for_episode(ep_idx)
                peg_xy = (
                    episode_peg_xy_from_meta(meta, ep_idx) if is_table_place else None
                )
                if is_table_place and peg_xy is None:
                    self._log(
                        f"table-place replay: episode {ep_idx} has no peg_xy in meta; "
                        "keeping current cylinder pose"
                    )
                if not is_table_place and handle is None:
                    self._log(
                        f"reach-ik replay: episode {ep_idx} has no handle in meta; "
                        "keeping current sword pose"
                    )

                # Action-only replay: read parquet via hf_dataset so we never
                # touch torchcodec/FFmpeg (default video backend is broken here).
                # video_backend=pyav is a safety net if anything still decodes.
                ds = LeRobotDataset(
                    repo_id,
                    root=root,
                    episodes=[int(ep_idx)],
                    download_videos=False,
                    video_backend="pyav",
                    return_uint8=True,
                )
                hf = ds.hf_dataset
                n_frames = len(hf)

                with self.lock:
                    self.reach_ik_replay_status.episode = int(ep_idx)
                    self.reach_ik_replay_status.n_episodes = int(n_episodes)
                    self.reach_ik_replay_status.n_frames = int(n_frames)
                    self.reach_ik_replay_status.frame = 0
                    self.reach_ik_replay_status.handle = (
                        list(handle) if handle is not None else None
                    )
                    self.reach_ik_replay_status.peg_xy = (
                        [float(peg_xy[0]), float(peg_xy[1])] if peg_xy is not None else None
                    )
                    self.reach_ik_replay_status.message = (
                        f"episode {ep_idx + 1}/{n_episodes} frames={n_frames}"
                    )
                    if self.reach_ik_replay_status.state != "paused":
                        self.reach_ik_replay_status.state = "playing"
                    # Grasp-only replay: never show execution point.
                    self.env.set_execution_point(enable=False)
                    if is_table_place:
                        reset_opts = {"peg_xy": peg_xy} if peg_xy is not None else None
                        self.obs, _ = self.env.reset(options=reset_opts)
                        if peg_xy is not None:
                            self.env.place_peg_xy(peg_xy)
                        if hasattr(self.env, "_peg_attached"):
                            self.env._peg_attached = False
                        if hasattr(self.env, "_peg_falling"):
                            self.env._peg_falling = False
                    else:
                        if handle is not None:
                            place_sword_handle(self.env, handle)
                        self.obs, _ = self.env.reset()
                        if handle is not None:
                            place_sword_handle(self.env, handle)
                        # Detach so Three.js pins sword at sampled handle (not TCP follow).
                        if hasattr(self.env, "_sword_attached"):
                            self.env._sword_attached = False
                        if hasattr(self.env, "_shield_attached"):
                            self.env._shield_attached = False

                for i in range(n_frames):
                    if self.reach_ik_replay_stop.is_set():
                        stopped = True
                        break
                    while self.reach_ik_replay_pause.is_set() and not self.reach_ik_replay_stop.is_set():
                        time.sleep(0.05)
                    if self.reach_ik_replay_stop.is_set():
                        stopped = True
                        break
                    action = hf[i]["action"]
                    if hasattr(action, "detach"):
                        action = action.detach().cpu().numpy()
                    action = np.asarray(action, dtype=np.float32).reshape(16)
                    with self.lock:
                        self.obs, _r, _t, _tr, _info = self.env.step(action)
                        self.reach_ik_replay_status.frame = i + 1
                        self.step = i + 1
                    time.sleep(dt)

                if stopped:
                    break

            with self.lock:
                if stopped or self.reach_ik_replay_stop.is_set():
                    self.reach_ik_replay_status.state = "idle"
                    self.reach_ik_replay_status.message = "stopped"
                else:
                    self.reach_ik_replay_status.state = "done"
                    self.reach_ik_replay_status.message = "finished"
                if self.status == "replay":
                    self.status = "idle"
        except Exception as exc:
            with self.lock:
                self.reach_ik_replay_status.state = "error"
                self.reach_ik_replay_status.message = f"{type(exc).__name__}: {exc}"
                if self.status == "replay":
                    self.status = "idle"
            self._log(f"reach-ik replay failed: {exc}")
