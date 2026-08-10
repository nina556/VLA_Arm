"""Persistent web console settings (JSON on disk)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from gym_unoarm.constants import DEFAULT_EXECUTION_POINT_POS
from gym_unoarm.sword_pose import (
    default_shield_euler_deg,
    default_shield_handle_pos,
    default_sword_euler_deg,
    default_sword_handle_pos,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = ROOT / "data" / "web_settings.json"

# Shown in GET responses / forms when a real key is stored but not re-sent.
API_KEY_MASK = "********"


def default_settings() -> dict[str, Any]:
    return {
        "checkpoint": str(
            ROOT.parent.parent
            / "outputs"
            / "train"
            / "act_table_place_50"
            / "checkpoints"
            / "020000"
            / "pretrained_model"
        ),
        "vlm_model_name": "",
        "device": "",
        "llm_model": "THUDM/GLM-4-9B-0414",
        "api_key": "",
        "api_key_env": "SILICONFLOW_API_KEY",
        "api_base_url": "https://api.siliconflow.cn/v1",
        "llm_timeout": 20.0,
        "allowed_tasks": [
            "Pick up the cylinder and place it on the target circle",
        ],
        "scene": "table_place",
        "remove_sword": False,
        "remove_shield": False,
        "enable_execution_point": False,
        "execution_point_pos": list(DEFAULT_EXECUTION_POINT_POS),
        "sword_handle_pos": default_sword_handle_pos(),
        "sword_euler_deg": default_sword_euler_deg(),
        "shield_handle_pos": default_shield_handle_pos(),
        "shield_euler_deg": default_shield_euler_deg(),
        "max_steps": 400,
        "n_action_steps": 50,
        "action_ema": 0.0,
        "max_action_delta": 0.0,
        "terminate_on_success": False,
        "reach_success_threshold": 0.05,
        "speed": 1.0,
        "stream_fps": 12.0,
        "display_width": 1280,
        "display_height": 720,
        "jpeg_quality": 90,
        "quiet_router_logs": False,
        "designs_dir": str(ROOT / "data" / "designs"),
        "bridge_enabled": False,
        "bridge_base_url": "http://192.168.10.38:8765",
        "bridge_arms": "both",
        "bridge_execute": True,
        "bridge_result_timeout_sec": 5.0,
        "bridge_http_timeout_sec": 8.0,
    }


def _normalize_vec3(raw: Any, default: list[float]) -> list[float]:
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        parts = [p for p in raw.replace(",", " ").split() if p]
        raw = parts
    try:
        vals = [float(x) for x in list(raw)]
    except (TypeError, ValueError):
        return list(default)
    if len(vals) != 3:
        return list(default)
    return vals


def load_settings(path: Path | None = None) -> dict[str, Any]:
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    base = default_settings()
    if not settings_path.exists():
        return base
    raw = json.loads(settings_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Settings file must be a JSON object: {settings_path}")
    merged = deepcopy(base)
    merged.update(raw)
    merged.pop("sword_pos", None)
    # Normalize types
    tasks = merged.get("allowed_tasks") or []
    if isinstance(tasks, str):
        tasks = [line.strip() for line in tasks.splitlines() if line.strip()]
    merged["allowed_tasks"] = [str(t).strip() for t in tasks if str(t).strip()]
    merged["checkpoint"] = str(merged.get("checkpoint") or "").strip()
    merged["vlm_model_name"] = str(merged.get("vlm_model_name") or "").strip()
    merged["device"] = str(merged.get("device") or "").strip()
    merged["api_key"] = str(merged.get("api_key") or "")
    merged["bridge_enabled"] = bool(merged.get("bridge_enabled", False))
    merged["bridge_base_url"] = str(merged.get("bridge_base_url") or "http://192.168.10.38:8765").strip()
    merged["bridge_arms"] = str(merged.get("bridge_arms") or "both").strip().lower()
    merged["bridge_execute"] = bool(merged.get("bridge_execute", True))
    merged["bridge_result_timeout_sec"] = float(merged.get("bridge_result_timeout_sec", 5.0))
    merged["bridge_http_timeout_sec"] = float(merged.get("bridge_http_timeout_sec", 8.0))
    merged["sword_handle_pos"] = _normalize_vec3(merged.get("sword_handle_pos"), default_sword_handle_pos())
    merged["sword_euler_deg"] = _normalize_vec3(merged.get("sword_euler_deg"), default_sword_euler_deg())
    merged["shield_handle_pos"] = _normalize_vec3(
        merged.get("shield_handle_pos"), default_shield_handle_pos()
    )
    merged["shield_euler_deg"] = _normalize_vec3(merged.get("shield_euler_deg"), default_shield_euler_deg())
    merged["enable_execution_point"] = bool(merged.get("enable_execution_point", False))
    merged["execution_point_pos"] = _normalize_vec3(
        merged.get("execution_point_pos"), list(DEFAULT_EXECUTION_POINT_POS)
    )
    return merged


def save_settings(data: dict[str, Any], path: Path | None = None) -> Path:
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(default_settings())
    payload.update(data)
    payload.pop("sword_pos", None)
    tasks = payload.get("allowed_tasks") or []
    if isinstance(tasks, str):
        tasks = [line.strip() for line in tasks.splitlines() if line.strip()]
    payload["allowed_tasks"] = [str(t).strip() for t in tasks if str(t).strip()]
    payload["sword_handle_pos"] = _normalize_vec3(payload.get("sword_handle_pos"), default_sword_handle_pos())
    payload["sword_euler_deg"] = _normalize_vec3(payload.get("sword_euler_deg"), default_sword_euler_deg())
    payload["shield_handle_pos"] = _normalize_vec3(
        payload.get("shield_handle_pos"), default_shield_handle_pos()
    )
    payload["shield_euler_deg"] = _normalize_vec3(payload.get("shield_euler_deg"), default_shield_euler_deg())
    payload["enable_execution_point"] = bool(payload.get("enable_execution_point", False))
    payload["execution_point_pos"] = _normalize_vec3(
        payload.get("execution_point_pos"), list(DEFAULT_EXECUTION_POINT_POS)
    )
    settings_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return settings_path


def public_settings(data: dict[str, Any]) -> dict[str, Any]:
    """Copy for API responses; mask api_key when present."""
    out = deepcopy(data)
    out.pop("sword_pos", None)
    key = str(out.get("api_key") or "")
    out["api_key_set"] = bool(key)
    out["api_key"] = API_KEY_MASK if key else ""
    return out


def merge_settings_patch(
    current: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Apply UI patch onto current settings. Keep old api_key if patch is blank/masked."""
    merged = deepcopy(current)
    for key, value in patch.items():
        if key == "sword_pos":
            continue
        if key == "api_key":
            text = "" if value is None else str(value)
            if text.strip() in ("", API_KEY_MASK):
                continue
            merged["api_key"] = text.strip()
            continue
        if key == "allowed_tasks" and isinstance(value, str):
            merged["allowed_tasks"] = [line.strip() for line in value.splitlines() if line.strip()]
            continue
        if key == "sword_handle_pos":
            merged["sword_handle_pos"] = _normalize_vec3(value, default_sword_handle_pos())
            continue
        if key == "sword_euler_deg":
            merged["sword_euler_deg"] = _normalize_vec3(value, default_sword_euler_deg())
            continue
        if key == "shield_handle_pos":
            merged["shield_handle_pos"] = _normalize_vec3(value, default_shield_handle_pos())
            continue
        if key == "shield_euler_deg":
            merged["shield_euler_deg"] = _normalize_vec3(value, default_shield_euler_deg())
            continue
        if key == "execution_point_pos":
            merged["execution_point_pos"] = _normalize_vec3(value, list(DEFAULT_EXECUTION_POINT_POS))
            continue
        if key == "enable_execution_point":
            merged["enable_execution_point"] = bool(value)
            continue
        merged[key] = value
    merged.pop("sword_pos", None)
    return merged
