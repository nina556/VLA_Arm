"""HTTP client for external MuJoCo VLA joint-chunk bridge."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

DEFAULT_BRIDGE_BASE_URL = "http://192.168.10.38:8765"
JOINT_CHUNK_PATH = "/api/mujoco/vla_joint_chunk"
VALID_ARMS = frozenset({"right", "left", "both"})


@dataclass
class BridgeConfig:
    enabled: bool = False
    base_url: str = DEFAULT_BRIDGE_BASE_URL
    arms: str = "both"
    fps: float = 20.0
    execute: bool = True
    result_timeout_sec: float = 5.0
    http_timeout_sec: float = 8.0

    def normalized(self) -> BridgeConfig:
        arms = str(self.arms or "both").strip().lower()
        if arms not in VALID_ARMS:
            raise ValueError(f"arms must be one of {sorted(VALID_ARMS)}, got {self.arms!r}")
        base = str(self.base_url or DEFAULT_BRIDGE_BASE_URL).strip().rstrip("/")
        if not base:
            raise ValueError("base_url cannot be empty")
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        if self.result_timeout_sec <= 0:
            raise ValueError(f"result_timeout_sec must be > 0, got {self.result_timeout_sec}")
        if self.http_timeout_sec <= 0:
            raise ValueError(f"http_timeout_sec must be > 0, got {self.http_timeout_sec}")
        return BridgeConfig(
            enabled=bool(self.enabled),
            base_url=base,
            arms=arms,
            fps=float(self.fps),
            execute=bool(self.execute),
            result_timeout_sec=float(self.result_timeout_sec),
            http_timeout_sec=float(self.http_timeout_sec),
        )

    def to_dict(self) -> dict[str, Any]:
        cfg = self.normalized()
        return {
            "enabled": cfg.enabled,
            "base_url": cfg.base_url,
            "arms": cfg.arms,
            "fps": cfg.fps,
            "execute": cfg.execute,
            "result_timeout_sec": cfg.result_timeout_sec,
            "http_timeout_sec": cfg.http_timeout_sec,
            "endpoint": f"{cfg.base_url}{JOINT_CHUNK_PATH}",
        }


def denormalize_actions(
    actions: np.ndarray | Sequence[Sequence[float]],
    joint_limits: np.ndarray,
) -> np.ndarray:
    """Map normalized [-1, 1] actions to absolute joint targets in radians."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"actions must have shape (N, 16), got {arr.shape}")
    limits = np.asarray(joint_limits, dtype=np.float32)
    if limits.shape != (16, 2):
        raise ValueError(f"joint_limits must have shape (16, 2), got {limits.shape}")
    clipped = np.clip(arr, -1.0, 1.0)
    low = limits[:, 0]
    high = limits[:, 1]
    return low + (clipped + 1.0) * 0.5 * (high - low)


def build_joint_chunk_body(
    actions_rad: np.ndarray | Sequence[Sequence[float]],
    *,
    arms: str = "both",
    fps: float = 20.0,
    execute: bool = True,
    result_timeout_sec: float = 5.0,
    instruction: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arr = np.asarray(actions_rad, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 16:
        raise ValueError(f"actions must be non-empty Nx16, got {arr.shape}")
    arms_norm = str(arms or "both").strip().lower()
    if arms_norm not in VALID_ARMS:
        raise ValueError(f"arms must be one of {sorted(VALID_ARMS)}, got {arms!r}")

    payload_meta: dict[str, Any] = dict(meta or {})
    if instruction is not None and "instruction" not in payload_meta:
        payload_meta["instruction"] = instruction

    body: dict[str, Any] = {
        "command_type": "joint_chunk",
        "unit": "rad",
        "arms": arms_norm,
        "fps": float(fps),
        "execute": bool(execute),
        "result_timeout_sec": float(result_timeout_sec),
        "actions": arr.tolist(),
    }
    if payload_meta:
        body["meta"] = payload_meta
    return body


class VlaBridgeClient:
    def __init__(self, cfg: BridgeConfig | None = None) -> None:
        self.cfg = (cfg or BridgeConfig()).normalized()

    def update(self, **kwargs: Any) -> BridgeConfig:
        data = self.cfg.to_dict()
        data.pop("endpoint", None)
        data.update(kwargs)
        self.cfg = BridgeConfig(**{k: data[k] for k in BridgeConfig.__dataclass_fields__}).normalized()
        return self.cfg

    @property
    def endpoint(self) -> str:
        return f"{self.cfg.base_url}{JOINT_CHUNK_PATH}"

    def post_joint_chunk(
        self,
        actions_rad: np.ndarray | Sequence[Sequence[float]],
        *,
        instruction: str | None = None,
        meta: dict[str, Any] | None = None,
        execute: bool | None = None,
    ) -> dict[str, Any]:
        body = build_joint_chunk_body(
            actions_rad,
            arms=self.cfg.arms,
            fps=self.cfg.fps,
            execute=self.cfg.execute if execute is None else bool(execute),
            result_timeout_sec=self.cfg.result_timeout_sec,
            instruction=instruction,
            meta=meta,
        )
        raw = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=raw,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.http_timeout_sec) as resp:
                payload = resp.read().decode("utf-8")
                status = int(getattr(resp, "status", 200) or 200)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"bridge HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"bridge unreachable ({self.endpoint}): {exc.reason}") from exc

        try:
            parsed = json.loads(payload) if payload else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"bridge returned non-JSON (HTTP {status}): {payload[:200]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"bridge returned unexpected JSON type: {type(parsed).__name__}")
        return parsed
