"""WebSocket helpers for Unoarm web console."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from gym_unoarm.constants import CONTROL_JOINTS

from webapp.modes import AppMode, ModeError
from webapp.runner import UnoarmWebRunner


async def stream_snapshots(websocket: WebSocket, runner: UnoarmWebRunner) -> None:
    await websocket.accept()
    delay = 1.0 / max(runner.cfg.stream_fps, 1.0)
    send_lock = asyncio.Lock()

    async def safe_send(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def sender() -> None:
        while True:
            await safe_send(runner.snapshot())
            await asyncio.sleep(delay)

    async def receiver() -> None:
        while True:
            raw = await websocket.receive_text()
            try:
                message: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if message.get("type") == "obs_preview":
                enabled = bool(message.get("enabled", True))
                result = await asyncio.to_thread(runner.set_obs_preview, enabled)
                await safe_send({"type": "obs_preview_ack", **result})
                continue
            if message.get("type") != "design_pose":
                continue
            pose = message.get("pose")
            if pose is None:
                continue
            try:
                if runner.modes.mode != AppMode.DESIGN:
                    await safe_send(
                        {
                            "type": "pose_ack",
                            "ok": False,
                            "error": f"当前模式为 {runner.modes.mode.value}，无法改姿态（请先切到动作设计）",
                        }
                    )
                    continue
                result = await asyncio.to_thread(runner.set_pose, pose)
                applied = result.get("pose")
                await safe_send(
                    {
                        "type": "pose_ack",
                        "ok": True,
                        "seq": message.get("seq"),
                        "pose": applied,
                        "reach_distance": result.get("reach_distance"),
                        "left_reach_distance": result.get("left_reach_distance"),
                        "right_reach_distance": result.get("right_reach_distance"),
                        "success": result.get("success"),
                        "left_success": result.get("left_success"),
                        "right_success": result.get("right_success"),
                        "attached": result.get("attached"),
                        "left_attached": result.get("left_attached"),
                        "right_attached": result.get("right_attached"),
                        "both_attached": result.get("both_attached"),
                        "joint_state": {
                            "name": list(CONTROL_JOINTS),
                            "position": applied,
                        },
                    }
                )
            except ModeError as exc:
                await safe_send({"type": "pose_ack", "ok": False, "error": str(exc)})
            except (ValueError, TypeError) as exc:
                await safe_send({"type": "pose_ack", "ok": False, "error": str(exc)})

    send_task = asyncio.create_task(sender())
    recv_task = asyncio.create_task(receiver())
    try:
        done, pending = await asyncio.wait(
            {send_task, recv_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                raise exc
    except WebSocketDisconnect:
        return
    finally:
        for task in (send_task, recv_task):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
