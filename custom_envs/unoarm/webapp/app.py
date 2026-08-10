"""FastAPI application for Unoarm VLA web console."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from webapp.modes import ModeError
from webapp.runner import CameraRequest, UnoarmWebRunner
from webapp.ws import stream_snapshots

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent.parent
STATIC_DIR = ROOT / "static" / "web"
DATA_DIR = REPO_ROOT / "data"


class MessageRequest(BaseModel):
    text: str


class ModeRequest(BaseModel):
    mode: str


class PoseRequest(BaseModel):
    pose: list[float] | dict[str, float]


class ProjectRequest(BaseModel):
    version: int = 1
    name: str
    task: str = ""
    keyframes: dict[str, dict[str, float]] = Field(default_factory=dict)
    playlist: list[str] = Field(default_factory=list)


class KeyframeRequest(BaseModel):
    action: str = "upsert"  # upsert | delete
    name: str
    pose: list[float] | dict[str, float] | None = None


class PlaylistRequest(BaseModel):
    playlist: list[str]


class SaveRequest(BaseModel):
    name: str | None = None
    task: str | None = None


class NameRequest(BaseModel):
    name: str


class PreviewStartRequest(BaseModel):
    segment_steps: int = 20


class GenerateRequest(BaseModel):
    name: str | None = None
    task: str | None = None
    episodes: int = 30
    segment_steps: int = 20
    hold_steps: int = 2
    midpoint_noise_std: float = 0.0
    hold_noise_std: float = 0.0
    pose_jitter_std: float = 0.05
    seed: int = 0
    output_root: str | None = None
    repo_id: str | None = None
    overwrite: bool = False


class ValidateRequest(BaseModel):
    episodes: int = 10
    max_steps: int | None = None
    seed: int = 0
    task: str | None = None
    handle_jitter: list[float] | None = None
    randomize_euler: bool = False
    shield_handle_jitter: list[float] | None = None
    randomize_shield_euler: bool = False


class ReachIkGenerateRequest(BaseModel):
    mode: str = "json"
    targets_json_text: str | None = None
    bbox_min: list[float] | None = None
    bbox_max: list[float] | None = None
    num_targets: int = 0
    episodes_per_target: int = 1
    segment_steps: int = 20
    hold_steps: int = 2
    pose_jitter_std: float = 0.0
    seed: int = 0
    output_name: str = "reach_ik_web"
    repo_id: str | None = None
    overwrite: bool = False
    execution_point: list[float] | None = None
    approach_offset: float = 0.06
    task: str | None = None


class ReachIkReplayStartRequest(BaseModel):
    root: str
    episode: int = 0
    speed: float = 1.0


class ReachIkReplayPauseRequest(BaseModel):
    paused: bool = True


class CameraBody(BaseModel):
    delta_azimuth: float = 0.0
    delta_elevation: float = 0.0
    delta_distance: float = 0.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    reset: bool = False


class SettingsRequest(BaseModel):
    checkpoint: str | None = None
    vlm_model_name: str | None = None
    device: str | None = None
    llm_model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    api_base_url: str | None = None
    llm_timeout: float | None = None
    allowed_tasks: list[str] | str | None = None
    scene: str | None = None
    remove_sword: bool | None = None
    remove_shield: bool | None = None
    enable_execution_point: bool | None = None
    execution_point_pos: list[float] | None = None
    max_steps: int | None = None
    n_action_steps: int | None = None
    action_ema: float | None = None
    max_action_delta: float | None = None
    terminate_on_success: bool | None = None
    reach_success_threshold: float | None = None
    sword_handle_pos: list[float] | None = None
    sword_euler_deg: list[float] | None = None
    shield_handle_pos: list[float] | None = None
    shield_euler_deg: list[float] | None = None
    speed: float | None = None
    stream_fps: float | None = None
    display_width: int | None = None
    display_height: int | None = None
    jpeg_quality: int | None = None
    quiet_router_logs: bool | None = None
    designs_dir: str | None = None
    bridge_enabled: bool | None = None
    bridge_base_url: str | None = None
    bridge_arms: str | None = None
    bridge_execute: bool | None = None
    bridge_result_timeout_sec: float | None = None
    bridge_http_timeout_sec: float | None = None


class BridgeConfigRequest(BaseModel):
    enabled: bool | None = None
    base_url: str | None = None
    arms: str | None = None
    fps: float | None = None
    execute: bool | None = None
    result_timeout_sec: float | None = None
    http_timeout_sec: float | None = None
    # Prefixed aliases (same as settings keys)
    bridge_enabled: bool | None = None
    bridge_base_url: str | None = None
    bridge_arms: str | None = None
    bridge_execute: bool | None = None
    bridge_result_timeout_sec: float | None = None
    bridge_http_timeout_sec: float | None = None


def _http_mode_error(exc: ModeError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


def build_app(runner: UnoarmWebRunner) -> FastAPI:
    app = FastAPI(title="Unoarm VLA Web")
    app.mount("/urdf", StaticFiles(directory=str(ROOT / "gym_unoarm")), name="urdf")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if DATA_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=str(DATA_DIR)), name="assets")

    @app.middleware("http")
    async def no_cache_frontend_js(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/js/") or path.endswith(".js"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/favicon.ico")
    def favicon() -> Response:
        # Browsers always request this; avoid noisy 404 in the console.
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
            "<rect width='32' height='32' rx='6' fill='#34d17c'/>"
            "<text x='16' y='22' text-anchor='middle' font-size='16' "
            "font-family='sans-serif' font-weight='700' fill='#06140c'>U</text>"
            "</svg>"
        )
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return await asyncio.to_thread(runner.get_settings)

    @app.post("/api/settings")
    async def post_settings(request: SettingsRequest) -> dict[str, Any]:
        patch = {k: v for k, v in request.model_dump().items() if v is not None}
        try:
            return await asyncio.to_thread(runner.apply_settings, patch)
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.get("/api/bridge")
    async def get_bridge() -> dict[str, Any]:
        return await asyncio.to_thread(runner.get_bridge_config)

    @app.post("/api/bridge")
    async def post_bridge(request: BridgeConfigRequest) -> dict[str, Any]:
        patch = {k: v for k, v in request.model_dump().items() if v is not None}
        try:
            return await asyncio.to_thread(runner.set_bridge_config, patch)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/ws")
    async def websocket_stream(websocket: WebSocket) -> None:
        await stream_snapshots(websocket, runner)

    @app.post("/api/mode")
    async def set_mode(request: ModeRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(runner.set_mode, request.mode)
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/message")
    async def message(request: MessageRequest) -> dict[str, str]:
        try:
            return await asyncio.to_thread(runner.handle_message, request.text)
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stop")
    async def stop() -> dict[str, str]:
        await asyncio.to_thread(runner.stop)
        await asyncio.to_thread(runner.stop_preview)
        return {"ok": "true"}

    @app.post("/api/reset")
    async def reset() -> dict[str, str]:
        await asyncio.to_thread(runner.reset)
        return {"ok": "true"}

    @app.post("/api/camera")
    async def camera(request: CameraBody) -> dict[str, str]:
        patch = CameraRequest(**request.model_dump())
        await asyncio.to_thread(runner.adjust_camera, patch)
        return {"ok": "true"}

    @app.get("/api/design/pose")
    async def get_pose() -> dict[str, Any]:
        return await asyncio.to_thread(runner.get_pose)

    @app.post("/api/design/pose")
    async def post_pose(request: PoseRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(runner.set_pose, request.pose)
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/design/project")
    async def get_project() -> dict[str, Any]:
        return await asyncio.to_thread(runner.get_project)

    @app.post("/api/design/project")
    async def post_project(request: ProjectRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(runner.set_project, request.model_dump())
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/design/keyframes")
    async def keyframes(request: KeyframeRequest) -> dict[str, Any]:
        try:
            if request.action == "delete":
                return await asyncio.to_thread(runner.delete_keyframe, request.name)
            return await asyncio.to_thread(runner.upsert_keyframe, request.name, request.pose)
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/design/playlist")
    async def playlist(request: PlaylistRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(runner.set_playlist, request.playlist)
        except ModeError as exc:
            raise _http_mode_error(exc) from exc

    @app.post("/api/design/preview/start")
    async def preview_start(request: PreviewStartRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(runner.start_preview, request.segment_steps)
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/design/preview/stop")
    async def preview_stop() -> dict[str, str]:
        await asyncio.to_thread(runner.stop_preview)
        return {"ok": "true"}

    @app.post("/api/design/save")
    async def save(request: SaveRequest) -> dict[str, str]:
        try:
            return await asyncio.to_thread(runner.save_project, request.name, request.task)
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/design/list")
    async def list_designs() -> list[dict[str, Any]]:
        return await asyncio.to_thread(runner.list_designs)

    @app.post("/api/design/load")
    async def load_design(request: NameRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(runner.load_design, request.name)
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/design/delete")
    async def delete_design(name: str) -> dict[str, str]:
        try:
            await asyncio.to_thread(runner.delete_design, name)
            return {"ok": "true"}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/design/generate")
    async def generate(request: GenerateRequest) -> dict[str, Any]:
        try:
            params = request.model_dump()
            return await asyncio.to_thread(runner.start_generate, params)
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            code = 404 if isinstance(exc, FileNotFoundError) else 400
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    @app.get("/api/design/generate/status")
    async def generate_status() -> dict[str, Any]:
        return await asyncio.to_thread(runner.get_generate_status)

    @app.get("/api/validate/status")
    async def validate_status() -> dict[str, Any]:
        return await asyncio.to_thread(runner.get_validate_status)

    @app.post("/api/validate/start")
    async def validate_start(request: ValidateRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                runner.start_validate,
                episodes=request.episodes,
                max_steps=request.max_steps,
                seed=request.seed,
                task=request.task,
                handle_jitter=request.handle_jitter,
                randomize_euler=request.randomize_euler,
                shield_handle_jitter=request.shield_handle_jitter,
                randomize_shield_euler=request.randomize_shield_euler,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/validate/stop")
    async def validate_stop() -> dict[str, Any]:
        return await asyncio.to_thread(runner.stop_validate)

    @app.post("/api/reach-ik/generate")
    async def reach_ik_generate(request: ReachIkGenerateRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(runner.start_reach_ik_generate, request.model_dump())
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            code = 409 if isinstance(exc, RuntimeError) else 400
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    @app.get("/api/reach-ik/generate/status")
    async def reach_ik_generate_status() -> dict[str, Any]:
        return await asyncio.to_thread(runner.get_reach_ik_generate_status)

    @app.get("/api/reach-ik/datasets")
    async def reach_ik_datasets() -> dict[str, Any]:
        items = await asyncio.to_thread(runner.list_reach_ik_datasets)
        return {"datasets": items}

    @app.get("/api/reach-ik/datasets/info")
    async def reach_ik_dataset_info(root: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(runner.get_reach_ik_dataset_info, root)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/reach-ik/replay/start")
    async def reach_ik_replay_start(request: ReachIkReplayStartRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                runner.start_reach_ik_replay,
                root=request.root,
                episode=request.episode,
                speed=request.speed,
            )
        except ModeError as exc:
            raise _http_mode_error(exc) from exc
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            code = 409 if isinstance(exc, RuntimeError) else 400
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    @app.post("/api/reach-ik/replay/pause")
    async def reach_ik_replay_pause(request: ReachIkReplayPauseRequest) -> dict[str, Any]:
        return await asyncio.to_thread(runner.pause_reach_ik_replay, request.paused)

    @app.post("/api/reach-ik/replay/stop")
    async def reach_ik_replay_stop() -> dict[str, Any]:
        return await asyncio.to_thread(runner.stop_reach_ik_replay)

    @app.get("/api/reach-ik/replay/status")
    async def reach_ik_replay_status() -> dict[str, Any]:
        return await asyncio.to_thread(runner.get_reach_ik_replay_status)

    @app.on_event("shutdown")
    def shutdown() -> None:
        runner.close()

    return app
