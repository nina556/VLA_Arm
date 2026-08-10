"""Build the reach-sword MJCF from the free-space Unoarm XML."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from .constants import (
    HELMET_BODY_POS,
    HELMET_BODY_QUAT,
    HELMET_HANDLE_LOCAL,
    HELMET_MESH_PATH,
    HELMET_MESH_SCALE,
    HELMET_RGBA,
    LEFT_TCP_LOCAL,
    REACH_SWORD_XML_PATH,
    RIGHT_TCP_LOCAL,
    SWORD_BODY_POS,
    SWORD_BODY_QUAT,
    SWORD_HANDLE_LOCAL,
    SWORD_MESH_PATH,
    SWORD_MESH_SCALE,
    SWORD_RGBA,
    XML_PATH,
)


def evaluate_reach(
    tcp_pos: np.ndarray,
    handle_pos: np.ndarray,
    threshold: float,
) -> tuple[float, bool]:
    """Return (distance_m, success) for TCP vs target handle (red marker)."""
    tcp = np.asarray(tcp_pos, dtype=np.float64).reshape(3)
    handle = np.asarray(handle_pos, dtype=np.float64).reshape(3)
    dist = float(np.linalg.norm(tcp - handle))
    return dist, dist < float(threshold)


def _ensure_mesh(dest: Path, repo_name: str, *, max_faces: int = 180_000) -> None:
    """Copy mesh from data/; decimate STL if MuJoCo face limit would be exceeded."""
    repo_mesh = Path(__file__).resolve().parents[3] / "data" / repo_name
    need_copy = not dest.exists()
    if need_copy:
        if not repo_mesh.exists():
            raise FileNotFoundError(
                f"Missing mesh at {dest} and {repo_mesh}. "
                f"Place data/{repo_name} in the repo or copy it into gym_unoarm/meshes/."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_mesh, dest)

    # MuJoCo STL decoder rejects >200k faces; keep a safe margin.
    try:
        import struct

        raw = dest.read_bytes()
        if len(raw) < 84 or raw[:5] == b"solid":
            return
        nfaces = struct.unpack_from("<I", raw, 80)[0]
        if nfaces <= max_faces:
            return
    except Exception:
        return

    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            f"Mesh {dest} has too many faces for MuJoCo and trimesh is unavailable to decimate."
        ) from exc

    mesh = trimesh.load_mesh(dest, force="mesh")
    simplified = mesh.simplify_quadric_decimation(face_count=max_faces)
    simplified.export(dest)


def _ensure_sword_mesh() -> None:
    _ensure_mesh(SWORD_MESH_PATH, "sword.stl")


def _ensure_helmet_mesh() -> None:
    _ensure_mesh(HELMET_MESH_PATH, "helmet.stl")


def _strip_reach_sword_blocks(text: str) -> str:
    """Remove previously injected sword/shield mesh / body so pose can be rewritten."""
    for mesh_name in ("sword", "helmet", "shield"):
        text = re.sub(
            rf'\s*<mesh name="{mesh_name}"[^/]*/>\s*',
            "\n",
            text,
            count=1,
        )
    text = re.sub(
        r'\s*<!-- Reach-sword:.*?-->\s*<body name="sword"[\s\S]*?</body>\s*',
        "\n",
        text,
        count=1,
    )
    text = re.sub(
        r'\s*<!-- Reach-shield:.*?-->\s*<body name="shield"[\s\S]*?</body>\s*',
        "\n",
        text,
        count=1,
    )
    # Fallbacks if comments were removed
    text = re.sub(r'\s*<body name="sword"[\s\S]*?</body>\s*', "\n", text, count=1)
    text = re.sub(r'\s*<body name="shield"[\s\S]*?</body>\s*', "\n", text, count=1)
    return text


def ensure_reach_sword_xml(
    *,
    base_xml: Path | None = None,
    out_xml: Path | None = None,
    body_pos: tuple[float, float, float] | None = None,
    body_quat: tuple[float, float, float, float] | None = None,
    handle_local: tuple[float, float, float] | None = None,
    mesh_scale: float | None = None,
    rgba: tuple[float, float, float, float] | None = None,
    shield_body_pos: tuple[float, float, float] | None = None,
    shield_body_quat: tuple[float, float, float, float] | None = None,
    shield_handle_local: tuple[float, float, float] | None = None,
    shield_mesh_scale: float | None = None,
    shield_rgba: tuple[float, float, float, float] | None = None,
) -> Path:
    """Inject TCP/handle sites, sword (right), and shield/helmet (left)."""
    base = Path(base_xml) if base_xml is not None else XML_PATH
    out = Path(out_xml) if out_xml is not None else REACH_SWORD_XML_PATH
    if not base.exists():
        raise FileNotFoundError(
            f"Missing base MJCF: {base}. Run "
            "`uv run python custom_envs/unoarm/scripts/01_convert_urdf_to_mjcf.py` first."
        )
    _ensure_sword_mesh()
    _ensure_helmet_mesh()

    pos = tuple(float(x) for x in (body_pos if body_pos is not None else SWORD_BODY_POS))
    quat = tuple(float(x) for x in (body_quat if body_quat is not None else SWORD_BODY_QUAT))
    handle = tuple(float(x) for x in (handle_local if handle_local is not None else SWORD_HANDLE_LOCAL))
    scale = float(mesh_scale if mesh_scale is not None else SWORD_MESH_SCALE)
    color = tuple(float(x) for x in (rgba if rgba is not None else SWORD_RGBA))

    shield_pos = tuple(
        float(x) for x in (shield_body_pos if shield_body_pos is not None else HELMET_BODY_POS)
    )
    shield_quat = tuple(
        float(x) for x in (shield_body_quat if shield_body_quat is not None else HELMET_BODY_QUAT)
    )
    shield_handle = tuple(
        float(x) for x in (shield_handle_local if shield_handle_local is not None else HELMET_HANDLE_LOCAL)
    )
    shield_scale = float(shield_mesh_scale if shield_mesh_scale is not None else HELMET_MESH_SCALE)
    shield_color = tuple(float(x) for x in (shield_rgba if shield_rgba is not None else HELMET_RGBA))

    text = base.read_text(encoding="utf-8")
    text = _strip_reach_sword_blocks(text)

    if 'name="left_tcp"' not in text:
        marker = '<camera name="left_wrist"'
        if marker not in text:
            raise ValueError("Could not find left_wrist camera to insert left_tcp site")
        lx, ly, lz = LEFT_TCP_LOCAL
        site = (
            f'                  <site name="left_tcp" pos="{lx} {ly} {lz}" '
            f'size="0.012" rgba="0.1 0.9 0.2 0.6" />\n                  '
        )
        text = text.replace(marker, site + marker, 1)

    if 'name="right_tcp"' not in text:
        marker = '<camera name="right_wrist"'
        if marker not in text:
            raise ValueError("Could not find right_wrist camera to insert right_tcp site")
        rx, ry, rz = RIGHT_TCP_LOCAL
        site = (
            f'                  <site name="right_tcp" pos="{rx} {ry} {rz}" '
            f'size="0.012" rgba="0.1 0.55 0.95 0.6" />\n                  '
        )
        text = text.replace(marker, site + marker, 1)

    bx, by, bz = pos
    qw, qx, qy, qz = quat
    hx, hy, hz = handle
    rr, gg, bb, aa = color

    sbx, sby, sbz = shield_pos
    sqw, sqx, sqy, sqz = shield_quat
    shx, shy, shz = shield_handle
    srr, sgg, sbb, saa = shield_color

    asset_insert = (
        f'    <mesh name="sword" content_type="model/stl" '
        f'file="meshes/sword.stl" scale="{scale} {scale} {scale}" />\n'
        f'    <mesh name="helmet" content_type="model/stl" '
        f'file="meshes/helmet.stl" scale="{shield_scale} {shield_scale} {shield_scale}" />\n'
        f"  </asset>"
    )
    if "  </asset>" not in text:
        raise ValueError("Could not find </asset> in base MJCF")
    text = text.replace("  </asset>", asset_insert, 1)

    props = f"""    <!-- Reach-sword: upright floating target (kinematic; no contacts) -->
    <body name="sword" pos="{bx} {by} {bz}" quat="{qw} {qx} {qy} {qz}">
      <geom name="sword_geom" type="mesh" mesh="sword" rgba="{rr} {gg} {bb} {aa}"
            contype="0" conaffinity="0" />
      <site name="sword_handle" pos="{hx} {hy} {hz}" size="0.02" rgba="0.95 0.15 0.1 0.85" />
    </body>
    <!-- Reach-shield: left-arm helmet/shield target (kinematic; no contacts) -->
    <body name="shield" pos="{sbx} {sby} {sbz}" quat="{sqw} {sqx} {sqy} {sqz}">
      <geom name="shield_geom" type="mesh" mesh="helmet" rgba="{srr} {sgg} {sbb} {saa}"
            contype="0" conaffinity="0" />
      <site name="shield_handle" pos="{shx} {shy} {shz}" size="0.02" rgba="0.95 0.15 0.1 0.85" />
    </body>
"""
    if '    <camera name="top"' in text:
        text = text.replace('    <camera name="top"', props + '    <camera name="top"', 1)
    elif "  </worldbody>" in text:
        text = text.replace("  </worldbody>", props + "  </worldbody>", 1)
    else:
        raise ValueError("Could not find insertion point for sword props in worldbody")

    out.write_text(text, encoding="utf-8")
    return out
