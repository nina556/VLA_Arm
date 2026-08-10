"""Build the table-place MJCF from the free-space Unoarm XML."""

from __future__ import annotations

import re
from pathlib import Path

from .constants import (
    LEFT_TCP_LOCAL,
    PEG_DEFAULT_XY,
    PEG_HALF_HEIGHT,
    PEG_RADIUS,
    PEG_RGBA,
    RIGHT_TCP_LOCAL,
    TABLE_CIRCLE_CENTER_XY,
    TABLE_CIRCLE_RADIUS,
    TABLE_CIRCLE_RGBA,
    TABLE_PLACE_HALF,
    TABLE_PLACE_POS,
    TABLE_PLACE_RGBA,
    TABLE_PLACE_XML_PATH,
    XML_PATH,
    peg_body_pos_from_xy,
    table_place_top_z,
)


def _strip_table_place_blocks(text: str) -> str:
    text = re.sub(
        r'\s*<!-- Table-place:.*?-->\s*<body name="table"[\s\S]*?</body>\s*',
        "\n",
        text,
        count=1,
    )
    text = re.sub(
        r'\s*<!-- Table-place peg:.*?-->\s*<body name="peg"[\s\S]*?</body>\s*',
        "\n",
        text,
        count=1,
    )
    text = re.sub(
        r'\s*<!-- Table-place circle:.*?-->\s*<site name="target_circle"[\s\S]*?/>\s*',
        "\n",
        text,
        count=1,
    )
    text = re.sub(r'\s*<body name="table"[\s\S]*?</body>\s*', "\n", text, count=1)
    text = re.sub(r'\s*<body name="peg"[\s\S]*?</body>\s*', "\n", text, count=1)
    text = re.sub(r'\s*<site name="target_circle"[^/]*/>\s*', "\n", text, count=1)
    return text


def ensure_table_place_xml(
    *,
    base_xml: Path | None = None,
    out_xml: Path | None = None,
    peg_xy: tuple[float, float] | None = None,
) -> Path:
    """Inject right/left TCP sites, table, peg cylinder, and fixed target circle."""
    base = Path(base_xml) if base_xml is not None else XML_PATH
    out = Path(out_xml) if out_xml is not None else TABLE_PLACE_XML_PATH
    if not base.exists():
        raise FileNotFoundError(
            f"Missing base MJCF: {base}. Run "
            "`uv run python custom_envs/unoarm/scripts/01_convert_urdf_to_mjcf.py` first."
        )

    text = base.read_text(encoding="utf-8")
    text = _strip_table_place_blocks(text)

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

    tx, ty, tz = (float(v) for v in TABLE_PLACE_POS)
    hx, hy, hz = (float(v) for v in TABLE_PLACE_HALF)
    tr, tg, tb, ta = (float(v) for v in TABLE_PLACE_RGBA)

    xy = peg_xy if peg_xy is not None else PEG_DEFAULT_XY
    px, py, pz = peg_body_pos_from_xy(xy)
    pr, pg, pb, pa = (float(v) for v in PEG_RGBA)
    peg_r = float(PEG_RADIUS)
    peg_hh = float(PEG_HALF_HEIGHT)

    cx, cy = (float(v) for v in TABLE_CIRCLE_CENTER_XY)
    cz = table_place_top_z() + 0.001
    cr = float(TABLE_CIRCLE_RADIUS)
    crr, cgg, cbb, caa = (float(v) for v in TABLE_CIRCLE_RGBA)

    props = f"""    <!-- Table-place: fixed table (kinematic; no contacts) -->
    <body name="table" pos="{tx} {ty} {tz}">
      <geom name="table_geom" type="box" size="{hx} {hy} {hz}"
            rgba="{tr} {tg} {tb} {ta}" contype="0" conaffinity="0" />
    </body>
    <!-- Table-place peg: upright cylinder; grasp site at top -->
    <body name="peg" pos="{px} {py} {pz}">
      <geom name="peg_geom" type="cylinder" size="{peg_r} {peg_hh}"
            rgba="{pr} {pg} {pb} {pa}" contype="0" conaffinity="0" />
      <site name="peg_grasp" pos="0 0 {peg_hh}" size="0.012" rgba="0.95 0.15 0.1 0.85" />
    </body>
    <!-- Table-place circle: fixed target marker on table top -->
    <site name="target_circle" pos="{cx} {cy} {cz}" size="{cr} 0.002"
          type="cylinder" rgba="{crr} {cgg} {cbb} {caa}" />
"""
    if '    <camera name="top"' in text:
        text = text.replace('    <camera name="top"', props + '    <camera name="top"', 1)
    elif "  </worldbody>" in text:
        text = text.replace("  </worldbody>", props + "  </worldbody>", 1)
    else:
        raise ValueError("Could not find insertion point for table-place props in worldbody")

    out.write_text(text, encoding="utf-8")
    return out
