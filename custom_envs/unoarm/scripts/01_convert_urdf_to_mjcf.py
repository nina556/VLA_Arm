from __future__ import annotations

import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "gym_unoarm"
DEFAULT_MODEL_DIR = Path("/mnt/d/work/code/unoarm_model")
MODEL_DIR = Path(os.environ.get("UNOARM_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser()
SRC_URDF = MODEL_DIR / "unoarm_mujoco.urdf"
SRC_MESH_DIR = MODEL_DIR / "mujoco_meshes"
STAGED_URDF = PACKAGE_DIR / "unoarm_mujoco.urdf"
MESH_DST = PACKAGE_DIR / "meshes"
OUT_XML = PACKAGE_DIR / "mujoco_unoarm.xml"


def copy_assets() -> None:
    if not SRC_URDF.exists():
        raise FileNotFoundError(f"URDF not found: {SRC_URDF}")
    if not SRC_MESH_DIR.exists():
        raise FileNotFoundError(f"Mesh directory not found: {SRC_MESH_DIR}")

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    MESH_DST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_URDF, STAGED_URDF)
    for src in SRC_MESH_DIR.iterdir():
        if src.is_file():
            shutil.copy2(src, MESH_DST / src.name)


def ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def remove_named(parent: ET.Element, tag: str, names: set[str]) -> None:
    for child in list(parent.findall(tag)):
        if child.get("name") in names:
            parent.remove(child)


def find_body(element: ET.Element, name: str) -> ET.Element | None:
    for body in element.iter("body"):
        if body.get("name") == name:
            return body
    return None


def add_cameras(xml_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = ensure_child(root, "worldbody")

    remove_named(worldbody, "camera", {"top"})
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "top",
            "pos": "0 -1.4 1.7",
            "xyaxes": "1 0 0 0 0.65 0.76",
            "fovy": "60",
        },
    )

    for body_name, camera_name, pos, xyaxes in (
        ("Left_Link7", "left_wrist", "0.16 0 0.08", "0 -1 0 0 0 1"),
        ("Right_Link7", "right_wrist", "0.16 0 0.08", "0 -1 0 0 0 1"),
    ):
        body = find_body(worldbody, body_name)
        if body is None:
            raise ValueError(f"Body not found in MJCF: {body_name}")
        remove_named(body, "camera", {camera_name})
        ET.SubElement(
            body,
            "camera",
            {
                "name": camera_name,
                "pos": pos,
                "xyaxes": xyaxes,
                "fovy": "70",
            },
        )

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=False)


def add_position_actuators(xml_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    actuator = ensure_child(root, "actuator")
    joint_names = (
        "Left_Joint1",
        "Left_Joint2",
        "Left_Joint3",
        "Left_Joint4",
        "Left_Joint5",
        "Left_Joint6",
        "Left_Joint7",
        "Left_Gripper_Joint",
        "Right_Joint1",
        "Right_Joint2",
        "Right_Joint3",
        "Right_Joint4",
        "Right_Joint5",
        "Right_Joint6",
        "Right_Joint7",
        "Right_Gripper_Joint",
    )
    remove_named(actuator, "position", {f"{name}_actuator" for name in joint_names})
    for name in joint_names:
        ET.SubElement(actuator, "position", {"name": f"{name}_actuator", "joint": name, "kp": "80"})
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    copy_assets()
    model = mujoco.MjModel.from_xml_path(str(STAGED_URDF))
    mujoco.mj_saveLastXML(str(OUT_XML), model)
    add_cameras(OUT_XML)
    add_position_actuators(OUT_XML)
    model = mujoco.MjModel.from_xml_path(str(OUT_XML))
    print(f"wrote {OUT_XML}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}, ncam={model.ncam}")
    for i in range(model.ncam):
        print(f"camera {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)}")


if __name__ == "__main__":
    sys.exit(main())
