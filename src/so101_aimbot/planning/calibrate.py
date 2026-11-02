"""Calibration helpers: mouse↔cursor scale and robot-only pad teach."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from so101_aimbot.config import CONFIG_DIR, REPO_ROOT
from so101_aimbot.control.arm import ArmController


@dataclass(frozen=True, slots=True)
class MouseCursorCalibration:
    """Linear map from mouse pad Δmm to cursor Δpx."""

    px_per_mm_x: float = 1.0
    px_per_mm_y: float = 1.0


def load_calibration() -> MouseCursorCalibration:
    """Load a saved cursor calibration (identity if missing)."""
    path = CONFIG_DIR / "mouse_cursor_calibration.yaml"
    if not path.is_file():
        return MouseCursorCalibration()
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return MouseCursorCalibration(
        px_per_mm_x=float(data.get("px_per_mm_x", 1.0)),
        px_per_mm_y=float(data.get("px_per_mm_y", 1.0)),
    )


def pixel_delta_to_mouse_mm(
    dx_px: float,
    dy_px: float,
    calibration: MouseCursorCalibration | None = None,
) -> tuple[float, float]:
    """Convert a desired cursor pixel delta into mouse pad millimetres."""
    cal = calibration or load_calibration()
    return dx_px / cal.px_per_mm_x, dy_px / cal.px_per_mm_y


def teach_pad_home(
    arm: ArmController | None = None,
    *,
    save: bool = True,
    robot_yaml: Path | None = None,
) -> dict[str, Any]:
    """Teach home / contact Z / tip orientation from the current arm pose.

    Robot-only: no camera. Place the mouse on the pad at the preferred rest pose,
    then call this to write ``home_joints``, ``mouse_pad_contact_z_m``, ``z_lift_m``,
    and ``tip_wxyz`` into ``configs/robot.yaml``.
    """
    from so101_aimbot.control.mouse import PadMouseController

    own_arm = arm is None
    controller = arm or ArmController()
    if own_arm:
        controller.connect()
    try:
        pad = PadMouseController(controller)
        pad.connect(warm_up_ik=True)
        taught = pad.teach_home_from_current()
        taught["max_radius_mm"] = pad.max_radius_mm
        taught["lift_clearance_m"] = pad.lift_clearance_m
        taught["pad_x_axis"] = pad.pad_x_axis.tolist()
        taught["pad_y_axis"] = pad.pad_y_axis.tolist()
        if save:
            path = robot_yaml or (CONFIG_DIR / "robot.yaml")
            _merge_robot_yaml(path, taught)
            taught["saved_to"] = str(path)
        return taught
    finally:
        if own_arm:
            controller.disconnect()


def set_max_radius_mm(radius_mm: float, *, robot_yaml: Path | None = None) -> None:
    """Update ``max_radius_mm`` in ``configs/robot.yaml``."""
    if radius_mm <= 0.0:
        raise ValueError("max_radius_mm must be positive")
    path = robot_yaml or (CONFIG_DIR / "robot.yaml")
    _merge_robot_yaml(path, {"max_radius_mm": float(radius_mm)})


def teach_gripper(
    arm: ArmController | None = None,
    *,
    open_pos: float | None = None,
    closed_pos: float | None = None,
    save: bool = True,
    sync_home: bool = True,
    robot_yaml: Path | None = None,
    prompt: Callable[[str], str] = input,
) -> dict[str, Any]:
    """Teach gripper open/closed positions and write them to ``configs/robot.yaml``.

    If ``open_pos`` / ``closed_pos`` are omitted, the gripper motor torque is
    disabled so you can move it by hand; press Enter at each pose to capture.
    On SO-101 the gripper is normalized to [0, 100] (open is typically larger).
    """
    own_arm = arm is None
    controller = arm or ArmController()
    if own_arm:
        controller.connect()
    try:
        grip_cfg = dict(controller._cfg.get("gripper", {}) or {})
        click_hold_s = float(grip_cfg.get("click_hold_s", 0.12))
        captured_open = open_pos
        captured_closed = closed_pos

        if captured_open is None or captured_closed is None:
            print(
                "Gripper torque will be disabled so you can move it by hand.\n"
                "Leave the arm joints alone; only adjust the gripper.\n"
            )
            controller.set_gripper_torque(False)
            try:
                if captured_open is None:
                    prompt(
                        "Move the gripper to the OPEN pose (loose on the mouse / fully open), "
                        "then press ENTER…"
                    )
                    captured_open = float(controller.read_joints()[5])
                    print(f"  captured open={captured_open:.4f}")
                if captured_closed is None:
                    prompt(
                        "Move the gripper to the CLOSED pose (firm click on the mouse), "
                        "then press ENTER…"
                    )
                    captured_closed = float(controller.read_joints()[5])
                    print(f"  captured closed={captured_closed:.4f}")
            finally:
                controller.set_gripper_torque(True)
                hold = captured_closed if captured_closed is not None else captured_open
                if hold is not None:
                    controller.set_gripper(float(hold))

        if captured_open is None or captured_closed is None:
            raise RuntimeError("Failed to capture both open and closed gripper positions")

        open_val = float(captured_open)
        closed_val = float(captured_closed)
        if not (0.0 <= closed_val <= open_val <= 100.0):
            raise ValueError(
                f"Expected 0 <= closed ({closed_val}) <= open ({open_val}) <= 100"
            )

        taught: dict[str, Any] = {
            "gripper": {
                "open": open_val,
                "closed": closed_val,
                "click_hold_s": click_hold_s,
            }
        }
        if sync_home:
            home = list(controller._cfg.get("home_joints", [0.0] * 6))
            if len(home) != 6:
                home = [0.0] * 6
            home[5] = open_val
            taught["home_joints"] = home

        controller._gripper_open = open_val
        controller._gripper_closed = closed_val
        controller._cfg["gripper"] = taught["gripper"]
        if "home_joints" in taught:
            controller._cfg["home_joints"] = taught["home_joints"]

        if save:
            path = robot_yaml or (CONFIG_DIR / "robot.yaml")
            _merge_robot_yaml(path, taught)
            taught["saved_to"] = str(path)
        return taught
    finally:
        if own_arm:
            controller.disconnect()


def _merge_robot_yaml(path: Path, updates: dict[str, Any]) -> None:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    existing: dict[str, Any] = {}
    if path.is_file():
        with path.open() as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                existing = loaded
    # Drop bookkeeping keys that should not be written as config.
    skip = {"home_xyz_m", "saved_to"}
    for key, value in updates.items():
        if key in skip:
            continue
        existing[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(existing, f, default_flow_style=False, sort_keys=False)


def save_cursor_calibration(cal: MouseCursorCalibration) -> Path:
    """Persist mouse↔cursor scale to ``configs/mouse_cursor_calibration.yaml``."""
    path = CONFIG_DIR / "mouse_cursor_calibration.yaml"
    with path.open("w") as f:
        yaml.safe_dump(asdict(cal), f, default_flow_style=False)
    return path
