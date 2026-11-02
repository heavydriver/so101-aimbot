"""Pad mouse control: polar slides, home resets, and gripper click."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from so101_aimbot.config import load_config
from so101_aimbot.control.arm import ArmController
from so101_aimbot.planning.kinematics import So101Kinematics
from so101_aimbot.planning.pad_geometry import (
    polar_to_xy_mm,
    ray_to_disk_boundary_mm,
)
from so101_aimbot.planning.ruckig_traj import limits_from_config, run_ruckig
from so101_aimbot.types import MouseCommand


class PadMouseController:
    """Vision-free pad controller: ``move_polar`` / ``go_home`` / ``click``."""

    def __init__(
        self,
        arm: ArmController,
        *,
        kinematics: So101Kinematics | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.arm = arm
        self._cfg = dict(config) if config is not None else load_config("robot")
        urdf = self._cfg.get("urdf_path", "assets/so101/so101_new_calib.urdf")
        target = self._cfg.get("target_link", "gripper_frame_link")
        self.kin = kinematics or So101Kinematics(urdf, target_link=str(target))

        self.max_radius_mm = float(self._cfg.get("max_radius_mm", 40.0))
        contact = self._cfg.get("mouse_pad_contact_z_m")
        self.z_contact_m = float(contact) if contact is not None else None
        lift = self._cfg.get("z_lift_m")
        self.z_lift_m = float(lift) if lift is not None else None
        self.lift_clearance_m = float(self._cfg.get("lift_clearance_m", 0.03))
        tip = self._cfg.get("tip_wxyz")
        self.tip_wxyz = (
            np.asarray(tip, dtype=np.float64)
            if tip is not None
            else None
        )
        self.pad_x_axis = _unit3(self._cfg.get("pad_x_axis", [1.0, 0.0, 0.0]))
        self.pad_y_axis = _unit3(self._cfg.get("pad_y_axis", [0.0, 1.0, 0.0]))
        self.dt_s = float(self._cfg.get("control_dt_s", 0.01))
        self.position_tolerance_mm = float(self._cfg.get("position_tolerance_mm", 0.8))
        self._ruckig = dict(self._cfg.get("ruckig", {}))

        self._home_xyz_m: np.ndarray | None = None
        self._warmed = False

    def connect(self, *, warm_up_ik: bool = True) -> None:
        """Connect the arm and optionally JIT-warm PyRoki IK."""
        if not self.arm.connected:
            self.arm.connect()
        self._refresh_home_from_joints(self.arm.read_joints())
        if warm_up_ik and not self._warmed:
            guess = self.arm.joints_deg_to_rad(self.arm.read_joints())
            self.kin.warm_up(guess)
            self._warmed = True

    def _refresh_home_from_joints(self, joints_deg: Sequence[float]) -> None:
        """Set pad origin from current (or taught home) joint pose FK."""
        home_joints = self._cfg.get("home_joints")
        q_deg = list(home_joints) if home_joints is not None else list(joints_deg)
        pose = self.kin.forward_kinematics(self.arm.joints_deg_to_rad(q_deg))
        self._home_xyz_m = pose.position_m.copy()
        if self.z_contact_m is None:
            self.z_contact_m = float(pose.position_m[2])
        if self.z_lift_m is None:
            self.z_lift_m = float(self.z_contact_m) + self.lift_clearance_m
        if self.tip_wxyz is None:
            self.tip_wxyz = pose.wxyz.copy()

    def teach_home_from_current(self) -> dict[str, Any]:
        """Capture home from the live arm pose (joints + tip)."""
        joints = self.arm.read_joints()
        pose = self.kin.forward_kinematics(self.arm.joints_deg_to_rad(joints))
        self._home_xyz_m = pose.position_m.copy()
        self.z_contact_m = float(pose.position_m[2])
        self.z_lift_m = self.z_contact_m + self.lift_clearance_m
        self.tip_wxyz = pose.wxyz.copy()
        self._cfg["home_joints"] = list(joints)
        self._cfg["mouse_pad_contact_z_m"] = self.z_contact_m
        self._cfg["z_lift_m"] = self.z_lift_m
        self._cfg["tip_wxyz"] = self.tip_wxyz.tolist()
        return {
            "home_joints": list(joints),
            "home_xyz_m": self._home_xyz_m.tolist(),
            "mouse_pad_contact_z_m": self.z_contact_m,
            "z_lift_m": self.z_lift_m,
            "tip_wxyz": self.tip_wxyz.tolist(),
        }

    def pad_xy_mm(self, position_m: Sequence[float]) -> tuple[float, float]:
        """Project a base-frame tip position into pad millimetres relative to home."""
        home = self._require_home()
        delta = np.asarray(position_m, dtype=np.float64) - home
        x = float(np.dot(delta, self.pad_x_axis) * 1000.0)
        y = float(np.dot(delta, self.pad_y_axis) * 1000.0)
        return x, y

    def pad_xy_to_base_m(self, x_mm: float, y_mm: float, z_m: float) -> np.ndarray:
        """Map pad millimetres + absolute Z to a base-frame tip position."""
        home = self._require_home()
        # Home defines pad origin in XY; Z is commanded absolutely.
        xy = home + (x_mm / 1000.0) * self.pad_x_axis + (y_mm / 1000.0) * self.pad_y_axis
        out = xy.copy()
        out[2] = float(z_m)
        return out

    def current_pad_xy_mm(self) -> tuple[float, float]:
        joints = self.arm.read_joints()
        pose = self.kin.forward_kinematics(self.arm.joints_deg_to_rad(joints))
        return self.pad_xy_mm(pose.position_m)

    def move_polar(self, distance_mm: float, direction_deg: float) -> None:
        """Slide on the pad; auto lift→home→drop when hitting ``max_radius_mm``."""
        remaining = float(distance_mm)
        eps = self.position_tolerance_mm

        while remaining > eps:
            x, y = self.current_pad_xy_mm()
            step = min(
                remaining,
                ray_to_disk_boundary_mm((x, y), direction_deg, self.max_radius_mm),
            )
            if step <= eps:
                self.go_home()
                continue

            self._slide_polar(step, direction_deg)
            remaining -= step
            if remaining > eps:
                self.go_home()

    def move_xy_mm(self, dx_mm: float, dy_mm: float) -> None:
        """Relative pad move expressed as Cartesian millimetres."""
        distance, direction = _cart_to_polar(dx_mm, dy_mm)
        self.move_polar(distance, direction)

    def go_home(self) -> None:
        """Lift → air-move to home XY → drop onto the pad."""
        self._lift()
        self._air_to_home()
        self._drop()

    def click(self) -> None:
        """Actuate the gripper only (arm joints held)."""
        self.arm.click()

    def _slide_polar(self, distance_mm: float, direction_deg: float) -> None:
        if self.z_contact_m is None:
            raise RuntimeError("Contact Z not initialized; call connect()")
        dx, dy = polar_to_xy_mm(distance_mm, direction_deg)
        x0, y0 = self.current_pad_xy_mm()
        x1, y1 = x0 + dx, y0 + dy
        # Clamp end to disk for safety.
        r = math_hypot(x1, y1)
        if r > self.max_radius_mm:
            scale = self.max_radius_mm / r
            x1, y1 = x1 * scale, y1 * scale

        vmax, amax, jmax = limits_from_config(self._ruckig["slide"], 2)
        z_contact = float(self.z_contact_m)

        def on_sample(xy_m: list[float]) -> None:
            # Ruckig works in metres in pad XY relative to home.
            self._goto_pose(
                self.pad_xy_to_base_m(xy_m[0] * 1000.0, xy_m[1] * 1000.0, z_contact)
            )

        run_ruckig(
            dofs=2,
            dt_s=self.dt_s,
            current=[x0 / 1000.0, y0 / 1000.0],
            target=[x1 / 1000.0, y1 / 1000.0],
            max_velocity=vmax,
            max_acceleration=amax,
            max_jerk=jmax,
            on_sample=on_sample,
            realtime=not self.arm.dry_run,
        )
        self._finish_segment()

    def _lift(self) -> None:
        if self.z_lift_m is None:
            raise RuntimeError("Lift Z not initialized; call connect()")
        x, y = self.current_pad_xy_mm()
        z0 = self._current_z_m()
        vmax, amax, jmax = limits_from_config(self._ruckig["lift"], 1)
        z_lift = float(self.z_lift_m)

        def on_sample(z: list[float]) -> None:
            self._goto_pose(self.pad_xy_to_base_m(x, y, z[0]))

        run_ruckig(
            dofs=1,
            dt_s=self.dt_s,
            current=[z0],
            target=[z_lift],
            max_velocity=vmax,
            max_acceleration=amax,
            max_jerk=jmax,
            on_sample=on_sample,
            realtime=not self.arm.dry_run,
        )
        self._finish_segment()

    def _drop(self) -> None:
        if self.z_contact_m is None:
            raise RuntimeError("Contact Z not initialized; call connect()")
        x, y = self.current_pad_xy_mm()
        z0 = self._current_z_m()
        vmax, amax, jmax = limits_from_config(self._ruckig["lift"], 1)
        z_contact = float(self.z_contact_m)

        def on_sample(z: list[float]) -> None:
            self._goto_pose(self.pad_xy_to_base_m(x, y, z[0]))

        run_ruckig(
            dofs=1,
            dt_s=self.dt_s,
            current=[z0],
            target=[z_contact],
            max_velocity=vmax,
            max_acceleration=amax,
            max_jerk=jmax,
            on_sample=on_sample,
            realtime=not self.arm.dry_run,
        )
        self._finish_segment()

    def _air_to_home(self) -> None:
        if self.z_lift_m is None:
            raise RuntimeError("Lift Z not initialized; call connect()")
        x0, y0 = self.current_pad_xy_mm()
        vmax, amax, jmax = limits_from_config(self._ruckig["air"], 2)
        z_lift = float(self.z_lift_m)

        def on_sample(xy_m: list[float]) -> None:
            self._goto_pose(
                self.pad_xy_to_base_m(xy_m[0] * 1000.0, xy_m[1] * 1000.0, z_lift)
            )

        run_ruckig(
            dofs=2,
            dt_s=self.dt_s,
            current=[x0 / 1000.0, y0 / 1000.0],
            target=[0.0, 0.0],
            max_velocity=vmax,
            max_acceleration=amax,
            max_jerk=jmax,
            on_sample=on_sample,
            realtime=not self.arm.dry_run,
        )
        self._finish_segment()

    def _finish_segment(self) -> None:
        """Wait for motion to stop, then resync IK seed to measured joints."""
        self.arm.wait_until_settled()
        self.arm.resync_from_present()

    def _goto_pose(self, position_m: Sequence[float]) -> None:
        if self.tip_wxyz is None or self.z_contact_m is None or self.z_lift_m is None:
            raise RuntimeError("Pad heights/orientation not initialized; call connect()")
        # Use last commanded joints as IK seed so streaming stays continuous even
        # when present position lags the setpoint slightly.
        joints_deg = self.arm.commanded_joints
        gripper_hold = joints_deg[5]
        guess = self.arm.joints_deg_to_rad(joints_deg)
        q_rad = self.kin.inverse_kinematics(
            position_m,
            self.tip_wxyz,
            guess_rad_project=guess,
        )
        # Keep gripper out of IK result.
        cmd = self.arm.joints_rad_to_deg(q_rad, gripper_hold=gripper_hold)
        self.arm.move_joints(cmd)

    def _current_z_m(self) -> float:
        joints = self.arm.read_joints()
        pose = self.kin.forward_kinematics(self.arm.joints_deg_to_rad(joints))
        return float(pose.position_m[2])

    def _require_home(self) -> np.ndarray:
        if self._home_xyz_m is None:
            raise RuntimeError("Home pose is not set; call connect() or teach_home_from_current()")
        return self._home_xyz_m


def execute_mouse_command(arm: ArmController, command: MouseCommand) -> None:
    """Execute a relative mouse move and optional click via the pad controller."""
    cfg = load_config("robot")
    pad = PadMouseController(arm, config=cfg)
    if not arm.connected:
        pad.connect(warm_up_ik=True)
    else:
        # Ensure home / kinematics are ready even if arm was connected externally.
        if pad._home_xyz_m is None:
            pad._refresh_home_from_joints(arm.read_joints())
        if not pad._warmed:
            pad.kin.warm_up(arm.joints_deg_to_rad(arm.read_joints()))
            pad._warmed = True

    if abs(command.dx) > 1e-9 or abs(command.dy) > 1e-9:
        pad.move_xy_mm(command.dx, command.dy)
    if command.click:
        pad.click()


def _unit3(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(list(values), dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("axis vector must be non-zero")
    return v / n


def _cart_to_polar(dx_mm: float, dy_mm: float) -> tuple[float, float]:
    import math

    return math.hypot(dx_mm, dy_mm), math.degrees(math.atan2(dy_mm, dx_mm))


def math_hypot(x: float, y: float) -> float:
    import math

    return math.hypot(x, y)
