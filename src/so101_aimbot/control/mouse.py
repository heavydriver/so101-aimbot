"""Mouse motion built from FK, one IK solve, and one joint trajectory."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from so101_aimbot.config import load_config
from so101_aimbot.control.arm import ArmController
from so101_aimbot.planning.kinematics import So101Kinematics
from so101_aimbot.planning.pad_geometry import (
    polar_to_xy_mm,
    ray_to_disk_boundary_mm,
)
from so101_aimbot.planning.ruckig_traj import JointLimits, run_joint_trajectory
from so101_aimbot.types import MouseCommand


class MouseController:
    """Move the mouse in pad coordinates and click with the gripper."""

    def __init__(
        self,
        arm: ArmController,
        *,
        kinematics: So101Kinematics | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.arm = arm
        self.config = dict(config) if config is not None else load_config("robot")
        self.home_arm_deg = _home_arm(self.config)
        self.wrist_roll_deg = self.home_arm_deg[4]
        self.kinematics = kinematics or So101Kinematics(
            self.config.get("urdf_path", "assets/so101/so101_new_calib.urdf"),
            self.home_arm_deg,
            target_frame=str(self.config.get("target_frame", "gripper_frame_link")),
        )

        self.home_xyz_mm = self.kinematics.home_pose.position_mm.copy()
        self.contact_z_mm = float(self.home_xyz_mm[2])
        self.max_radius_mm = float(self.config.get("max_radius_mm", 150.0))
        self.lift_mm = float(self.config.get("lift_mm", 50.0))
        self.position_tolerance_mm = float(
            self.config.get("position_tolerance_mm", 0.8)
        )
        motion = dict(self.config.get("motion", {}))
        self.dt_s = float(motion.get("dt_s", 0.01))
        self.limits = JointLimits.from_config(motion)

    def connect(self) -> None:
        if not self.arm.connected:
            self.arm.connect()

    def teach_home_from_current(self) -> dict[str, Any]:
        """Return the calibration values that should be persisted."""
        arm_deg = self.arm.read_arm()
        return {
            "home_arm_deg": arm_deg,
            "max_radius_mm": self.max_radius_mm,
            "lift_mm": self.lift_mm,
        }

    def current_xyz_mm(self) -> np.ndarray:
        return self.kinematics.forward(self.arm.read_arm()[:4]).position_mm

    def current_pad_xy_mm(self) -> tuple[float, float]:
        delta = self.current_xyz_mm() - self.home_xyz_mm
        return float(delta[0]), float(delta[1])

    def move_xy_mm(self, dx_mm: float, dy_mm: float) -> None:
        distance = math.hypot(dx_mm, dy_mm)
        direction = math.degrees(math.atan2(dy_mm, dx_mm))
        self.move_polar(distance, direction)

    def move_polar(self, distance_mm: float, direction_deg: float) -> None:
        """Move by distance and angle, resetting home at the pad boundary."""
        remaining = float(distance_mm)
        if remaining < 0.0:
            raise ValueError("distance_mm must be non-negative")

        resets = 0
        while remaining > self.position_tolerance_mm:
            x_mm, y_mm = self.current_pad_xy_mm()
            available = ray_to_disk_boundary_mm(
                (x_mm, y_mm), direction_deg, self.max_radius_mm
            )
            step_mm = min(remaining, available)
            if step_mm <= self.position_tolerance_mm:
                if resets >= 2:
                    raise RuntimeError("Mouse motion made no progress after homing")
                self.go_home()
                resets += 1
                continue

            dx_mm, dy_mm = polar_to_xy_mm(step_mm, direction_deg)
            self._move_to(
                [x_mm + dx_mm, y_mm + dy_mm, self.contact_z_mm],
                relative_to_home=True,
            )
            remaining -= step_mm
            resets = 0
            if remaining > self.position_tolerance_mm:
                self.go_home()

    def go_home(self) -> None:
        """Lift, move above home, then drop onto home."""
        current = self.current_xyz_mm()
        lift_z = self.contact_z_mm + self.lift_mm
        self._move_to([current[0], current[1], lift_z])
        self._move_to([self.home_xyz_mm[0], self.home_xyz_mm[1], lift_z])
        self._move_to(self.home_xyz_mm)

    def click(self) -> None:
        self.arm.click()

    def _move_to(
        self,
        xyz_mm: Sequence[float],
        *,
        relative_to_home: bool = False,
    ) -> None:
        target = np.asarray(xyz_mm, dtype=float).reshape(3)
        if relative_to_home:
            target[:2] += self.home_xyz_mm[:2]

        current_arm = self.arm.read_arm()
        target_active = self.kinematics.inverse(
            target,
            initial_deg=current_arm[:4],
            tolerance_mm=self.position_tolerance_mm,
        )

        def command(active_deg: list[float]) -> None:
            self.arm.move_arm([*active_deg, self.wrist_roll_deg])

        run_joint_trajectory(
            dt_s=self.dt_s,
            current_deg=current_arm[:4],
            target_deg=target_active,
            limits=self.limits,
            on_sample=command,
            realtime=not self.arm.dry_run,
        )
        self.arm.wait_until_settled()
        self.arm.resync_from_present()


def execute_mouse_command(arm: ArmController, command: MouseCommand) -> None:
    """Execute a relative move and optional click."""
    mouse = MouseController(arm)
    mouse.connect()
    if abs(command.dx) > 1e-9 or abs(command.dy) > 1e-9:
        mouse.move_xy_mm(command.dx, command.dy)
    if command.click:
        mouse.click()


def _home_arm(config: Mapping[str, Any]) -> list[float]:
    values = config.get("home_arm_deg")
    if values is None:
        raise ValueError("robot config is missing home_arm_deg")
    home = [float(value) for value in values]
    if len(home) != 5:
        raise ValueError("home_arm_deg must contain five angles")
    return home
