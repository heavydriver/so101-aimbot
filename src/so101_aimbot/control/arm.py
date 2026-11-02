"""SO-101 arm interface via LeRobot SO101Follower (with dry-run fallback)."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from so101_aimbot.config import REPO_ROOT, load_config

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ALL_JOINT_NAMES = ARM_JOINT_NAMES + ("gripper",)


def _as_float_list(values: Sequence[float], n: int, label: str) -> list[float]:
    if len(values) != n:
        raise ValueError(f"{label} must have length {n}, got {len(values)}")
    return [float(v) for v in values]


class ArmController:
    """Send joint commands to the SO-101 and read proprioceptive state.

    When ``dry_run`` is true (default if no serial port), joint state is
    simulated in software so planning/control can be exercised without hardware.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._cfg = dict(config) if config is not None else load_config("robot")
        self._connected = False
        self._robot: Any | None = None
        self._dry_run = bool(self._cfg.get("dry_run", True)) or not self._cfg.get("port")
        home = self._cfg.get("home_joints", [0.0] * 6)
        self._joints_deg = _as_float_list(home, 6, "home_joints")
        grip = self._cfg.get("gripper", {})
        self._gripper_open = float(grip.get("open", 40.0))
        self._gripper_closed = float(grip.get("closed", 5.0))
        self._click_hold_s = float(grip.get("click_hold_s", 0.12))
        self._joint_tol_deg = float(self._cfg.get("joint_tolerance_deg", 3.0))
        self._gripper_tol = float(self._cfg.get("gripper_tolerance", 2.0))
        self._settle_timeout_s = float(self._cfg.get("settle_timeout_s", 8.0))
        self._settle_poll_s = float(self._cfg.get("settle_poll_s", 0.02))
        self._settle_vel_tol_deg = float(self._cfg.get("settle_vel_tol_deg", 0.4))
        self._settle_stable_reads = int(self._cfg.get("settle_stable_reads", 3))

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def commanded_joints(self) -> list[float]:
        """Last joint targets written to the bus (or dry-run state)."""
        return list(self._joints_deg)

    def connect(self, *, calibrate: bool = False) -> None:
        """Open the serial / SDK connection, or arm the dry-run simulator."""
        if self._connected:
            return
        if self._dry_run:
            self._connected = True
            return

        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        port = self._cfg["port"]
        robot_id = self._cfg.get("robot_id") or "so101_pad"
        cal_dir = self._cfg.get("calibration_dir")
        config = SO101FollowerConfig(
            port=str(port),
            id=str(robot_id),
            use_degrees=True,
            cameras={},
            calibration_dir=Path(cal_dir) if cal_dir else None,
        )
        self._robot = SO101Follower(config)
        self._robot.connect(calibrate=calibrate)
        self._connected = True
        self._joints_deg = self.read_joints()

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
        self._connected = False

    def read_joints(self) -> list[float]:
        """Return [pan, lift, elbow, wrist_flex, wrist_roll, gripper] in LeRobot units.

        Arm joints are degrees; gripper is in [0, 100].
        """
        self._require_connected()
        if self._dry_run or self._robot is None:
            return list(self._joints_deg)

        obs = self._robot.get_observation()
        return [float(obs[f"{name}.pos"]) for name in ALL_JOINT_NAMES]

    def move_joints(self, joints: Sequence[float], *, wait: bool = False) -> None:
        """Command joint targets in LeRobot units (degrees + gripper 0–100).

        When ``wait`` is true, block until present joints are within tolerance
        of the commanded targets (or until ``settle_timeout_s``).
        """
        self._require_connected()
        targets = _as_float_list(joints, 6, "joints")
        if self._dry_run or self._robot is None:
            self._joints_deg = targets
            return

        action = {f"{name}.pos": targets[i] for i, name in enumerate(ALL_JOINT_NAMES)}
        sent = self._robot.send_action(action)
        self._joints_deg = [float(sent[f"{name}.pos"]) for name in ALL_JOINT_NAMES]
        if wait:
            self.wait_for_joints(self._joints_deg)

    def resync_from_present(self) -> None:
        """Align the commanded-joint cache with the latest proprioceptive reading."""
        self._require_connected()
        self._joints_deg = self.read_joints()

    def wait_for_joints(
        self,
        targets: Sequence[float] | None = None,
        *,
        tol_deg: float | None = None,
        gripper_tol: float | None = None,
        timeout_s: float | None = None,
        poll_s: float | None = None,
    ) -> None:
        """Block until present joints match ``targets`` (default: last commanded)."""
        self._require_connected()
        goal = (
            _as_float_list(targets, 6, "targets")
            if targets is not None
            else list(self._joints_deg)
        )
        if self._dry_run or self._robot is None:
            self._joints_deg = goal
            return

        arm_tol = self._joint_tol_deg if tol_deg is None else float(tol_deg)
        grip_tol = self._gripper_tol if gripper_tol is None else float(gripper_tol)
        timeout = self._settle_timeout_s if timeout_s is None else float(timeout_s)
        poll = self._settle_poll_s if poll_s is None else float(poll_s)
        deadline = time.perf_counter() + timeout
        while True:
            current = self.read_joints()
            arm_ok = all(abs(current[i] - goal[i]) <= arm_tol for i in range(5))
            grip_ok = abs(current[5] - goal[5]) <= grip_tol
            if arm_ok and grip_ok:
                return
            if time.perf_counter() >= deadline:
                errs = [abs(current[i] - goal[i]) for i in range(6)]
                raise TimeoutError(
                    f"Arm did not reach target within {timeout:.1f}s "
                    f"(abs errors deg/[0-100]: {errs})"
                )
            time.sleep(poll)

    def wait_until_settled(
        self,
        *,
        vel_tol_deg: float | None = None,
        stable_reads: int | None = None,
        timeout_s: float | None = None,
        poll_s: float | None = None,
    ) -> None:
        """Block until joint readings stop changing (motion finished).

        Prefer this after Cartesian / IK-streamed moves: servos often hold a
        few degrees of steady-state error vs the last IK setpoint, so absolute
        ``wait_for_joints`` can hang even though the tip has arrived.
        """
        self._require_connected()
        if self._dry_run or self._robot is None:
            return

        vel_tol = self._settle_vel_tol_deg if vel_tol_deg is None else float(vel_tol_deg)
        need_stable = (
            self._settle_stable_reads if stable_reads is None else int(stable_reads)
        )
        timeout = self._settle_timeout_s if timeout_s is None else float(timeout_s)
        poll = self._settle_poll_s if poll_s is None else float(poll_s)
        deadline = time.perf_counter() + timeout
        prev = self.read_joints()
        stable = 0
        while True:
            time.sleep(poll)
            cur = self.read_joints()
            delta = max(abs(cur[i] - prev[i]) for i in range(5))
            prev = cur
            if delta <= vel_tol:
                stable += 1
                if stable >= need_stable:
                    return
            else:
                stable = 0
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"Arm did not settle within {timeout:.1f}s "
                    f"(last inter-sample Δ={delta:.2f} deg)"
                )

    def home(self) -> None:
        """Move to the configured home joint pose and wait until settled."""
        home = self._cfg.get("home_joints", [0.0] * 6)
        self.move_joints(_as_float_list(home, 6, "home_joints"), wait=True)

    def set_gripper(self, position: float, *, wait: bool = True) -> None:
        """Set gripper only; keep arm joints at their last commanded pose."""
        joints = self.read_joints()
        joints[5] = float(position)
        self.move_joints(joints, wait=wait)

    def set_gripper_torque(self, enabled: bool) -> None:
        """Enable or disable gripper motor torque for manual positioning."""
        self._require_connected()
        if self._dry_run or self._robot is None:
            return
        if enabled:
            self._robot.bus.enable_torque("gripper")
        else:
            self._robot.bus.disable_torque("gripper")

    def click(self, *, hold_s: float | None = None) -> None:
        """Pulse the gripper closed then restore the open pose."""
        hold = self._click_hold_s if hold_s is None else float(hold_s)
        open_pos = self._gripper_open
        closed_pos = self._gripper_closed
        self.set_gripper(closed_pos, wait=True)
        time.sleep(hold)
        self.set_gripper(open_pos, wait=True)

    def joints_deg_to_rad(self, joints_deg: Sequence[float]) -> list[float]:
        """Convert LeRobot arm degrees (+ gripper) to radians for kinematics.

        Gripper stays in [0, 100] for the bus, but PyRoki models it as revolute
        radians. We map the normalized gripper to a small synthetic angle for FK/IK
        continuity, then overwrite gripper after IK from the held bus value.
        """
        vals = _as_float_list(joints_deg, 6, "joints_deg")
        arm_rad = [math.radians(v) for v in vals[:5]]
        # Map gripper 0–100 → ~0–1 rad for the URDF revolute joint placeholder.
        grip_rad = (vals[5] / 100.0) * 1.0
        return arm_rad + [grip_rad]

    def joints_rad_to_deg(self, joints_rad: Sequence[float], *, gripper_hold: float) -> list[float]:
        """Convert PyRoki radians back to LeRobot degrees; restore held gripper."""
        vals = _as_float_list(joints_rad, 6, "joints_rad")
        arm_deg = [math.degrees(v) for v in vals[:5]]
        return arm_deg + [float(gripper_hold)]

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Arm is not connected")
