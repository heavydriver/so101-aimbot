"""SO-101 arm interface via LeRobot SO101Follower (with dry-run fallback)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from so101_aimbot.config import load_config
from so101_aimbot.control.pid import PID, PIDBank, PIDGains

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
        home = self._cfg.get("home_arm_deg")
        if home is None:
            raise ValueError("robot config is missing home_arm_deg")
        self._home_arm_deg = _as_float_list(home, 5, "home_arm_deg")
        self._arm_deg = list(self._home_arm_deg)
        grip = self._cfg.get("gripper", {})
        self._gripper_open = float(grip.get("open", 40.0))
        self._gripper_closed = float(grip.get("closed", 5.0))
        self._gripper_position = self._gripper_open
        self._click_hold_s = float(grip.get("click_hold_s", 0.12))
        self._settle_timeout_s = float(self._cfg.get("settle_timeout_s", 8.0))
        self._settle_poll_s = float(self._cfg.get("settle_poll_s", 0.02))
        self._settle_vel_tol_deg = float(self._cfg.get("settle_vel_tol_deg", 0.4))
        self._settle_stable_reads = int(self._cfg.get("settle_stable_reads", 3))
        pid_cfg = dict(self._cfg.get("pid", {}) or {})
        self._pid_enabled = bool(pid_cfg.get("enabled", False))
        arm_pid = dict(pid_cfg.get("arm", {}) or {})
        grip_pid = dict(pid_cfg.get("gripper", {}) or {})
        self._arm_pid = PIDBank.from_config(arm_pid, 5)
        self._gripper_pid = PID(
            PIDGains(
                float(grip_pid.get("kp", 0.0)),
                float(grip_pid.get("ki", 0.0)),
                float(grip_pid.get("kd", 0.0)),
            ),
            i_clamp=float(grip_pid["i_clamp"])
            if grip_pid.get("i_clamp") is not None
            else None,
            output_clamp=float(grip_pid.get("output_clamp", 5.0)),
        )
        self._gripper_pid_tolerance = float(grip_pid.get("tolerance", 2.0))
        self._last_arm_cmd_time: float | None = None

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def commanded_arm(self) -> list[float]:
        """Last arm targets written to the bus, in degrees."""
        return list(self._arm_deg)

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
        self._reset_pid()
        self._arm_deg = self.read_arm()
        self._gripper_position = self.read_gripper()

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
        self._connected = False

    def read_joints(self) -> list[float]:
        """Return five arm angles in degrees followed by gripper position."""
        self._require_connected()
        if self._dry_run or self._robot is None:
            return [*self._arm_deg, self._gripper_position]

        obs = self._robot.get_observation()
        return [float(obs[f"{name}.pos"]) for name in ALL_JOINT_NAMES]

    def read_arm(self) -> list[float]:
        return self.read_joints()[:5]

    def read_gripper(self) -> float:
        return self.read_joints()[5]

    def move_arm(self, arm_deg: Sequence[float]) -> None:
        """Command only the five arm joints; never include the gripper."""
        self._require_connected()
        targets = _as_float_list(arm_deg, 5, "arm_deg")
        if self._dry_run or self._robot is None:
            self._arm_deg = targets
            return

        send_targets = targets
        if self._pid_enabled:
            now = time.perf_counter()
            dt_s = (
                now - self._last_arm_cmd_time
                if self._last_arm_cmd_time is not None
                else self._settle_poll_s
            )
            self._last_arm_cmd_time = now
            measured = self.read_arm()
            errors = [targets[i] - measured[i] for i in range(5)]
            corrections = self._arm_pid.update(errors, dt_s)
            send_targets = [
                targets[i] + corrections[i] for i in range(5)
            ]

        action = {
            f"{name}.pos": send_targets[i] for i, name in enumerate(ARM_JOINT_NAMES)
        }
        sent = self._robot.send_action(action)
        self._arm_deg = [float(sent[f"{name}.pos"]) for name in ARM_JOINT_NAMES]

    def resync_from_present(self) -> None:
        """Align the command cache with the measured arm state."""
        self._require_connected()
        self._reset_pid()
        self._arm_deg = self.read_arm()

    def wait_until_settled(
        self,
        *,
        vel_tol_deg: float | None = None,
        stable_reads: int | None = None,
        timeout_s: float | None = None,
        poll_s: float | None = None,
    ) -> None:
        """Block until joint readings stop changing (motion finished).

        This is more reliable than waiting for exact servo setpoints, because
        the physical arm can settle with a small steady-state error.
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
        prev = self.read_arm()
        stable = 0
        while True:
            time.sleep(poll)
            cur = self.read_arm()
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
        """Move the arm to its configured home angles."""
        self._reset_pid()
        self.move_arm(self._home_arm_deg)
        self.wait_until_settled()

    def set_gripper(self, position: float, *, wait: bool = True) -> None:
        """Command only the gripper on its native 0–100 scale."""
        self._require_connected()
        target = float(position)
        if not 0.0 <= target <= 100.0:
            raise ValueError("gripper position must be between 0 and 100")
        if self._dry_run or self._robot is None:
            self._gripper_position = target
            return

        if self._pid_enabled and wait:
            self._gripper_pid.reset()
            self._wait_for_gripper_pid(target)
            return

        sent = self._robot.send_action({"gripper.pos": target})
        self._gripper_position = float(sent["gripper.pos"])
        if wait:
            self._wait_for_gripper(self._gripper_position)

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

    def _wait_for_gripper(self, target: float) -> None:
        tol = self._gripper_pid_tolerance
        deadline = time.perf_counter() + self._settle_timeout_s
        while True:
            current = self.read_gripper()
            if abs(current - target) <= tol:
                self._gripper_position = current
                return
            if time.perf_counter() >= deadline:
                raise TimeoutError("Gripper did not reach its target")
            time.sleep(self._settle_poll_s)

    def _wait_for_gripper_pid(self, target: float) -> None:
        tol = self._gripper_pid_tolerance
        deadline = time.perf_counter() + self._settle_timeout_s
        prev_t = time.perf_counter()
        while True:
            current = self.read_gripper()
            error = target - current
            if abs(error) <= tol:
                self._gripper_position = current
                return
            if time.perf_counter() >= deadline:
                raise TimeoutError("Gripper did not reach its target")
            now = time.perf_counter()
            dt_s = now - prev_t
            prev_t = now
            correction = self._gripper_pid.update(error, dt_s)
            cmd = max(0.0, min(100.0, target + correction))
            sent = self._robot.send_action({"gripper.pos": cmd})
            self._gripper_position = float(sent["gripper.pos"])
            time.sleep(self._settle_poll_s)

    def _reset_pid(self) -> None:
        self._arm_pid.reset()
        self._gripper_pid.reset()
        self._last_arm_cmd_time = None

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Arm is not connected")
