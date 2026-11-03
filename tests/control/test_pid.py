"""Tests for discrete PID helpers and ArmController PID wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from so101_aimbot.control.arm import ALL_JOINT_NAMES, ArmController
from so101_aimbot.control.pid import PID, PIDBank, PIDGains


def test_pid_proportional_only() -> None:
    ctrl = PID(PIDGains(kp=2.0), output_clamp=10.0)
    assert ctrl.update(1.0, 0.01) == pytest.approx(2.0)
    assert ctrl.update(1.0, 0.01) == pytest.approx(2.0)


def test_pid_integral_accumulates() -> None:
    ctrl = PID(PIDGains(ki=1.0), i_clamp=1.0, output_clamp=10.0)
    out1 = ctrl.update(1.0, 0.5)
    assert out1 == pytest.approx(0.5)
    out2 = ctrl.update(1.0, 0.5)
    assert out2 == pytest.approx(1.0)


def test_pid_derivative() -> None:
    ctrl = PID(PIDGains(kd=1.0), output_clamp=10.0)
    ctrl.update(0.0, 0.1)
    assert ctrl.update(2.0, 0.1) == pytest.approx(10.0)


def test_pid_output_clamp() -> None:
    ctrl = PID(PIDGains(kp=10.0), output_clamp=1.5)
    assert ctrl.update(1.0, 0.01) == pytest.approx(1.5)


def test_pid_reset_clears_state() -> None:
    ctrl = PID(PIDGains(kp=1.0))
    ctrl.update(3.0, 0.1)
    ctrl.reset()
    assert ctrl.update(1.0, 0.1) == pytest.approx(1.0)


def test_pid_bank_from_config_scalar() -> None:
    bank = PIDBank.from_config({"kp": 0.2, "ki": 0.0, "kd": 0.0}, 5)
    assert len(bank) == 5
    assert bank.update([1.0] * 5, 0.01) == pytest.approx([0.2] * 5)


def test_move_arm_applies_pid_correction() -> None:
    cfg: dict[str, Any] = {
        "dry_run": False,
        "port": "/dev/null",
        "home_arm_deg": [0.0, 0.0, 0.0, 0.0, 0.0],
        "pid": {
            "enabled": True,
            "arm": {
                "kp": [1.0, 0.0, 0.0, 0.0, 0.0],
                "ki": [0.0] * 5,
                "kd": [0.0] * 5,
                "output_clamp_deg": 5.0,
            },
            "gripper": {"kp": 0.0, "ki": 0.0, "kd": 0.0},
        },
    }
    arm = ArmController(cfg)
    arm._connected = True
    arm._robot = MagicMock()
    measured = [0.0, 10.0, 0.0, 0.0, 0.0, 50.0]
    arm._robot.get_observation.return_value = {
        f"{name}.pos": measured[i] for i, name in enumerate(ALL_JOINT_NAMES)
    }

    def send_action(action: dict[str, float]) -> dict[str, float]:
        return dict(action)

    arm._robot.send_action.side_effect = send_action

    arm.move_arm([5.0, 20.0, 0.0, 0.0, 0.0])
    sent = arm._robot.send_action.call_args[0][0]
    assert sent["shoulder_pan.pos"] == pytest.approx(10.0)
    assert sent["shoulder_lift.pos"] == pytest.approx(20.0)


def test_move_arm_skips_pid_when_disabled() -> None:
    cfg: dict[str, Any] = {
        "dry_run": False,
        "port": "/dev/null",
        "home_arm_deg": [0.0] * 5,
        "pid": {"enabled": False},
    }
    arm = ArmController(cfg)
    arm._connected = True
    arm._robot = MagicMock()
    arm._robot.get_observation.return_value = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 50.0,
    }
    arm._robot.send_action.side_effect = lambda a: dict(a)
    arm.move_arm([5.0] * 5)
    sent = arm._robot.send_action.call_args[0][0]
    assert sent["shoulder_pan.pos"] == pytest.approx(5.0)
