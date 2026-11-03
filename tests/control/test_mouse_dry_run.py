"""Dry-run integration tests for the mouse and gripper controllers."""

from __future__ import annotations

import numpy as np
import pytest

from so101_aimbot.config import load_config
from so101_aimbot.control import ArmController, MouseController


@pytest.fixture()
def mouse() -> MouseController:
    config = {**load_config("robot"), "dry_run": True, "port": None}
    config["motion"] = {**config["motion"], "dt_s": 0.05}
    arm = ArmController(config)
    controller = MouseController(arm, config=config)
    controller.connect()
    arm.home()
    return controller


def test_move_uses_fixed_contact_height(mouse: MouseController) -> None:
    mouse.move_xy_mm(5.0, 0.0)
    xyz = mouse.current_xyz_mm()
    assert xyz[0] - mouse.home_xyz_mm[0] == pytest.approx(5.0, abs=1.0)
    assert xyz[1] - mouse.home_xyz_mm[1] == pytest.approx(0.0, abs=1.0)
    assert xyz[2] == pytest.approx(mouse.contact_z_mm, abs=1.0)
    assert mouse.arm.read_arm()[4] == pytest.approx(mouse.wrist_roll_deg)


def test_pad_boundary_resets_home(mouse: MouseController) -> None:
    mouse.max_radius_mm = 5.0
    mouse.move_polar(7.5, 0.0)
    x_mm, y_mm = mouse.current_pad_xy_mm()
    assert x_mm == pytest.approx(2.5, abs=1.0)
    assert y_mm == pytest.approx(0.0, abs=1.0)


def test_go_home_uses_lift_translate_drop(
    mouse: MouseController, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = mouse.current_xyz_mm()
    targets: list[np.ndarray] = []
    monkeypatch.setattr(
        mouse,
        "_move_to",
        lambda target, **_: targets.append(np.asarray(target, dtype=float)),
    )

    mouse.go_home()

    lift_z = mouse.contact_z_mm + mouse.lift_mm
    assert np.allclose(targets[0], [current[0], current[1], lift_z])
    assert np.allclose(
        targets[1], [mouse.home_xyz_mm[0], mouse.home_xyz_mm[1], lift_z]
    )
    assert np.allclose(targets[2], mouse.home_xyz_mm)


def test_click_does_not_change_arm(mouse: MouseController) -> None:
    before = mouse.arm.read_arm()
    mouse.click()
    assert mouse.arm.read_arm() == before
    assert mouse.arm.read_gripper() == pytest.approx(mouse.arm._gripper_open)


def test_gripper_range_is_validated(mouse: MouseController) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        mouse.arm.set_gripper(101.0)
