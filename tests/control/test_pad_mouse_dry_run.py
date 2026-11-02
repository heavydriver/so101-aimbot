"""Dry-run integration for PadMouseController (no hardware)."""

from __future__ import annotations

import pytest

from so101_aimbot.config import load_config
from so101_aimbot.control import ArmController, PadMouseController


@pytest.fixture(scope="module")
def pad() -> PadMouseController:
    cfg = load_config("robot")
    cfg = {**cfg, "dry_run": True, "port": None, "control_dt_s": 0.05}
    # Larger dt → fewer IK samples in tests.
    arm = ArmController(cfg)
    controller = PadMouseController(arm, config=cfg)
    controller.connect(warm_up_ik=True)
    arm.home()
    return controller


def test_short_move_stays_near_command(pad: PadMouseController) -> None:
    pad.arm.home()
    pad.move_polar(5.0, 0.0)
    x, y = pad.current_pad_xy_mm()
    assert x == pytest.approx(5.0, abs=2.0)
    assert y == pytest.approx(0.0, abs=2.0)


def test_oversize_triggers_reset_and_consumes_distance(pad: PadMouseController) -> None:
    pad.arm.home()
    # Force a reset: travel more than max radius along +X.
    distance = pad.max_radius_mm * 1.5
    pad.move_polar(distance, 0.0)
    x, y = pad.current_pad_xy_mm()
    # After final segment from home, tip should be near (remaining-after-first-leg).
    # First leg uses full radius, reset, second leg = 0.5 * radius.
    assert abs(x) <= pad.max_radius_mm + 2.0
    assert y == pytest.approx(0.0, abs=2.0)


def test_click_changes_gripper(pad: PadMouseController) -> None:
    open_pos = pad.arm._gripper_open
    pad.click()
    assert pad.arm.read_joints()[5] == pytest.approx(open_pos)
