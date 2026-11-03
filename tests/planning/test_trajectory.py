"""Tests for cursor-to-target command planning."""

from __future__ import annotations

from so101_aimbot.planning.trajectory import plan_mouse_command
from so101_aimbot.types import CursorState, Target


def test_does_not_click_while_cursor_is_outside_target() -> None:
    command = plan_mouse_command(
        CursorState(x=80.0, y=100.0),
        Target(x=100.0, y=100.0, radius_px=12.0),
    )

    assert command is not None
    assert command.dx > 0.0
    assert command.click is False


def test_clicks_when_cursor_is_inside_target() -> None:
    command = plan_mouse_command(
        CursorState(x=95.0, y=104.0),
        Target(x=100.0, y=100.0, radius_px=12.0),
    )

    assert command is not None
    assert command.click is True


def test_clicks_when_cursor_is_on_target_edge() -> None:
    command = plan_mouse_command(
        CursorState(x=88.0, y=100.0),
        Target(x=100.0, y=100.0, radius_px=12.0),
    )

    assert command is not None
    assert command.click is True
