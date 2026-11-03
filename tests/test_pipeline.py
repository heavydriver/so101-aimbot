"""Tests for the synchronous one-target pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from so101_aimbot.pipeline import Pipeline
from so101_aimbot.planning.calibrate import pixel_delta_to_mouse_mm
from so101_aimbot.types import CursorState, FrameBundle, PerceptionResult, Target


class FakeMouse:
    def __init__(self) -> None:
        self.events: list[tuple[str, float, float] | tuple[str]] = []

    def move_xy_mm(self, dx_mm: float, dy_mm: float) -> None:
        self.events.append(("move", dx_mm, dy_mm))

    def click(self) -> None:
        self.events.append(("click",))


def test_run_once_moves_in_planned_direction_without_early_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mouse = FakeMouse()
    pipeline = Pipeline(
        mouse=mouse,  # type: ignore[arg-type]
        perception_config={},
        robot_config={},
        actuate=True,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    perception = PerceptionResult(
        targets=[Target(130.0, 70.0)],
        cursor=CursorState(100.0, 100.0),
        selected=Target(130.0, 70.0),
    )
    monkeypatch.setattr(
        pipeline,
        "perceive",
        lambda _: (FrameBundle(raw=frame), perception),
    )

    step = pipeline.run_once(frame)

    expected_dx, expected_dy = pixel_delta_to_mouse_mm(30.0, -30.0)
    assert step.command is not None
    move = mouse.events[0]
    assert move[0] == "move"
    assert move[1:] == pytest.approx((expected_dx, expected_dy))
    assert mouse.events == [move]


def test_run_once_does_nothing_without_cursor_or_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mouse = FakeMouse()
    pipeline = Pipeline(
        mouse=mouse,  # type: ignore[arg-type]
        perception_config={},
        robot_config={},
        actuate=True,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    perception = PerceptionResult()
    monkeypatch.setattr(
        pipeline,
        "perceive",
        lambda _: (FrameBundle(raw=frame), perception),
    )

    step = pipeline.run_once(frame)

    assert step.command is None
    assert mouse.events == []
