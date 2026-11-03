"""Synchronous perception → planning → mouse-control pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from so101_aimbot.config import load_config
from so101_aimbot.control.arm import ArmController
from so101_aimbot.control.mouse import MouseController
from so101_aimbot.perception.detect import detect_cursor, detect_targets
from so101_aimbot.perception.preprocess import find_game_corners, warp_game_region
from so101_aimbot.perception.select import select_nearest_target
from so101_aimbot.planning.trajectory import plan_mouse_command
from so101_aimbot.types import FrameBundle, MouseCommand, PerceptionResult


@dataclass(slots=True)
class PipelineStep:
    """Everything produced from one camera frame."""

    frames: FrameBundle
    perception: PerceptionResult
    command: MouseCommand | None


class Pipeline:
    """Process exactly one target at a time.

    ``run_once`` does not return until the mouse move has settled and the click
    has completed. The caller can therefore safely capture a fresh frame as
    soon as it returns.
    """

    def __init__(
        self,
        arm: ArmController | None = None,
        mouse: MouseController | None = None,
        *,
        perception_config: dict[str, Any] | None = None,
        robot_config: dict[str, Any] | None = None,
        actuate: bool = False,
    ) -> None:
        self.perception_config = (
            perception_config if perception_config is not None else load_config("perception")
        )
        self.robot_config = (
            robot_config if robot_config is not None else load_config("robot")
        )
        self.mouse = mouse
        if self.mouse is None and arm is not None:
            self.mouse = MouseController(arm, config=self.robot_config)

        self.actuate = bool(actuate)
        if self.actuate and self.mouse is None:
            raise ValueError("actuate=True requires a mouse controller or arm")

        self._corners: NDArray[np.float32] | None = None

    def perceive(self, frame: NDArray[np.uint8]) -> tuple[FrameBundle, PerceptionResult]:
        """Detect the cursor and targets, then choose the nearest target."""
        if self._corners is None:
            self._corners = find_game_corners(frame, config=self.perception_config)

        warped: NDArray[np.uint8] | None = None
        work = frame
        if self._corners is not None:
            size = (
                int(self.perception_config.get("warp_width", 960)),
                int(self.perception_config.get("warp_height", 540)),
            )
            warped = warp_game_region(frame, self._corners, size=size)
            work = warped

        targets = detect_targets(work, config=self.perception_config)
        cursor = detect_cursor(work, config=self.perception_config)
        selected = select_nearest_target(targets, cursor)
        return (
            FrameBundle(raw=frame, warped=warped),
            PerceptionResult(targets=targets, cursor=cursor, selected=selected),
        )

    @staticmethod
    def plan_perception(perception: PerceptionResult) -> MouseCommand | None:
        """Convert target-minus-cursor pixels into mouse-pad millimetres."""
        return plan_mouse_command(perception.cursor, perception.selected)

    def plan(self, frame: NDArray[np.uint8]) -> MouseCommand | None:
        """Plan from a frame without moving the robot."""
        _, perception = self.perceive(frame)
        return self.plan_perception(perception)

    def execute(self, command: MouseCommand) -> None:
        """Move, wait for the controller to settle, and then click."""
        if not self.actuate:
            return
        if self.mouse is None:
            raise RuntimeError("No mouse controller is configured")

        # MouseCommand is already expressed in the controller's pad axes.
        # move_xy_mm is synchronous and waits for the arm to settle.
        if abs(command.dx) > 1e-9 or abs(command.dy) > 1e-9:
            self.mouse.move_xy_mm(-command.dy, -command.dx)

        # This is deliberately after the blocking move.
        if command.click:
            self.mouse.click()

    def run_once(self, frame: NDArray[np.uint8]) -> PipelineStep:
        """Process and, when enabled, execute one target from one frame."""
        frames, perception = self.perceive(frame)
        command = self.plan_perception(perception)
        if command is not None:
            self.execute(command)
        return PipelineStep(frames=frames, perception=perception, command=command)

    def tick(self, frame: NDArray[np.uint8]) -> MouseCommand | None:
        """Compatibility wrapper returning only the generated command."""
        return self.run_once(frame).command
