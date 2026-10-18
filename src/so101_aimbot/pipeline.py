"""One control-loop tick wiring perception → planning → control."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from so101_aimbot.config import load_config
from so101_aimbot.control.arm import ArmController
from so101_aimbot.control.mouse import execute_mouse_command
from so101_aimbot.perception.detect import detect_cursor, detect_targets
from so101_aimbot.perception.preprocess import find_game_corners, warp_game_region
from so101_aimbot.perception.select import select_nearest_target
from so101_aimbot.planning.trajectory import plan_mouse_command
from so101_aimbot.types import FrameBundle, MouseCommand, PerceptionResult


class Pipeline:
    """Live Perception → Planning → Control loop for Gridshot."""

    def __init__(
        self,
        arm: ArmController | None = None,
        *,
        perception_config: dict[str, Any] | None = None,
        robot_config: dict[str, Any] | None = None,
        actuate: bool = False,
    ) -> None:
        self.arm = arm
        self.perception_config = perception_config or load_config("perception")
        self.robot_config = robot_config or load_config("robot")
        self.actuate = actuate
        self._corners: NDArray[np.float32] | None = None

    def perceive(self, frame: NDArray[np.uint8]) -> tuple[FrameBundle, PerceptionResult]:
        if self._corners is None:
            self._corners = find_game_corners(frame, config=self.perception_config)

        warped = None
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

        bundle = FrameBundle(raw=frame, warped=warped)
        result = PerceptionResult(targets=targets, cursor=cursor, selected=selected)
        return bundle, result

    def tick(self, frame: NDArray[np.uint8]) -> MouseCommand | None:
        """Run one full pipeline tick; optionally actuate the arm."""
        _, perception = self.perceive(frame)
        click_threshold = float(self.perception_config.get("click_threshold_px", 8.0))
        command = plan_mouse_command(
            perception.cursor,
            perception.selected,
            click_threshold_px=click_threshold,
        )
        if command is not None and self.actuate and self.arm is not None:
            execute_mouse_command(self.arm, command)
        return command
