"""Run the one-frame, one-target SO-101 aimbot loop."""

from __future__ import annotations

# temporary work around
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2

from so101_aimbot.config import load_config
from so101_aimbot.control import ArmController, MouseController
from so101_aimbot.perception import Camera
from so101_aimbot.pipeline import Pipeline

WINDOW_NAME = "SO-101 Aimbot"


def main() -> None:
    robot_config = load_config("robot")
    arm = ArmController(config=robot_config)
    mouse = MouseController(arm, config=robot_config)
    camera: Camera | None = None

    try:
        mouse.connect()
        arm.home()
        camera = Camera()
        pipeline = Pipeline(
            mouse=mouse,
            robot_config=robot_config,
            actuate=True,
        )

        while True:
            # This happens only after the previous move and click have completed.
            frame = camera.read_latest()
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            step = pipeline.run_once(frame)
            command = step.command
            if command is None:
                continue

            cursor = step.perception.cursor
            target = step.perception.selected
            if cursor is not None and target is not None:
                print(
                    "target "
                    f"px=({target.x - cursor.x:+.1f}, "
                    f"{target.y - cursor.y:+.1f}) "
                    f"mm=({command.dx:+.2f}, {command.dy:+.2f})"
                )
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.release()
        arm.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
