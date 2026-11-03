"""Thin entrypoint for the live perception → planning → control loop."""

from __future__ import annotations

import cv2

from so101_aimbot.config import load_config
from so101_aimbot.control import ArmController, MouseController
from so101_aimbot.perception import Camera
from so101_aimbot.pipeline import Pipeline


def main() -> None:
    robot_config = load_config("robot")
    arm = ArmController(config=robot_config)
    mouse = MouseController(arm, config=robot_config)
    cam = Camera()
    try:
        mouse.connect()
        pipeline = Pipeline(
            arm=arm,
            mouse=mouse,
            robot_config=robot_config,
            actuate=True,
        )
        while True:
            frame = cam.read()
            command = pipeline.tick(frame)
            cv2.imshow("SO-101 Aimbot", frame)
            if command is not None:
                print(f"cmd dx={command.dx:.2f} dy={command.dy:.2f} click={command.click}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        arm.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
