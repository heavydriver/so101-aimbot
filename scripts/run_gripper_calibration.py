"""Calibrate gripper open / closed angles and write them to configs/robot.yaml."""

from __future__ import annotations

import argparse

from so101_aimbot.control import ArmController
from so101_aimbot.planning.calibrate import teach_gripper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open",
        type=float,
        default=None,
        help="Open position in [0, 100]. Omit to capture interactively.",
    )
    parser.add_argument(
        "--closed",
        type=float,
        default=None,
        help="Closed position in [0, 100]. Omit to capture interactively.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print captured values without writing robot.yaml.",
    )
    parser.add_argument(
        "--test-click",
        action="store_true",
        help="After teaching, pulse click() once to verify the range.",
    )
    args = parser.parse_args()

    arm = ArmController()
    arm.connect()
    try:
        taught = teach_gripper(
            arm,
            open_pos=args.open,
            closed_pos=args.closed,
            save=not args.no_save,
        )
        print("Taught gripper:")
        for key, value in taught.items():
            print(f"  {key}: {value}")

        if args.test_click:
            print("Testing click()…")
            arm.set_gripper(float(taught["gripper"]["open"]))
            arm.click()
            print("  click OK")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
