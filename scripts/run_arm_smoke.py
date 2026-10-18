"""Smoke-test arm connect / home / disconnect without perception."""

from __future__ import annotations

from so101_aimbot.control import ArmController


def main() -> None:
    arm = ArmController()
    arm.connect()
    try:
        arm.home()
        print("Arm smoke test OK (stub controller).")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
