"""Step-response logging to tune PID gains in configs/robot.yaml."""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

from so101_aimbot.config import CONFIG_DIR, load_config
from so101_aimbot.control.arm import ARM_JOINT_NAMES, ArmController
from so101_aimbot.planning.calibrate import _merge_robot_yaml


def _cal_cfg(config: dict[str, Any]) -> dict[str, Any]:
    pid = dict(config.get("pid", {}) or {})
    return dict(pid.get("cal", {}) or {})


def _summarize(
    errors: list[float],
    *,
    ref_start: float,
    ref_end: float,
) -> None:
    if not errors:
        print("No samples recorded.")
        return
    abs_err = [abs(e) for e in errors]
    max_err = max(abs_err)
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    print(f"  max |error| = {max_err:.3f}")
    print(f"  RMS error   = {rms:.3f}")
    step = ref_end - ref_start
    if abs(step) > 1e-6:
        overshoot = 0.0
        if step > 0:
            overshoot = max(0.0, min(errors) * -1.0)
        else:
            overshoot = max(0.0, max(errors))
        pct = 100.0 * overshoot / abs(step)
        print(f"  overshoot   = {overshoot:.3f} ({pct:.1f}% of step)")


def run_arm_step(arm: ArmController, joint: int, config: dict[str, Any]) -> None:
    cal = _cal_cfg(config)
    step_deg = float(cal.get("step_deg", 4.0))
    duration_s = float(cal.get("duration_s", 2.0))
    poll_s = float(cal.get("poll_s", 0.02))
    name = ARM_JOINT_NAMES[joint]

    current = arm.read_arm()
    ref_start = current[joint]
    ref_end = ref_start + step_deg
    target = list(current)
    target[joint] = ref_end

    print(f"Arm joint {joint} ({name}): step {ref_start:.2f} -> {ref_end:.2f} deg")
    print("t_s\tref\tmeas\terror\tcorr_sent")
    errors: list[float] = []
    t0 = time.perf_counter()
    deadline = t0 + duration_s
    while time.perf_counter() < deadline:
        meas = arm.read_arm()
        ref = target[joint]
        error = ref - meas[joint]
        errors.append(error)
        corr = target[joint] - arm.commanded_arm[joint]
        print(
            f"{time.perf_counter() - t0:.3f}\t{ref:.3f}\t{meas[joint]:.3f}\t"
            f"{error:.3f}\t{corr:.3f}"
        )
        arm.move_arm(target)
        time.sleep(poll_s)

    print("Summary:")
    _summarize(errors, ref_start=ref_start, ref_end=ref_end)


def run_gripper_step(
    arm: ArmController, config: dict[str, Any], *, toward: str
) -> None:
    cal = _cal_cfg(config)
    step = float(cal.get("gripper_step", 8.0))
    duration_s = float(cal.get("duration_s", 2.0))
    poll_s = float(cal.get("poll_s", 0.02))
    grip = dict(config.get("gripper", {}) or {})
    open_pos = float(grip.get("open", 40.0))
    closed_pos = float(grip.get("closed", 5.0))

    current = arm.read_gripper()
    if toward == "open":
        ref_end = min(100.0, current + step)
        if ref_end <= current + 0.5:
            ref_end = open_pos
    else:
        ref_end = max(0.0, current - step)
        if ref_end >= current - 0.5:
            ref_end = closed_pos

    ref_start = current
    target = ref_end
    print(f"Gripper step {ref_start:.2f} -> {target:.2f} ({toward})")
    print("t_s\tref\tmeas\terror")
    errors: list[float] = []
    t0 = time.perf_counter()
    deadline = t0 + duration_s
    arm.set_gripper(target, wait=False)
    while time.perf_counter() < deadline:
        meas = arm.read_gripper()
        error = target - meas
        errors.append(error)
        print(f"{time.perf_counter() - t0:.3f}\t{target:.3f}\t{meas:.3f}\t{error:.3f}")
        if config.get("pid", {}).get("enabled"):
            arm.set_gripper(target, wait=False)
        time.sleep(poll_s)

    print("Summary:")
    _summarize(errors, ref_start=ref_start, ref_end=target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", type=int, choices=range(5), metavar="N")
    parser.add_argument(
        "--gripper",
        action="store_true",
        help="Run a gripper step toward open (default) or closed with --closed.",
    )
    parser.add_argument(
        "--closed",
        action="store_true",
        help="With --gripper, step toward closed instead of open.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--write-gains",
        action="store_true",
        help="Merge pid.arm / pid.gripper from loaded config into robot.yaml (no-op if unchanged).",
    )
    parser.add_argument("--kp", type=float, default=None, help="Set arm kp scalar when writing.")
    args = parser.parse_args()

    if args.joint is None and not args.gripper:
        parser.error("Specify --joint N or --gripper")

    cfg = load_config("robot")
    if args.dry_run:
        cfg.update(dry_run=True, port=None)
    if not cfg.get("pid", {}).get("enabled"):
        print(
            "Note: pid.enabled is false — arm moves are open-loop; "
            "set pid.enabled: true in robot.yaml to test corrections."
        )

    arm = ArmController(cfg)
    arm.connect()
    try:
        if args.joint is not None:
            run_arm_step(arm, args.joint, cfg)
        if args.gripper:
            run_gripper_step(arm, cfg, toward="closed" if args.closed else "open")

        if args.write_gains:
            pid = dict(cfg.get("pid", {}) or {})
            updates: dict[str, Any] = {"pid": pid}
            if args.kp is not None:
                arm_block = dict(pid.get("arm", {}) or {})
                arm_block["kp"] = [float(args.kp)] * 5
                pid["arm"] = arm_block
                updates["pid"] = pid
            _merge_robot_yaml(CONFIG_DIR / "robot.yaml", updates)
            print(f"Wrote pid block to {CONFIG_DIR / 'robot.yaml'}")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
