# SO-101 Aimbot

An SO-101 arm plays Gridshot. Each frame goes through perception (find the screen, detect targets and the cursor), planning (mouse delta and click), then control.

## Setup

Requires [Pixi](https://pixi.prefix.dev/latest/)

```bash
pixi install
```

The frame source is `configs/camera.yaml`. It defaults to a recorded clip (`clips/clip-1.mp4`). Set `source: webcam` to use a live camera.

Update `port` and `robot_id` in `configs/robot.yaml`

Set the physical mouse `dpi`, in-game `sensitivity`, and native game
resolution in `configs/mouse.yaml`. The planner uses them with the perception
warp size to convert cursor pixel errors into mouse-pad millimetres.

## Run

```bash
pixi run main
```

## Components

Perception viewer. Shows the raw camera and a warped overlay of targets and the cursor. Press `q` to quit.

```bash
pixi run perception
```

Robot calibration - teach home from the current pose, or set max radius:

```bash
pixi run python scripts/run_calibration.py --teach-home
pixi run python scripts/run_calibration.py --max-radius-mm 40
```

Gripper open / closed calibration. Disables gripper torque so you can move it by hand, then writes `gripper.open` / `gripper.closed` into `configs/robot.yaml`:

```bash
pixi run python scripts/run_gripper_calibration.py
pixi run python scripts/run_gripper_calibration.py --test-click
pixi run python scripts/run_gripper_calibration.py --open 70 --closed 5
```

Arm / pad smoke test (dry-run by default). Short `move_polar`, oversize move with lift→home→continue, then `click`.

```bash
pixi run python scripts/run_arm_smoke.py
```

Set `port` and `dry_run: false` in `configs/robot.yaml` to talk to a real SO-101.

Tests:

```bash
pixi run test
pixi run -e dev pytest tests/perception
pixi run -e dev pytest tests/planning
pixi run -e dev pytest tests/control
```
