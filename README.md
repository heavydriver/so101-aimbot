# SO-101 Aimbot

An SO-101 arm plays Gridshot. Each frame goes through perception (find the screen, detect targets and the cursor), planning (mouse delta and click), then control.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

That creates `.venv` and installs the package plus dev tools (`pytest`).

The frame source is `configs/camera.yaml`. It defaults to a recorded clip (`clips/clip-1.mp4`). Set `source: webcam` to use a live camera.

## Run

Full loop. Prints planned mouse commands and does not move the arm. Press `q` to quit.

```bash
uv run python main.py
```

## Components

Perception viewer. Shows the raw camera and a warped overlay of targets and the cursor. Press `q` to quit.

```bash
uv run python scripts/run_perception.py
```

Mouse-to-cursor calibration. Prints the current scale from pad millimeters to screen pixels.

```bash
uv run python scripts/run_calibration.py
```

Arm smoke test. Connects, homes, and disconnects.

```bash
uv run python scripts/run_arm_smoke.py
```

Tests:

```bash
uv run pytest
uv run pytest tests/perception
uv run pytest tests/planning
```
