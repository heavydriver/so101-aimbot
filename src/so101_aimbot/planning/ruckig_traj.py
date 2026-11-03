"""Jerk-limited trajectories for the four moving arm joints."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ruckig import InputParameter, OutputParameter, Result, Ruckig

DOFS = 4


def _values(values: Sequence[float], label: str) -> list[float]:
    vals = [float(v) for v in values]
    if len(vals) != DOFS:
        raise ValueError(f"{label} must contain {DOFS} values")
    return vals


@dataclass(frozen=True, slots=True)
class JointLimits:
    velocity_deg_s: list[float]
    acceleration_deg_s2: list[float]
    jerk_deg_s3: list[float]

    @classmethod
    def from_config(cls, config: dict[str, object]) -> JointLimits:
        return cls(
            _values(config["max_velocity_deg_s"], "max_velocity_deg_s"),  # type: ignore[arg-type]
            _values(config["max_acceleration_deg_s2"], "max_acceleration_deg_s2"),  # type: ignore[arg-type]
            _values(config["max_jerk_deg_s3"], "max_jerk_deg_s3"),  # type: ignore[arg-type]
        )


def run_joint_trajectory(
    *,
    dt_s: float,
    current_deg: Sequence[float],
    target_deg: Sequence[float],
    limits: JointLimits,
    on_sample: Callable[[list[float]], None],
    realtime: bool = True,
) -> None:
    """Stream a four-joint trajectory in degrees."""
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    otg = Ruckig(DOFS, float(dt_s))
    inp = InputParameter(DOFS)
    out = OutputParameter(DOFS)

    inp.current_position = _values(current_deg, "current_deg")
    inp.current_velocity = [0.0] * DOFS
    inp.current_acceleration = [0.0] * DOFS
    inp.target_position = _values(target_deg, "target_deg")
    inp.target_velocity = [0.0] * DOFS
    inp.max_velocity = limits.velocity_deg_s
    inp.max_acceleration = limits.acceleration_deg_s2
    inp.max_jerk = limits.jerk_deg_s3

    res = Result.Working
    next_t = time.perf_counter()
    while res == Result.Working:
        res = otg.update(inp, out)
        if res not in (Result.Working, Result.Finished):
            raise ValueError(f"Ruckig failed with result {res}")
        on_sample(list(out.new_position))
        out.pass_to_input(inp)
        if realtime:
            next_t += float(dt_s)
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
