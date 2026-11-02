"""Ruckig online trajectory helpers for pad slides and lift/home/drop."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from ruckig import InputParameter, OutputParameter, Result, Ruckig


def _as_list(values: Sequence[float], n: int) -> list[float]:
    vals = [float(v) for v in values]
    if len(vals) != n:
        raise ValueError(f"expected {n} values, got {len(vals)}")
    return vals


def run_ruckig(
    *,
    dofs: int,
    dt_s: float,
    current: Sequence[float],
    target: Sequence[float],
    max_velocity: Sequence[float],
    max_acceleration: Sequence[float],
    max_jerk: Sequence[float],
    on_sample: Callable[[list[float]], None],
    current_velocity: Sequence[float] | None = None,
    current_acceleration: Sequence[float] | None = None,
    target_velocity: Sequence[float] | None = None,
    max_iterations: int = 100_000,
    realtime: bool = True,
) -> None:
    """Generate a jerk-limited trajectory and invoke ``on_sample`` each cycle.

    When ``realtime`` is true (default), each sample is paced to wall-clock
    ``dt_s`` so hardware has time to track the stream. Set ``realtime=False``
    for dry-run / unit tests that only need kinematic samples.
    """
    otg = Ruckig(dofs, float(dt_s))
    inp = InputParameter(dofs)
    out = OutputParameter(dofs)

    inp.current_position = _as_list(current, dofs)
    inp.current_velocity = (
        _as_list(current_velocity, dofs) if current_velocity is not None else [0.0] * dofs
    )
    inp.current_acceleration = (
        _as_list(current_acceleration, dofs)
        if current_acceleration is not None
        else [0.0] * dofs
    )
    inp.target_position = _as_list(target, dofs)
    inp.target_velocity = (
        _as_list(target_velocity, dofs) if target_velocity is not None else [0.0] * dofs
    )
    inp.max_velocity = _as_list(max_velocity, dofs)
    inp.max_acceleration = _as_list(max_acceleration, dofs)
    inp.max_jerk = _as_list(max_jerk, dofs)

    res = Result.Working
    iterations = 0
    next_t = time.perf_counter()
    while res == Result.Working:
        res = otg.update(inp, out)
        on_sample(list(out.new_position))
        out.pass_to_input(inp)
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError("Ruckig trajectory exceeded max_iterations")
        if realtime:
            next_t += float(dt_s)
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    if res == Result.ErrorInvalidInput:
        raise ValueError("Ruckig rejected input parameters")


def limits_from_config(section: dict[str, Any], dofs: int) -> tuple[list[float], list[float], list[float]]:
    """Read max velocity / acceleration / jerk lists from a robot.yaml ruckig section."""
    vmax = [float(v) for v in section["max_velocity_mps"]]
    amax = [float(v) for v in section["max_acceleration_mps2"]]
    jmax = [float(v) for v in section["max_jerk_mps3"]]
    if not (len(vmax) == len(amax) == len(jmax) == dofs):
        raise ValueError(f"ruckig limits must each have length {dofs}")
    return vmax, amax, jmax
