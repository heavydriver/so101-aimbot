"""Discrete PID helpers for joint and gripper correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _clamp(value: float, limit: float | None) -> float:
    if limit is None or limit <= 0.0:
        return value
    return max(-limit, min(limit, value))


def _expand_gain(value: float | Sequence[float], n: int, key: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * n
    vals = [float(v) for v in value]
    if len(vals) != n:
        raise ValueError(f"{key} must be a scalar or length-{n} list, got {len(vals)}")
    return vals


@dataclass(frozen=True, slots=True)
class PIDGains:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0


class PID:
    """Simple discrete PID with optional integral and output clamps."""

    def __init__(
        self,
        gains: PIDGains,
        *,
        i_clamp: float | None = None,
        output_clamp: float | None = None,
    ) -> None:
        self._gains = gains
        self._i_clamp = i_clamp
        self._output_clamp = output_clamp
        self._integral = 0.0
        self._prev_error: float | None = None

    @property
    def gains(self) -> PIDGains:
        return self._gains

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, error: float, dt_s: float) -> float:
        if dt_s <= 0.0:
            dt_s = 1e-6
        p = self._gains.kp * error
        self._integral += error * dt_s
        self._integral = _clamp(self._integral, self._i_clamp)
        i = self._gains.ki * self._integral
        d = 0.0
        if self._gains.kd != 0.0 and self._prev_error is not None:
            d = self._gains.kd * (error - self._prev_error) / dt_s
        self._prev_error = error
        return _clamp(p + i + d, self._output_clamp)


class PIDBank:
    """One PID instance per joint."""

    def __init__(self, controllers: Sequence[PID]) -> None:
        self._controllers = list(controllers)

    def __len__(self) -> int:
        return len(self._controllers)

    def reset(self) -> None:
        for ctrl in self._controllers:
            ctrl.reset()

    def update(self, errors: Sequence[float], dt_s: float) -> list[float]:
        if len(errors) != len(self._controllers):
            raise ValueError(
                f"errors length {len(errors)} != {len(self._controllers)} controllers"
            )
        return [c.update(float(e), dt_s) for c, e in zip(self._controllers, errors)]

    @classmethod
    def from_config(
        cls,
        section: Mapping[str, Any],
        n: int,
        *,
        default_output_clamp: float | None = None,
        default_i_clamp: float | None = None,
    ) -> PIDBank:
        kp = _expand_gain(section.get("kp", 0.0), n, "kp")
        ki = _expand_gain(section.get("ki", 0.0), n, "ki")
        kd = _expand_gain(section.get("kd", 0.0), n, "kd")
        out_key = "output_clamp_deg" if n > 1 else "output_clamp"
        i_key = "i_clamp_deg" if n > 1 else "i_clamp"
        output_clamp = section.get(out_key, default_output_clamp)
        i_clamp = section.get(i_key, default_i_clamp)
        if output_clamp is not None:
            output_clamp = float(output_clamp)
        if i_clamp is not None:
            i_clamp = float(i_clamp)
        controllers = [
            PID(
                PIDGains(kp[i], ki[i], kd[i]),
                i_clamp=i_clamp,
                output_clamp=output_clamp,
            )
            for i in range(n)
        ]
        return cls(controllers)
