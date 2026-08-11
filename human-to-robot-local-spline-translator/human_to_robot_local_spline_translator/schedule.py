from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PiecewiseLinearSchedule:
    milestones: tuple[float, ...]
    values: tuple[float, ...]

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "PiecewiseLinearSchedule":
        if not isinstance(payload, dict):
            raise ValueError("Schedule config must be a mapping.")
        milestones = tuple(float(value) for value in payload.get("milestones", []))
        values = tuple(float(value) for value in payload.get("values", []))
        if not milestones or not values or len(milestones) != len(values):
            raise ValueError("Schedule config must contain same-length milestones and values.")
        if any(milestones[index] > milestones[index + 1] for index in range(len(milestones) - 1)):
            raise ValueError("Schedule milestones must be nondecreasing.")
        return cls(milestones=milestones, values=values)

    def value(self, progress: float) -> float:
        progress = float(progress)
        if progress <= self.milestones[0]:
            return float(self.values[0])
        if progress >= self.milestones[-1]:
            return float(self.values[-1])
        for index in range(len(self.milestones) - 1):
            left_m = self.milestones[index]
            right_m = self.milestones[index + 1]
            if left_m <= progress <= right_m:
                left_v = self.values[index]
                right_v = self.values[index + 1]
                if right_m <= left_m:
                    return float(right_v)
                alpha = (progress - left_m) / (right_m - left_m)
                return float((1.0 - alpha) * left_v + alpha * right_v)
        return float(self.values[-1])
