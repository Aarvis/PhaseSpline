from __future__ import annotations

from openpi_client import base_policy as _base_policy

from .runtime import SplineRuntime


class SplineRuntimePolicy(_base_policy.BasePolicy):
    def __init__(self, runtime: SplineRuntime) -> None:
        self._runtime = runtime

    def infer(self, obs: dict) -> dict:
        return self._runtime.infer(obs)

    def reset(self) -> None:
        return None
