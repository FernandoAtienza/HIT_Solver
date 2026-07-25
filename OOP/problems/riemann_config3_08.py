from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from OOP.problems.riemann_config3 import RiemannConfig3, run_riemann


@dataclass(frozen=True)
class RiemannConfig3Offset08(RiemannConfig3):
    """Configuration 3 with the quadrant interface at x0=y0=0.8."""

    scenario_name: ClassVar[str] = "riemann_config3_08"
    configuration_label: ClassVar[str] = "Configuration 3 (x0=y0=0.8)"

    x_split: float = 0.8
    y_split: float = 0.8
    backend: str = "cupy"
    hyperviscosity_interval: int = 5


def run_riemann_config3_08(config: RiemannConfig3Offset08):
    return run_riemann(config)
