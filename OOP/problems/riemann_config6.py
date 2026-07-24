from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from OOP.parallel.backend import select_array_module
from OOP.problems.riemann_config3 import (
    RiemannConfig3,
    run_riemann,
)


@dataclass(frozen=True)
class RiemannConfig6(RiemannConfig3):
    """Lax-Liu Configuration 6: four interacting contact discontinuities."""

    configuration_number: ClassVar[int] = 6

    tfinal: float = 0.25
    backend: str = "cupy"

    def initial_state(self):
        xp = select_array_module(self.backend)
        domain = self.domain
        x = xp.asarray(domain.x)
        y = xp.asarray(domain.y)
        X, Y = xp.meshgrid(x, y, indexing="xy")

        rho = xp.empty((self.ny, self.nx), dtype=xp.float64)
        u = xp.empty_like(rho)
        v = xp.empty_like(rho)
        pressure = xp.ones_like(rho)

        top = Y >= self.y_split
        right = X >= self.x_split

        # Quadrants follow the standard ordering:
        # II | I
        # -------
        # III| IV
        rho[...] = 1.0
        u[...] = -0.75
        v[...] = 0.50

        quadrant_i = top & right
        rho[quadrant_i] = 1.0
        u[quadrant_i] = 0.75
        v[quadrant_i] = -0.50

        quadrant_ii = top & (~right)
        rho[quadrant_ii] = 2.0
        u[quadrant_ii] = 0.75
        v[quadrant_ii] = 0.50

        quadrant_iv = (~top) & right
        rho[quadrant_iv] = 3.0
        u[quadrant_iv] = -0.75
        v[quadrant_iv] = -0.50

        return self.equation.conservative_from_primitive(rho, u, v, pressure)


def load_riemann_config6_npz(path: Path):
    data = np.load(path, allow_pickle=False)
    config_data = json.loads(str(data["config_json"])) if "config_json" in data else {}
    fields = RiemannConfig6.__dataclass_fields__
    config = RiemannConfig6(
        **{
            key: value
            for key, value in config_data.items()
            if key != "configuration_number" and key in fields
        }
    )
    return data, config


def run_riemann_config6(config: RiemannConfig6):
    return run_riemann(config)
