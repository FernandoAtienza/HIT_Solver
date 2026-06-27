from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Domain1D:
    """Uniform 1D domain used by the current Burgers and shock-tube scripts."""

    x_min: float
    x_max: float
    nx: int
    endpoint: bool = False

    def __post_init__(self) -> None:
        if self.nx <= 0:
            raise ValueError("nx must be positive")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def dx(self) -> float:
        intervals = self.nx - 1 if self.endpoint else self.nx
        return self.length / intervals

    @property
    def x(self) -> np.ndarray:
        if self.endpoint:
            return np.linspace(self.x_min, self.x_max, self.nx)
        return self.x_min + self.dx * np.arange(self.nx)

    def mask(self, x_min: float, x_max: float) -> np.ndarray:
        return (self.x >= x_min) & (self.x <= x_max)

    @classmethod
    def from_dx(cls, x_min: float, x_max: float, dx: float) -> "Domain1D":
        if dx <= 0.0:
            raise ValueError("dx must be positive")
        nx = int(round((x_max - x_min) / dx))
        return cls(x_min=x_min, x_max=x_max, nx=nx)


@dataclass(frozen=True)
class Domain2D:
    """Uniform cell-centered 2D domain for finite-volume Euler tests."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def x(self) -> np.ndarray:
        return self.x_min + self.dx * (np.arange(self.nx) + 0.5)

    @property
    def y(self) -> np.ndarray:
        return self.y_min + self.dy * (np.arange(self.ny) + 0.5)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return X, Y with shape (ny, nx), matching state[:, y, x]."""

        return np.meshgrid(self.x, self.y, indexing="xy")
