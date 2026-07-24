from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BoundaryCondition:
    """Base interface for boundary conditions on arrays whose last axis is x."""

    def apply(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class PeriodicBoundary(BoundaryCondition):
    """Periodic boundary condition.

    The current compact and WENO operators use np.roll, so no ghost cells are
    needed here. The copy keeps the solver from accidentally sharing state.
    """

    def apply(self, state: np.ndarray) -> np.ndarray:
        return np.array(state, copy=True)


@dataclass(frozen=True)
class ExtrapolatedBoundary(BoundaryCondition):
    """Zero-gradient guard cells for padded shock-tube domains."""

    guard_cells: int = 0

    def apply(self, state: np.ndarray) -> np.ndarray:
        arr = np.array(state, copy=True)
        if self.guard_cells == 0:
            return arr
        if arr.shape[-1] <= 2 * self.guard_cells:
            raise ValueError("state is too small for the requested guard cells")

        g = self.guard_cells
        arr[..., :g] = arr[..., g][..., None]
        arr[..., -g:] = arr[..., -g - 1][..., None]
        return arr


@dataclass(frozen=True)
class ExtrapolatedBoundary2D(BoundaryCondition):
    """Zero-gradient guard cells for arrays whose last axes are (y, x)."""

    guard_cells: int = 0

    def apply(self, state: np.ndarray) -> np.ndarray:
        arr = np.array(state, copy=True)
        if self.guard_cells == 0:
            return arr
        if arr.shape[-1] <= 2 * self.guard_cells or arr.shape[-2] <= 2 * self.guard_cells:
            raise ValueError("state is too small for the requested guard cells")

        g = self.guard_cells
        arr[..., :, :g] = arr[..., :, g][..., None]
        arr[..., :, -g:] = arr[..., :, -g - 1][..., None]
        arr[..., :g, :] = arr[..., g, :][..., None, :]
        arr[..., -g:, :] = arr[..., -g - 1, :][..., None, :]
        return arr
