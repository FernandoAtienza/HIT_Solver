"""Backend-aware array classes for optional parallel/GPU HIT runs."""

from OOP.parallel.backend import array_module, select_array_module, to_numpy
from OOP.parallel.equations import ParallelCompressibleNavierStokes2D, ParallelEulerEquation2D
from OOP.parallel.spatial_operator import (
    ParallelPeriodicEulerShockSensor2D,
    ParallelPeriodicHybridEuler2DOperator,
    ParallelPeriodicHyperviscosity2D,
    ParallelPeriodicLineCompactDerivative,
)

__all__ = [
    "ParallelCompressibleNavierStokes2D",
    "ParallelEulerEquation2D",
    "ParallelPeriodicEulerShockSensor2D",
    "ParallelPeriodicHybridEuler2DOperator",
    "ParallelPeriodicHyperviscosity2D",
    "ParallelPeriodicLineCompactDerivative",
    "array_module",
    "select_array_module",
    "to_numpy",
]
