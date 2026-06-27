"""OOP structure for the HIT2D solver subset."""

from OOP.domain import Domain1D, Domain2D
from OOP.equations import BurgersEquation, CompressibleNavierStokes2D, EulerEquation, EulerEquation2D
from OOP.forcing import IsotropicShellOUForcing2D
from OOP.postprocess import HIT2DSpectra, IsotropyDiagnostics2D, TwoPointCorrelation2D
from OOP.spatial_operator import (
    BurgersHybridSpatialOperator,
    EulerHybridSpatialOperator,
    EulerShockSensor,
    PeriodicEulerShockSensor2D,
    PeriodicHybridEuler2DOperator,
    PeriodicHyperviscosity2D,
    WangHyperviscosity,
)
from OOP.time_operator import SSPRK3, TimeOperator

__all__ = [
    "BurgersHybridSpatialOperator",
    "BurgersEquation",
    "CompressibleNavierStokes2D",
    "Domain1D",
    "Domain2D",
    "EulerEquation",
    "EulerEquation2D",
    "EulerHybridSpatialOperator",
    "EulerShockSensor",
    "HIT2DSpectra",
    "IsotropyDiagnostics2D",
    "IsotropicShellOUForcing2D",
    "PeriodicEulerShockSensor2D",
    "PeriodicHybridEuler2DOperator",
    "PeriodicHyperviscosity2D",
    "SSPRK3",
    "TimeOperator",
    "TwoPointCorrelation2D",
    "WangHyperviscosity",
]
