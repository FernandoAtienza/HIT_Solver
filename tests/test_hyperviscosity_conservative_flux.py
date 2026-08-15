from __future__ import annotations

import numpy as np

from OOP.domain import Domain2D
from OOP.equations import EulerEquation2D
from OOP.spatial_operator import PeriodicHyperviscosity2D


def _smooth_state(domain: Domain2D, equation: EulerEquation2D) -> np.ndarray:
    x, y = domain.mesh()
    rho = 1.0 + 0.02 * np.sin(x) * np.cos(y)
    u = 0.1 * np.sin(y)
    v = 0.1 * np.cos(x)
    p = 1.0 + 0.01 * np.cos(x + y)
    return equation.conservative_from_primitive(rho, u, v, p)


def test_conservative_flux_hyperviscosity_conserves_periodic_sums():
    domain = Domain2D(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 32, 32)
    equation = EulerEquation2D(gamma=1.4)
    q = _smooth_state(domain, equation)
    mask = np.ones((domain.ny, domain.nx), dtype=bool)
    mask[8:12, 10:15] = False
    hv = PeriodicHyperviscosity2D(domain, mn=0.01, mode="conservative_flux")
    before = np.sum(q, axis=(-2, -1))
    after = hv.apply(q, equation, active_mask=mask)
    after_sum = np.sum(after, axis=(-2, -1))
    np.testing.assert_allclose(after_sum, before, rtol=0.0, atol=5e-12)


def test_conservative_flux_has_no_flux_across_weno_boundary():
    domain = Domain2D(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 32, 32)
    equation = EulerEquation2D(gamma=1.4)
    q = _smooth_state(domain, equation)
    compact = np.zeros((domain.ny, domain.nx), dtype=bool)
    compact[:, :16] = True
    hv = PeriodicHyperviscosity2D(domain, mn=0.01, mode="conservative_flux")
    after = hv.apply(q, equation, active_mask=compact)
    # Nodes deep in the WENO half cannot receive a hyperviscous correction.
    np.testing.assert_allclose(after[:, :, 19:29], q[:, :, 19:29], rtol=0.0, atol=1e-13)


def test_legacy_mode_remains_available():
    domain = Domain2D(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 16, 16)
    equation = EulerEquation2D(gamma=1.4)
    q = _smooth_state(domain, equation)
    mask = np.ones((domain.ny, domain.nx), dtype=bool)
    hv = PeriodicHyperviscosity2D(domain, mn=0.001, mode="legacy_node")
    result = hv.apply(q, equation, active_mask=mask)
    assert result.shape == q.shape
    assert np.isfinite(result).all()
