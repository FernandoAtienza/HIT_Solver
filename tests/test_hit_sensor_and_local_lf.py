from __future__ import annotations

import unittest

import numpy as np

from OOP.domain import Domain2D
from OOP.equations import EulerEquation2D
from OOP.spatial_operator import (
    PeriodicEulerShockSensor2D,
    weno7_flux_axis,
    weno7_flux_axis_local_lf,
)


class LocalLaxFriedrichsTests(unittest.TestCase):
    def test_local_matches_global_for_constant_wave_speed(self):
        rng = np.random.default_rng(7)
        q = rng.normal(size=(4, 9, 32))
        flux = rng.normal(size=q.shape)
        alpha = 1.75
        wave_speed = np.full(q.shape[-2:], alpha)
        global_flux = weno7_flux_axis(q, flux, alpha)
        local_flux = weno7_flux_axis_local_lf(q, flux, wave_speed)
        np.testing.assert_allclose(local_flux, global_flux, rtol=1e-13, atol=1e-13)


class CompressionGatedSensorTests(unittest.TestCase):
    def setUp(self):
        self.domain = Domain2D(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 64, 64)
        self.eq = EulerEquation2D(gamma=1.4)
        self.x, self.y = self.domain.mesh()

    def _state(self, rho, u, v, pressure):
        return self.eq.conservative_from_primitive(rho, u, v, pressure)

    def test_gated_sensor_rejects_smooth_vortical_motion(self):
        u = np.sin(self.y)
        v = -np.sin(self.x)
        rho = np.ones_like(u)
        p = np.full_like(u, 1.0 / 1.4)
        q = self._state(rho, u, v, p)
        sensor = PeriodicEulerShockSensor2D(
            self.domain,
            self.eq,
            width=2,
            compression_threshold=2.0,
            jump_threshold=0.02,
            mode="compression_gated",
            ducros_threshold=0.5,
        )
        self.assertEqual(int(np.count_nonzero(sensor.detect(q))), 0)

    def test_directional_sensor_localizes_x_normal_compression(self):
        # A smooth but narrow x-normal compression coupled to a pressure/density
        # jump should activate the x reconstruction much more than y.
        x0 = np.pi
        width = 0.10
        transition = np.tanh((self.x - x0) / width)
        u = -0.35 * transition
        v = np.zeros_like(u)
        rho = 1.0 + 0.12 * transition
        p = (1.0 / 1.4) * (1.0 + 0.12 * transition)
        q = self._state(rho, u, v, p)
        sensor = PeriodicEulerShockSensor2D(
            self.domain,
            self.eq,
            width=2,
            compression_threshold=1.0,
            jump_threshold=0.005,
            mode="directional",
            ducros_threshold=0.4,
        )
        mask_x, mask_y = sensor.detect_directional(q)
        self.assertGreater(np.count_nonzero(mask_x), 0)
        self.assertEqual(int(np.count_nonzero(mask_y)), 0)
        np.testing.assert_array_equal(sensor.detect(q), mask_x)


if __name__ == "__main__":
    unittest.main()
