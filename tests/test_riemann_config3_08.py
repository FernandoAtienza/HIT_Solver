from __future__ import annotations

import unittest

import numpy as np

from OOP.problems.riemann_config3 import RiemannConfig3
from OOP.problems.riemann_config3_08 import RiemannConfig3Offset08


class RiemannConfig3Offset08Tests(unittest.TestCase):
    def test_tutor_defaults(self) -> None:
        config = RiemannConfig3Offset08()

        self.assertEqual(config.x_split, 0.8)
        self.assertEqual(config.y_split, 0.8)
        self.assertEqual(config.tfinal, 0.85)
        self.assertEqual(config.cfl, 0.4)
        self.assertEqual(config.backend, "cupy")
        self.assertEqual(config.scheme, "hybrid")
        self.assertEqual(config.mn, 0.001)
        self.assertEqual(config.hyperviscosity_interval, 5)
        self.assertEqual(config.boundary_guard, 4)
        self.assertEqual(config.guard_cells, 4)

    def test_quadrants_are_split_at_point_eight(self) -> None:
        config = RiemannConfig3Offset08(nx=10, ny=10, backend="numpy")
        q = config.initial_state()
        rho, u, v, pressure = config.equation.primitive_from_conservative(q)

        expected_states = {
            (0, 0): (0.138, 1.206, 1.206, 0.029),
            (0, 9): (0.5323, 0.0, 1.206, 0.3),
            (9, 0): (0.5323, 1.206, 0.0, 0.3),
            (9, 9): (1.5, 0.0, 0.0, 1.5),
        }
        for index, expected in expected_states.items():
            actual = (rho[index], u[index], v[index], pressure[index])
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)

        self.assertEqual(np.count_nonzero(rho == 0.138), 64)
        self.assertEqual(np.count_nonzero(rho == 0.5323), 32)
        self.assertEqual(np.count_nonzero(rho == 1.5), 4)

    def test_centered_configuration_remains_unchanged(self) -> None:
        centered = RiemannConfig3()
        self.assertEqual(centered.x_split, 0.5)
        self.assertEqual(centered.y_split, 0.5)
        self.assertEqual(centered.cfl, 0.4)


if __name__ == "__main__":
    unittest.main()
