from __future__ import annotations

import unittest

import numpy as np

from OOP.problems.riemann_config6 import RiemannConfig6


class RiemannConfig6InitialStateTests(unittest.TestCase):
    def test_reference_quadrant_states(self) -> None:
        config = RiemannConfig6(nx=4, ny=4, backend="numpy")
        q = config.initial_state()
        rho, u, v, pressure = config.equation.primitive_from_conservative(q)

        np.testing.assert_array_equal(
            rho,
            [
                [1.0, 1.0, 3.0, 3.0],
                [1.0, 1.0, 3.0, 3.0],
                [2.0, 2.0, 1.0, 1.0],
                [2.0, 2.0, 1.0, 1.0],
            ],
        )
        np.testing.assert_array_equal(
            u,
            [
                [-0.75, -0.75, -0.75, -0.75],
                [-0.75, -0.75, -0.75, -0.75],
                [0.75, 0.75, 0.75, 0.75],
                [0.75, 0.75, 0.75, 0.75],
            ],
        )
        np.testing.assert_array_equal(
            v,
            [
                [0.50, 0.50, -0.50, -0.50],
                [0.50, 0.50, -0.50, -0.50],
                [0.50, 0.50, -0.50, -0.50],
                [0.50, 0.50, -0.50, -0.50],
            ],
        )
        np.testing.assert_array_equal(pressure, np.ones((4, 4)))

    def test_reference_defaults(self) -> None:
        config = RiemannConfig6()
        self.assertEqual(config.configuration_number, 6)
        self.assertEqual(config.tfinal, 0.25)
        self.assertEqual(config.cfl, 0.4)
        self.assertEqual(config.backend, "cupy")


if __name__ == "__main__":
    unittest.main()
