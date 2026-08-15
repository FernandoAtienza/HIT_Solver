from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from OOP.domain import Domain2D
from OOP.equations import EulerEquation2D
from OOP.hit2d import HIT2DConfig, save_run_config
from OOP.parallel.spatial_operator import ParallelPeriodicHyperviscosity2D
from OOP.problems.riemann_config3 import (
    RiemannConfig3,
    hyperviscosity_enabled_for_config,
)
from OOP.spatial_operator import PeriodicHyperviscosity2D


class HITRunConfigTests(unittest.TestCase):
    def test_config_json_records_resolved_configuration_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            config = HIT2DConfig(
                nx=32,
                ny=24,
                target_mach=0.25,
                output_dir=output_dir,
                backend="numpy",
            )

            path = save_run_config(output_dir, config)
            payload = json.loads(path.read_text())

            self.assertEqual(path, output_dir / "config.json")
            self.assertEqual(payload["nx"], 32)
            self.assertEqual(payload["ny"], 24)
            self.assertEqual(payload["target_mach"], 0.25)
            self.assertEqual(payload["output_dir"], str(output_dir))
            self.assertEqual(payload["problem"], "hit2d")
            self.assertEqual(payload["forcing_type"], "solenoidal_shell_ou")
            self.assertEqual(payload["spatial_discretization"], "hybrid_compact_weno7")
            self.assertEqual(payload["hyperviscosity_policy"], "compact_region_faces_only")
            self.assertEqual(payload["hyperviscosity_time_policy"], "every_N_steps_no_dt_scaling")
            self.assertEqual(payload["hyperviscosity_mode"], "conservative_flux")


class CompactNodeHyperviscosityTests(unittest.TestCase):
    def setUp(self):
        self.domain = Domain2D(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 12, 10)
        self.equation = EulerEquation2D(gamma=1.4)
        x, y = self.domain.mesh()
        rho = 1.0 + 0.01 * np.sin(x) * np.cos(y)
        u = 0.02 * np.sin(y)
        v = 0.02 * np.cos(x)
        pressure = np.full_like(rho, 1.0 / 1.4)
        self.q = self.equation.conservative_from_primitive(rho, u, v, pressure)
        self.compact_mask = np.zeros((self.domain.ny, self.domain.nx), dtype=bool)
        self.compact_mask[3:7, 4:9] = True

    def _assert_filter_respects_mask(self, filter_object):
        filtered = filter_object.apply(
            self.q.copy(),
            self.equation,
            active_mask=self.compact_mask,
        )
        difference = filtered - self.q

        self.assertTrue(np.any(np.abs(difference[:, self.compact_mask]) > 0.0))
        self.assertTrue(np.allclose(difference[:, ~self.compact_mask], 0.0))

        with self.assertRaises(ValueError):
            filter_object.apply(self.q.copy(), self.equation)

    def test_numpy_filter_updates_only_compact_nodes(self):
        self._assert_filter_respects_mask(
            PeriodicHyperviscosity2D(self.domain, mn=0.002)
        )

    def test_backend_aware_filter_updates_only_compact_nodes(self):
        self._assert_filter_respects_mask(
            ParallelPeriodicHyperviscosity2D(self.domain, mn=0.002)
        )

    def test_weno_only_riemann_disables_hyperviscosity(self):
        hybrid = RiemannConfig3(scheme="hybrid", mn=0.001, hyperviscosity_interval=1)
        weno = RiemannConfig3(scheme="weno", mn=0.001, hyperviscosity_interval=1)

        self.assertTrue(hyperviscosity_enabled_for_config(hybrid))
        self.assertFalse(hyperviscosity_enabled_for_config(weno))


if __name__ == "__main__":
    unittest.main()
