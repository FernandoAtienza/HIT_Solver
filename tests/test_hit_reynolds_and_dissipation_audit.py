from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from OOP.domain import Domain2D
from OOP.equations import CompressibleNavierStokes2D
from OOP.hit2d import (
    HIT2DConfig,
    compute_diagnostics,
    initialize_hit_and_resolve_viscosity,
    save_run_config,
)


class InitialTaylorReynoldsControlTests(unittest.TestCase):
    def test_requested_initial_re_lambda_resolves_viscosity(self):
        config = HIT2DConfig(
            nx=32,
            ny=32,
            target_mach=0.25,
            initial_kmin=3,
            initial_kmax=5,
            initial_re_lambda=80.0,
            backend="numpy",
        )
        domain = Domain2D(0.0, config.length, 0.0, config.length, 32, 32)
        q, resolved_config, metadata = initialize_hit_and_resolve_viscosity(
            config,
            domain,
            np.random.default_rng(1234),
            xp=np,
        )

        self.assertEqual(q.shape, (4, 32, 32))
        self.assertGreater(resolved_config.viscosity, 0.0)
        self.assertAlmostEqual(metadata["resolved_initial_re_lambda_2d"], 80.0, places=10)
        self.assertAlmostEqual(
            metadata["resolved_dynamic_viscosity"],
            resolved_config.viscosity,
        )

    def test_resolved_metadata_is_written_to_config_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            config = HIT2DConfig(
                initial_re_lambda=60.0,
                viscosity=2.5e-3,
                output_dir=output_dir,
            )
            metadata = {
                "requested_initial_re_lambda_2d": 60.0,
                "resolved_initial_re_lambda_2d": 60.0,
                "resolved_dynamic_viscosity": 2.5e-3,
                "initial_taylor_microscale_2d": 0.5,
            }
            path = save_run_config(output_dir, config, metadata)
            text = path.read_text()
            self.assertIn('"resolved_initial_re_lambda_2d": 60.0', text)
            self.assertIn('"resolved_dynamic_viscosity": 0.0025', text)


class DissipationDiagnosticTests(unittest.TestCase):
    def test_compute_diagnostics_reports_resolution_and_hyperviscosity(self):
        domain = Domain2D(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 24, 24)
        equation = CompressibleNavierStokes2D(
            gamma=1.4,
            viscosity=1.0e-3,
            prandtl=0.72,
        )
        x, y = domain.mesh()
        rho = np.ones_like(x)
        u = 0.2 * np.sin(3.0 * x)
        v = 0.2 * np.sin(3.0 * y)
        pressure = np.full_like(rho, 1.0 / 1.4)
        q = equation.conservative_from_primitive(rho, u, v, pressure)

        diagnostics = compute_diagnostics(
            q,
            equation,
            domain,
            hyperviscosity_drain_power=2.0e-5,
            hyperviscosity_energy_removed_cumulative=1.0e-4,
            hyperviscosity_nominal_rate=0.25,
        )

        self.assertGreater(diagnostics["re_lambda_2d"], 0.0)
        self.assertGreater(diagnostics["physical_viscous_dissipation"], 0.0)
        self.assertGreater(diagnostics["eta_over_dx"], 0.0)
        self.assertAlmostEqual(diagnostics["hyperviscosity_drain_power"], 2.0e-5)
        self.assertAlmostEqual(
            diagnostics["hyperviscosity_energy_removed_cumulative"],
            1.0e-4,
        )
        self.assertAlmostEqual(diagnostics["hyperviscosity_nominal_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
