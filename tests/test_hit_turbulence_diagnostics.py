from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from OOP.postprocess.pdfs import HIT2DPDFDiagnostics
from OOP.postprocess.spectra import HIT2DSpectra
from OOP.turbulence_statistics import helmholtz_fourier_2d


class TurbulencePostprocessingTests(unittest.TestCase):
    def setUp(self):
        self.nx = 32
        self.ny = 32
        self.length = 2.0 * np.pi
        self.dx = self.length / self.nx
        self.dy = self.length / self.ny
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")

    def test_helmholtz_decomposition_closes(self):
        u = np.sin(3.0 * self.Y) + 0.3 * np.cos(2.0 * self.X)
        v = np.sin(2.0 * self.X) + 0.2 * np.cos(3.0 * self.Y)
        us, vs, ud, vd = helmholtz_fourier_2d(u, v, self.dx, self.dy)
        u_hat = np.fft.fft2(u - np.mean(u))
        v_hat = np.fft.fft2(v - np.mean(v))
        self.assertTrue(np.allclose(us + ud, u_hat, atol=1.0e-10))
        self.assertTrue(np.allclose(vs + vd, v_hat, atol=1.0e-10))

    @staticmethod
    def _make_run(directory: Path) -> None:
        nx = ny = 32
        length = 2.0 * np.pi
        x = np.arange(nx) * length / nx
        y = np.arange(ny) * length / ny
        X, Y = np.meshgrid(x, y, indexing="xy")
        (directory / "config.json").write_text(
            json.dumps({"forcing_kmin": 3.0, "forcing_kmax": 5.0})
        )
        for step in range(6):
            phase = 0.1 * step
            u = np.sin(3.0 * Y + phase) + 0.08 * np.cos(4.0 * X)
            v = np.sin(3.0 * X - phase) + 0.05 * np.cos(4.0 * Y)
            rho = 1.0 + 0.01 * np.sin(X + Y + phase)
            pressure = 1.0 / 1.4 + 0.005 * np.cos(2.0 * X - phase)
            u_hat = np.fft.fft2(u)
            v_hat = np.fft.fft2(v)
            kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
            ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=length / ny)
            KX, KY = np.meshgrid(kx, ky, indexing="xy")
            divergence = np.fft.ifft2(1j * (KX * u_hat + KY * v_hat)).real
            vorticity = np.fft.ifft2(1j * (KX * v_hat - KY * u_hat)).real
            np.savez_compressed(
                directory / f"hit2d_step{step:07d}.npz",
                step=step,
                time=float(step),
                x=x,
                y=y,
                rho=rho,
                u=u,
                v=v,
                pressure=pressure,
                divergence=divergence,
                vorticity=vorticity,
            )

    def test_spectra_parseval_and_helmholtz_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._make_run(run_dir)
            spectra = HIT2DSpectra(run_dir)
            result = spectra.compute()
            self.assertLess(result.parseval_energy_error, 1.0e-12)
            self.assertLess(result.helmholtz_closure_error, 1.0e-12)
            self.assertAlmostEqual(float(np.sum(result.energy_normalized_mean)), 1.0)
            self.assertTrue(spectra.save().exists())
            self.assertTrue(spectra.plot().exists())

    def test_pdf_diagnostics_are_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._make_run(run_dir)
            pdfs = HIT2DPDFDiagnostics(run_dir, bins=61, joint_bins=51)
            result = pdfs.compute()
            self.assertEqual(result.number_of_snapshots, 6)
            self.assertIn("dilatation", result.moments)
            self.assertIn("vorticity", result.moments)
            self.assertTrue(pdfs.save().exists())
            self.assertTrue(pdfs.plot().exists())


if __name__ == "__main__":
    unittest.main()
