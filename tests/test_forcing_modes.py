from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OOP.domain import Domain2D
from OOP.forcing import IsotropicShellOUForcing2D


def spectral_divergence_and_curl(fx: np.ndarray, fy: np.ndarray, domain: Domain2D):
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(domain.nx, d=domain.dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(domain.ny, d=domain.dy)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")
    fx_hat = np.fft.fft2(fx)
    fy_hat = np.fft.fft2(fy)
    divergence = np.fft.ifft2(1j * (kx * fx_hat + ky * fy_hat)).real
    curl = np.fft.ifft2(1j * (kx * fy_hat - ky * fx_hat)).real
    return divergence, curl


class TestForcingModes(unittest.TestCase):
    def setUp(self) -> None:
        self.domain = Domain2D(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 32, 32)
        rng = np.random.default_rng(17)
        self.rho = np.ones((32, 32))
        self.u = rng.standard_normal((32, 32))
        self.v = rng.standard_normal((32, 32))

    def make_force(self, mode: str):
        forcing = IsotropicShellOUForcing2D(
            domain=self.domain,
            k_min=3.0,
            k_max=5.0,
            mode=mode,
            correlation_time=0.5,
            force_rms=1.0,
            target_power=None,
            seed=1235,
            xp=np,
        )
        return forcing.update(0.01, self.rho, self.u, self.v)

    def test_solenoidal_force_is_divergence_free(self) -> None:
        fx, fy, info = self.make_force("solenoidal")
        divergence, curl = spectral_divergence_and_curl(fx, fy, self.domain)
        self.assertLess(np.sqrt(np.mean(divergence**2)), 1.0e-11)
        self.assertGreater(np.sqrt(np.mean(curl**2)), 1.0e-6)
        self.assertEqual(info["forcing_solenoidal_fraction"], 1.0)
        self.assertEqual(info["forcing_dilatational_fraction"], 0.0)

    def test_dilatational_force_is_curl_free(self) -> None:
        fx, fy, info = self.make_force("dilatational")
        divergence, curl = spectral_divergence_and_curl(fx, fy, self.domain)
        self.assertLess(np.sqrt(np.mean(curl**2)), 1.0e-11)
        self.assertGreater(np.sqrt(np.mean(divergence**2)), 1.0e-6)
        self.assertEqual(info["forcing_solenoidal_fraction"], 0.0)
        self.assertEqual(info["forcing_dilatational_fraction"], 1.0)

    def test_compressive_alias(self) -> None:
        forcing = IsotropicShellOUForcing2D(
            domain=self.domain,
            k_min=3.0,
            k_max=5.0,
            mode="compressive",
            target_power=None,
        )
        self.assertEqual(forcing.mode, "dilatational")


if __name__ == "__main__":
    unittest.main()
