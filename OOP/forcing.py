from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from OOP.domain import Domain2D


@dataclass
class IsotropicShellOUForcing2D:
    """Finite-correlation-time solenoidal forcing on a complete Fourier shell.

    A real random scalar potential is transformed to Fourier space, restricted
    to k_min <= |k| <= k_max, and converted to acceleration through
    f_hat = (i*ky*phi_hat, -i*kx*phi_hat). This is perpendicular to every
    wavevector, includes all rotated/reflected shell modes, and retains
    Hermitian symmetry.
    """

    domain: Domain2D
    k_min: float
    k_max: float
    correlation_time: float = 1.0
    force_rms: float = 1.0
    target_power: float | None = 1.0e-3
    min_power: float = 1.0e-6
    max_rescale: float | None = 20.0
    alpha_memory: float = 0.2
    seed: int = 1234
    xp: object = np
    _potential_hat: object = field(default=None, init=False, repr=False)
    _alpha: float = field(default=1.0, init=False, repr=False)
    _rng: object = field(default=None, init=False, repr=False)
    _kx: object = field(default=None, init=False, repr=False)
    _ky: object = field(default=None, init=False, repr=False)
    _mask: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k_min < 0.0 or self.k_max <= 0.0 or self.k_min > self.k_max:
            raise ValueError("forcing shell must satisfy 0 <= k_min <= k_max")
        if self.correlation_time <= 0.0:
            raise ValueError("forcing correlation time must be positive")
        if not 0.0 <= self.alpha_memory < 1.0:
            raise ValueError("alpha_memory must be in [0, 1)")

        kx_values = 2.0 * np.pi * np.fft.fftfreq(self.domain.nx, d=self.domain.dx)
        ky_values = 2.0 * np.pi * np.fft.fftfreq(self.domain.ny, d=self.domain.dy)
        self._kx, self._ky = self.xp.meshgrid(
            self.xp.asarray(kx_values),
            self.xp.asarray(ky_values),
            indexing="xy",
        )
        magnitude = self.xp.sqrt(self._kx**2 + self._ky**2)
        self._mask = (
            (magnitude >= float(self.k_min))
            & (magnitude <= float(self.k_max))
            & (magnitude > 0.0)
        )
        if int(self.xp.count_nonzero(self._mask)) == 0:
            raise ValueError("forcing shell contains no Fourier modes on this grid")
        self._potential_hat = self.xp.zeros(
            (self.domain.ny, self.domain.nx), dtype=self.xp.complex128
        )
        if self.xp is np:
            self._rng = np.random.default_rng(self.seed)
        else:
            self._rng = self.xp.random.RandomState(self.seed)

    def update(self, dt: float, rho, u, v) -> tuple[object, object, dict[str, float]]:
        """Advance the OU state once and return one forcing field for this step."""

        if dt <= 0.0:
            raise ValueError("forcing update requires dt > 0")
        decay = float(np.exp(-dt / self.correlation_time))
        increment_scale = float(np.sqrt(max(1.0 - decay**2, 0.0)))
        random_real = self._rng.standard_normal((self.domain.ny, self.domain.nx))
        random_hat = self.xp.fft.fft2(random_real)
        random_hat = self.xp.where(self._mask, random_hat, 0.0)
        self._potential_hat = decay * self._potential_hat + increment_scale * random_hat

        fx_hat = 1j * self._ky * self._potential_hat
        fy_hat = -1j * self._kx * self._potential_hat
        fx = self.xp.fft.ifft2(fx_hat).real
        fy = self.xp.fft.ifft2(fy_hat).real

        vector_rms = float(self.xp.sqrt(self.xp.mean(fx**2 + fy**2)))
        if self.force_rms <= 0.0:
            fx.fill(0.0)
            fy.fill(0.0)
        elif vector_rms > np.finfo(float).eps:
            force_scale = self.force_rms / vector_rms
            fx *= force_scale
            fy *= force_scale

        power_before = float(self.xp.mean(rho * (fx * u + fy * v)))
        alpha_target = self._alpha
        if self.target_power is not None and abs(power_before) > self.min_power:
            alpha_target = self.target_power / power_before
            if self.max_rescale is not None:
                alpha_target = float(
                    np.clip(alpha_target, -self.max_rescale, self.max_rescale)
                )
            self._alpha = (
                self.alpha_memory * self._alpha
                + (1.0 - self.alpha_memory) * alpha_target
            )
        elif self.target_power is None:
            self._alpha = 1.0

        fx *= self._alpha
        fy *= self._alpha
        fxx = float(self.xp.mean(fx**2))
        fyy = float(self.xp.mean(fy**2))
        fxy = float(self.xp.mean(fx * fy))
        denominator = fxx + fyy
        forcing_anisotropy = abs(fxx - fyy) / denominator if denominator > 0.0 else 0.0
        injected_power = float(self.xp.mean(rho * (fx * u + fy * v)))
        return fx, fy, {
            "alpha": float(self._alpha),
            "alpha_target": float(alpha_target),
            "power_before_rescale": power_before,
            "injected_power": injected_power,
            "Fxx": fxx,
            "Fyy": fyy,
            "Fxy": fxy,
            "A_F": forcing_anisotropy,
            "ou_decay": decay,
        }
