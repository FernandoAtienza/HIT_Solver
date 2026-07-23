from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from OOP.domain import Domain2D


@dataclass
class IsotropicShellOUForcing2D:
    """Finite-correlation-time forcing on a complete Fourier shell.

    A real random scalar potential is transformed to Fourier space and
    restricted to ``k_min <= |k| <= k_max``. The vector forcing is then built
    from its solenoidal and dilatational Helmholtz components,

    ``f_s_hat = i (k_y, -k_x) phi_hat`` and
    ``f_d_hat = i (k_x,  k_y) phi_hat``.

    ``forcing_mode='solenoidal'`` selects only the divergence-free component,
    ``'dilatational'`` selects only the curl-free component, and ``'mixed'``
    combines both using ``dilatational_fraction`` as the prescribed fraction
    of forcing variance in the dilatational component. The same scalar OU state
    is used for both orthogonal projections, so the total spectral amplitude is
    independent of the selected mixture.
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
    forcing_mode: str = "solenoidal"
    dilatational_fraction: float = 0.5
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
        if self.forcing_mode not in {"solenoidal", "dilatational", "mixed"}:
            raise ValueError(
                "forcing_mode must be 'solenoidal', 'dilatational', or 'mixed'"
            )
        if not 0.0 <= self.dilatational_fraction <= 1.0:
            raise ValueError("dilatational_fraction must be in [0, 1]")

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

    @property
    def resolved_dilatational_fraction(self) -> float:
        if self.forcing_mode == "solenoidal":
            return 0.0
        if self.forcing_mode == "dilatational":
            return 1.0
        return float(self.dilatational_fraction)

    def _project_potential(self):
        """Return the shell forcing coefficients before scalar normalization."""

        chi = self.resolved_dilatational_fraction
        solenoidal_weight = float(np.sqrt(max(1.0 - chi, 0.0)))
        dilatational_weight = float(np.sqrt(max(chi, 0.0)))

        # The two vectors (k_y, -k_x) and (k_x, k_y) are orthogonal for every
        # non-zero Fourier mode. Square-root weights therefore prescribe the
        # variance fraction without changing the total modal amplitude.
        fx_hat = 1j * (
            solenoidal_weight * self._ky
            + dilatational_weight * self._kx
        ) * self._potential_hat
        fy_hat = 1j * (
            -solenoidal_weight * self._kx
            + dilatational_weight * self._ky
        ) * self._potential_hat
        return fx_hat, fy_hat

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

        fx_hat, fy_hat = self._project_potential()
        fx = self.xp.fft.ifft2(fx_hat).real
        fy = self.xp.fft.ifft2(fy_hat).real

        vector_rms = float(self.xp.sqrt(self.xp.mean(fx**2 + fy**2)))
        force_scale = 0.0
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

        total_spectral_scale = force_scale * self._alpha
        scaled_fx_hat = total_spectral_scale * fx_hat
        scaled_fy_hat = total_spectral_scale * fy_hat
        divergence_hat = 1j * (
            self._kx * scaled_fx_hat + self._ky * scaled_fy_hat
        )
        curl_hat = 1j * (
            self._kx * scaled_fy_hat - self._ky * scaled_fx_hat
        )
        number_of_points = float(self.domain.nx * self.domain.ny)
        forcing_divergence_rms = float(
            self.xp.sqrt(self.xp.sum(self.xp.abs(divergence_hat) ** 2))
            / number_of_points
        )
        forcing_curl_rms = float(
            self.xp.sqrt(self.xp.sum(self.xp.abs(curl_hat) ** 2))
            / number_of_points
        )

        fxx = float(self.xp.mean(fx**2))
        fyy = float(self.xp.mean(fy**2))
        fxy = float(self.xp.mean(fx * fy))
        denominator = fxx + fyy
        forcing_anisotropy = abs(fxx - fyy) / denominator if denominator > 0.0 else 0.0
        injected_power = float(self.xp.mean(rho * (fx * u + fy * v)))
        return fx, fy, {
            "alpha": float(self._alpha),
            "alpha_target": float(alpha_target),
            "target_power": (
                float("nan") if self.target_power is None else float(self.target_power)
            ),
            "power_before_rescale": power_before,
            "injected_power": injected_power,
            "Fxx": fxx,
            "Fyy": fyy,
            "Fxy": fxy,
            "A_F": forcing_anisotropy,
            "ou_decay": decay,
            "forcing_dilatational_fraction": self.resolved_dilatational_fraction,
            "forcing_solenoidal_fraction": 1.0 - self.resolved_dilatational_fraction,
            "forcing_divergence_rms": forcing_divergence_rms,
            "forcing_curl_rms": forcing_curl_rms,
        }
