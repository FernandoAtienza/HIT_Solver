from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from OOP.domain import Domain2D


@dataclass
class IsotropicShellOUForcing2D:
    """Finite-correlation-time forcing on a complete Fourier shell.

    The Ornstein-Uhlenbeck state is a real scalar potential transformed to
    Fourier space and restricted to ``k_min <= |k| <= k_max``. The requested
    Helmholtz component is then formed as

    - solenoidal:    ``f_hat = (i*ky*phi_hat, -i*kx*phi_hat)``;
    - dilatational:  ``f_hat = (i*kx*phi_hat,  i*ky*phi_hat)``.

    The first field is perpendicular to every nonzero wavevector and therefore
    divergence-free. The second is parallel to every nonzero wavevector and
    therefore curl-free. Both constructions preserve Hermitian symmetry and
    produce real physical-space acceleration fields after the inverse FFT.
    """

    domain: Domain2D
    k_min: float
    k_max: float
    mode: str = "solenoidal"
    correlation_time: float = 1.0
    force_rms: float = 1.0
    target_power: float | None = 1.0e-3
    min_power: float = 1.0e-6
    max_rescale: float | None = 20.0
    alpha_memory: float = 0.2
    alpha_response_time: float = 0.25
    seed: int = 1234
    xp: object = np
    _potential_hat: object = field(default=None, init=False, repr=False)
    _alpha: float = field(default=1.0, init=False, repr=False)
    _rng: object = field(default=None, init=False, repr=False)
    _kx: object = field(default=None, init=False, repr=False)
    _ky: object = field(default=None, init=False, repr=False)
    _mask: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        normalized_mode = self.mode.strip().lower()
        if normalized_mode == "compressive":
            normalized_mode = "dilatational"
        if normalized_mode not in {"solenoidal", "dilatational"}:
            raise ValueError(
                "forcing mode must be 'solenoidal' or 'dilatational' "
                "('compressive' is accepted as an alias)"
            )
        self.mode = normalized_mode

        if self.k_min < 0.0 or self.k_max <= 0.0 or self.k_min > self.k_max:
            raise ValueError("forcing shell must satisfy 0 <= k_min <= k_max")
        if self.correlation_time <= 0.0:
            raise ValueError("forcing correlation time must be positive")
        if not 0.0 <= self.alpha_memory < 1.0:
            raise ValueError("alpha_memory must be in [0, 1)")
        if self.alpha_response_time <= 0.0:
            raise ValueError("alpha_response_time must be positive")

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

    def _project_potential(self) -> tuple[object, object]:
        """Return the selected pure Helmholtz component in Fourier space."""

        if self.mode == "solenoidal":
            return (
                1j * self._ky * self._potential_hat,
                -1j * self._kx * self._potential_hat,
            )
        return (
            1j * self._kx * self._potential_hat,
            1j * self._ky * self._potential_hat,
        )

    def update(self, dt: float, rho, u, v) -> tuple[object, object, dict[str, float]]:
        """Advance the OU state once and return one acceleration field."""

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

            if self.mode == "dilatational":
                # Curl-free forcing can have a rapidly changing velocity-force
                # correlation because of acoustic oscillations. Smoothing a signed
                # coefficient across a zero crossing can temporarily extract energy
                # and drives the large power oscillations seen in long runs. Follow
                # the required sign immediately, while smoothing only its magnitude
                # in physical time.
                sign = 1.0 if alpha_target >= 0.0 else -1.0
                target_magnitude = abs(alpha_target)
                current_magnitude = abs(self._alpha)
                relaxation = 1.0 - float(np.exp(-dt / self.alpha_response_time))
                smoothed_magnitude = (
                    current_magnitude
                    + relaxation * (target_magnitude - current_magnitude)
                )
                self._alpha = sign * smoothed_magnitude
            else:
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
        is_dilatational = float(self.mode == "dilatational")
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
            "forcing_solenoidal_fraction": 1.0 - is_dilatational,
            "forcing_dilatational_fraction": is_dilatational,
            "ou_decay": decay,
        }
