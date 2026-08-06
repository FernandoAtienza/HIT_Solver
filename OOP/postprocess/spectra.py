from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from OOP.turbulence_statistics import helmholtz_fourier_2d


@dataclass
class SpectrumResults2D:
    k: np.ndarray
    energy_mean: np.ndarray
    energy_std: np.ndarray
    density_weighted_energy_mean: np.ndarray
    density_weighted_energy_std: np.ndarray
    solenoidal_energy_mean: np.ndarray
    solenoidal_energy_std: np.ndarray
    dilatational_energy_mean: np.ndarray
    dilatational_energy_std: np.ndarray
    energy_normalized_mean: np.ndarray
    solenoidal_energy_normalized_mean: np.ndarray
    dilatational_energy_normalized_mean: np.ndarray
    initial_energy_normalized: np.ndarray
    initial_solenoidal_energy_normalized: np.ndarray
    initial_dilatational_energy_normalized: np.ndarray
    enstrophy_mean: np.ndarray
    enstrophy_std: np.ndarray
    number_of_snapshots: int
    selected_time_interval: tuple[float, float]
    high_k_energy_ratio: float
    high_k_enstrophy_ratio: float
    complete_shell_max: int
    forcing_shell: tuple[float, float]
    parseval_energy_error: float
    helmholtz_closure_error: float
    density_weighted_parseval_error: float
    mean_dilatational_energy_fraction: float


class HIT2DSpectra:
    """Shell-integrated spectra from periodic HIT2D snapshots.

    The velocity spectrum is normalized with the FFT convention so that

    ``sum_k E(k) = 0.5 * rho_mean * <u'^2 + v'^2>``.

    This is the discrete Parseval normalization.  It is not an integral of the
    energy *flux* or cascade rate.  A separate density-weighted spectrum based on
    ``sqrt(rho) u'`` is also retained, while the Helmholtz split is performed on
    velocity and weighted by the mean density so that ``E = E_s + E_d`` closes
    mode by mode.

    Because this solver is two-dimensional and the forcing shell is located at
    low wavenumbers, the post-forcing range is compared with the forward
    enstrophy-cascade guide ``E(k) ~ k^-3``.  The corresponding enstrophy
    spectrum guide is ``Z(k) ~ k^-1`` because ``Z(k) = k^2 E(k)``.  These are
    reference slopes, not fitted laws.
    """

    def __init__(
        self,
        run_dir: str | Path,
        fluctuation_type: str = "reynolds",
        start_time: float | None = None,
        end_time: float | None = None,
        stride: int = 1,
        output_dir: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.fluctuation_type = fluctuation_type.lower()
        if self.fluctuation_type not in {"reynolds", "favre"}:
            raise ValueError("fluctuation_type must be 'reynolds' or 'favre'")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.start_time = start_time
        self.end_time = end_time
        self.stride = stride
        self.output_dir = Path(output_dir) if output_dir is not None else self.run_dir / "postprocess"
        self.results: SpectrumResults2D | None = None

    def compute(self) -> SpectrumResults2D:
        paths = sorted(self.run_dir.glob("hit2d_step*.npz"))
        if not paths:
            raise FileNotFoundError(f"No HIT2D snapshots found in {self.run_dir}")

        initial_fields = self._read_snapshot(paths[0])[1]
        selected: list[tuple[float, dict[str, np.ndarray]]] = []
        for path in paths:
            time, fields = self._read_snapshot(path)
            if self.start_time is not None and time < self.start_time:
                continue
            if self.end_time is not None and time > self.end_time:
                continue
            selected.append((time, fields))
        selected = selected[:: self.stride]
        if not selected:
            raise ValueError("No snapshots are available in the selected spectral interval")
        if len(selected) < 5:
            warnings.warn("Few snapshots selected for spectral averaging", RuntimeWarning)

        energy_samples: list[np.ndarray] = []
        density_weighted_samples: list[np.ndarray] = []
        solenoidal_samples: list[np.ndarray] = []
        dilatational_samples: list[np.ndarray] = []
        enstrophy_samples: list[np.ndarray] = []
        physical_energy_samples: list[float] = []
        density_weighted_physical_samples: list[float] = []
        k_shell: np.ndarray | None = None
        complete_shell_max = 0

        for _time, fields in selected:
            sample = self._snapshot_spectra(fields)
            energy_samples.append(sample["energy"])
            density_weighted_samples.append(sample["density_weighted_energy"])
            solenoidal_samples.append(sample["solenoidal_energy"])
            dilatational_samples.append(sample["dilatational_energy"])
            enstrophy_samples.append(sample["enstrophy"])
            physical_energy_samples.append(float(sample["physical_energy"]))
            density_weighted_physical_samples.append(
                float(sample["density_weighted_physical_energy"])
            )
            k_shell = sample["k"]
            complete_shell_max = int(sample["complete_shell_max"])

        initial = self._snapshot_spectra(initial_fields)
        if k_shell is None:
            raise RuntimeError("No spectra were computed")

        energy_stack = np.stack(energy_samples)
        density_weighted_stack = np.stack(density_weighted_samples)
        solenoidal_stack = np.stack(solenoidal_samples)
        dilatational_stack = np.stack(dilatational_samples)
        enstrophy_stack = np.stack(enstrophy_samples)

        energy_mean = np.mean(energy_stack, axis=0)
        density_weighted_mean = np.mean(density_weighted_stack, axis=0)
        solenoidal_mean = np.mean(solenoidal_stack, axis=0)
        dilatational_mean = np.mean(dilatational_stack, axis=0)
        enstrophy_mean = np.mean(enstrophy_stack, axis=0)

        complete_slice = slice(0, complete_shell_max + 1)
        high_k_energy_ratio = self._cutoff_pileup_ratio(energy_mean[complete_slice])
        high_k_enstrophy_ratio = self._cutoff_pileup_ratio(enstrophy_mean[complete_slice])
        if high_k_energy_ratio > 2.0:
            warnings.warn("Kinetic-energy spectrum may be piling up near the complete-shell cutoff", RuntimeWarning)
        if high_k_enstrophy_ratio > 2.0:
            warnings.warn("Enstrophy spectrum may be piling up near the complete-shell cutoff", RuntimeWarning)

        energy_integral = float(np.sum(energy_mean))
        density_weighted_integral = float(np.sum(density_weighted_mean))
        physical_energy_mean = float(np.mean(physical_energy_samples))
        density_weighted_physical_mean = float(np.mean(density_weighted_physical_samples))
        helmholtz_integral = float(np.sum(solenoidal_mean + dilatational_mean))
        tiny = np.finfo(float).tiny
        parseval_error = abs(energy_integral - physical_energy_mean) / max(
            abs(physical_energy_mean), tiny
        )
        density_weighted_parseval_error = abs(
            density_weighted_integral - density_weighted_physical_mean
        ) / max(abs(density_weighted_physical_mean), tiny)
        helmholtz_closure_error = float(
            np.linalg.norm(energy_mean - solenoidal_mean - dilatational_mean)
            / max(np.linalg.norm(energy_mean), tiny)
        )
        if parseval_error > 1.0e-10 or helmholtz_closure_error > 1.0e-10:
            warnings.warn(
                "Unexpected spectral normalization or Helmholtz closure error",
                RuntimeWarning,
            )

        initial_total = float(np.sum(initial["energy"]))
        energy_normalized = energy_mean / max(energy_integral, tiny)
        solenoidal_normalized = solenoidal_mean / max(energy_integral, tiny)
        dilatational_normalized = dilatational_mean / max(energy_integral, tiny)
        initial_energy_normalized = initial["energy"] / max(initial_total, tiny)
        initial_solenoidal_normalized = initial["solenoidal_energy"] / max(
            initial_total, tiny
        )
        initial_dilatational_normalized = initial["dilatational_energy"] / max(
            initial_total, tiny
        )
        mean_dilatational_fraction = float(np.sum(dilatational_mean) / max(helmholtz_integral, tiny))

        times = [item[0] for item in selected]
        self.results = SpectrumResults2D(
            k=k_shell,
            energy_mean=energy_mean,
            energy_std=np.std(energy_stack, axis=0),
            density_weighted_energy_mean=density_weighted_mean,
            density_weighted_energy_std=np.std(density_weighted_stack, axis=0),
            solenoidal_energy_mean=solenoidal_mean,
            solenoidal_energy_std=np.std(solenoidal_stack, axis=0),
            dilatational_energy_mean=dilatational_mean,
            dilatational_energy_std=np.std(dilatational_stack, axis=0),
            energy_normalized_mean=energy_normalized,
            solenoidal_energy_normalized_mean=solenoidal_normalized,
            dilatational_energy_normalized_mean=dilatational_normalized,
            initial_energy_normalized=initial_energy_normalized,
            initial_solenoidal_energy_normalized=initial_solenoidal_normalized,
            initial_dilatational_energy_normalized=initial_dilatational_normalized,
            enstrophy_mean=enstrophy_mean,
            enstrophy_std=np.std(enstrophy_stack, axis=0),
            number_of_snapshots=len(selected),
            selected_time_interval=(float(times[0]), float(times[-1])),
            high_k_energy_ratio=high_k_energy_ratio,
            high_k_enstrophy_ratio=high_k_enstrophy_ratio,
            complete_shell_max=complete_shell_max,
            forcing_shell=self._forcing_shell(
                k=np.asarray(initial["k"], dtype=float),
                initial_energy=np.asarray(initial["energy"], dtype=float),
            ),
            parseval_energy_error=float(parseval_error),
            helmholtz_closure_error=float(helmholtz_closure_error),
            density_weighted_parseval_error=float(density_weighted_parseval_error),
            mean_dilatational_energy_fraction=mean_dilatational_fraction,
        )
        return self.results

    def save(self, output_path: str | Path | None = None) -> Path:
        result = self._require_results()
        path = Path(output_path) if output_path is not None else self.run_dir / "spectra_diagnostics.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            k=result.k,
            energy_mean=result.energy_mean,
            energy_std=result.energy_std,
            density_weighted_energy_mean=result.density_weighted_energy_mean,
            density_weighted_energy_std=result.density_weighted_energy_std,
            solenoidal_energy_mean=result.solenoidal_energy_mean,
            solenoidal_energy_std=result.solenoidal_energy_std,
            dilatational_energy_mean=result.dilatational_energy_mean,
            dilatational_energy_std=result.dilatational_energy_std,
            energy_normalized_mean=result.energy_normalized_mean,
            solenoidal_energy_normalized_mean=result.solenoidal_energy_normalized_mean,
            dilatational_energy_normalized_mean=result.dilatational_energy_normalized_mean,
            initial_energy_normalized=result.initial_energy_normalized,
            initial_solenoidal_energy_normalized=result.initial_solenoidal_energy_normalized,
            initial_dilatational_energy_normalized=result.initial_dilatational_energy_normalized,
            enstrophy_mean=result.enstrophy_mean,
            enstrophy_std=result.enstrophy_std,
            number_of_snapshots=np.asarray(result.number_of_snapshots),
            selected_time_interval=np.asarray(result.selected_time_interval),
            high_k_energy_ratio=np.asarray(result.high_k_energy_ratio),
            high_k_enstrophy_ratio=np.asarray(result.high_k_enstrophy_ratio),
            complete_shell_max=np.asarray(result.complete_shell_max),
            forcing_shell=np.asarray(result.forcing_shell),
            parseval_energy_error=np.asarray(result.parseval_energy_error),
            helmholtz_closure_error=np.asarray(result.helmholtz_closure_error),
            density_weighted_parseval_error=np.asarray(result.density_weighted_parseval_error),
            mean_dilatational_energy_fraction=np.asarray(
                result.mean_dilatational_energy_fraction
            ),
            fluctuation_type=np.asarray(self.fluctuation_type),
            normalization=np.asarray(
                "sum(E)=0.5*rho_mean*mean(u_prime^2+v_prime^2); Helmholtz split uses mean density"
            ),
        )
        return path

    def plot(self, output_path: str | Path | None = None) -> Path:
        """Plot Helmholtz-decomposed energy and enstrophy spectra.

        Only one physical cascade law is used for this configuration:
        ``E(k) ~ k^-3`` in the post-forcing forward-enstrophy range.  The right
        panel shows the equivalent enstrophy scaling ``Z(k) ~ k^-1`` since
        ``Z(k) = k^2 E(k)``.  Each guide is normalized at an actual point of the
        computed curve, so it intersects that curve at the selected anchor.
        """
        result = self._require_results()
        path = (
            Path(output_path)
            if output_path is not None
            else self.output_dir / "energy_enstrophy_spectra.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        shell_number = np.arange(result.k.size)
        valid_energy = (
            (result.k > 0.0)
            & (shell_number <= result.complete_shell_max)
            & (result.energy_mean > np.max(result.energy_mean) * 1.0e-14)
        )
        valid_enstrophy = (
            (result.k > 0.0)
            & (shell_number <= result.complete_shell_max)
            & (result.enstrophy_mean > np.max(result.enstrophy_mean) * 1.0e-14)
        )
        sol_visible = valid_energy & (result.solenoidal_energy_mean > 0.0)
        dil_visible = valid_energy & (result.dilatational_energy_mean > 0.0)

        forcing_shell = self._resolved_forcing_shell(result)
        kmin, kmax = forcing_shell
        preferred_anchor = None
        if np.isfinite(kmax):
            preferred_anchor = max(kmax + 3.0, 2.0 * kmax)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13.0, 5.4),
            constrained_layout=True,
        )

        ax = axes[0]
        ax.loglog(
            result.k[valid_energy],
            result.energy_mean[valid_energy],
            linewidth=2.0,
            label=r"$E_s(k)+E_d(k)$",
        )
        ax.loglog(
            result.k[sol_visible],
            result.solenoidal_energy_mean[sol_visible],
            linewidth=1.6,
            label=r"$E_s(k)$",
        )
        ax.loglog(
            result.k[dil_visible],
            result.dilatational_energy_mean[dil_visible],
            linewidth=1.6,
            label=r"$E_d(k)$",
        )
        self._shade_forcing_shell(ax, forcing_shell)
        forward_energy = valid_energy.copy()
        if np.isfinite(kmax):
            forward_energy &= result.k > kmax
        self._add_local_power_law_guide(
            ax,
            result.k,
            result.energy_mean,
            exponent=-3.0,
            label=r"$k^{-3}$ guide",
            visible=forward_energy,
            preferred_anchor=preferred_anchor,
        )
        ax.set_title("Helmholtz-decomposed kinetic-energy spectrum")
        ax.set_xlabel("k")
        ax.set_ylabel("shell energy")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

        ax = axes[1]
        ax.loglog(
            result.k[valid_enstrophy],
            result.enstrophy_mean[valid_enstrophy],
            linewidth=2.0,
            label=r"$Z(k)$",
        )
        self._shade_forcing_shell(ax, forcing_shell)
        forward_enstrophy = valid_enstrophy.copy()
        if np.isfinite(kmax):
            forward_enstrophy &= result.k > kmax
        self._add_local_power_law_guide(
            ax,
            result.k,
            result.enstrophy_mean,
            exponent=-1.0,
            label=r"$k^{-1}$ guide ($E\sim k^{-3}$)",
            visible=forward_enstrophy,
            preferred_anchor=preferred_anchor,
        )
        ax.set_title("Enstrophy spectrum")
        ax.set_xlabel("k")
        ax.set_ylabel(r"$Z(k)$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    def _snapshot_spectra(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray | float | int]:
        x = fields["x"]
        y = fields["y"]
        rho = fields["rho"]
        u_fluct, v_fluct = self._fluctuations(rho, fields["u"], fields["v"])
        nx = x.size
        ny = y.size
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        k_magnitude = np.sqrt(KX**2 + KY**2)
        dk = min(2.0 * np.pi / (nx * dx), 2.0 * np.pi / (ny * dy))
        shell_index = np.rint(k_magnitude / dk).astype(int)
        max_shell = int(np.max(shell_index))
        complete_shell_max = min(nx, ny) // 2
        normalization = float((nx * ny) ** 2)
        rho_mean = float(np.mean(rho))

        u_hat = np.fft.fft2(u_fluct)
        v_hat = np.fft.fft2(v_fluct)
        energy_mode = 0.5 * rho_mean * (
            np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2
        ) / normalization

        us_hat, vs_hat, ud_hat, vd_hat = helmholtz_fourier_2d(
            u_fluct, v_fluct, dx, dy
        )
        solenoidal_mode = 0.5 * rho_mean * (
            np.abs(us_hat) ** 2 + np.abs(vs_hat) ** 2
        ) / normalization
        dilatational_mode = 0.5 * rho_mean * (
            np.abs(ud_hat) ** 2 + np.abs(vd_hat) ** 2
        ) / normalization

        weighted_u_hat = np.fft.fft2(np.sqrt(rho) * u_fluct)
        weighted_v_hat = np.fft.fft2(np.sqrt(rho) * v_fluct)
        density_weighted_mode = 0.5 * (
            np.abs(weighted_u_hat) ** 2 + np.abs(weighted_v_hat) ** 2
        ) / normalization

        omega_hat = 1j * (KX * v_hat - KY * u_hat)
        enstrophy_mode = 0.5 * np.abs(omega_hat) ** 2 / normalization

        def shell_sum(mode: np.ndarray) -> np.ndarray:
            return np.bincount(
                shell_index.ravel(),
                weights=mode.ravel(),
                minlength=max_shell + 1,
            )[: max_shell + 1]

        return {
            "k": dk * np.arange(max_shell + 1),
            "energy": shell_sum(energy_mode),
            "density_weighted_energy": shell_sum(density_weighted_mode),
            "solenoidal_energy": shell_sum(solenoidal_mode),
            "dilatational_energy": shell_sum(dilatational_mode),
            "enstrophy": shell_sum(enstrophy_mode),
            "physical_energy": 0.5 * rho_mean * float(
                np.mean(u_fluct**2 + v_fluct**2)
            ),
            "density_weighted_physical_energy": 0.5 * float(
                np.mean(rho * (u_fluct**2 + v_fluct**2))
            ),
            "complete_shell_max": complete_shell_max,
        }

    def _read_snapshot(self, path: Path) -> tuple[float, dict[str, np.ndarray]]:
        with np.load(path) as data:
            return float(data["time"]), {
                "x": np.asarray(data["x"], dtype=float),
                "y": np.asarray(data["y"], dtype=float),
                "rho": np.asarray(data["rho"], dtype=float),
                "u": np.asarray(data["u"], dtype=float),
                "v": np.asarray(data["v"], dtype=float),
            }

    def _fluctuations(
        self, rho: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.fluctuation_type == "favre":
            rho_mean = float(np.mean(rho))
            u_mean = float(np.mean(rho * u) / rho_mean)
            v_mean = float(np.mean(rho * v) / rho_mean)
        else:
            u_mean = float(np.mean(u))
            v_mean = float(np.mean(v))
        return u - u_mean, v - v_mean

    def _forcing_shell(
        self,
        k: np.ndarray | None = None,
        initial_energy: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Read the forcing shell, with a fallback for older HIT runs.

        Older result folders may not contain ``config.json`` or may use legacy
        key names.  When metadata cannot be recovered, the initialized spectral
        support is used: the initial velocity is shell-restricted, so shells
        whose energy exceeds a small relative threshold identify ``k_f,min`` and
        ``k_f,max`` robustly.
        """
        config_path = self.run_dir / "config.json"
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text())
                containers = [payload]
                for name in ("config", "hit2d_config", "parameters"):
                    nested = payload.get(name)
                    if isinstance(nested, dict):
                        containers.append(nested)

                kmin_keys = (
                    "forcing_kmin",
                    "kf_min",
                    "forcing_k_min",
                    "k_force_min",
                )
                kmax_keys = (
                    "forcing_kmax",
                    "kf_max",
                    "forcing_k_max",
                    "k_force_max",
                )

                def first_value(keys: tuple[str, ...]):
                    for container in containers:
                        for key in keys:
                            value = container.get(key)
                            if value is not None:
                                return float(value)
                    return None

                kmin = first_value(kmin_keys)
                kmax = first_value(kmax_keys)
                if (
                    kmin is not None
                    and kmax is not None
                    and np.isfinite(kmin)
                    and np.isfinite(kmax)
                    and kmax >= kmin >= 0.0
                ):
                    return (float(kmin), float(kmax))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        if k is not None and initial_energy is not None:
            k = np.asarray(k, dtype=float)
            initial_energy = np.asarray(initial_energy, dtype=float)
            if k.shape == initial_energy.shape and initial_energy.size:
                peak = float(np.max(initial_energy))
                if peak > 0.0:
                    support = (k > 0.0) & (initial_energy > peak * 1.0e-10)
                    if np.any(support):
                        return (
                            float(np.min(k[support])),
                            float(np.max(k[support])),
                        )

        return (float("nan"), float("nan"))

    @staticmethod
    def _resolved_forcing_shell(
        result: SpectrumResults2D,
    ) -> tuple[float, float]:
        """Return saved forcing metadata or infer it from the initial spectrum."""
        kmin, kmax = result.forcing_shell
        if (
            np.isfinite(kmin)
            and np.isfinite(kmax)
            and kmax >= kmin >= 0.0
        ):
            return (float(kmin), float(kmax))

        initial = np.asarray(result.initial_energy_normalized, dtype=float)
        k = np.asarray(result.k, dtype=float)
        if initial.shape == k.shape and initial.size:
            peak = float(np.max(initial))
            if peak > 0.0:
                support = (k > 0.0) & (initial > peak * 1.0e-10)
                if np.any(support):
                    return (
                        float(np.min(k[support])),
                        float(np.max(k[support])),
                    )
        return (float("nan"), float("nan"))

    @staticmethod
    def _shade_forcing_shell(ax, forcing_shell: tuple[float, float]) -> None:
        kmin, kmax = forcing_shell
        if np.isfinite(kmin) and np.isfinite(kmax) and kmax > kmin:
            ax.axvspan(max(kmin, np.finfo(float).tiny), kmax, alpha=0.08, label="forced shell")

    @staticmethod
    def _add_local_power_law_guide(
        ax,
        k: np.ndarray,
        values: np.ndarray,
        exponent: float,
        label: str,
        visible: np.ndarray,
        preferred_anchor: float | None,
    ) -> None:
        """Draw a localized slope guide that intersects the computed spectrum.

        The guide is normalized at one actual spectrum point, so it necessarily
        touches the curve at the anchor. It is limited to the selected range to
        avoid implying a power law over the full resolved spectrum.
        """
        indices = np.flatnonzero(visible & (values > 0.0))
        if indices.size < 2:
            return

        if preferred_anchor is None:
            anchor_index = indices[indices.size // 2]
        else:
            anchor_index = indices[np.argmin(np.abs(k[indices] - preferred_anchor))]

        # Keep the visual guide local: at most about one decade and no more than
        # the available points in the physically relevant range.
        k_anchor = float(k[anchor_index])
        value_anchor = float(values[anchor_index])
        k_low = k_anchor / np.sqrt(10.0)
        k_high = k_anchor * np.sqrt(10.0)
        local = indices[(k[indices] >= k_low) & (k[indices] <= k_high)]
        if local.size < 2:
            local = indices

        k_line = k[local]
        guide = value_anchor * (k_line / k_anchor) ** exponent
        ax.loglog(k_line, guide, linestyle=":", linewidth=1.4, label=label)

    @staticmethod
    def _cutoff_pileup_ratio(values: np.ndarray) -> float:
        nonzero = values[1:]
        width = max(3, nonzero.size // 10)
        if nonzero.size < 2 * width:
            return 0.0
        reference = float(np.mean(nonzero[-2 * width : -width]))
        cutoff = float(np.mean(nonzero[-width:]))
        if float(np.sum(nonzero[-width:])) < 1.0e-8 * float(np.sum(nonzero)):
            return 0.0
        return cutoff / max(reference, np.finfo(float).tiny)

    def _require_results(self) -> SpectrumResults2D:
        if self.results is None:
            raise RuntimeError("Call compute() before save() or plot()")
        return self.results
