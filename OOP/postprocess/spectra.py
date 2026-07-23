from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from OOP.postprocess.turnover import resolve_turnover_window, turnover_for_snapshots


@dataclass
class SpectrumResults2D:
    k: np.ndarray
    energy_mean: np.ndarray
    energy_std: np.ndarray
    enstrophy_mean: np.ndarray
    enstrophy_std: np.ndarray
    number_of_snapshots: int
    selected_time_interval: tuple[float, float]
    selected_turnover_interval: tuple[float, float]
    high_k_energy_ratio: float
    high_k_enstrophy_ratio: float


class HIT2DSpectra:
    """Shell-averaged kinetic-energy and enstrophy spectra from HIT2D snapshots."""

    def __init__(
        self,
        run_dir: str | Path,
        fluctuation_type: str = "reynolds",
        start_time: float | None = None,
        end_time: float | None = None,
        start_turnover: float | None = None,
        end_turnover: float | None = None,
        turnover_length: float | None = None,
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
        self.start_turnover = start_turnover
        self.end_turnover = end_turnover
        self.turnover_length = turnover_length
        self.stride = stride
        self.output_dir = Path(output_dir) if output_dir is not None else self.run_dir / "postprocess"
        self.results: SpectrumResults2D | None = None

    def compute(self) -> SpectrumResults2D:
        paths = sorted(self.run_dir.glob("hit2d_step*.npz"))
        if not paths:
            raise FileNotFoundError(f"No HIT2D snapshots found in {self.run_dir}")

        _snapshot_times, snapshot_turnovers, _length = turnover_for_snapshots(
            self.run_dir,
            paths,
            length_ref=self.turnover_length,
        )
        self.start_turnover, self.end_turnover = resolve_turnover_window(
            self.run_dir,
            self.start_turnover,
            self.end_turnover,
        )

        selected: list[tuple[float, float, dict[str, np.ndarray]]] = []
        for path, turnover in zip(paths, snapshot_turnovers):
            with np.load(path) as data:
                time = float(data["time"])
                if self.start_time is not None and time < self.start_time:
                    continue
                if self.end_time is not None and time > self.end_time:
                    continue
                if self.start_turnover is not None:
                    tolerance = 1.0e-6 * max(1.0, abs(self.start_turnover))
                    if turnover < self.start_turnover - tolerance:
                        continue
                if self.end_turnover is not None:
                    tolerance = 1.0e-6 * max(1.0, abs(self.end_turnover))
                    if turnover > self.end_turnover + tolerance:
                        continue
                selected.append(
                    (
                        time,
                        float(turnover),
                        {
                            "x": np.asarray(data["x"], dtype=float),
                            "y": np.asarray(data["y"], dtype=float),
                            "rho": np.asarray(data["rho"], dtype=float),
                            "u": np.asarray(data["u"], dtype=float),
                            "v": np.asarray(data["v"], dtype=float),
                        },
                    )
                )
        selected = selected[:: self.stride]
        if not selected:
            raise ValueError("No snapshots are available in the selected spectral interval")
        if len(selected) < 5:
            warnings.warn("Few snapshots selected for spectral averaging", RuntimeWarning)

        energy_samples = []
        enstrophy_samples = []
        k_shell = None
        for _time, _turnover, fields in selected:
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
            max_shell = min(nx, ny) // 2

            weighted_u_hat = np.fft.fft2(np.sqrt(rho) * u_fluct)
            weighted_v_hat = np.fft.fft2(np.sqrt(rho) * v_fluct)
            normalization = float((nx * ny) ** 2)
            energy_mode = (
                0.5
                * (np.abs(weighted_u_hat) ** 2 + np.abs(weighted_v_hat) ** 2)
                / normalization
            )

            u_hat = np.fft.fft2(u_fluct)
            v_hat = np.fft.fft2(v_fluct)
            omega_hat = 1j * (KX * v_hat - KY * u_hat)
            enstrophy_mode = 0.5 * np.abs(omega_hat) ** 2 / normalization
            energy_shell = np.bincount(
                shell_index.ravel(),
                weights=energy_mode.ravel(),
                minlength=max_shell + 1,
            )[: max_shell + 1]
            enstrophy_shell = np.bincount(
                shell_index.ravel(),
                weights=enstrophy_mode.ravel(),
                minlength=max_shell + 1,
            )[: max_shell + 1]
            energy_samples.append(energy_shell)
            enstrophy_samples.append(enstrophy_shell)
            k_shell = dk * np.arange(max_shell + 1)

        energy_stack = np.stack(energy_samples)
        enstrophy_stack = np.stack(enstrophy_samples)
        energy_mean = np.mean(energy_stack, axis=0)
        enstrophy_mean = np.mean(enstrophy_stack, axis=0)
        high_k_energy_ratio = self._cutoff_pileup_ratio(energy_mean)
        high_k_enstrophy_ratio = self._cutoff_pileup_ratio(enstrophy_mean)
        if high_k_energy_ratio > 2.0:
            warnings.warn("Kinetic-energy spectrum may be piling up near k_max", RuntimeWarning)
        if high_k_enstrophy_ratio > 2.0:
            warnings.warn("Enstrophy spectrum may be piling up near k_max", RuntimeWarning)

        times = [item[0] for item in selected]
        turnovers = [item[1] for item in selected]
        self.results = SpectrumResults2D(
            k=k_shell,
            energy_mean=energy_mean,
            energy_std=np.std(energy_stack, axis=0),
            enstrophy_mean=enstrophy_mean,
            enstrophy_std=np.std(enstrophy_stack, axis=0),
            number_of_snapshots=len(selected),
            selected_time_interval=(float(times[0]), float(times[-1])),
            selected_turnover_interval=(float(turnovers[0]), float(turnovers[-1])),
            high_k_energy_ratio=high_k_energy_ratio,
            high_k_enstrophy_ratio=high_k_enstrophy_ratio,
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
            enstrophy_mean=result.enstrophy_mean,
            enstrophy_std=result.enstrophy_std,
            number_of_snapshots=np.asarray(result.number_of_snapshots),
            selected_time_interval=np.asarray(result.selected_time_interval),
            selected_turnover_interval=np.asarray(result.selected_turnover_interval),
            high_k_energy_ratio=np.asarray(result.high_k_energy_ratio),
            high_k_enstrophy_ratio=np.asarray(result.high_k_enstrophy_ratio),
            fluctuation_type=np.asarray(self.fluctuation_type),
        )
        return path

    def plot(self, output_path: str | Path | None = None) -> Path:
        result = self._require_results()
        path = (
            Path(output_path)
            if output_path is not None
            else self.output_dir / "energy_enstrophy_spectra.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        positive = result.k > 0.0
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True, constrained_layout=True)
        panels = [
            (result.energy_mean, result.energy_std, "Kinetic-energy spectrum", "E(k)"),
            (result.enstrophy_mean, result.enstrophy_std, "Enstrophy spectrum", "Z(k)"),
        ]
        for ax, (mean, std, title, ylabel) in zip(axes, panels):
            lower = np.maximum(mean - std, 0.05 * mean)
            upper = mean + std
            visible = positive & (mean > np.max(mean) * 1.0e-14)
            ax.loglog(result.k[visible], mean[visible], linewidth=1.8)
            ax.fill_between(
                result.k[visible],
                lower[visible],
                upper[visible],
                alpha=0.18,
            )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)
        axes[-1].set_xlabel("k")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

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
