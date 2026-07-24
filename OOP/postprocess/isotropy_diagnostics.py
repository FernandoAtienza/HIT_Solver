from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from OOP.postprocess.two_point_correlation import TwoPointCorrelation2D
from OOP.postprocess.turnover import cumulative_turnover, infer_turnover_length


@dataclass
class IsotropyResults2D:
    """Time histories and directional correlations for a saved HIT2D run."""

    time: np.ndarray
    turnover: np.ndarray
    selected_mask: np.ndarray
    K: np.ndarray
    Mt: np.ndarray
    Kx: np.ndarray
    Ky: np.ndarray
    A_K: np.ndarray
    C_uv: np.ndarray
    r: np.ndarray
    Ruu_x_mean: np.ndarray
    Rvv_y_mean: np.ndarray
    Rvv_x_mean: np.ndarray
    Ruu_y_mean: np.ndarray
    Ruu_x_std: np.ndarray
    Rvv_y_std: np.ndarray
    Rvv_x_std: np.ndarray
    Ruu_y_std: np.ndarray
    E_LL: float
    E_NN: float
    E_LL_normalized: float
    E_NN_normalized: float
    fluctuation_type: str
    selected_time_interval: tuple[float, float]
    selected_turnover_interval: tuple[float, float]
    turnover_length: float
    number_of_snapshots: int
    nx: int
    ny: int
    Lx: float
    Ly: float


class IsotropyDiagnostics2D:
    """Assess stationarity and directional isotropy from HIT2D snapshots.

    Reynolds fluctuations subtract the spatial mean velocity. Favre
    fluctuations subtract the density-weighted mean velocity. K, Kx, Ky,
    anisotropy, and velocity covariance remain density weighted for both
    definitions, matching the compressible kinetic-energy convention.
    """

    def __init__(
        self,
        run_dir: str | Path,
        fluctuation_type: str = "reynolds",
        gamma: float = 1.4,
        start_time: float | None = None,
        end_time: float | None = None,
        start_turnover: float | None = None,
        end_turnover: float | None = None,
        turnover_length: float | None = None,
        start_snapshot: int | None = None,
        end_snapshot: int | None = None,
        stride: int = 1,
        Lx: float | None = None,
        Ly: float | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.fluctuation_type = fluctuation_type.lower()
        if self.fluctuation_type not in {"reynolds", "favre"}:
            raise ValueError("fluctuation_type must be 'reynolds' or 'favre'")
        if gamma <= 1.0:
            raise ValueError("gamma must be greater than one")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.gamma = gamma
        self.start_time = start_time
        self.end_time = end_time
        self.start_turnover = start_turnover
        self.end_turnover = end_turnover
        self.turnover_length = turnover_length
        self.start_snapshot = start_snapshot
        self.end_snapshot = end_snapshot
        self.stride = stride
        self.Lx = Lx
        self.Ly = Ly
        self.output_dir = Path(output_dir) if output_dir is not None else self.run_dir / "postprocess"
        self.snapshot_paths: list[Path] = []
        self.results: IsotropyResults2D | None = None

    def compute(self) -> IsotropyResults2D:
        """Compute histories from all snapshots and correlations from the selected interval."""

        self.snapshot_paths = sorted(self.run_dir.glob("hit2d_step*.npz"))
        if not self.snapshot_paths:
            raise FileNotFoundError(f"No HIT2D snapshots found in {self.run_dir}")

        snapshots = [self._load_snapshot(path) for path in self.snapshot_paths]
        time = np.asarray([snapshot["time"] for snapshot in snapshots], dtype=float)
        first = snapshots[0]
        x = first["x"]
        y = first["y"]
        nx = int(x.size)
        ny = int(y.size)
        dx = self._spacing(x)
        dy = self._spacing(y)
        Lx = float(self.Lx if self.Lx is not None else nx * dx)
        Ly = float(self.Ly if self.Ly is not None else ny * dy)
        self._validate_snapshot_shapes(snapshots, ny, nx)

        K = np.empty(time.size)
        Mt = np.empty(time.size)
        Kx = np.empty(time.size)
        Ky = np.empty(time.size)
        A_K = np.empty(time.size)
        C_uv = np.empty(time.size)
        u_rms = np.empty(time.size)

        correlation_samples = {
            "Ruu_x": [],
            "Rvv_y": [],
            "Rvv_x": [],
            "Ruu_y": [],
        }
        r = self._common_separations(nx, ny, dx, dy, Lx, Ly)

        for index, snapshot in enumerate(snapshots):
            rho = snapshot["rho"]
            u = snapshot["u"]
            v = snapshot["v"]
            pressure = snapshot["pressure"]
            u_fluct, v_fluct = self._velocity_fluctuations(rho, u, v)

            uu = float(np.mean(rho * u_fluct**2))
            vv = float(np.mean(rho * v_fluct**2))
            uv = float(np.mean(rho * u_fluct * v_fluct))
            Kx[index] = 0.5 * uu
            Ky[index] = 0.5 * vv
            K[index] = Kx[index] + Ky[index]
            denominator = uu + vv
            A_K[index] = abs(uu - vv) / denominator if denominator > 0.0 else 0.0
            covariance_scale = np.sqrt(max(uu * vv, 0.0))
            C_uv[index] = uv / covariance_scale if covariance_scale > 0.0 else 0.0
            u_rms[index] = float(np.sqrt(np.mean(u_fluct**2 + v_fluct**2)))

            sound_speed = np.sqrt(
                np.maximum(self.gamma * pressure / rho, np.finfo(float).tiny)
            )
            Mt[index] = float(u_rms[index] / np.mean(sound_speed))

        length_ref = (
            infer_turnover_length(self.run_dir)
            if self.turnover_length is None
            else float(self.turnover_length)
        )
        turnover = cumulative_turnover(time, u_rms, length_ref)
        selected_mask = self._selection_mask(time, turnover)
        selected_indices = np.flatnonzero(selected_mask)[:: self.stride]
        if selected_indices.size == 0:
            raise ValueError("The selected statistically stationary interval contains no snapshots")
        if selected_indices.size < 5:
            warnings.warn(
                f"Only {selected_indices.size} snapshots selected; isotropy statistics may be noisy",
                RuntimeWarning,
            )

        for index in selected_indices:
            snapshot = snapshots[index]
            rho = snapshot["rho"]
            u = snapshot["u"]
            v = snapshot["v"]
            u_fluct, v_fluct = self._velocity_fluctuations(rho, u, v)
            Ruu = TwoPointCorrelation2D.autocorrelation(u_fluct, normalize=True)
            Rvv = TwoPointCorrelation2D.autocorrelation(v_fluct, normalize=True)
            rx = dx * np.arange(nx // 2 + 1)
            ry = dy * np.arange(ny // 2 + 1)
            correlation_samples["Ruu_x"].append(np.interp(r, rx, Ruu[0, : rx.size]))
            correlation_samples["Rvv_y"].append(np.interp(r, ry, Rvv[: ry.size, 0]))
            correlation_samples["Rvv_x"].append(np.interp(r, rx, Rvv[0, : rx.size]))
            correlation_samples["Ruu_y"].append(np.interp(r, ry, Ruu[: ry.size, 0]))

        stacks = {name: np.stack(values) for name, values in correlation_samples.items()}
        means = {name: np.mean(values, axis=0) for name, values in stacks.items()}
        stds = {name: np.std(values, axis=0) for name, values in stacks.items()}
        E_LL = float(np.sqrt(np.mean((means["Ruu_x"] - means["Rvv_y"]) ** 2)))
        E_NN = float(np.sqrt(np.mean((means["Rvv_x"] - means["Ruu_y"]) ** 2)))
        E_LL_normalized = self._normalized_mismatch(
            E_LL, means["Ruu_x"], means["Rvv_y"]
        )
        E_NN_normalized = self._normalized_mismatch(
            E_NN, means["Rvv_x"], means["Ruu_y"]
        )

        selected_times = time[selected_indices]
        selected_turnovers = turnover[selected_indices]
        results = IsotropyResults2D(
            time=time,
            turnover=turnover,
            selected_mask=selected_mask,
            K=K,
            Mt=Mt,
            Kx=Kx,
            Ky=Ky,
            A_K=A_K,
            C_uv=np.clip(C_uv, -1.0, 1.0),
            r=r,
            Ruu_x_mean=means["Ruu_x"],
            Rvv_y_mean=means["Rvv_y"],
            Rvv_x_mean=means["Rvv_x"],
            Ruu_y_mean=means["Ruu_y"],
            Ruu_x_std=stds["Ruu_x"],
            Rvv_y_std=stds["Rvv_y"],
            Rvv_x_std=stds["Rvv_x"],
            Ruu_y_std=stds["Ruu_y"],
            E_LL=E_LL,
            E_NN=E_NN,
            E_LL_normalized=E_LL_normalized,
            E_NN_normalized=E_NN_normalized,
            fluctuation_type=self.fluctuation_type,
            selected_time_interval=(float(selected_times[0]), float(selected_times[-1])),
            selected_turnover_interval=(
                float(selected_turnovers[0]),
                float(selected_turnovers[-1]),
            ),
            turnover_length=length_ref,
            number_of_snapshots=int(selected_indices.size),
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
        )
        self._verify(results)
        self.results = results
        return results

    def save(self, output_path: str | Path | None = None) -> Path:
        """Save all numerical isotropy diagnostics to a compressed NPZ file."""

        result = self._require_results()
        output_path = Path(output_path) if output_path is not None else self.run_dir / "isotropy_diagnostics.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            time=result.time,
            turnover=result.turnover,
            selected_mask=result.selected_mask,
            K=result.K,
            Mt=result.Mt,
            Kx=result.Kx,
            Ky=result.Ky,
            A_K=result.A_K,
            C_uv=result.C_uv,
            r=result.r,
            Ruu_x_mean=result.Ruu_x_mean,
            Rvv_y_mean=result.Rvv_y_mean,
            Rvv_x_mean=result.Rvv_x_mean,
            Ruu_y_mean=result.Ruu_y_mean,
            Ruu_x_std=result.Ruu_x_std,
            Rvv_y_std=result.Rvv_y_std,
            Rvv_x_std=result.Rvv_x_std,
            Ruu_y_std=result.Ruu_y_std,
            E_LL=np.asarray(result.E_LL),
            E_NN=np.asarray(result.E_NN),
            E_LL_normalized=np.asarray(result.E_LL_normalized),
            E_NN_normalized=np.asarray(result.E_NN_normalized),
            fluctuation_type=np.asarray(result.fluctuation_type),
            selected_time_interval=np.asarray(result.selected_time_interval),
            selected_turnover_interval=np.asarray(result.selected_turnover_interval),
            turnover_length=np.asarray(result.turnover_length),
            number_of_snapshots=np.asarray(result.number_of_snapshots),
            Nx=np.asarray(result.nx),
            Ny=np.asarray(result.ny),
            Lx=np.asarray(result.Lx),
            Ly=np.asarray(result.Ly),
        )
        return output_path

    def plot_stationarity(
        self,
        output_path: str | Path | None = None,
        x_axis: str = "turnover",
    ) -> Path:
        """Plot the complete K and Mt histories."""

        result = self._require_results()
        output_path = self._output_path(output_path, "stationarity_K_Mt.png")
        x_values, x_label = self._history_axis(result, x_axis)
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=True)

        panels = [
            (result.K, "Turbulent kinetic energy", "K"),
            (result.Mt, "Turbulent Mach number", "Mt"),
        ]
        for ax, (values, title, ylabel) in zip(axes, panels):
            ax.plot(x_values, values, linewidth=1.6)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel(x_label)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_directional_correlations(self, output_path: str | Path | None = None) -> Path:
        """Compare correlations that should match under a 90-degree rotation."""

        result = self._require_results()
        output_path = self._output_path(
            output_path, "directional_isotropy_correlations.png"
        )
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.0), sharex=True, constrained_layout=True)

        self._plot_correlation_pair(
            axes[0],
            result.r,
            result.Ruu_x_mean,
            result.Ruu_x_std,
            "Ruu_x = R_LL^x",
            result.Rvv_y_mean,
            result.Rvv_y_std,
            "Rvv_y = R_LL^y",
        )
        axes[0].set_title(f"Longitudinal rotational equivalents, E_LL={result.E_LL:.3e}")

        self._plot_correlation_pair(
            axes[1],
            result.r,
            result.Rvv_x_mean,
            result.Rvv_x_std,
            "Rvv_x = R_NN^x",
            result.Ruu_y_mean,
            result.Ruu_y_std,
            "Ruu_y = R_NN^y",
        )
        axes[1].set_title(f"Transverse rotational equivalents, E_NN={result.E_NN:.3e}")
        axes[1].set_xlabel("r")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_component_anisotropy(
        self,
        output_path: str | Path | None = None,
        x_axis: str = "turnover",
    ) -> Path:
        """Plot component energies, anisotropy index, and cross covariance."""

        result = self._require_results()
        output_path = self._output_path(output_path, "component_energy_anisotropy.png")
        x_values, x_label = self._history_axis(result, x_axis)
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=True)

        axes[0].plot(x_values, result.Kx, label="Kx", linewidth=1.5)
        axes[0].plot(x_values, result.Ky, label="Ky", linewidth=1.5)
        axes[0].set_ylabel("component energy")
        axes[0].set_title("Velocity-component turbulent energy")
        axes[0].legend()

        axes[1].plot(x_values, result.A_K, label="A_K", linewidth=1.5)
        axes[1].plot(x_values, result.C_uv, label="C_uv", linewidth=1.3)
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("anisotropy measure")
        axes[1].set_title("Component-energy anisotropy and covariance")
        axes[1].legend()

        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_all(self, filename_suffix: str | None = None, x_axis: str = "turnover") -> list[Path]:
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        return [
            self.plot_stationarity(
                self.output_dir / f"stationarity_K_Mt{suffix}.png",
                x_axis=x_axis,
            ),
            self.plot_directional_correlations(
                self.output_dir / f"directional_isotropy_correlations{suffix}.png"
            ),
            self.plot_component_anisotropy(
                self.output_dir / f"component_energy_anisotropy{suffix}.png",
                x_axis=x_axis,
            ),
        ]

    def _selection_mask(self, time: np.ndarray, turnover: np.ndarray) -> np.ndarray:
        mask = np.ones(time.size, dtype=bool)
        interval_supplied = any(
            value is not None
            for value in (
                self.start_time,
                self.end_time,
                self.start_turnover,
                self.end_turnover,
                self.start_snapshot,
                self.end_snapshot,
            )
        )
        if not interval_supplied:
            warnings.warn(
                "No stationary interval supplied; processing all snapshots, including the transient",
                RuntimeWarning,
            )
        if self.start_time is not None:
            mask &= time >= self.start_time
        if self.end_time is not None:
            mask &= time <= self.end_time
        if self.start_turnover is not None:
            mask &= turnover >= self.start_turnover
        if self.end_turnover is not None:
            mask &= turnover <= self.end_turnover
        index_mask = np.zeros(time.size, dtype=bool)
        start = 0 if self.start_snapshot is None else self.start_snapshot
        stop = time.size if self.end_snapshot is None else self.end_snapshot
        index_mask[slice(start, stop)] = True
        if self.start_snapshot is not None or self.end_snapshot is not None:
            mask &= index_mask
        return mask

    @staticmethod
    def _history_axis(result: IsotropyResults2D, x_axis: str) -> tuple[np.ndarray, str]:
        axis = x_axis.lower()
        if axis in {"turnover", "eddy", "eddy-turnover"}:
            return result.turnover, r"$N_{eddy}=\int u_{rms}/L_{ref}\,dt$"
        if axis in {"time", "t"}:
            return result.time, "t"
        raise ValueError("x_axis must be 'turnover' or 'time'")

    def _velocity_fluctuations(
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
    def _load_snapshot(path: Path) -> dict[str, np.ndarray | float]:
        with np.load(path) as data:
            required = ("time", "x", "y", "rho", "u", "v", "pressure")
            missing = [name for name in required if name not in data]
            if missing:
                raise KeyError(f"{path} is missing {missing}")
            return {
                "time": float(data["time"]),
                "x": np.asarray(data["x"], dtype=float),
                "y": np.asarray(data["y"], dtype=float),
                "rho": np.asarray(data["rho"], dtype=float),
                "u": np.asarray(data["u"], dtype=float),
                "v": np.asarray(data["v"], dtype=float),
                "pressure": np.asarray(data["pressure"], dtype=float),
            }

    @staticmethod
    def _spacing(values: np.ndarray) -> float:
        return float(values[1] - values[0]) if values.size > 1 else 1.0

    @staticmethod
    def _validate_snapshot_shapes(
        snapshots: list[dict[str, np.ndarray | float]], ny: int, nx: int
    ) -> None:
        expected = (ny, nx)
        for snapshot in snapshots:
            for name in ("rho", "u", "v", "pressure"):
                if np.asarray(snapshot[name]).shape != expected:
                    raise ValueError(f"All {name} fields must have shape {expected}")

    @staticmethod
    def _common_separations(
        nx: int, ny: int, dx: float, dy: float, Lx: float, Ly: float
    ) -> np.ndarray:
        maximum = 0.5 * min(Lx, Ly)
        spacing = max(dx, dy)
        return spacing * np.arange(int(np.floor(maximum / spacing)) + 1)

    @staticmethod
    def _normalized_mismatch(error: float, first: np.ndarray, second: np.ndarray) -> float:
        scale = np.sqrt(0.5 * np.mean(first**2 + second**2))
        return error / scale if scale > np.finfo(float).eps else 0.0

    @staticmethod
    def _plot_correlation_pair(
        ax: plt.Axes,
        r: np.ndarray,
        first: np.ndarray,
        first_std: np.ndarray,
        first_label: str,
        second: np.ndarray,
        second_std: np.ndarray,
        second_label: str,
    ) -> None:
        ax.plot(r, first, label=first_label, linewidth=1.7)
        ax.fill_between(r, first - first_std, first + first_std, alpha=0.16)
        ax.plot(r, second, label=second_label, linewidth=1.7)
        ax.fill_between(r, second - second_std, second + second_std, alpha=0.16)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("normalized correlation")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _output_path(self, output_path: str | Path | None, filename: str) -> Path:
        path = Path(output_path) if output_path is not None else self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _require_results(self) -> IsotropyResults2D:
        if self.results is None:
            raise RuntimeError("Call compute() before saving or plotting")
        return self.results

    @staticmethod
    def _verify(result: IsotropyResults2D) -> None:
        arrays = (
            result.K,
            result.Mt,
            result.Kx,
            result.Ky,
            result.A_K,
            result.C_uv,
            result.r,
            result.Ruu_x_mean,
            result.Rvv_y_mean,
            result.Rvv_x_mean,
            result.Ruu_y_mean,
        )
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("Isotropy diagnostics contain NaN or infinite values")
        if not np.allclose(result.K, result.Kx + result.Ky, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("K is inconsistent with Kx + Ky")
        if np.any(result.A_K < -1.0e-14):
            raise ValueError("A_K must be nonnegative")
        if np.any(np.abs(result.C_uv) > 1.0 + 1.0e-12):
            raise ValueError("|C_uv| exceeds one")
        correlation_arrays = (
            result.Ruu_x_mean,
            result.Rvv_y_mean,
            result.Rvv_x_mean,
            result.Ruu_y_mean,
        )
        if any(values.shape != result.r.shape for values in correlation_arrays):
            raise ValueError("Directional correlation arrays have inconsistent shapes")
        if not all(np.isclose(values[0], 1.0, atol=1.0e-12) for values in correlation_arrays):
            raise ValueError("Normalized autocorrelations must equal one at r=0")
        if result.r[-1] > 0.5 * min(result.Lx, result.Ly) + 1.0e-12:
            raise ValueError("Correlation separations exceed half the domain")
