from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CorrelationResults2D:
    """Container for averaged two-point correlation results."""

    r: np.ndarray
    rx: np.ndarray
    ry: np.ndarray
    Ruu_2d: np.ndarray
    Rvv_2d: np.ndarray
    Ruv_2d: np.ndarray
    Rrhop_2d: np.ndarray | None
    Rp_theta_2d: np.ndarray | None
    Rrho_2d: np.ndarray | None
    Rp_2d: np.ndarray | None
    Romega_2d: np.ndarray | None
    Rtheta_2d: np.ndarray | None
    Ruu_x: np.ndarray
    Ruu_y: np.ndarray
    Rvv_x: np.ndarray
    Rvv_y: np.ndarray
    R_LL: np.ndarray
    R_NN: np.ndarray
    R_LL_std: np.ndarray
    R_NN_std: np.ndarray
    L_integral: float
    lambda_taylor: float
    fluctuation_type: str
    number_of_snapshots: int
    metadata: dict[str, float | int | str] = field(default_factory=dict)


class TwoPointCorrelation2D:
    """Compute periodic two-point correlations from saved 2D HIT snapshots.

    The two-point correlation R_aa(rx, ry) = <a(x, y) a(x + rx, y + ry)>
    measures the spatial memory of a turbulent field. In a periodic HIT box the
    FFT formula is natural because every separation wraps around the domain:

        R = ifft2(fft2(a) * conj(fft2(a))).real / (Nx * Ny)

    Reynolds fluctuations subtract the spatial mean velocity. Favre
    fluctuations subtract the density-weighted mean velocity, which is often
    more meaningful once density fluctuations become dynamically important.
    """

    def __init__(
        self,
        run_dir: str | Path | None = None,
        snapshot_paths: list[str | Path] | None = None,
        fluctuation_type: str = "reynolds",
        Lx: float | None = None,
        Ly: float | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.snapshot_paths = [Path(path) for path in snapshot_paths] if snapshot_paths else []
        self.fluctuation_type = fluctuation_type.lower()
        if self.fluctuation_type not in {"reynolds", "favre"}:
            raise ValueError("fluctuation_type must be 'reynolds' or 'favre'")
        self.Lx = Lx
        self.Ly = Ly
        self.output_dir = Path(output_dir) if output_dir is not None else self.run_dir
        self.results: CorrelationResults2D | None = None

    def load_snapshots(
        self,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
        max_snapshots: int | None = None,
    ) -> list[Path]:
        """Select snapshots from a HIT2D execution folder.

        start, stop, and stride are applied to the sorted snapshot list by
        index, not by solver step number. This makes it easy to select a
        statistically stationary interval after inspecting the saved run.
        """

        if self.run_dir is None:
            if not self.snapshot_paths:
                raise ValueError("Provide run_dir or snapshot_paths before loading")
            selected = self.snapshot_paths
        else:
            selected = sorted(self.run_dir.glob("hit2d_step*.npz"))
            if not selected:
                raise FileNotFoundError(f"No HIT2D .npz snapshots found in {self.run_dir}")

        selected = selected[slice(start, stop, max(stride, 1))]
        if max_snapshots is not None:
            selected = selected[:max_snapshots]
        if not selected:
            raise ValueError("Snapshot selection is empty")
        self.snapshot_paths = selected
        return selected

    @staticmethod
    def autocorrelation(field: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Return periodic autocorrelation of one 2D field using FFTs."""

        values = np.asarray(field, dtype=float)
        spectrum = np.fft.fft2(values)
        correlation = np.fft.ifft2(spectrum * np.conj(spectrum)).real / values.size
        if normalize:
            zero_lag = float(correlation[0, 0])
            if abs(zero_lag) > np.finfo(float).eps:
                correlation = correlation / zero_lag
            else:
                correlation = np.zeros_like(correlation)
                correlation[0, 0] = 1.0
        return correlation

    @staticmethod
    def cross_correlation(a: np.ndarray, b: np.ndarray, normalize: bool = False) -> np.ndarray:
        """Return periodic cross-correlation <a(x) b(x + r)> using FFTs."""

        a_values = np.asarray(a, dtype=float)
        b_values = np.asarray(b, dtype=float)
        if a_values.shape != b_values.shape:
            raise ValueError("cross-correlation fields must have the same shape")
        a_hat = np.fft.fft2(a_values)
        b_hat = np.fft.fft2(b_values)
        correlation = np.fft.ifft2(a_hat * np.conj(b_hat)).real / a_values.size
        if normalize:
            scale = np.sqrt(float(np.mean(a_values**2) * np.mean(b_values**2)))
            if scale > np.finfo(float).eps:
                correlation = correlation / scale
            else:
                correlation = np.zeros_like(correlation)
        return correlation

    def compute(self) -> CorrelationResults2D:
        """Compute and average two-point correlations over selected snapshots."""

        if not self.snapshot_paths:
            self.load_snapshots()

        per_snapshot: dict[str, list[np.ndarray]] = {
            "Ruu_2d": [],
            "Rvv_2d": [],
            "Ruv_2d": [],
            "Rrhop_2d": [],
            "Rp_theta_2d": [],
            "Rrho_2d": [],
            "Rp_2d": [],
            "Romega_2d": [],
            "Rtheta_2d": [],
            "R_LL": [],
            "R_NN": [],
        }

        metadata: dict[str, float | int | str] | None = None
        rx = ry = r = None
        Ruu_x = Ruu_y = Rvv_x = Rvv_y = None

        for snapshot_path in self.snapshot_paths:
            fields = self._load_snapshot_fields(snapshot_path)
            rho = fields["rho"]
            u = fields["u"]
            v = fields["v"]
            pressure = fields["pressure"]
            x = fields["x"]
            y = fields["y"]
            dx = self._spacing(x)
            dy = self._spacing(y)
            Lx = self.Lx if self.Lx is not None else dx * x.size
            Ly = self.Ly if self.Ly is not None else dy * y.size

            if metadata is None:
                metadata = {
                    "nx": int(x.size),
                    "ny": int(y.size),
                    "Lx": float(Lx),
                    "Ly": float(Ly),
                    "dx": float(dx),
                    "dy": float(dy),
                    "run_dir": str(self.run_dir) if self.run_dir is not None else "",
                }
                rx, ry = self._half_separations(x.size, y.size, float(dx), float(dy))

            u_fluct, v_fluct = self._velocity_fluctuations(rho, u, v)
            rho_fluct = rho - float(np.mean(rho))
            p_fluct = pressure - float(np.mean(pressure))
            omega = fields.get("vorticity")
            theta = fields.get("divergence")
            if omega is None or theta is None:
                theta, omega = self._divergence_and_vorticity(u, v, dx, dy)

            Ruu = self.autocorrelation(u_fluct, normalize=True)
            Rvv = self.autocorrelation(v_fluct, normalize=True)
            Ruv = self.cross_correlation(u_fluct, v_fluct, normalize=True)
            Rrho = self.autocorrelation(rho_fluct, normalize=True)
            Rp = self.autocorrelation(p_fluct, normalize=True)
            omega_fluct = omega - float(np.mean(omega))
            theta_fluct = theta - float(np.mean(theta))
            Romega = self.autocorrelation(omega_fluct, normalize=True)
            Rtheta = self.autocorrelation(theta_fluct, normalize=True)
            Rrhop = self.cross_correlation(rho_fluct, p_fluct, normalize=True)
            Rp_theta = self.cross_correlation(p_fluct, theta_fluct, normalize=True)

            curves = self._directional_curves(Ruu, Rvv, rx.size, ry.size)
            common = min(curves["Ruu_x"].size, curves["Rvv_y"].size)
            r_snapshot = 0.5 * (rx[:common] + ry[:common])
            R_LL = 0.5 * (curves["Ruu_x"][:common] + curves["Rvv_y"][:common])
            R_NN = 0.5 * (curves["Rvv_x"][:common] + curves["Ruu_y"][:common])
            R_LL = self._normalize_curve(R_LL)
            R_NN = self._normalize_curve(R_NN)

            per_snapshot["Ruu_2d"].append(Ruu)
            per_snapshot["Rvv_2d"].append(Rvv)
            per_snapshot["Ruv_2d"].append(Ruv)
            per_snapshot["Rrhop_2d"].append(Rrhop)
            per_snapshot["Rp_theta_2d"].append(Rp_theta)
            per_snapshot["Rrho_2d"].append(Rrho)
            per_snapshot["Rp_2d"].append(Rp)
            per_snapshot["Romega_2d"].append(Romega)
            per_snapshot["Rtheta_2d"].append(Rtheta)
            per_snapshot["R_LL"].append(R_LL)
            per_snapshot["R_NN"].append(R_NN)
            r = r_snapshot
            Ruu_x = curves["Ruu_x"]
            Ruu_y = curves["Ruu_y"]
            Rvv_x = curves["Rvv_x"]
            Rvv_y = curves["Rvv_y"]

        if metadata is None or rx is None or ry is None or r is None:
            raise RuntimeError("No snapshots were processed")

        R_LL_stack = np.stack(per_snapshot["R_LL"])
        R_NN_stack = np.stack(per_snapshot["R_NN"])
        R_LL_mean = np.mean(R_LL_stack, axis=0)
        R_NN_mean = np.mean(R_NN_stack, axis=0)
        R_LL_std = np.std(R_LL_stack, axis=0)
        R_NN_std = np.std(R_NN_stack, axis=0)
        L_integral = self.integral_length_scale(r, R_LL_mean)
        lambda_taylor = self.taylor_microscale(r, R_LL_mean)

        results = CorrelationResults2D(
            r=r,
            rx=rx,
            ry=ry,
            Ruu_2d=np.mean(np.stack(per_snapshot["Ruu_2d"]), axis=0),
            Rvv_2d=np.mean(np.stack(per_snapshot["Rvv_2d"]), axis=0),
            Ruv_2d=np.mean(np.stack(per_snapshot["Ruv_2d"]), axis=0),
            Rrhop_2d=np.mean(np.stack(per_snapshot["Rrhop_2d"]), axis=0),
            Rp_theta_2d=np.mean(np.stack(per_snapshot["Rp_theta_2d"]), axis=0),
            Rrho_2d=np.mean(np.stack(per_snapshot["Rrho_2d"]), axis=0),
            Rp_2d=np.mean(np.stack(per_snapshot["Rp_2d"]), axis=0),
            Romega_2d=np.mean(np.stack(per_snapshot["Romega_2d"]), axis=0),
            Rtheta_2d=np.mean(np.stack(per_snapshot["Rtheta_2d"]), axis=0),
            Ruu_x=np.mean(
                np.stack([self._directional_curves(item, item, rx.size, ry.size)["Ruu_x"] for item in per_snapshot["Ruu_2d"]]),
                axis=0,
            ),
            Ruu_y=np.mean(
                np.stack([self._directional_curves(item, item, rx.size, ry.size)["Ruu_y"] for item in per_snapshot["Ruu_2d"]]),
                axis=0,
            ),
            Rvv_x=np.mean(
                np.stack([self._directional_curves(item, item, rx.size, ry.size)["Rvv_x"] for item in per_snapshot["Rvv_2d"]]),
                axis=0,
            ),
            Rvv_y=np.mean(
                np.stack([self._directional_curves(item, item, rx.size, ry.size)["Rvv_y"] for item in per_snapshot["Rvv_2d"]]),
                axis=0,
            ),
            R_LL=R_LL_mean,
            R_NN=R_NN_mean,
            R_LL_std=R_LL_std,
            R_NN_std=R_NN_std,
            L_integral=L_integral,
            lambda_taylor=lambda_taylor,
            fluctuation_type=self.fluctuation_type,
            number_of_snapshots=len(self.snapshot_paths),
            metadata=metadata,
        )
        self.results = results
        return results

    def save(self, output_path: str | Path | None = None) -> Path:
        """Save the current results to a compressed .npz file."""

        if self.results is None:
            raise RuntimeError("Call compute() before save()")
        if output_path is None:
            if self.output_dir is None:
                output_path = Path("two_point_correlation_results.npz")
            else:
                output_path = self.output_dir / "two_point_correlation_results.npz"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = self.results
        save_data = {
            "r": result.r,
            "rx": result.rx,
            "ry": result.ry,
            "Ruu_2d": result.Ruu_2d,
            "Rvv_2d": result.Rvv_2d,
            "Ruv_2d": result.Ruv_2d,
            "Rrhop_2d": result.Rrhop_2d,
            "Rp_theta_2d": result.Rp_theta_2d,
            "Rrho_2d": result.Rrho_2d,
            "Rp_2d": result.Rp_2d,
            "Romega_2d": result.Romega_2d,
            "Rtheta_2d": result.Rtheta_2d,
            "Ruu_x": result.Ruu_x,
            "Ruu_y": result.Ruu_y,
            "Rvv_x": result.Rvv_x,
            "Rvv_y": result.Rvv_y,
            "R_LL": result.R_LL,
            "R_NN": result.R_NN,
            "R_LL_std": result.R_LL_std,
            "R_NN_std": result.R_NN_std,
            "L_integral": np.asarray(result.L_integral),
            "lambda_taylor": np.asarray(result.lambda_taylor),
            "fluctuation_type": np.asarray(result.fluctuation_type),
            "number_of_snapshots": np.asarray(result.number_of_snapshots),
        }
        for key, value in result.metadata.items():
            save_data[f"metadata_{key}"] = np.asarray(value)
        np.savez_compressed(output_path, **save_data)
        return output_path

    def plot(
        self,
        output_dir: str | Path | None = None,
        show: bool = False,
        filename_suffix: str | None = None,
    ) -> list[Path]:
        """Create standard correlation plots and save them as PNG files."""

        if self.results is None:
            raise RuntimeError("Call compute() before plot()")
        output_dir = Path(output_dir) if output_dir is not None else self.output_dir
        if output_dir is None:
            output_dir = Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self.results
        saved_paths: list[Path] = []
        suffix = f"_{filename_suffix}" if filename_suffix else ""

        saved_paths.append(self._plot_LL_NN(output_dir / f"two_point_RLL_RNN{suffix}.png", result))
        saved_paths.append(self._plot_directional(output_dir / f"two_point_directional_velocity{suffix}.png", result))
        saved_paths.append(self._plot_velocity_maps(output_dir / f"two_point_Ruu_Rvv_2d{suffix}.png", result))

        if show:
            plt.show(block=True)
        else:
            plt.close("all")
        return saved_paths

    @staticmethod
    def integral_length_scale(r: np.ndarray, R_LL: np.ndarray) -> float:
        """Integrate normalized R_LL to its first zero crossing."""

        if r.size < 2 or R_LL.size < 2:
            return float("nan")
        negative = np.flatnonzero(R_LL < 0.0)
        if negative.size == 0:
            warnings.warn("R_LL has no zero crossing; integrating to L/2", RuntimeWarning)
            return float(np.trapezoid(R_LL, r))
        first = int(negative[0])
        if first == 0:
            return 0.0

        r_segment = list(r[:first])
        R_segment = list(R_LL[:first])
        r0, r1 = r[first - 1], r[first]
        R0, R1 = R_LL[first - 1], R_LL[first]
        if R1 != R0:
            zero_r = r0 - R0 * (r1 - r0) / (R1 - R0)
            r_segment.append(float(zero_r))
            R_segment.append(0.0)
        return float(np.trapezoid(np.asarray(R_segment), np.asarray(r_segment)))

    @staticmethod
    def taylor_microscale(r: np.ndarray, R_LL: np.ndarray, max_points: int = 6) -> float:
        """Estimate Taylor microscale from R_LL(r) = 1 - a r^2 near r = 0."""

        if r.size < 4 or R_LL.size < 4:
            warnings.warn("Not enough points for Taylor microscale fit", RuntimeWarning)
            return float("nan")
        count = min(max_points, r.size - 1)
        x_fit = r[1 : count + 1] ** 2
        y_fit = 1.0 - R_LL[1 : count + 1]
        valid = np.isfinite(x_fit) & np.isfinite(y_fit) & (x_fit > 0.0)
        if np.count_nonzero(valid) < 3:
            warnings.warn("Invalid points for Taylor microscale fit", RuntimeWarning)
            return float("nan")
        slope = float(np.dot(x_fit[valid], y_fit[valid]) / np.dot(x_fit[valid], x_fit[valid]))
        if slope <= 0.0 or not np.isfinite(slope):
            warnings.warn("Taylor microscale fit has non-positive curvature", RuntimeWarning)
            return float("nan")
        return float(np.sqrt(1.0 / (2.0 * slope)))

    def _velocity_fluctuations(
        self,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.fluctuation_type == "favre":
            rho_mean = float(np.mean(rho))
            if rho_mean <= np.finfo(float).eps:
                raise ValueError("Cannot compute Favre fluctuation with zero mean density")
            u_mean = float(np.mean(rho * u) / rho_mean)
            v_mean = float(np.mean(rho * v) / rho_mean)
        else:
            u_mean = float(np.mean(u))
            v_mean = float(np.mean(v))
        return u - u_mean, v - v_mean

    @staticmethod
    def _load_snapshot_fields(snapshot_path: Path) -> dict[str, np.ndarray]:
        with np.load(snapshot_path) as data:
            required = ("x", "y", "rho", "u", "v", "pressure")
            missing = [name for name in required if name not in data]
            if missing:
                raise KeyError(f"{snapshot_path} is missing {missing}")
            fields = {name: np.asarray(data[name], dtype=float) for name in required}
            if "vorticity" in data:
                fields["vorticity"] = np.asarray(data["vorticity"], dtype=float)
            if "divergence" in data:
                fields["divergence"] = np.asarray(data["divergence"], dtype=float)
            if "time" in data:
                fields["time"] = np.asarray(data["time"])
            if "step" in data:
                fields["step"] = np.asarray(data["step"])
        return fields

    @staticmethod
    def _spacing(values: np.ndarray) -> float:
        return float(values[1] - values[0]) if values.size > 1 else 1.0

    @staticmethod
    def _half_separations(nx: int, ny: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
        return dx * np.arange(nx // 2 + 1), dy * np.arange(ny // 2 + 1)

    @staticmethod
    def _directional_curves(
        Ruu: np.ndarray,
        Rvv: np.ndarray,
        nx_half: int,
        ny_half: int,
    ) -> dict[str, np.ndarray]:
        # Saved HIT2D arrays use shape (ny, nx), so x-separations are columns.
        return {
            "Ruu_x": Ruu[0, :nx_half],
            "Ruu_y": Ruu[:ny_half, 0],
            "Rvv_x": Rvv[0, :nx_half],
            "Rvv_y": Rvv[:ny_half, 0],
        }

    @staticmethod
    def _normalize_curve(values: np.ndarray) -> np.ndarray:
        zero = float(values[0])
        if abs(zero) <= np.finfo(float).eps:
            return np.full_like(values, np.nan)
        return values / zero

    @staticmethod
    def _divergence_and_vorticity(
        u: np.ndarray,
        v: np.ndarray,
        dx: float,
        dy: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
        du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dy)
        dv_dx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dx)
        dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dy)
        divergence = du_dx + dv_dy
        vorticity = dv_dx - du_dy
        return divergence, vorticity

    @staticmethod
    def _plot_LL_NN(output_path: Path, result: CorrelationResults2D) -> Path:
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        ax.plot(result.r, result.R_LL, label="R_LL", linewidth=2.0)
        ax.fill_between(
            result.r,
            result.R_LL - result.R_LL_std,
            result.R_LL + result.R_LL_std,
            alpha=0.2,
        )
        ax.plot(result.r, result.R_NN, label="R_NN", linewidth=2.0)
        ax.fill_between(
            result.r,
            result.R_NN - result.R_NN_std,
            result.R_NN + result.R_NN_std,
            alpha=0.2,
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("r")
        ax.set_ylabel("normalized correlation")
        ax.set_title("Longitudinal and transverse correlations")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        return output_path

    @staticmethod
    def _plot_directional(output_path: Path, result: CorrelationResults2D) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
        axes[0].plot(result.rx, result.Ruu_x, label="Ruu_x")
        axes[0].plot(result.ry, result.Ruu_y, label="Ruu_y")
        axes[0].set_title("u-fluctuation directional correlations")
        axes[0].set_xlabel("r")
        axes[0].set_ylabel("Ruu")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(result.rx, result.Rvv_x, label="Rvv_x")
        axes[1].plot(result.ry, result.Rvv_y, label="Rvv_y")
        axes[1].set_title("v-fluctuation directional correlations")
        axes[1].set_xlabel("r")
        axes[1].set_ylabel("Rvv")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        return output_path

    @staticmethod
    def _plot_velocity_maps(output_path: Path, result: CorrelationResults2D) -> Path:
        Lx = float(result.metadata.get("Lx", result.Ruu_2d.shape[1]))
        Ly = float(result.metadata.get("Ly", result.Ruu_2d.shape[0]))
        extent = [0.0, Lx, 0.0, Ly]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)
        panels = [
            (result.Ruu_2d, "Ruu(rx, ry)"),
            (result.Rvv_2d, "Rvv(rx, ry)"),
        ]

        for ax, (values, title) in zip(axes, panels):
            image = ax.imshow(
                values,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
            )
            ax.set_title(title)
            ax.set_xlabel("rx")
            ax.set_ylabel("ry")
            fig.colorbar(image, ax=ax, label="normalized correlation")

        fig.suptitle("2D velocity autocorrelation maps")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        return output_path

    @staticmethod
    def _plot_map(
        output_path: Path,
        values: np.ndarray,
        title: str,
        result: CorrelationResults2D,
    ) -> Path:
        Lx = float(result.metadata.get("Lx", values.shape[1]))
        Ly = float(result.metadata.get("Ly", values.shape[0]))
        extent = [0.0, Lx, 0.0, Ly]
        fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("rx")
        ax.set_ylabel("ry")
        fig.colorbar(image, ax=ax, label="normalized correlation")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        return output_path
