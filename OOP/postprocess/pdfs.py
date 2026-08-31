from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


@dataclass
class PDFResults2D:
    bin_centers: dict[str, np.ndarray]
    pdfs: dict[str, np.ndarray]
    moments: dict[str, dict[str, float]]
    joint_x_edges: np.ndarray
    joint_y_edges: np.ndarray
    joint_pdf: np.ndarray
    number_of_snapshots: int
    selected_time_interval: tuple[float, float]


class HIT2DPDFDiagnostics:
    """One-point PDFs for stationary periodic HIT2D snapshots.

    The primary variables are the normalized dilatation ``theta/theta_rms`` and
    normalized out-of-plane vorticity ``omega/omega_rms``.  Pressure and density
    fluctuations are included because their PDF asymmetry is a standard
    compressible-turbulence diagnostic.  A joint dilatation-vorticity PDF is also
    produced to show how compressive and rotational events coexist.
    """

    def __init__(
        self,
        run_dir: str | Path,
        start_time: float | None = None,
        end_time: float | None = None,
        stride: int = 1,
        bins: int = 161,
        joint_bins: int = 121,
        robust_percentile: float = 99.95,
        output_dir: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.start_time = start_time
        self.end_time = end_time
        if stride <= 0:
            raise ValueError("stride must be positive")
        if bins < 21 or joint_bins < 21:
            raise ValueError("PDF bin counts must be at least 21")
        if not 90.0 < robust_percentile <= 100.0:
            raise ValueError("robust_percentile must be in (90, 100]")
        self.stride = stride
        self.bins = bins
        self.joint_bins = joint_bins
        self.robust_percentile = robust_percentile
        self.output_dir = Path(output_dir) if output_dir is not None else self.run_dir / "postprocess"
        self.results: PDFResults2D | None = None

    def compute(self) -> PDFResults2D:
        paths = sorted(self.run_dir.glob("hit2d_step*.npz"))
        if not paths:
            raise FileNotFoundError(f"No HIT2D snapshots found in {self.run_dir}")

        selected: list[Path] = []
        times: list[float] = []
        for path in paths:
            with np.load(path) as data:
                time = float(data["time"])
            if self.start_time is not None and time < self.start_time:
                continue
            if self.end_time is not None and time > self.end_time:
                continue
            selected.append(path)
            times.append(time)
        selected = selected[:: self.stride]
        times = times[:: self.stride]
        if not selected:
            raise ValueError("No snapshots are available in the selected PDF interval")
        if len(selected) < 5:
            warnings.warn("Few snapshots selected for PDF averaging", RuntimeWarning)

        pooled: dict[str, list[np.ndarray]] = {
            "dilatation": [],
            "vorticity": [],
            "pressure": [],
            "density": [],
        }
        for path in selected:
            with np.load(path) as data:
                divergence = np.asarray(data["divergence"], dtype=float)
                vorticity = np.asarray(data["vorticity"], dtype=float)
                pressure = np.asarray(data["pressure"], dtype=float)
                density = np.asarray(data["rho"], dtype=float)
            pooled["dilatation"].append(divergence.ravel())
            pooled["vorticity"].append(vorticity.ravel())
            pooled["pressure"].append((pressure - np.mean(pressure)).ravel())
            pooled["density"].append((density - np.mean(density)).ravel())

        standardized: dict[str, np.ndarray] = {}
        moments: dict[str, dict[str, float]] = {}
        bin_centers: dict[str, np.ndarray] = {}
        pdfs: dict[str, np.ndarray] = {}

        for name, chunks in pooled.items():
            values = np.concatenate(chunks)
            values -= float(np.mean(values))
            rms = float(np.sqrt(np.mean(values**2)))
            if rms <= np.finfo(float).tiny:
                normalized = np.zeros_like(values)
            else:
                normalized = values / rms
            standardized[name] = normalized
            moments[name] = self._moments(normalized, dimensional_rms=rms)
            edges = self._symmetric_edges(normalized, self.bins)
            histogram, edges = np.histogram(normalized, bins=edges, density=True)
            bin_centers[name] = 0.5 * (edges[:-1] + edges[1:])
            pdfs[name] = histogram

        theta = standardized["dilatation"]
        omega = standardized["vorticity"]
        theta_edges = self._symmetric_edges(theta, self.joint_bins)
        omega_edges = self._symmetric_edges(omega, self.joint_bins)
        joint_hist, theta_edges, omega_edges = np.histogram2d(
            theta,
            omega,
            bins=(theta_edges, omega_edges),
            density=True,
        )

        self.results = PDFResults2D(
            bin_centers=bin_centers,
            pdfs=pdfs,
            moments=moments,
            joint_x_edges=theta_edges,
            joint_y_edges=omega_edges,
            joint_pdf=joint_hist,
            number_of_snapshots=len(selected),
            selected_time_interval=(float(times[0]), float(times[-1])),
        )
        return self.results

    def save(self, output_path: str | Path | None = None) -> Path:
        result = self._require_results()
        path = Path(output_path) if output_path is not None else self.run_dir / "pdf_diagnostics.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "joint_theta_edges": result.joint_x_edges,
            "joint_omega_edges": result.joint_y_edges,
            "joint_theta_omega_pdf": result.joint_pdf,
            "number_of_snapshots": np.asarray(result.number_of_snapshots),
            "selected_time_interval": np.asarray(result.selected_time_interval),
            "normalization": np.asarray("zero mean and unit rms for each pooled variable"),
        }
        for name in result.pdfs:
            payload[f"{name}_bin_centers"] = result.bin_centers[name]
            payload[f"{name}_pdf"] = result.pdfs[name]
            for moment_name, value in result.moments[name].items():
                payload[f"{name}_{moment_name}"] = np.asarray(value)
        np.savez_compressed(path, **payload)
        return path

    def plot(self, output_path: str | Path | None = None) -> Path:
        result = self._require_results()
        path = Path(output_path) if output_path is not None else self.output_dir / "one_point_pdfs.png"
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.4), constrained_layout=True)
        definitions = [
            ("dilatation", r"$\theta/\theta_{rms}$", "Dilatation PDF"),
            ("vorticity", r"$\omega_z/\omega_{rms}$", "Vorticity PDF"),
            ("pressure", r"$p'/p_{rms}$", "Pressure-fluctuation PDF"),
            ("density", r"$\rho'/\rho_{rms}$", "Density-fluctuation PDF"),
        ]
        for ax, (name, xlabel, title) in zip(axes.flat, definitions):
            x = result.bin_centers[name]
            y = result.pdfs[name]
            visible = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
            ax.semilogy(x[visible], y[visible], linewidth=2.1, label="DNS samples")
            gaussian = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
            ax.semilogy(x, gaussian, linestyle="--", linewidth=1.7, label="Gaussian")
            ax.set_xlabel(xlabel, fontsize=17)
            ax.set_ylabel("p.d.f.", fontsize=17)
            ax.set_title(title, fontsize=19)
            ax.tick_params(labelsize=14)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=12, loc="best")

        fig.suptitle("One-point statistics of 2-D solenoidally forced compressible HIT", fontsize=20)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def _symmetric_edges(self, values: np.ndarray, bins: int) -> np.ndarray:
        if values.size == 0:
            return np.linspace(-1.0, 1.0, bins + 1)
        q = float(np.percentile(np.abs(values), self.robust_percentile))
        limit = max(q, 3.0)
        return np.linspace(-limit, limit, bins + 1)

    @staticmethod
    def _moments(values: np.ndarray, dimensional_rms: float) -> dict[str, float]:
        variance = float(np.mean(values**2))
        if variance <= np.finfo(float).tiny:
            return {
                "rms": float(dimensional_rms),
                "skewness": 0.0,
                "flatness": 0.0,
            }
        return {
            "rms": float(dimensional_rms),
            "skewness": float(np.mean(values**3) / variance**1.5),
            "flatness": float(np.mean(values**4) / variance**2),
        }

    def _require_results(self) -> PDFResults2D:
        if self.results is None:
            raise RuntimeError("Call compute() before save() or plot()")
        return self.results
