from __future__ import annotations

from pathlib import Path

import numpy as np


def infer_turnover_length(run_dir: str | Path, fallback_length: float = 0.5 * np.pi) -> float:
    """Infer the large-eddy turnover length from run metadata.

    For the HIT2D setup, the natural reference length is the wavelength
    associated with the center of the forced shell:

        L_ref = 2*pi / k_ref,   k_ref = 0.5 * (k_min + k_max).

    If metadata are missing, use the shell 3 <= |k| <= 5 fallback, for which
    L_ref = 2*pi/4 = pi/2.
    """

    history_path = Path(run_dir) / "diagnostic_history.npz"
    if not history_path.exists():
        return fallback_length

    with np.load(history_path) as history:
        if "forcing_kmin" not in history.files or "forcing_kmax" not in history.files:
            return fallback_length
        k_min = float(np.asarray(history["forcing_kmin"]))
        k_max = float(np.asarray(history["forcing_kmax"]))

    if not np.isfinite(k_min) or not np.isfinite(k_max) or k_max <= 0.0:
        return fallback_length
    if k_min <= 0.0:
        k_ref = k_max
    else:
        k_ref = 0.5 * (k_min + k_max)
    return float(2.0 * np.pi / k_ref)


def cumulative_turnover(time: np.ndarray, u_rms: np.ndarray, length_ref: float) -> np.ndarray:
    """Compute cumulative eddy-turnover time N(t)=integral u_rms/L_ref dt."""

    time = np.asarray(time, dtype=float)
    u_rms = np.asarray(u_rms, dtype=float)
    if time.ndim != 1 or u_rms.ndim != 1 or time.size != u_rms.size:
        raise ValueError("time and u_rms must be one-dimensional arrays with matching size")
    if time.size == 0:
        return np.asarray([], dtype=float)
    if length_ref <= 0.0:
        raise ValueError("turnover reference length must be positive")

    rate = np.maximum(u_rms, 0.0) / length_ref
    turnover = np.zeros_like(time, dtype=float)
    if time.size > 1:
        dt = np.diff(time)
        turnover[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)
    return turnover


def turnover_from_history(
    run_dir: str | Path,
    snapshot_times: np.ndarray | None = None,
    length_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load diagnostic history and return time, cumulative turnover, length."""

    run_dir = Path(run_dir)
    history_path = run_dir / "diagnostic_history.npz"
    if not history_path.exists():
        raise FileNotFoundError(f"No diagnostic history found at {history_path}")

    with np.load(history_path) as history:
        time = np.asarray(history["time"], dtype=float)
        if "rms_velocity" in history.files:
            u_rms = np.asarray(history["rms_velocity"], dtype=float)
        elif "kinetic_energy" in history.files:
            u_rms = np.sqrt(np.maximum(2.0 * np.asarray(history["kinetic_energy"], dtype=float), 0.0))
        else:
            raise KeyError("diagnostic history must contain rms_velocity or kinetic_energy")

    length = infer_turnover_length(run_dir) if length_ref is None else float(length_ref)
    turnover = cumulative_turnover(time, u_rms, length)
    if snapshot_times is not None:
        snapshot_times = np.asarray(snapshot_times, dtype=float)
        turnover = np.interp(snapshot_times, time, turnover)
        time = snapshot_times
    return time, turnover, length
