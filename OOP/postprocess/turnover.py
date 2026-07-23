from __future__ import annotations

from pathlib import Path

import numpy as np


def _scalar_or_none(values: np.lib.npyio.NpzFile, name: str) -> float | None:
    if name not in values.files:
        return None
    value = float(np.asarray(values[name]))
    return value if np.isfinite(value) else None


def infer_turnover_length(run_dir: str | Path, fallback_length: float = 0.5 * np.pi) -> float:
    """Infer the large-eddy turnover length from run metadata.

    For the HIT2D setup, the natural reference length is the wavelength
    associated with the center of the forced shell,

        L_ref = 2*pi/k_ref,  k_ref = 0.5*(k_min + k_max).

    New runs save the exact reference length used online. Older runs are
    reconstructed from the forcing shell. If metadata are unavailable, the
    shell 3 <= |k| <= 5 fallback gives L_ref = pi/2.
    """

    history_path = Path(run_dir) / "diagnostic_history.npz"
    if not history_path.exists():
        return fallback_length

    with np.load(history_path) as history:
        saved_length = _scalar_or_none(history, "turnover_length")
        if saved_length is not None and saved_length > 0.0:
            return saved_length
        if "forcing_kmin" not in history.files or "forcing_kmax" not in history.files:
            return fallback_length
        k_min = float(np.asarray(history["forcing_kmin"]))
        k_max = float(np.asarray(history["forcing_kmax"]))

    if not np.isfinite(k_min) or not np.isfinite(k_max) or k_max <= 0.0:
        return fallback_length
    k_ref = k_max if k_min <= 0.0 else 0.5 * (k_min + k_max)
    return float(2.0 * np.pi / k_ref)


def load_turnover_window(run_dir: str | Path) -> tuple[float | None, float | None]:
    """Return the default post-processing turnover interval saved by a run.

    The values correspond to ``turnover_data_start`` and
    ``turnover_final_target`` in ``diagnostic_history.npz``. Older runs simply
    return ``(None, None)`` and require explicit post-processing bounds.
    """

    history_path = Path(run_dir) / "diagnostic_history.npz"
    if not history_path.exists():
        return None, None
    with np.load(history_path) as history:
        start = _scalar_or_none(history, "turnover_data_start")
        end = _scalar_or_none(history, "turnover_final_target")
    return start, end


def resolve_turnover_window(
    run_dir: str | Path,
    start_turnover: float | None = None,
    end_turnover: float | None = None,
) -> tuple[float | None, float | None]:
    """Resolve explicit bounds, falling back to the run's saved defaults."""

    saved_start, saved_end = load_turnover_window(run_dir)
    start = saved_start if start_turnover is None else float(start_turnover)
    end = saved_end if end_turnover is None else float(end_turnover)
    if start is not None and start < 0.0:
        raise ValueError("start turnover must be non-negative")
    if end is not None and end <= 0.0:
        raise ValueError("end turnover must be positive")
    if start is not None and end is not None and start >= end:
        raise ValueError("start turnover must be smaller than end turnover")
    return start, end


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
    if np.any(np.diff(time) < 0.0):
        raise ValueError("time values must be monotonically non-decreasing")

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
    """Load diagnostic history and return time, turnover and reference length.

    New runs integrate turnover online at every solver step and store it in the
    diagnostic history. For backward compatibility, older histories are
    reconstructed from ``rms_velocity`` using trapezoidal integration.
    """

    run_dir = Path(run_dir)
    history_path = run_dir / "diagnostic_history.npz"
    if not history_path.exists():
        raise FileNotFoundError(f"No diagnostic history found at {history_path}")

    with np.load(history_path) as history:
        time = np.asarray(history["time"], dtype=float)
        if "turnover" in history.files:
            turnover = np.asarray(history["turnover"], dtype=float)
        else:
            if "rms_velocity" in history.files:
                u_rms = np.asarray(history["rms_velocity"], dtype=float)
            elif "kinetic_energy" in history.files:
                u_rms = np.sqrt(
                    np.maximum(2.0 * np.asarray(history["kinetic_energy"], dtype=float), 0.0)
                )
            else:
                raise KeyError("diagnostic history must contain turnover, rms_velocity or kinetic_energy")
            length = infer_turnover_length(run_dir) if length_ref is None else float(length_ref)
            turnover = cumulative_turnover(time, u_rms, length)

    length = infer_turnover_length(run_dir) if length_ref is None else float(length_ref)
    if time.size != turnover.size:
        raise ValueError("diagnostic time and turnover arrays must have matching size")
    if snapshot_times is not None:
        snapshot_times = np.asarray(snapshot_times, dtype=float)
        turnover = np.interp(snapshot_times, time, turnover)
        time = snapshot_times
    return time, turnover, length


def turnover_for_snapshots(
    run_dir: str | Path,
    snapshot_paths: list[str | Path],
    length_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return time and turnover arrays aligned with a list of snapshots."""

    times: list[float] = []
    saved_turnovers: list[float] = []
    all_have_turnover = True
    for path in snapshot_paths:
        with np.load(path) as data:
            times.append(float(data["time"]))
            if "turnover" in data.files:
                saved_turnovers.append(float(data["turnover"]))
            else:
                all_have_turnover = False

    time = np.asarray(times, dtype=float)
    length = infer_turnover_length(run_dir) if length_ref is None else float(length_ref)
    if all_have_turnover:
        return time, np.asarray(saved_turnovers, dtype=float), length
    return turnover_from_history(run_dir, snapshot_times=time, length_ref=length)
