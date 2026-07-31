#!/usr/bin/env python3
"""Patch HIT_Solver to stabilize dilatationally forced HIT.

The patch is deliberately source-based and idempotent. It backs up modified
files, adds the pure dilatational forcing option when it is absent, and replaces
both nested power controllers with bounded physical-time feedback.
"""
from __future__ import annotations

import argparse
import py_compile
import re
import shutil
from pathlib import Path

BACKUP_SUFFIX = ".before_stable_dilatational"


def backup(path: Path) -> None:
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label}; found {count}")
    return text.replace(old, new, 1)


def add_dilatational_forcing_support(text: str) -> str:
    if 'mode: str = "solenoidal"' not in text:
        text = replace_once(
            text,
            "    k_min: float\n    k_max: float\n",
            '    k_min: float\n    k_max: float\n    mode: str = "solenoidal"\n',
            "forcing mode dataclass insertion point",
        )

    if "normalized_mode = self.mode.strip().lower()" not in text:
        marker = "    def __post_init__(self) -> None:\n"
        addition = '''    def __post_init__(self) -> None:\n        normalized_mode = self.mode.strip().lower()\n        if normalized_mode == "compressive":\n            normalized_mode = "dilatational"\n        if normalized_mode not in {"solenoidal", "dilatational"}:\n            raise ValueError(\n                "forcing mode must be 'solenoidal' or 'dilatational' "\n                "('compressive' is accepted as an alias)"\n            )\n        self.mode = normalized_mode\n\n'''
        text = replace_once(text, marker, addition, "forcing __post_init__")

    if "def _project_potential" not in text:
        marker = "    def update(self, dt: float, rho, u, v)"
        project_method = '''    def _project_potential(self) -> tuple[object, object]:\n        """Return the selected pure Helmholtz component in Fourier space."""\n\n        if self.mode == "solenoidal":\n            return (\n                1j * self._ky * self._potential_hat,\n                -1j * self._kx * self._potential_hat,\n            )\n        return (\n            1j * self._kx * self._potential_hat,\n            1j * self._ky * self._potential_hat,\n        )\n\n'''
        index = text.find(marker)
        if index < 0:
            raise RuntimeError("Could not find forcing update method")
        text = text[:index] + project_method + text[index:]

    old_projection = (
        "        fx_hat = 1j * self._ky * self._potential_hat\n"
        "        fy_hat = -1j * self._kx * self._potential_hat\n"
    )
    if old_projection in text:
        text = text.replace(
            old_projection,
            "        fx_hat, fy_hat = self._project_potential()\n",
            1,
        )

    return text


def stabilize_inner_power_control(text: str) -> str:
    if "_alpha_initialized" not in text:
        text = replace_once(
            text,
            "    _alpha: float = field(default=1.0, init=False, repr=False)\n",
            "    _alpha: float = field(default=1.0, init=False, repr=False)\n"
            "    _alpha_initialized: bool = field(default=False, init=False, repr=False)\n",
            "alpha state field",
        )

    pattern = re.compile(
        r"        power_before = float\(self\.xp\.mean\(rho \* \(fx \* u \+ fy \* v\)\)\)\n"
        r"        alpha_target = self\._alpha\n"
        r"        if self\.target_power is not None and abs\(power_before\) > self\.min_power:\n"
        r".*?"
        r"        fx \*= self\._alpha\n"
        r"        fy \*= self\._alpha\n",
        re.DOTALL,
    )

    replacement = '''        power_before_signed = float(self.xp.mean(rho * (fx * u + fy * v)))
        alpha_target = self._alpha

        if self.mode == "dilatational":
            # A curl-free realization may have negative or nearly zero
            # instantaneous velocity correlation. Orienting both force
            # components together preserves curl-free forcing while avoiding
            # sign changes in the amplitude controller.
            if self.target_power is not None and power_before_signed < 0.0:
                fx *= -1.0
                fy *= -1.0
                power_before = -power_before_signed
            else:
                power_before = power_before_signed

            if self.target_power is not None and power_before > self.min_power:
                alpha_target = self.target_power / power_before
                upper_alpha = (
                    float("inf") if self.max_rescale is None else self.max_rescale
                )
                alpha_target = float(np.clip(alpha_target, 0.0, upper_alpha))

                if not self._alpha_initialized:
                    self._alpha = alpha_target
                    self._alpha_initialized = True
                else:
                    # Relax in physical time.  A small fraction of the OU
                    # correlation time tracks the requested power without the
                    # per-step sign chatter of the original controller.
                    relaxation_time = max(
                        0.05 * self.correlation_time, 10.0 * dt
                    )
                    blend = 1.0 - float(np.exp(-dt / relaxation_time))
                    self._alpha += blend * (alpha_target - self._alpha)
                    self._alpha = max(self._alpha, 0.0)
            elif self.target_power is None:
                self._alpha = 1.0
                self._alpha_initialized = True
            elif not self._alpha_initialized:
                # Avoid a unit-amplitude startup impulse when the OU field is
                # almost orthogonal to the velocity.
                self._alpha = 0.0
        else:
            # Preserve the accepted solenoidal campaign behavior exactly.
            power_before = power_before_signed
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
'''

    text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        if "power_before_signed" not in text:
            raise RuntimeError("Could not locate the original inner power controller")

    # Add forcing-family diagnostics when the older source does not have them.
    if '"forcing_dilatational_fraction"' not in text:
        old = '            "A_F": forcing_anisotropy,\n            "ou_decay": decay,\n'
        new = (
            '            "A_F": forcing_anisotropy,\n'
            '            "forcing_solenoidal_fraction": float(self.mode == "solenoidal"),\n'
            '            "forcing_dilatational_fraction": float(self.mode == "dilatational"),\n'
            '            "ou_decay": decay,\n'
        )
        text = replace_once(text, old, new, "forcing diagnostics")

    return text


CONTROLLER_FUNCTION = '''def update_mach_controlled_power(\n    config: HIT2DConfig,\n    current_power: float | None,\n    rho: np.ndarray,\n    u: np.ndarray,\n    v: np.ndarray,\n    pressure: np.ndarray,\n    dt: float,\n    state: dict[str, float | None],\n) -> tuple[float | None, dict[str, float]]:\n    """Bounded physical-time feedback for stationary K and turbulent Mach.\n\n    Dilatational forcing has a delayed acoustic response. A multiplicative\n    controller applied every numerical time step therefore winds up and creates\n    the large power/energy oscillations seen in the original runs. This\n    controller filters K and Mt in physical time, uses a dead band, changes\n    power logarithmically, rate-limits that change, and applies anti-windup\n    bounds.\n    """\n\n    xp = array_module(rho)\n    target_mach = (\n        config.target_mach\n        if config.mach_control_target is None\n        else config.mach_control_target\n    )\n    current_mach = turbulent_mach_from_primitive(\n        rho, u, v, pressure, config.gamma\n    )\n    current_ke = 0.5 * float(xp.mean(rho * (u**2 + v**2)))\n\n    # The initialized state has rho=1, p=1/gamma and therefore c_ref=1.\n    # Holding this K target together with pressure cooling makes K and Mt\n    # mutually consistent instead of letting pressure drift hide a K error.\n    target_ke = 0.5 * float(xp.mean(rho)) * target_mach**2\n\n    desired_power = np.nan if current_power is None else float(current_power)\n    if not config.mach_control or current_power is None:\n        return current_power, {\n            "mach_control_mach": current_mach,\n            "mach_control_target": float(target_mach),\n            "mach_control_power_desired": desired_power,\n            "mach_control_filtered_mach": current_mach,\n            "mach_control_filtered_ke": current_ke,\n            "mach_control_target_ke": target_ke,\n        }\n\n    if target_mach <= 0.0 or target_ke <= 0.0:\n        raise ValueError("Mach/energy control requires positive targets")\n    if dt <= 0.0:\n        raise ValueError("Mach/energy control requires dt > 0")\n\n    # Filter over at least one forcing-correlation time.\n    filter_time = max(config.forcing_correlation_time, 0.25)\n    filter_decay = float(np.exp(-dt / filter_time))\n    filtered_mach = state.get("filtered_mach")\n    filtered_ke = state.get("filtered_ke")\n    if filtered_mach is None or filtered_ke is None:\n        filtered_mach = current_mach\n        filtered_ke = current_ke\n    else:\n        filtered_mach = filter_decay * filtered_mach + (1.0 - filter_decay) * current_mach\n        filtered_ke = filter_decay * filtered_ke + (1.0 - filter_decay) * current_ke\n    state["filtered_mach"] = float(filtered_mach)\n    state["filtered_ke"] = float(filtered_ke)\n\n    tiny = 1.0e-14\n    mach_error = float(np.log(target_mach / max(filtered_mach, tiny)))\n    ke_error = float(np.log(target_ke / max(filtered_ke, tiny)))\n\n    # Both quantities matter. Pressure relaxation makes the two targets\n    # compatible; equal weighting prevents either one from dominating.\n    combined_error = 0.5 * mach_error + 0.5 * ke_error\n    deadband = 0.02\n    if abs(mach_error) < deadband and abs(ke_error) < deadband:\n        combined_error = 0.0\n\n    # The outer loop acts over several acoustic/forcing times, not every step.\n    control_time = max(8.0 * config.forcing_correlation_time, 4.0)\n    max_log_rate = 0.35\n    requested_log_rate = combined_error / control_time\n    log_rate = float(np.clip(requested_log_rate, -max_log_rate, max_log_rate))\n    desired_power = float(current_power * np.exp(log_rate * dt))\n\n    # Anti-windup. Explicit CLI bounds take precedence; otherwise keep the\n    # adaptive power within two decades around the campaign's initial value.\n    reference_power = max(float(config.target_energy_injection or current_power), tiny)\n    minimum_power = (\n        0.1 * reference_power\n        if config.mach_control_min_power is None\n        else config.mach_control_min_power\n    )\n    maximum_power = (\n        20.0 * reference_power\n        if config.mach_control_max_power is None\n        else config.mach_control_max_power\n    )\n    if maximum_power <= minimum_power:\n        raise ValueError("mach-control maximum power must exceed minimum power")\n    next_power = float(np.clip(desired_power, minimum_power, maximum_power))\n\n    return next_power, {\n        "mach_control_mach": current_mach,\n        "mach_control_target": float(target_mach),\n        "mach_control_power_desired": desired_power,\n        "mach_control_filtered_mach": float(filtered_mach),\n        "mach_control_filtered_ke": float(filtered_ke),\n        "mach_control_target_ke": target_ke,\n    }\n\n\n'''


def add_hit_forcing_mode_support(text: str) -> str:
    if 'forcing_mode: str = "solenoidal"' not in text:
        text = replace_once(
            text,
            "    target_mach: float = 0.1\n",
            '    target_mach: float = 0.1\n    forcing_mode: str = "solenoidal"\n',
            "HIT forcing mode config",
        )

    constructor_window = text[text.find("forcing = IsotropicShellOUForcing2D("):]
    constructor_window = constructor_window[:constructor_window.find(")\n", 0) + 2]
    if "mode=config.forcing_mode" not in constructor_window:
        text = replace_once(
            text,
            "        k_max=config.forcing_kmax,\n        correlation_time=config.forcing_correlation_time,\n",
            "        k_max=config.forcing_kmax,\n        mode=config.forcing_mode,\n"
            "        correlation_time=config.forcing_correlation_time,\n",
            "forcing constructor mode",
        )

    if 'parser.add_argument("--forcing-mode"' not in text:
        marker = '    parser.add_argument("--force-rms", type=float, default=1.0)\n'
        addition = marker + '''    parser.add_argument(\n        "--forcing-mode",\n        choices=("solenoidal", "dilatational"),\n        default="solenoidal",\n        help="Helmholtz component receiving the large-scale forcing.",\n    )\n'''
        text = replace_once(text, marker, addition, "forcing-mode CLI")

    # Add the CLI value to the main HIT2DConfig construction. Do not assume
    # which keyword follows target_mach because repository revisions may
    # reorder the configuration arguments.
    if "forcing_mode=args.forcing_mode" not in text:
        pattern = re.compile(
            r"^(?P<indent>\s*)target_mach\s*=\s*args\.mach\s*,\s*$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise RuntimeError(
                "Expected one target_mach=args.mach entry in the main "
                f"HIT2DConfig construction; found {len(matches)}"
            )
        match = matches[0]
        indent = match.group("indent")
        insertion = (
            match.group(0)
            + "\n"
            + indent
            + "forcing_mode=args.forcing_mode,"
        )
        text = text[:match.start()] + insertion + text[match.end():]
    return text


def stabilize_outer_control(text: str) -> str:
    pattern = re.compile(
        r"def update_mach_controlled_power\(.*?\n(?=def _spectral_band_filter\()",
        re.DOTALL,
    )
    text, count = pattern.subn(CONTROLLER_FUNCTION, text, count=1)
    if count != 1:
        if "Bounded physical-time feedback" not in text:
            raise RuntimeError("Could not replace the outer Mach controller")

    if 'mach_control_state: dict[str, float | None]' not in text:
        text = replace_once(
            text,
            "    adaptive_target_power = config.target_energy_injection\n",
            "    adaptive_target_power = config.target_energy_injection\n"
            "    mach_control_state: dict[str, float | None] = {\n"
            '        "filtered_mach": None,\n'
            '        "filtered_ke": None,\n'
            "    }\n",
            "Mach control state initialization",
        )

    old_call = '''        adaptive_target_power, mach_control_info = update_mach_controlled_power(\n            config,\n            adaptive_target_power,\n            rho,\n            u,\n            v,\n            pressure,\n        )\n'''
    new_call = '''        adaptive_target_power, mach_control_info = update_mach_controlled_power(\n            config,\n            adaptive_target_power,\n            rho,\n            u,\n            v,\n            pressure,\n            dt,\n            mach_control_state,\n        )\n'''
    if old_call in text:
        text = text.replace(old_call, new_call, 1)
    elif "mach_control_state," not in text[text.find("update_mach_controlled_power(", text.find("while time")):]:
        raise RuntimeError("Could not update the outer-controller call")
    return text



def add_controller_diagnostics(text: str) -> str:
    start = text.find("def compute_diagnostics(")
    end = text.find("def save_snapshot(", start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not locate compute_diagnostics")
    section = text[start:end]

    if '"mach_control_filtered_ke"' not in section:
        old = '''            "mach_control_power_desired": 0.0,
'''
        new = '''            "mach_control_power_desired": 0.0,
            "mach_control_filtered_mach": diagnostics["turbulent_mach"],
            "mach_control_filtered_ke": diagnostics["kinetic_energy"],
            "mach_control_target_ke": diagnostics["kinetic_energy"],
'''
        text = replace_once(text, old, new, "default controller diagnostics")

        old = '''                "mach_control_power_desired": forcing_info.get(
                    "mach_control_power_desired", 0.0
                ),
'''
        new = '''                "mach_control_power_desired": forcing_info.get(
                    "mach_control_power_desired", 0.0
                ),
                "mach_control_filtered_mach": forcing_info.get(
                    "mach_control_filtered_mach", diagnostics["turbulent_mach"]
                ),
                "mach_control_filtered_ke": forcing_info.get(
                    "mach_control_filtered_ke", diagnostics["kinetic_energy"]
                ),
                "mach_control_target_ke": forcing_info.get(
                    "mach_control_target_ke", diagnostics["kinetic_energy"]
                ),
'''
        text = replace_once(text, old, new, "forcing controller diagnostics")
    return text

def patch_repository(repo_root: Path) -> None:
    forcing_path = repo_root / "OOP" / "forcing.py"
    hit_path = repo_root / "OOP" / "hit2d.py"
    for path in (forcing_path, hit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        backup(path)

    forcing_text = forcing_path.read_text(encoding="utf-8")
    forcing_text = add_dilatational_forcing_support(forcing_text)
    forcing_text = stabilize_inner_power_control(forcing_text)
    forcing_path.write_text(forcing_text, encoding="utf-8")

    hit_text = hit_path.read_text(encoding="utf-8")
    hit_text = add_hit_forcing_mode_support(hit_text)
    hit_text = stabilize_outer_control(hit_text)
    hit_text = add_controller_diagnostics(hit_text)
    hit_path.write_text(hit_text, encoding="utf-8")

    py_compile.compile(str(forcing_path), doraise=True)
    py_compile.compile(str(hit_path), doraise=True)
    print(f"Patched and syntax-checked: {forcing_path}")
    print(f"Patched and syntax-checked: {hit_path}")
    print(f"Backups use suffix: {BACKUP_SUFFIX}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "repo_root",
        nargs="?",
        type=Path,
        default=Path.home() / "github" / "HIT_Solver",
    )
    args = parser.parse_args()
    patch_repository(args.repo_root.expanduser().resolve())


if __name__ == "__main__":
    main()
