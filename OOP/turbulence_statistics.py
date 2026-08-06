from __future__ import annotations

"""Spectral diagnostics for periodic two-dimensional turbulence fields.

The solver is two-dimensional.  The Taylor-microscale and Kolmogorov-scale
quantities returned here are therefore two-dimensional analogues intended for
internal resolution monitoring.  They should not be compared numerically with
three-dimensional HIT databases without stating that distinction.
"""

import numpy as np

from OOP.parallel.backend import array_module


def spectral_wavenumbers_2d(nx: int, ny: int, dx: float, dy: float, xp=np):
    kx_1d = xp.asarray(2.0 * np.pi * np.fft.fftfreq(nx, d=dx))
    ky_1d = xp.asarray(2.0 * np.pi * np.fft.fftfreq(ny, d=dy))
    return xp.meshgrid(kx_1d, ky_1d, indexing="xy")


def spectral_velocity_gradients_2d(u, v, dx: float, dy: float):
    """Return du/dx, du/dy, dv/dx and dv/dy using periodic FFT derivatives."""

    xp = array_module(u)
    ny, nx = u.shape
    kx, ky = spectral_wavenumbers_2d(nx, ny, dx, dy, xp=xp)
    u_hat = xp.fft.fft2(u)
    v_hat = xp.fft.fft2(v)
    du_dx = xp.fft.ifft2(1j * kx * u_hat).real
    du_dy = xp.fft.ifft2(1j * ky * u_hat).real
    dv_dx = xp.fft.ifft2(1j * kx * v_hat).real
    dv_dy = xp.fft.ifft2(1j * ky * v_hat).real
    return du_dx, du_dy, dv_dx, dv_dy


def helmholtz_fourier_2d(u, v, dx: float, dy: float):
    """Helmholtz-decompose a periodic velocity field in Fourier space.

    Returns ``(us_hat, vs_hat, ud_hat, vd_hat)``.  The solenoidal component is
    perpendicular to each non-zero wavevector and the dilatational component is
    parallel to it.  The mean mode is assigned to neither fluctuation component.
    """

    xp = array_module(u)
    ny, nx = u.shape
    kx, ky = spectral_wavenumbers_2d(nx, ny, dx, dy, xp=xp)
    k2 = kx**2 + ky**2
    u_hat = xp.fft.fft2(u - xp.mean(u))
    v_hat = xp.fft.fft2(v - xp.mean(v))
    k_dot_u = kx * u_hat + ky * v_hat
    k2_safe = xp.where(k2 > 0.0, k2, 1.0)
    ud_hat = xp.where(k2 > 0.0, kx * k_dot_u / k2_safe, 0.0)
    vd_hat = xp.where(k2 > 0.0, ky * k_dot_u / k2_safe, 0.0)
    us_hat = u_hat - ud_hat
    vs_hat = v_hat - vd_hat
    return us_hat, vs_hat, ud_hat, vd_hat


def helmholtz_velocity_2d(u, v, dx: float, dy: float):
    """Return physical-space solenoidal and dilatational velocity components."""

    xp = array_module(u)
    us_hat, vs_hat, ud_hat, vd_hat = helmholtz_fourier_2d(u, v, dx, dy)
    us = xp.fft.ifft2(us_hat).real
    vs = xp.fft.ifft2(vs_hat).real
    ud = xp.fft.ifft2(ud_hat).real
    vd = xp.fft.ifft2(vd_hat).real
    return us, vs, ud, vd


def turbulence_scales_2d(rho, u, v, dynamic_viscosity: float, dx: float, dy: float):
    """Compute 2-D Taylor/Kolmogorov diagnostics and Helmholtz energy partition.

    Definitions
    -----------
    ``u_component_rms`` is the rms of one velocity component estimated by
    averaging the two in-plane components.  The longitudinal gradient variance is
    likewise averaged between ``du/dx`` and ``dv/dy``.  The resulting Taylor scale
    and Reynolds number are useful for comparisons *within this 2-D solver*.

    The physical dissipation rate is computed from the Newtonian stress with
    Stokes' hypothesis.  The Kolmogorov scale uses the mean kinematic viscosity and
    the physical viscous dissipation only; WENO and numerical hyperviscosity are not
    included in ``epsilon_physical``.
    """

    xp = array_module(u)
    rho_mean = float(xp.mean(rho))
    u_fluct = u - xp.mean(u)
    v_fluct = v - xp.mean(v)
    component_variance = 0.5 * xp.mean(u_fluct**2 + v_fluct**2)
    u_component_rms = float(xp.sqrt(xp.maximum(component_variance, 0.0)))

    du_dx, du_dy, dv_dx, dv_dy = spectral_velocity_gradients_2d(u, v, dx, dy)
    longitudinal_gradient_variance = 0.5 * xp.mean(du_dx**2 + dv_dy**2)
    longitudinal_gradient_rms = float(
        xp.sqrt(xp.maximum(longitudinal_gradient_variance, 0.0))
    )
    taylor_microscale = (
        u_component_rms / longitudinal_gradient_rms
        if longitudinal_gradient_rms > np.finfo(float).eps
        else float("nan")
    )

    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy

    if dynamic_viscosity > 0.0:
        mu = float(dynamic_viscosity)
        tau_xx = 2.0 * mu * du_dx - (2.0 / 3.0) * mu * divergence
        tau_yy = 2.0 * mu * dv_dy - (2.0 / 3.0) * mu * divergence
        tau_xy = mu * (du_dy + dv_dx)
        dissipation_per_volume = float(
            xp.mean(
                tau_xx * du_dx
                + tau_xy * du_dy
                + tau_xy * dv_dx
                + tau_yy * dv_dy
            )
        )
        epsilon_physical = dissipation_per_volume / max(rho_mean, np.finfo(float).tiny)
        nu_mean = mu / max(rho_mean, np.finfo(float).tiny)
        kolmogorov_length = (
            (nu_mean**3 / epsilon_physical) ** 0.25
            if epsilon_physical > np.finfo(float).tiny
            else float("inf")
        )

        # In two-dimensional turbulence, the forward small-scale cascade is
        # normally associated with enstrophy rather than kinetic energy.  The
        # following Kraichnan microscale is therefore reported alongside the
        # conventional Kolmogorov analogue.  In compressible 2-D flow it is an
        # indicative resolution measure, not an exact similarity result.
        omega_hat = xp.fft.fft2(vorticity)
        ny, nx = vorticity.shape
        kx, ky = spectral_wavenumbers_2d(nx, ny, dx, dy, xp=xp)
        domega_dx = xp.fft.ifft2(1j * kx * omega_hat).real
        domega_dy = xp.fft.ifft2(1j * ky * omega_hat).real
        enstrophy_dissipation = float(
            nu_mean * xp.mean(domega_dx**2 + domega_dy**2)
        )
        kraichnan_length = (
            (nu_mean**3 / enstrophy_dissipation) ** (1.0 / 6.0)
            if enstrophy_dissipation > np.finfo(float).tiny
            else float("inf")
        )
        re_lambda = (
            rho_mean * u_component_rms * taylor_microscale / mu
            if np.isfinite(taylor_microscale)
            else float("nan")
        )
    else:
        dissipation_per_volume = 0.0
        epsilon_physical = 0.0
        nu_mean = 0.0
        kolmogorov_length = float("inf")
        enstrophy_dissipation = 0.0
        kraichnan_length = float("inf")
        re_lambda = float("inf")

    us, vs, ud, vd = helmholtz_velocity_2d(u, v, dx, dy)
    ks = 0.5 * rho_mean * float(xp.mean(us**2 + vs**2))
    kd = 0.5 * rho_mean * float(xp.mean(ud**2 + vd**2))
    helmholtz_total = ks + kd
    chi_d = kd / helmholtz_total if helmholtz_total > np.finfo(float).tiny else 0.0
    delta_d = np.sqrt(kd / ks) if ks > np.finfo(float).tiny else float("inf")

    h = min(dx, dy)
    kmax_nominal = np.pi / h
    eta_over_dx = kolmogorov_length / h
    kmax_eta = kmax_nominal * kolmogorov_length
    kraichnan_over_dx = kraichnan_length / h
    kmax_kraichnan = kmax_nominal * kraichnan_length

    return {
        "u_component_rms": u_component_rms,
        "longitudinal_gradient_rms": longitudinal_gradient_rms,
        "taylor_microscale_2d": float(taylor_microscale),
        "re_lambda_2d": float(re_lambda),
        "dissipation_per_volume_physical": float(dissipation_per_volume),
        "epsilon_physical": float(epsilon_physical),
        "mean_kinematic_viscosity": float(nu_mean),
        "kolmogorov_length": float(kolmogorov_length),
        "eta_over_dx": float(eta_over_dx),
        "kmax_nominal": float(kmax_nominal),
        "kmax_eta": float(kmax_eta),
        "enstrophy_dissipation_2d": float(enstrophy_dissipation),
        "kraichnan_length_2d": float(kraichnan_length),
        "kraichnan_over_dx": float(kraichnan_over_dx),
        "kmax_kraichnan": float(kmax_kraichnan),
        "solenoidal_kinetic_energy": float(ks),
        "dilatational_kinetic_energy": float(kd),
        "helmholtz_kinetic_energy": float(helmholtz_total),
        "dilatational_energy_fraction": float(chi_d),
        "dilatational_to_solenoidal_rms": float(delta_d),
        "divergence_rms_spectral": float(xp.sqrt(xp.mean(divergence**2))),
        "vorticity_rms_spectral": float(xp.sqrt(xp.mean(vorticity**2))),
    }


def viscosity_for_target_initial_re_lambda_2d(
    rho,
    u,
    v,
    target_re_lambda: float,
    dx: float,
    dy: float,
) -> tuple[float, dict[str, float]]:
    """Return the constant dynamic viscosity giving the requested initial 2-D Rλ."""

    if target_re_lambda <= 0.0:
        raise ValueError("target_re_lambda must be positive")
    baseline = turbulence_scales_2d(rho, u, v, 1.0, dx, dy)
    xp = array_module(rho)
    numerator = (
        float(xp.mean(rho))
        * baseline["u_component_rms"]
        * baseline["taylor_microscale_2d"]
    )
    viscosity = numerator / target_re_lambda
    if not np.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("could not infer a positive viscosity from the initial field")
    resolved = turbulence_scales_2d(rho, u, v, viscosity, dx, dy)
    return float(viscosity), resolved
