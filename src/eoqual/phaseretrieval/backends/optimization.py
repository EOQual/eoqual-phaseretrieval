#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrouvé de phase par optimisation directe (moindres carrés non-linéaires).

Contrairement à Gerchberg-Saxton, les coefficients de Zernike sont ajustés
*directement* de façon à ce que la PSF simulée reproduise la PSF mesurée,
sans étape intermédiaire de retrouvé de phase pixel-à-pixel. Une seule PSF
en entrée.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from ..optics import build_zernike_basis, phase_from_coefficients, psf_from_phase


@dataclass(slots=True)
class OpticalConfig:
    """
    Optical / optimisation configuration.

    Parameters
    ----------
    wavelength : float
        Wavelength [m].
    zernike_order : int
        Number of Zernike modes.
    fit_piston_tilt : bool
        If False, Z1-Z3 (piston, tip, tilt) are held at zero during the fit.
    max_nfev : int, optional
        Maximum number of residual-function evaluations passed to
        ``scipy.optimize.least_squares``. ``None`` uses scipy's default.
    verbose : bool
        Print the solver's convergence trace.
    core_weight_radius : float, optional
        If set, down-weight residual pixels away from the PSF centre with a
        Gaussian taper of this radius [px] (``weight = exp(-(r/radius)^2)``)
        — prioritises the (high-SNR) core over noisy wings. ``None``
        (default) weights every pixel equally, as before.
    l2_reg : float, default 0.0
        Tikhonov (L2) regularisation strength on the fitted coefficients
        (in radians); penalises large excursions and improves robustness
        when the fit is under-constrained (e.g. high ``zernike_order`` with
        a noisy or low-flux PSF). ``0.0`` (default) disables it, reproducing
        the original unregularised fit.
    """
    wavelength: float = 550e-9
    zernike_order: int = 36
    fit_piston_tilt: bool = False
    max_nfev: Optional[int] = None
    verbose: bool = True
    core_weight_radius: Optional[float] = None
    l2_reg: float = 0.0


def _core_weight(shape: Tuple[int, int], radius: float) -> NDArray[np.float64]:
    """
    Gaussian taper centred on the array, for :func:`residuals`'
    ``core_weight_radius`` option.

    Parameters
    ----------
    shape : tuple of int
        Array shape.
    radius : float
        Taper radius [px].

    Returns
    -------
    NDArray
        Weight map, shape ``shape``, values in ``(0, 1]`` (``1`` at centre).
    """
    y, x = np.indices(shape)
    cy, cx = (np.array(shape) - 1) / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-r2 / radius ** 2)


def residuals(
    coeffs_rad: NDArray[np.float64],
    measured_psf: NDArray[np.float64],
    basis: NDArray[np.float64],
    mask: NDArray[np.float64],
    *,
    weight: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Least-squares residuals between a simulated and a measured PSF.

    Parameters
    ----------
    coeffs_rad : NDArray
        Zernike coefficients [rad].
    measured_psf : NDArray
        Experimental PSF.
    basis : NDArray
        Zernike basis.
    mask : NDArray
        Pupil mask.
    weight : NDArray, optional
        Optional per-pixel weight map (e.g. from ``OpticalConfig.core_weight_radius``
        via :func:`_core_weight`), same shape as ``measured_psf``. Applied as
        ``sqrt(weight)`` to keep the weighted sum of squares equal to
        ``sum(weight * (model - measured) ** 2)``. ``None`` weights every
        pixel equally (original behaviour).

    Returns
    -------
    NDArray
        Flattened residual vector (amplitude domain: ``sqrt`` of the PSF).
    """
    phase = phase_from_coefficients(coeffs_rad, basis, mask)
    model_psf = psf_from_phase(phase, mask)

    residual = np.sqrt(model_psf) - np.sqrt(measured_psf)
    if weight is not None:
        residual = residual * np.sqrt(weight)

    return residual.ravel()


def fit_zernike_from_psf(
    psf: NDArray[np.float64],
    mask: NDArray[np.float64],
    cfg: OpticalConfig,
) -> Tuple[NDArray[np.float64], Dict[str, float]]:
    """
    Estimate Zernike coefficients by directly fitting a single PSF.

    Parameters
    ----------
    psf : NDArray
        Measured PSF (energy-normalised).
    mask : NDArray
        Pupil mask.
    cfg : OpticalConfig
        Configuration.

    Returns
    -------
    coeffs_nm : NDArray
        Zernike coefficients (Noll order), in nanometres of WFE.
    diagnostics : dict
        Optimisation diagnostics: final cost, number of function
        evaluations, solver status message and PSF-domain RMSE.
    """
    basis = build_zernike_basis(npix=psf.shape[0], nterms=cfg.zernike_order)

    if cfg.fit_piston_tilt:
        active = np.arange(cfg.zernike_order)
    else:
        active = np.arange(3, cfg.zernike_order)

    weight = _core_weight(psf.shape, cfg.core_weight_radius) if cfg.core_weight_radius else None

    def objective(x: NDArray[np.float64]) -> NDArray[np.float64]:
        coeffs = np.zeros(cfg.zernike_order)
        coeffs[active] = x
        r = residuals(coeffs, psf, basis, mask, weight=weight)
        if cfg.l2_reg > 0.0:
            # Tikhonov (L2) regularisation embedded as extra pseudo-residuals:
            # sum(r**2) + l2_reg*sum(x**2) == sum(r**2) + sum((sqrt(l2_reg)*x)**2)
            r = np.concatenate([r, np.sqrt(cfg.l2_reg) * x])
        return r

    result = least_squares(
        objective,
        np.zeros(active.size),
        method="trf",
        max_nfev=cfg.max_nfev,
        verbose=2 if cfg.verbose else 0,
    )

    coeffs_rad = np.zeros(cfg.zernike_order)
    coeffs_rad[active] = result.x

    rad_to_nm = cfg.wavelength * 1e9 / (2 * np.pi)
    coeffs_nm = coeffs_rad * rad_to_nm

    model_psf = psf_from_phase(phase_from_coefficients(coeffs_rad, basis, mask), mask)
    psf_rmse = float(np.sqrt(np.mean((model_psf - psf) ** 2)))

    diagnostics = {
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "status": int(result.status),
        "message": str(result.message),
        "psf_rmse": psf_rmse,
    }

    return coeffs_nm, diagnostics
