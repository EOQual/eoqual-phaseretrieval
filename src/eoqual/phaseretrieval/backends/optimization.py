#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Récupération de phase par optimisation directe (moindres carrés non-linéaires).

Contrairement à Gerchberg-Saxton, les coefficients de Zernike sont ajustés
*directement* de façon à ce que la PSF simulée reproduise la PSF mesurée,
sans étape intermédiaire de récupération de phase pixel-à-pixel. Une seule PSF
en entrée.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from ..optics import phase_from_coefficients, psf_from_phase, wfe_map_to_phase
from ..zernike_basis_orthogonal import resolve_zernike_basis
from ..zernike_priors import build_prior_bounds, build_prior_penalty, feasible_start


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
        the original unregularised fit. Equivalent to a *zero-mean, equal-weight*
        Gaussian prior on every mode — see ``zernike_priors`` below for the
        general (per-mode, informative) case, of which this is a special
        case.
    zernike_priors : dict, optional
        Per-mode informative prior, as ``{noll_index: (mean_nm, sigma_nm)}``
        — e.g. ``{11: (0.0, 5.0)}`` for a known-small spherical aberration.
        Added to the residual vector as a per-mode L2 penalty
        (:func:`eoqual.phaseretrieval.zernike_priors.build_prior_penalty`):
        mathematically a Bayesian MAP estimate with a Gaussian prior
        ``c_k ~ N(mean_nm, sigma_nm**2)`` on each listed mode — a strict
        generalisation of ``l2_reg`` (arbitrary mean and per-mode weight,
        instead of zero and a single shared weight). Typical source: an
        as-built optical model, a prior calibration, or an error budget —
        **not** a generic assumption that high-order modes are small (see
        ``REFERENCE_TECHNIQUE.md`` §6.5bis for why a Kolmogorov-turbulence-style
        decreasing-with-radial-order prior is *not*, in general, justified
        for a satellite instrument's own (non-atmospheric) WFE). ``None``
        (default) disables it. Raises ``ValueError`` if a key is not among
        the modes actually being fitted (``zernike_order``/``fit_piston_tilt``).
    zernike_bounds : dict, optional
        Per-mode hard box constraint, as
        ``{noll_index: (low_nm, high_nm)}`` — the same idea as
        ``zernike_priors`` but with a *uniform* prior (zero outside the
        interval) instead of a Gaussian one, passed directly to
        ``scipy.optimize.least_squares(bounds=...)``
        (:func:`eoqual.phaseretrieval.zernike_priors.build_prior_bounds`).
        Can be combined with ``zernike_priors`` (e.g. a Gaussian prior
        truncated to a physically plausible range). ``None`` (default)
        leaves every mode unbounded.
    orthogonal_basis : bool, default False
        Fit on a Gram-Schmidt basis orthogonalised on the actual mask
        (see :class:`zernike_fit.ZernikeConfig` and §10 of
        ``REFERENCE_TECHNIQUE.md``) instead of the standard Noll basis —
        recommended for an obstructed pupil (central obstruction,
        spiders). ``False`` (default) keeps the historical behaviour.
    aperture_px : int, optional
        Diameter [px] of the aperture within ``mask``, when it does not
        fill the whole ``psf``/``mask`` array — e.g. the ``N_pup_target``
        returned by ``io_utils.load_and_prepare_mask`` whenever it differs
        from ``psf.shape[0]`` (multi-resolution / oversampling, §12 of
        ``REFERENCE_TECHNIQUE.md``), or a ``PupilGenerator(npix,
        aperture_px=...)`` mask built directly in memory. **Must match**
        whatever was actually used to build ``mask``: otherwise the
        Zernike basis is evaluated as if the aperture filled the whole
        array — a silent scale mismatch (see §12.5 and
        ``optics.build_zernike_basis``). ``None`` (default): the aperture
        fills the array (historical behaviour, correct only when that is
        actually the case).
    known_ho_wfe_nm : NDArray, optional
        Known, **fixed** wavefront-error map [nm] — typically a high-order
        (HO) contribution beyond ``zernike_order``, measured independently
        (interferogram, as-built optical model, ground calibration) —
        added to the fitted low-order phase before computing the model PSF
        (see §11.3 of ``REFERENCE_TECHNIQUE.md``). Same shape as the PSF/
        mask. Never optimised: only the Zernike coefficients up to
        ``zernike_order`` remain free parameters, exactly as if this
        contribution were absent from the pupil altogether — the point is
        to stop it from silently biasing those low-order coefficients.
        ``None`` (default) disables it, reproducing the original model.
    """
    wavelength: float = 550e-9
    zernike_order: int = 36
    fit_piston_tilt: bool = False
    max_nfev: Optional[int] = None
    verbose: bool = True
    core_weight_radius: Optional[float] = None
    l2_reg: float = 0.0
    zernike_priors: Optional[Dict[int, Tuple[float, float]]] = None
    zernike_bounds: Optional[Dict[int, Tuple[float, float]]] = None
    orthogonal_basis: bool = False
    aperture_px: Optional[int] = None
    known_ho_wfe_nm: Optional[NDArray[np.float64]] = None


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
    known_phase_rad: Optional[NDArray[np.float64]] = None,
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
    known_phase_rad : NDArray, optional
        Known, fixed phase contribution [rad] added to the modelled phase
        before computing the PSF (see ``OpticalConfig.known_ho_wfe_nm``).
        ``None`` (default) adds nothing, reproducing the original model.

    Returns
    -------
    NDArray
        Flattened residual vector (amplitude domain: ``sqrt`` of the PSF).

    Notes
    -----
    ``measured_psf`` is clipped to non-negative before the square root —
    a real (or realistically simulated, e.g.
    :func:`eoqual.phaseretrieval.simulation.noise.apply_detector_noise`)
    acquisition can carry slightly negative pixels (read noise, background
    subtraction) that would otherwise make ``least_squares`` fail outright
    (``sqrt`` of a negative number). Same precaution already taken in
    :func:`backends.gerchberg_saxton.gerchberg_saxton`'s target amplitude.
    ``model_psf`` needs no such clipping: it is ``|FFT|**2`` by
    construction (:func:`optics.psf_from_phase`), always non-negative.
    """
    phase = phase_from_coefficients(coeffs_rad, basis, mask)
    if known_phase_rad is not None:
        phase = phase + known_phase_rad
    model_psf = psf_from_phase(phase, mask)

    residual = np.sqrt(model_psf) - np.sqrt(np.maximum(measured_psf, 0.0))
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
    basis = resolve_zernike_basis(
        npix=psf.shape[0], nterms=cfg.zernike_order, mask=mask,
        orthogonalize=cfg.orthogonal_basis, aperture_px=cfg.aperture_px,
    )

    known_phase_rad = None
    if cfg.known_ho_wfe_nm is not None:
        known_phase_rad = wfe_map_to_phase(cfg.known_ho_wfe_nm, cfg.wavelength, mask)

    if cfg.fit_piston_tilt:
        active = np.arange(cfg.zernike_order)
    else:
        active = np.arange(3, cfg.zernike_order)

    weight = _core_weight(psf.shape, cfg.core_weight_radius) if cfg.core_weight_radius else None

    # A priori individuels par mode (généralisation de l2_reg ci-dessus) :
    # voir zernike_priors.py pour la dérivation (MAP gaussien = pénalité L2,
    # MAP uniforme = borne dure) et REFERENCE_TECHNIQUE.md §6.5bis.
    # active contient des positions de tableau 0-based (coeffs[active] = x) ;
    # zernike_priors/zernike_bounds sont indexés en Noll (1-based) -> +1.
    prior_mean_rad, prior_weight = build_prior_penalty(
        cfg.zernike_priors or {}, active + 1, cfg.wavelength,
    )
    has_priors = np.any(prior_weight > 0.0)
    lower_rad, upper_rad = build_prior_bounds(
        cfg.zernike_bounds or {}, active + 1, cfg.wavelength,
    )
    x0 = feasible_start(lower_rad, upper_rad)

    def objective(x: NDArray[np.float64]) -> NDArray[np.float64]:
        coeffs = np.zeros(cfg.zernike_order)
        coeffs[active] = x
        r = residuals(coeffs, psf, basis, mask, weight=weight, known_phase_rad=known_phase_rad)
        if cfg.l2_reg > 0.0:
            # Tikhonov (L2) regularisation embedded as extra pseudo-residuals:
            # sum(r**2) + l2_reg*sum(x**2) == sum(r**2) + sum((sqrt(l2_reg)*x)**2)
            r = np.concatenate([r, np.sqrt(cfg.l2_reg) * x])
        if has_priors:
            r = np.concatenate([r, prior_weight * (x - prior_mean_rad)])
        return r

    result = least_squares(
        objective,
        x0,
        method="trf",
        bounds=(lower_rad, upper_rad),
        max_nfev=cfg.max_nfev,
        verbose=2 if cfg.verbose else 0,
    )

    coeffs_rad = np.zeros(cfg.zernike_order)
    coeffs_rad[active] = result.x

    rad_to_nm = cfg.wavelength * 1e9 / (2 * np.pi)
    coeffs_nm = coeffs_rad * rad_to_nm

    model_phase = phase_from_coefficients(coeffs_rad, basis, mask)
    if known_phase_rad is not None:
        model_phase = model_phase + known_phase_rad
    model_psf = psf_from_phase(model_phase, mask)
    psf_rmse = float(np.sqrt(np.mean((model_psf - psf) ** 2)))

    diagnostics = {
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "status": int(result.status),
        "message": str(result.message),
        "psf_rmse": psf_rmse,
    }

    return coeffs_nm, diagnostics
