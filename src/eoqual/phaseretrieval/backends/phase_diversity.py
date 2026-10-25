#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Récupération de phase par diversité de phase modale.

Contrairement aux deux autres méthodes (:mod:`backends.gerchberg_saxton`,
:mod:`backends.optimization`), celle-ci nécessite **deux** PSF en entrée :
une PSF focalisée et une seconde PSF acquise avec un défocus (ou toute
autre aberration) connu et volontairement introduit. Les coefficients de
Zernike sont ajustés conjointement sur les deux images, ce qui lève
l'ambiguïté de signe inhérente à une reconstruction à partir d'une seule
PSF.

Cette contrainte de conception (deux fichiers d'entrée, une configuration
dédiée) est la raison pour laquelle cette méthode n'est *pas* fondue dans
la même fonction bas niveau que les méthodes mono-PSF : elle reste un
module à part, orchestré au même niveau que les autres par
:func:`pipeline.psf_to_wfe`.
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
class PhaseDiversityConfig:
    """
    Configuration parameters for the phase-diversity fit.

    Parameters
    ----------
    wavelength : float
        Wavelength [m].
    zernike_order : int
        Number of fitted Zernike modes (piston/tip/tilt excluded from the fit).
    defocus : float
        Known diversity defocus [m] applied on the second PSF.
    roi_radius : int
        Half-width [px] of the square region of interest used in the cost
        function (centred on the PSF): the residual is computed only on the
        ``2*roi_radius`` × ``2*roi_radius`` central window, not the full
        frame. Almost all of the PSF's energy sits near its centre, so the
        outer pixels are mostly background/noise (and can carry edge
        artefacts — FFT wraparound, other sources in the field, hot
        pixels); including them would dilute the useful signal in the sum
        of residuals and needlessly slow down the solver. Too small a
        value instead crops part of the signal that constrains the
        higher-order Zernike modes (which spread the PSF further out) —
        see ``REFERENCE_TECHNIQUE.md`` §6.3 for the trade-off and
        a worked example.
    loss : str
        Robust loss function name passed to ``scipy.optimize.least_squares``
        (e.g. ``"linear"``, ``"soft_l1"``, ``"huber"``).
    poisson_weighting : bool
        If True, weight residuals by the (photon) shot-noise standard
        deviation instead of using the raw amplitude difference.
    max_nfev : int, optional
        Maximum number of residual-function evaluations. ``None`` uses
        scipy's default.
    verbose : bool
        Print the solver's convergence trace.
    orthogonal_basis : bool, default False
        Fit on a Gram-Schmidt basis orthogonalised on the actual mask
        (see :class:`zernike_fit.ZernikeConfig` and §10 of
        ``REFERENCE_TECHNIQUE.md``) instead of the standard Noll basis —
        recommended for an obstructed pupil (central obstruction,
        spiders). ``False`` (default) keeps the historical behaviour.
    aperture_px : int, optional
        Diameter [px] of the aperture within ``mask``, when it does not
        fill the whole array — see
        ``backends.optimization.OpticalConfig.aperture_px`` and §12.5 of
        ``REFERENCE_TECHNIQUE.md``. ``None`` (default): the aperture fills
        the array (historical behaviour).
    known_ho_wfe_nm : NDArray, optional
        Known, **fixed** wavefront-error map [nm] — typically a high-order
        (HO) contribution beyond ``zernike_order``, measured independently
        — added to the fitted low-order phase in **both** channels (focus
        and defocus): it is a static contribution of the optics, present
        regardless of the diversity term (see §11.3 of
        ``REFERENCE_TECHNIQUE.md``). Same shape as the PSF/mask. Never
        optimised, exactly like ``defocus`` itself. ``None`` (default)
        disables it, reproducing the original model.
    zernike_priors : dict, optional
        Per-mode informative prior, as ``{noll_index: (mean_nm, sigma_nm)}``
        — a Bayesian MAP Gaussian prior added to the residual vector as a
        per-mode L2 penalty, exactly like
        ``backends.optimization.OpticalConfig.zernike_priors`` (same helper,
        :func:`eoqual.phaseretrieval.zernike_priors.build_prior_penalty`;
        see its docstring and ``REFERENCE_TECHNIQUE.md`` §6.5bis for the
        derivation). Unlike ``OpticalConfig``, there is no ``l2_reg`` here
        to generalise from — this is the only prior mechanism on this
        method. **Caveat specific to this method:** the pseudo-residuals
        encoding the prior go through the *same* robust ``loss`` as the
        data residuals (below) — the exact Gaussian-MAP equivalence holds
        only for ``loss="linear"``; with the default ``"soft_l1"`` the
        prior is effectively robustified too (large deviations from
        ``mean_nm`` are penalised less than a pure quadratic would, same
        rationale as robustifying the data term against outlier pixels).
        ``None`` (default) disables it.
    zernike_bounds : dict, optional
        Per-mode hard box constraint, as
        ``{noll_index: (low_nm, high_nm)}`` — a uniform-prior MAP,
        equivalent in spirit to ``zernike_priors`` but as a hard bound
        (:func:`eoqual.phaseretrieval.zernike_priors.build_prior_bounds`).
        ``None`` (default) leaves every mode unbounded.
    """
    wavelength: float = 550e-9
    zernike_order: int = 36
    defocus: float = 275e-9
    roi_radius: int = 32
    loss: str = "soft_l1"
    poisson_weighting: bool = True
    max_nfev: Optional[int] = None
    verbose: bool = True
    orthogonal_basis: bool = False
    aperture_px: Optional[int] = None
    known_ho_wfe_nm: Optional[NDArray[np.float64]] = None
    zernike_priors: Optional[Dict[int, Tuple[float, float]]] = None
    zernike_bounds: Optional[Dict[int, Tuple[float, float]]] = None


def build_roi_mask(shape: Tuple[int, int], radius: int) -> NDArray[np.bool_]:
    """
    Crée une région d'intérêt (ROI) carrée centrée.

    Restreindre le résidu des moindres carrés (:func:`residuals`) à cette
    ROI plutôt qu'à l'image entière concentre le coût sur les pixels où se
    trouve l'essentiel de l'énergie de la PSF, plutôt que de le diluer dans
    du bruit de fond ; ça accélère aussi le solveur (moins de pixels par
    évaluation). Voir la docstring de ``roi_radius`` dans
    :class:`PhaseDiversityConfig` pour le compromis taille de ROI / ordre
    de Zernike maximal représentable.

    Parameters
    ----------
    shape : tuple of int
        Taille de l'image ``(ny, nx)``.
    radius : int
        Demi-largeur de la ROI, en pixels.

    Returns
    -------
    NDArray
        Masque booléen, ``True`` à l'intérieur de la ROI.
    """
    ny, nx = shape
    cy, cx = ny // 2, nx // 2

    roi = np.zeros(shape, dtype=bool)
    roi[cy - radius:cy + radius, cx - radius:cx + radius] = True

    return roi


def poisson_residual(
    model: NDArray[np.float64],
    data: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Résidu pondéré par le bruit de photon (Poisson).

    Parameters
    ----------
    model : NDArray
        PSF simulée.
    data : NDArray
        PSF mesurée.

    Returns
    -------
    NDArray
        Résidu ``(model - data) / sqrt(data)``.
    """
    eps = 1e-12
    sigma = np.sqrt(np.maximum(data, eps))
    return (model - data) / sigma


def diversity_phase(
    basis: NDArray[np.float64],
    defocus: float,
    wavelength: float,
) -> NDArray[np.float64]:
    """
    Build the known diversity phase (defocus) map to add to the fitted phase.

    Parameters
    ----------
    basis : NDArray
        Zernike basis; ``basis[3]`` is the (Noll) defocus mode.
    defocus : float
        Diversity defocus [m].
    wavelength : float
        Wavelength [m], used to convert the defocus to radians.

    Returns
    -------
    NDArray
        Diversity phase map [rad].
    """
    nm_to_rad = (2 * np.pi) / wavelength
    return defocus * nm_to_rad * basis[3]


def residuals(
    active_coeffs: NDArray[np.float64],
    psf_focus: NDArray[np.float64],
    psf_defocus: NDArray[np.float64],
    basis: NDArray[np.float64],
    mask: NDArray[np.float64],
    roi_mask: NDArray[np.bool_],
    cfg: PhaseDiversityConfig,
) -> NDArray[np.float64]:
    """
    Joint residual vector over the focused and defocused PSF.

    Parameters
    ----------
    active_coeffs : NDArray
        Zernike coefficients [rad] for modes 4..zernike_order (piston/tip/tilt
        excluded).
    psf_focus : NDArray
        Focused PSF.
    psf_defocus : NDArray
        Defocused (diversity) PSF.
    basis : NDArray
        Zernike basis.
    mask : NDArray
        Binary pupil mask.
    roi_mask : NDArray
        Region of interest over which the residuals are computed.
    cfg : PhaseDiversityConfig
        Configuration.

    Returns
    -------
    NDArray
        Concatenated, flattened residual vector for both PSF.
    """
    coeffs = np.zeros(cfg.zernike_order, dtype=np.float64)
    coeffs[3:] = active_coeffs

    phase = phase_from_coefficients(coeffs, basis, mask)
    if cfg.known_ho_wfe_nm is not None:
        phase = phase + wfe_map_to_phase(cfg.known_ho_wfe_nm, cfg.wavelength, mask)

    psf_model_focus = psf_from_phase(phase, mask)

    diversity = diversity_phase(basis, cfg.defocus, cfg.wavelength)
    psf_model_defocus = psf_from_phase(phase + diversity, mask)

    mf, md = psf_model_focus[roi_mask], psf_model_defocus[roi_mask]
    df, dd = psf_focus[roi_mask], psf_defocus[roi_mask]

    if cfg.poisson_weighting:
        r1 = poisson_residual(mf, df)
        r2 = poisson_residual(md, dd)
    else:
        r1 = np.sqrt(mf) - np.sqrt(df)
        r2 = np.sqrt(md) - np.sqrt(dd)

    return np.concatenate([r1.ravel(), r2.ravel()])


def fit_phase_diversity(
    psf_focus: NDArray[np.float64],
    psf_defocus: NDArray[np.float64],
    mask: NDArray[np.float64],
    cfg: PhaseDiversityConfig,
) -> Tuple[NDArray[np.float64], Dict[str, float]]:
    """
    Estimate Zernike coefficients from a focused/defocused PSF pair.

    Parameters
    ----------
    psf_focus : NDArray
        Focused PSF (energy-normalised).
    psf_defocus : NDArray
        Defocused PSF (energy-normalised).
    mask : NDArray
        Binary pupil mask.
    cfg : PhaseDiversityConfig
        Configuration.

    Returns
    -------
    coeffs_nm : NDArray
        Zernike coefficients (Noll order), in nanometres of WFE.
    diagnostics : dict
        Optimisation diagnostics: final cost, number of function
        evaluations, solver status message and PSF-domain RMSE (both
        channels).
    """
    basis = resolve_zernike_basis(
        npix=psf_focus.shape[0], nterms=cfg.zernike_order, mask=mask,
        orthogonalize=cfg.orthogonal_basis, aperture_px=cfg.aperture_px,
    )
    roi_mask = build_roi_mask(psf_focus.shape, cfg.roi_radius)

    active = np.arange(3, cfg.zernike_order)

    # A priori individuels par mode : même mécanisme que
    # backends.optimization (voir zernike_priors.py et
    # REFERENCE_TECHNIQUE.md §6.5bis pour la dérivation). active contient
    # des positions de tableau 0-based -> +1 pour l'indexation Noll.
    prior_mean_rad, prior_weight = build_prior_penalty(
        cfg.zernike_priors or {}, active + 1, cfg.wavelength,
    )
    has_priors = np.any(prior_weight > 0.0)
    lower_rad, upper_rad = build_prior_bounds(
        cfg.zernike_bounds or {}, active + 1, cfg.wavelength,
    )
    x0 = feasible_start(lower_rad, upper_rad)

    def objective(x: NDArray[np.float64]) -> NDArray[np.float64]:
        r = residuals(x, psf_focus, psf_defocus, basis, mask, roi_mask, cfg)
        if has_priors:
            r = np.concatenate([r, prior_weight * (x - prior_mean_rad)])
        return r

    result = least_squares(
        objective,
        x0=x0,
        method="trf",
        loss=cfg.loss,
        f_scale=1.0,
        bounds=(lower_rad, upper_rad),
        max_nfev=cfg.max_nfev,
        verbose=2 if cfg.verbose else 0,
    )

    coeffs_rad = np.zeros(cfg.zernike_order)
    coeffs_rad[3:] = result.x

    rad_to_nm = cfg.wavelength * 1e9 / (2 * np.pi)
    coeffs_nm = coeffs_rad * rad_to_nm

    phase = phase_from_coefficients(coeffs_rad, basis, mask)
    if cfg.known_ho_wfe_nm is not None:
        phase = phase + wfe_map_to_phase(cfg.known_ho_wfe_nm, cfg.wavelength, mask)
    diversity = diversity_phase(basis, cfg.defocus, cfg.wavelength)
    model_focus = psf_from_phase(phase, mask)
    model_defocus = psf_from_phase(phase + diversity, mask)

    diagnostics = {
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "status": int(result.status),
        "message": str(result.message),
        "psf_focus_rmse": float(np.sqrt(np.mean((model_focus - psf_focus) ** 2))),
        "psf_defocus_rmse": float(np.sqrt(np.mean((model_defocus - psf_defocus) ** 2))),
    }

    return coeffs_nm, diagnostics
