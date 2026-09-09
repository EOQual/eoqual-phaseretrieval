#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics de qualité optique d'une PSF : Strehl, WFE (Maréchal), profil
radial, MTF et ellipticité du cœur.

Complète les diagnostics déjà produits par le fit Zernike lui-même
(:mod:`zernike_fit`) avec des métriques indépendantes du modèle Zernike —
utiles en particulier pour un premier diagnostic visuel/quantitatif *avant*
de lancer une méthode de retrouvé de phase (§1 et nouvelle §9 de
``phase_retrieval_and_diversity.md``), ou pour comparer plusieurs méthodes
entre elles sur un pied d'égalité.

Origine : portage, adapté au style et aux conventions de ce projet
(NumPy docstrings, typing, réutilisation de ``optics.py``), des utilitaires
``PSFAnalyzer``/``PSFMetrics`` développés dans le prototype
``Star/star_PSF/psf_star_V2d.py`` (répertoire fourre-tout hors dépôt).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.special import j1


def airy_psf(
    shape: tuple[int, int],
    wavelength: float,
    diameter: float,
    pixel_scale: float,
) -> NDArray[np.float64]:
    """
    Generate a normalised, diffraction-limited Airy PSF (unobstructed
    circular aperture).

    Parameters
    ----------
    shape : tuple of int
        ``(ny, nx)`` shape of the output array.
    wavelength : float
        Wavelength [m].
    diameter : float
        Aperture (pupil) diameter [m].
    pixel_scale : float
        Angle subtended by one pixel [rad/pixel]. If the PSF being compared
        to is oversampled by a factor ``k``, pass ``pixel_scale / k``.

    Returns
    -------
    NDArray
        Airy PSF, shape ``shape``, normalised so that its sum is 1.
    """
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0

    y, x = np.indices(shape)
    r_pix = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_rad = r_pix * pixel_scale

    k = np.pi * diameter / wavelength
    kr = k * r_rad

    psf = np.ones_like(kr)
    nonzero = kr != 0
    psf[nonzero] = (2 * j1(kr[nonzero]) / kr[nonzero]) ** 2

    return psf / psf.sum()


def pixel_scale_from_pixel_pitch(pixel_pitch_m: float, focal_length_m: float) -> float:
    """
    Convert a detector pixel pitch to an angular pixel scale (small-angle).

    Parameters
    ----------
    pixel_pitch_m : float
        Detector pixel size [m].
    focal_length_m : float
        Effective focal length [m].

    Returns
    -------
    float
        Angular pixel scale [rad/pixel].
    """
    return pixel_pitch_m / focal_length_m


def strehl_ratio(
    psf_measured: NDArray[np.float64],
    psf_reference: NDArray[np.float64],
) -> float:
    """
    Compute the Strehl ratio of a measured PSF against a reference.

    Parameters
    ----------
    psf_measured : NDArray
        Measured (or simulated) PSF.
    psf_reference : NDArray
        Reference, diffraction-limited PSF (e.g. from :func:`airy_psf`), on
        the *same* grid, sampling and spectral band as ``psf_measured``.
        The two PSF need not be individually normalised: only the ratio of
        their peaks is used.

    Returns
    -------
    float
        Strehl ratio, ``S = max(psf_measured) / max(psf_reference)``.

    Notes
    -----
    The Strehl ratio only depends on the peak values, not on any spatial
    shift between the two PSF. It is only meaningful if both PSF share the
    same normalisation convention (e.g. both sum to 1), the same pixel
    scale, and the same spectral band — see ``phase_retrieval_and_diversity.md``
    §9 for the full list of conditions and the Maréchal quality table.

    A ratio significantly above 1 is not physically possible for a
    genuine (aberrated or not) PSF against its own diffraction-limited
    reference: it signals one of these conditions was violated (most
    often a ``pixel_scale`` inconsistent with how ``psf_measured`` was
    actually produced) rather than an unusually good optic — a warning is
    logged in that case so the mismatch isn't mistaken for a result.
    """
    strehl = float(psf_measured.max() / psf_reference.max())
    if strehl > 1.02:
        logger.warning(
            f"Strehl = {strehl:.3f} > 1 : la référence Airy n'est probablement pas "
            "cohérente avec la PSF mesurée (pixel_scale, normalisation ou bande "
            "spectrale différente) — voir les conditions listées dans la docstring."
        )
    return strehl


def wfe_rms_from_strehl(strehl: float, wavelength: float) -> float:
    """
    Estimate the WFE RMS from a Strehl ratio (Maréchal approximation).

    Parameters
    ----------
    strehl : float
        Strehl ratio, in ``(0, 1]``.
    wavelength : float
        Wavelength [m].

    Returns
    -------
    float
        WFE RMS [m].

    Raises
    ------
    ValueError
        If ``strehl`` is not in ``(0, 1]``.

    Notes
    -----
    Maréchal's approximation: :math:`S \\approx \\exp[-(2\\pi\\sigma/\\lambda)^2]`,
    inverted as :math:`\\sigma = (\\lambda/2\\pi)\\sqrt{-\\ln S}`. Only valid
    for small aberrations (typically :math:`S \\gtrsim 0.1`–0.3); see
    :func:`zernike_fit.fit_zernike` for a direct (non-approximated) RMS from
    an actual Zernike fit.
    """
    if not (0.0 < strehl <= 1.0):
        raise ValueError(f"Strehl invalide (attendu dans (0, 1]) : {strehl}")

    return (wavelength / (2.0 * np.pi)) * np.sqrt(-np.log(strehl))


def radial_profile(psf: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the azimuthally-averaged radial profile of a centred PSF.

    Parameters
    ----------
    psf : NDArray
        2-D array (assumed centred at ``(shape - 1) / 2``).

    Returns
    -------
    NDArray
        1-D profile, mean value per integer-pixel radius from the centre.
    """
    y, x = np.indices(psf.shape)
    cy, cx = (np.array(psf.shape) - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    tbin = np.bincount(r.ravel(), psf.ravel())
    nr = np.bincount(r.ravel())

    return tbin / np.maximum(nr, 1)


def compute_mtf(psf: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the (real-valued) modulation transfer function of a PSF.

    Parameters
    ----------
    psf : NDArray
        2-D PSF (any normalisation; renormalised internally).

    Returns
    -------
    NDArray
        MTF, i.e. ``abs(OTF)``, normalised so that its maximum is 1.
    """
    psf_norm = psf / psf.sum()
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf_norm)))
    mtf = np.abs(otf)
    return mtf / mtf.max()


def core_ellipticity(psf: NDArray[np.float64], radius: float = 5.0) -> float:
    """
    Measure the ellipticity of a PSF's core from its weighted 2nd moments.

    Parameters
    ----------
    psf : NDArray
        Centred PSF.
    radius : float, default 5.0
        Radius [px] of the disk used to compute the moments.

    Returns
    -------
    float
        Ellipticity ``sqrt((Ixx-Iyy)^2 + 4*Ixy^2) / (Ixx+Iyy)``.
        ``0`` = circular core; larger values indicate coma/astigmatism.
        Rule of thumb: ``< 0.05`` circular, ``0.05-0.15`` moderate
        astigmatism, ``> 0.15`` strong coma/astigmatism.
    """
    y, x = np.indices(psf.shape)
    cy, cx = (np.array(psf.shape) - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    inside = r < radius
    x0 = x[inside] - cx
    y0 = y[inside] - cy
    w = psf[inside]

    Ixx = np.sum(w * x0 ** 2)
    Iyy = np.sum(w * y0 ** 2)
    Ixy = np.sum(w * x0 * y0)

    denom = Ixx + Iyy
    if denom == 0:
        return 0.0

    return float(np.sqrt((Ixx - Iyy) ** 2 + 4.0 * Ixy ** 2) / denom)


@dataclass(slots=True)
class PSFMetrics:
    """
    Bundle of PSF quality metrics, as returned by :func:`compute_metrics`.

    Parameters
    ----------
    strehl : float
        Strehl ratio.
    wfe_rms : float
        WFE RMS [m] (Maréchal approximation from ``strehl``).
    ellipticity : float
        Core ellipticity (see :func:`core_ellipticity`).
    first_ring_ratio : float
        Ratio of the intensity at radius 10 px to the peak — a simple proxy
        for the strength of the diffraction rings (high for defocus, low
        for a clean Airy-like core).
    """
    strehl: float
    wfe_rms: float
    ellipticity: float
    first_ring_ratio: float


def compute_metrics(
    psf: NDArray[np.float64],
    wavelength: float,
    diameter: float,
    pixel_scale: float,
) -> PSFMetrics:
    """
    Compute the standard PSF quality metrics against an Airy reference.

    Parameters
    ----------
    psf : NDArray
        Measured/simulated PSF, normalised (sum = 1).
    wavelength : float
        Wavelength [m].
    diameter : float
        Aperture diameter [m].
    pixel_scale : float
        Angular pixel scale of ``psf`` [rad/pixel] (already accounting for
        any oversampling — see :func:`airy_psf`).

    Returns
    -------
    PSFMetrics
    """
    psf_ref = airy_psf(psf.shape, wavelength, diameter, pixel_scale)

    strehl = strehl_ratio(psf, psf_ref)  # logs a warning if > 1 (see docstring)
    wfe = wfe_rms_from_strehl(min(strehl, 1.0), wavelength) if strehl > 1e-3 else float("nan")
    ellipticity = core_ellipticity(psf)

    profile = radial_profile(psf)
    first_ring_ratio = float(profile[10] / profile[0]) if len(profile) > 10 else float("nan")

    return PSFMetrics(
        strehl=strehl,
        wfe_rms=wfe,
        ellipticity=ellipticity,
        first_ring_ratio=first_ring_ratio,
    )


def diagnose_aberration(metrics: PSFMetrics) -> str:
    """
    Heuristic, qualitative diagnosis of the dominant aberration type.

    Parameters
    ----------
    metrics : PSFMetrics
        Metrics from :func:`compute_metrics`.

    Returns
    -------
    str
        A short human-readable diagnosis. **Heuristic only** — always
        confirm with an actual Zernike fit (:mod:`zernike_fit`,
        :mod:`retrieval_optimization`, :mod:`retrieval_phase_diversity`)
        before drawing conclusions.
    """
    if metrics.first_ring_ratio > 0.25 and metrics.ellipticity < 0.05:
        return "Défocus dominant"
    if metrics.ellipticity > 0.15:
        return "Coma / astigmatisme dominant"
    if metrics.first_ring_ratio < 0.1 and metrics.ellipticity < 0.05:
        return "Sphérique dominant"
    return "Aberrations mixtes / bruit haute fréquence"
