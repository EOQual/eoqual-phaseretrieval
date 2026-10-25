#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics de qualité optique d'une PSF : Strehl, WFE (Maréchal), profil
radial, MTF et ellipticité du cœur.

Complète les diagnostics déjà produits par le fit Zernike lui-même
(:mod:`zernike_fit`) avec des métriques indépendantes du modèle Zernike —
utiles en particulier pour un premier diagnostic visuel/quantitatif *avant*
de lancer une méthode de récupération de phase (§1 et nouvelle §9 de
``REFERENCE_TECHNIQUE.md``), ou pour comparer plusieurs méthodes
entre elles sur un pied d'égalité.

Origine : utilitaires ``PSFAnalyzer``/``PSFMetrics`` d'un prototype
antérieur de l'auteur, adaptés au style et aux conventions de ce projet
(NumPy docstrings, typing, réutilisation de ``optics.py``).

``airy_psf``, ``strehl_ratio``, ``radial_profile`` et ``encircled_energy``
délèguent à ``eoqual.core.psf`` (partagé avec eoqual-measure) ;
``strehl_ratio`` garde ici son avertissement via loguru. ``fwhm`` délègue
à ``eoqual.core.edge.fwhm`` — conçue pour une LSF, mais purement
générique (une coupe 1D avec un seul pic) : s'applique directement à une
coupe de PSF (voir :func:`mtf_cuts`, le même principe sur une FTM).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from eoqual.core import edge as core_edge
from eoqual.core import psf as core_psf


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
    return core_psf.airy_psf(shape, wavelength, diameter, pixel_scale)


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
    scale, and the same spectral band — see ``REFERENCE_TECHNIQUE.md``
    §9 for the full list of conditions and the Maréchal quality table.

    A ratio significantly above 1 is not physically possible for a
    genuine (aberrated or not) PSF against its own diffraction-limited
    reference: it signals one of these conditions was violated (most
    often a ``pixel_scale`` inconsistent with how ``psf_measured`` was
    actually produced) rather than an unusually good optic — a warning is
    logged in that case so the mismatch isn't mistaken for a result.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # signalé ci-dessous via loguru
        strehl = core_psf.strehl_ratio(psf_measured, psf_reference)
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
    return core_psf.radial_profile(psf)


def encircled_energy(
    psf: NDArray[np.float64],
    radii: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Énergie encerclée EE(r) : fraction de l'énergie totale contenue dans
    le disque de rayon ``r`` centré sur la PSF — délègue à
    ``eoqual.core.psf.encircled_energy``.

    Parameters
    ----------
    psf : NDArray
        PSF 2D centrée (le fond doit avoir été retiré : un fond résiduel
        fausse la normalisation).
    radii : NDArray, optional
        Rayons [px] où évaluer EE. ``None`` (défaut) :
        ``0,5, 1,5, ...`` jusqu'au bord du plus grand cercle inscrit.

    Returns
    -------
    radii, ee : NDArray, NDArray
        ``ee`` croissante, égale à ``1`` si le disque couvre toute la
        PSF.
    """
    return core_psf.encircled_energy(psf, radii)


def fwhm(x: NDArray[np.float64], cut: NDArray[np.float64]) -> float:
    """
    Largeur à mi-hauteur d'une coupe 1D de la PSF — délègue à
    ``eoqual.core.edge.fwhm`` (conçue pour une LSF, mais purement
    générique : n'importe quelle coupe 1D avec un seul pic).

    Parameters
    ----------
    x : NDArray
        Axe des abscisses de la coupe (ex. position en pixels, centrée
        ou non — seule l'échelle relative compte).
    cut : NDArray
        Coupe à mesurer (ex. coupe horizontale/verticale d'une PSF, voir
        :func:`mtf_cuts` pour l'équivalent sur une FTM).

    Returns
    -------
    float
        FWHM, dans les unités de ``x`` ; ``0.0`` si la coupe ne redescend
        pas sous la moitié de son maximum des deux côtés du pic.
    """
    return core_edge.fwhm(x, cut)


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


def mtf_frequency_axis(npix: int) -> NDArray[np.float64]:
    """
    Axe de fréquence réduite pour une FTM de taille ``npix`` (celle
    renvoyée par :func:`compute_mtf`), en fraction de la fréquence
    d'échantillonnage du détecteur ``fe`` (pas de la fréquence de
    Nyquist ``fe/2`` — voir la mise en garde ci-dessous).

    Parameters
    ----------
    npix : int
        Taille (pixels) du tableau PSF/FTM, carré.

    Returns
    -------
    NDArray
        Fréquence normalisée, longueur ``npix`` : ``0`` au centre (même
        convention de centrage que ``compute_mtf``, qui utilise
        ``fftshift``), ``+-0,5`` au bord du tableau. Le bord du tableau
        correspond toujours exactement à ``fe/2`` (Nyquist), par
        construction du modèle PSF = ``|FFT(pupille)|**2`` (une PSF de
        ``npix`` pixels échantillonne jusqu'à ``0,5`` cycle/pixel, quelle
        que soit l'ouverture) — **pas** à ``1,0`` : c'est la fréquence
        d'échantillonnage ``fe`` elle-même (toujours hors du tableau,
        jamais représentable par une FFT de signal réel) qui vaut ``1,0``
        dans cette convention.
    """
    return np.fft.fftshift(np.fft.fftfreq(npix))  # dans [-0.5, 0.5), déjà en fraction de fe


def mtf_cutoff_frequency(npix: int, aperture_px: Optional[int] = None) -> float:
    """
    Fréquence de coupure optique ``fc`` (diffraction), en fraction de la
    fréquence d'échantillonnage du détecteur ``fe`` — la fréquence
    spatiale au-delà de laquelle la FTM diffractive est exactement
    nulle.

    Vaut ``aperture_px / npix``. Justification : l'ouverture (pupille en
    amplitude) a un diamètre ``aperture_px`` pixels dans un cadre
    ``npix`` (voir :class:`eoqual.phaseretrieval.pupil.PupilGenerator`,
    ``OpticalConfig.aperture_px``, §12.5 de ``REFERENCE_TECHNIQUE.md``),
    donc un **rayon** ``aperture_px / 2``. La FTM = ``|FFT(PSF)|`` est
    l'autocorrélation de cette pupille, dont le support a pour **rayon**
    le double de celui de la pupille, soit ``aperture_px`` exactement
    (vérifié numériquement) — en unités de fréquence où le bord du
    tableau (distance ``npix / 2`` du centre) vaut ``fe/2`` (voir
    :func:`mtf_frequency_axis`). D'où
    ``fc / fe = (aperture_px / (npix / 2)) / 2 = aperture_px / npix``.

    Parameters
    ----------
    npix : int
        Taille (pixels) du tableau PSF/pupille, carré.
    aperture_px : int, optional
        Diamètre [px] de l'ouverture dans le cadre ``npix`` — typiquement
        le ``N_pup_target``/``aperture_px`` déjà utilisé ailleurs dans ce
        projet pour la base de Zernike (``io_utils.load_and_prepare_mask``,
        ``pupil.PupilGenerator``). ``None`` (défaut) : l'ouverture remplit
        le cadre (``aperture_px = npix``), donnant ``fc = 1.0`` — *hors*
        de l'intervalle ``[0, 0,5]`` de Nyquist (donc hors de ce qu'une
        FFT peut même représenter) : le modèle PSF brut, sans marge de
        suréchantillonnage, échantillonne en-deçà du critère de Nyquist
        pour la diffraction (``Q=1``, voir §12.1/§12.5 de
        ``REFERENCE_TECHNIQUE.md``) ; une marge ``aperture_px = npix/2``
        donnerait au contraire ``fc = 0,5`` (échantillonnage de Nyquist
        exact, ``Q=2`` : ``fc`` coïncide pile avec ``fe/2``).

    Returns
    -------
    float
        ``fc`` en fraction de ``fe`` (``0,5`` = exactement à Nyquist,
        ``1,0`` = à la fréquence d'échantillonnage elle-même).
    """
    if aperture_px is None:
        aperture_px = npix
    return aperture_px / npix


def mtf_cuts(mtf: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Coupes horizontale et verticale d'une FTM 2D, passant par son centre
    (fréquence nulle) — même convention de centrage que
    :func:`compute_mtf`/:func:`mtf_frequency_axis`.

    Parameters
    ----------
    mtf : NDArray
        FTM 2D, telle que renvoyée par :func:`compute_mtf`.

    Returns
    -------
    cut_h, cut_v : NDArray, NDArray
        Coupe horizontale (ligne centrale, le long de l'axe des
        fréquences en colonne) et verticale (colonne centrale), chacune
        de même longueur que :func:`mtf_frequency_axis` pour ce ``npix``.
    """
    ny, nx = mtf.shape
    return mtf[ny // 2, :], mtf[:, nx // 2]


def mtf_value_at(
    freq: NDArray[np.float64],
    cut: NDArray[np.float64],
    frequency: float,
) -> float:
    """
    Valeur interpolée d'une coupe de FTM à une fréquence réduite donnée
    (fraction de ``fe``, voir :func:`mtf_frequency_axis`) — ex.
    ``mtf_value_at(freq, cut_h, 0.25)`` pour la valeur à ``fe/4``.

    Interpolation linéaire (``numpy.interp``) : ``freq`` n'a besoin
    d'être trié que de façon croissante (c'est le cas pour la sortie de
    :func:`mtf_frequency_axis`).

    Parameters
    ----------
    freq : NDArray
        Axe de fréquence, voir :func:`mtf_frequency_axis`.
    cut : NDArray
        Coupe de FTM correspondante (même longueur), voir :func:`mtf_cuts`.
    frequency : float
        Fréquence réduite à laquelle évaluer la FTM (fraction de ``fe``).

    Returns
    -------
    float
        Valeur interpolée de la FTM à ``frequency``.
    """
    return float(np.interp(frequency, freq, cut))


def mtf_integral(
    freq: NDArray[np.float64],
    cut: NDArray[np.float64],
    f_max: float = 0.5,
) -> float:
    """
    Intégrale d'une coupe de FTM entre ``0`` et ``f_max`` (fraction de
    ``fe``, voir :func:`mtf_frequency_axis`) — règle des trapèzes.

    Une grandeur simple pour résumer "combien de contraste la FTM
    conserve en moyenne" sur une bande de fréquence, par ex. ``[0,
    fe/2]`` (défaut) pour tout le domaine représentable par le
    détecteur.

    Parameters
    ----------
    freq : NDArray
        Axe de fréquence, voir :func:`mtf_frequency_axis`.
    cut : NDArray
        Coupe de FTM correspondante (même longueur), voir :func:`mtf_cuts`.
    f_max : float, default 0.5
        Borne supérieure de l'intégrale (fraction de ``fe``) — ``0,5``
        (défaut) correspond à ``fe/2``, Nyquist.

    Returns
    -------
    float
        ``∫[0, f_max] FTM(f) df``, en unités de ``f_max`` (sans
        dimension ici, puisque ``f`` est déjà une fraction de ``fe``).
    """
    mask = (freq >= 0) & (freq <= f_max)
    return float(np.trapezoid(cut[mask], freq[mask]))


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
        :mod:`backends.optimization`, :mod:`backends.phase_diversity`)
        before drawing conclusions.
    """
    if metrics.first_ring_ratio > 0.25 and metrics.ellipticity < 0.05:
        return "Défocus dominant"
    if metrics.ellipticity > 0.15:
        return "Coma / astigmatisme dominant"
    if metrics.first_ring_ratio < 0.1 and metrics.ellipticity < 0.05:
        return "Sphérique dominant"
    return "Aberrations mixtes / bruit haute fréquence"
