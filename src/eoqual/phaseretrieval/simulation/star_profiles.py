#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation d'étoile par profil empirique (gaussienne, Moffat, Airy),
avec intégration correcte sur le pixel et assemblage Bayer RGGB.

Complète le simulateur physique (Zernike + FFT) de ``simu_psf_mask.ipynb`` :
là où celui-ci simule une PSF *diffraction-limitée avec aberrations
connues* (nécessaire pour valider une récupération de phase, puisque la vérité
terrain doit être exprimable en coefficients de Zernike), ce module simule
une étoile par un profil d'intensité empirique simple — utile pour tester
rapidement la mécanique de la mosaïque Bayer, un estimateur de décalage
inter-canaux, ou un pipeline de recentrage, sans dépendre d'un modèle de
front d'onde.

Portage, adapté au style de ce projet (NumPy docstrings, typing), du
prototype ``Star/star_simu_psf_bayer/star_simu_psf_bayer.ipynb``.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf, j1

ProfileType = Literal["gaussian", "moffat", "airy"]


def integrated_star_profile(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    profile: ProfileType,
    params: Dict[str, float],
    oversample: int = 5,
) -> NDArray[np.float32]:
    """
    Simule une étoile ponctuelle avec un profil empirique, intégré
    correctement sur la surface de chaque pixel (pas un simple
    échantillonnage ponctuel).

    Parameters
    ----------
    shape : tuple of int
        ``(H, W)`` taille de l'image.
    center : tuple of float
        ``(y0, x0)`` centre sub-pixel de l'étoile.
    profile : {"gaussian", "moffat", "airy"}
        Type de profil.

        - ``"gaussian"`` : intégration **exacte** par fonction d'erreur
          (``erf``) — pas de sur-échantillonnage nécessaire.
        - ``"moffat"``, ``"airy"`` : pas de primitive simple → intégration
          par sur-échantillonnage (grille ``oversample`` fois plus fine,
          moyennée par pixel).
    params : dict
        Paramètres du profil (voir Notes).
    oversample : int, default 5
        Facteur de sur-échantillonnage pour ``"moffat"``/``"airy"``.

    Returns
    -------
    NDArray
        Image ``(H, W)``, dtype float32.

    Notes
    -----
    Paramètres par profil (clé, valeur par défaut) :

    - ``"gaussian"`` : ``sigma=(sigma_y, sigma_x)=(1.5, 1.5)`` [px],
      ``amplitude=1.0``.
    - ``"moffat"`` : ``alpha=2.0`` [px] (échelle), ``beta=4.765``
      (forme — grand beta → gaussienne, petit beta → ailes larges,
      ``beta=4.765`` est la valeur usuelle pour la turbulence
      atmosphérique), ``amplitude=1.0``.
    - ``"airy"`` : ``radius_zero=2.0`` [px] (rayon du premier zéro
      d'Airy — voir :func:`airy_radius_from_wavelength` pour le relier à
      $\\lambda$, $D$ et la focale), ``amplitude=1.0``.
    """
    h, w = shape
    y0, x0 = center

    if profile == "gaussian":
        sigma_y, sigma_x = params.get("sigma", (1.5, 1.5))
        amplitude = params.get("amplitude", 1.0)
        y_grid, x_grid = np.indices((h, w))
        val_x = (
            erf((x_grid + 0.5 - x0) / (sigma_x * np.sqrt(2)))
            - erf((x_grid - 0.5 - x0) / (sigma_x * np.sqrt(2)))
        )
        val_y = (
            erf((y_grid + 0.5 - y0) / (sigma_y * np.sqrt(2)))
            - erf((y_grid - 0.5 - y0) / (sigma_y * np.sqrt(2)))
        )
        return (amplitude * 0.25 * val_x * val_y).astype(np.float32)

    y_sub, x_sub = np.mgrid[0:h:1 / oversample, 0:w:1 / oversample]
    offset = (1 / oversample) / 2.0
    y_sub = y_sub + offset
    x_sub = x_sub + offset
    r = np.sqrt((x_sub - x0) ** 2 + (y_sub - y0) ** 2)

    if profile == "moffat":
        alpha = params.get("alpha", 2.0)
        beta = params.get("beta", 4.765)
        amp = params.get("amplitude", 1.0)
        high_res = amp * (1 + r ** 2 / alpha ** 2) ** (-beta)

    elif profile == "airy":
        r_zero = params.get("radius_zero", 2.0)
        amp = params.get("amplitude", 1.0)
        k = 3.8317 / r_zero  # 1er zéro de J1 -> rayon = radius_zero
        x_val = k * r + 1e-8
        high_res = amp * (2 * j1(x_val) / x_val) ** 2

    else:
        raise ValueError(f"profile inconnu : {profile!r} (attendu gaussian/moffat/airy)")

    return high_res.reshape(h, oversample, w, oversample).mean(axis=(1, 3)).astype(np.float32)


def airy_radius_from_wavelength(
    wavelength_nm: float,
    diameter_m: float,
    focal_length_m: float,
    pixel_pitch_um: float,
) -> float:
    """
    Rayon (en pixels) du premier zéro d'Airy pour une bande donnée.

    Permet de dériver ``radius_zero`` (pour :func:`integrated_star_profile`,
    profil ``"airy"``) de façon physique plutôt qu'à la main — et donc de
    faire varier correctement la taille de la tache avec $\\lambda$ entre
    bandes d'un même capteur Bayer (voir :func:`simulate_bayer_star`).

    Parameters
    ----------
    wavelength_nm : float
        Longueur d'onde [nm].
    diameter_m : float
        Diamètre de la pupille [m].
    focal_length_m : float
        Focale effective [m].
    pixel_pitch_um : float
        Taille d'un pixel détecteur [µm].

    Returns
    -------
    float
        Rayon du premier zéro d'Airy, en pixels détecteur
        (``1.22 * lambda / D * focale / pixel_pitch``).
    """
    theta_airy = 1.22 * (wavelength_nm * 1e-9) / diameter_m
    pixel_scale = (pixel_pitch_um * 1e-6) / focal_length_m
    return theta_airy / pixel_scale


def psf_from_pupil_field_wavelength_scaled(
    pupil_field: NDArray[np.complex128],
    wavelength_ratio: float,
) -> NDArray[np.float64]:
    """
    Compute a diffraction PSF whose core size correctly scales with
    wavelength, from a complex pupil field defined on a fixed reference
    grid — unlike a bare ``|FFT(pupil_field)|**2`` on that same grid, whose
    core size in pixels is, by construction, **independent** of the
    wavelength used to build ``pupil_field``'s phase (the λ's cancel out
    of the discrete-FFT pixel scale).

    Parameters
    ----------
    pupil_field : NDArray (complex)
        Complex pupil field ``mask * exp(i * phase_rad)`` on an
        ``(n_ref, n_ref)`` grid, with ``phase_rad`` already computed for
        the *actual* band wavelength (e.g.
        ``optics.phase_from_coefficients(coeffs_nm / rad_to_nm_band, ...)``).
        The illuminated pupil (mask) footprint, in pixels, must be the
        *same* across bands (same physical aperture ``D``).
    wavelength_ratio : float
        ``wavelength_band / wavelength_reference``. ``1.0`` reproduces the
        plain (uncorrected) ``|FFT|**2`` behaviour exactly.

    Returns
    -------
    NDArray
        PSF, shape ``(n_ref, n_ref)`` (same grid as the input), normalised
        to unit sum. Its diffraction core now spans
        ``wavelength_ratio`` times as many pixels as the reference band's,
        matching the real λ ∝ (core size in pixels) relationship at a
        *fixed, physical* detector pixel scale.

    Notes
    -----
    Technique: resample ``pupil_field`` to ``n = round(n_ref * wavelength_ratio)``
    pixels **before** the FFT — zero-padding (``n > n_ref``, redder band:
    same pupil sampling, more surrounding zeros) or cropping (``n < n_ref``,
    bluer band: same pupil sampling, less surrounding zeros; the pupil
    itself, of fixed radius, must fit within ``n`` — see ``Raises``) — then
    resample the FFT **output** back onto the common ``n_ref × n_ref`` grid
    the opposite way (crop if padded, zero-pad if cropped). Padding the
    input refines the FFT's angular pixel scale exactly enough to
    compensate the larger physical Airy radius at longer wavelengths, and
    symmetrically for cropping at shorter ones — see
    ``REFERENCE_TECHNIQUE.md`` §6 (note sur le simulateur Bayer)
    for the derivation.

    Raises
    ------
    ValueError
        If ``n < n_ref`` (bluer band) and the pupil's illuminated extent
        does not fit within an ``n × n`` crop — i.e. ``wavelength_ratio``
        is too small for the pupil's footprint on ``pupil_field``'s grid.
    """
    n_ref = pupil_field.shape[0]
    n = int(round(n_ref * wavelength_ratio))

    if n >= n_ref:
        # Redder band: pad the input (finer FFT pixel scale), crop the output.
        resampled_input = np.zeros((n, n), dtype=np.complex128)
        off = (n - n_ref) // 2
        resampled_input[off:off + n_ref, off:off + n_ref] = pupil_field
    else:
        # Bluer band: crop the input (coarser FFT pixel scale), pad the output.
        illuminated = np.abs(pupil_field) > 0
        ys, xs = np.nonzero(illuminated)
        cy = (n_ref - 1) / 2.0
        margin = max(np.abs(ys - cy).max(), np.abs(xs - cy).max()) if ys.size else 0.0
        if margin > n / 2.0:
            raise ValueError(
                f"wavelength_ratio={wavelength_ratio:.3f} trop petit : la pupille "
                f"(demi-étendue {margin:.1f} px) ne tient pas dans un recadrage "
                f"de {n} px. Réduire l'écart de longueur d'onde, ou marger "
                "davantage la pupille dans pupil_field (mask plus petit que la grille)."
            )
        c = n_ref // 2
        h = n // 2
        resampled_input = pupil_field[c - h:c - h + n, c - h:c - h + n]

    focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(resampled_input)))
    psf_n = np.abs(focal) ** 2

    if n >= n_ref:
        c = n // 2
        h = n_ref // 2
        psf = psf_n[c - h:c - h + n_ref, c - h:c - h + n_ref]
    else:
        psf = np.zeros((n_ref, n_ref), dtype=np.float64)
        off = (n_ref - n) // 2
        psf[off:off + n, off:off + n] = psf_n

    return psf / psf.sum()


def simulate_bayer_star(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    shifts_px: Dict[str, Tuple[float, float]],
    profile: ProfileType,
    params_per_band: Dict[str, Dict[str, float]],
    oversample: int = 5,
) -> NDArray[np.float32]:
    """
    Simule une image Bayer RGGB brute d'une étoile, avec un décalage
    chromatique connu et un profil (taille) potentiellement différent par
    bande.

    Parameters
    ----------
    shape : tuple of int
        ``(H, W)`` taille de l'image Bayer.
    center : tuple of float
        ``(y0, x0)`` centre nominal (bande de référence).
    shifts_px : dict[str, tuple of float]
        Décalage ``(dy, dx)`` par bande ``"R"``/``"G"``/``"B"``, relatif à
        ``center`` (typiquement ``(0, 0)`` pour la bande de référence).
    profile : {"gaussian", "moffat", "airy"}
        Profil d'intensité, commun aux trois bandes (seuls les paramètres
        varient — voir :func:`integrated_star_profile`).
    params_per_band : dict[str, dict]
        Paramètres du profil par bande ``{"R": {...}, "G": {...}, "B": {...}}``.
    oversample : int, default 5
        Voir :func:`integrated_star_profile`.

    Returns
    -------
    NDArray
        Image Bayer RGGB brute, forme ``shape``, dtype float32. Convention
        de motif identique à ``starpsfsr.telescope.BayerExtractor`` :
        ``(0,0)=R``, ``(0,1)=Gu``, ``(1,0)=Gl``, ``(1,1)=B``.
    """
    raw = np.zeros(shape, dtype=np.float32)

    mask_r = np.zeros(shape, dtype=bool)
    mask_r[0::2, 0::2] = True
    mask_g = np.zeros(shape, dtype=bool)
    mask_g[0::2, 1::2] = True
    mask_g[1::2, 0::2] = True
    mask_b = np.zeros(shape, dtype=bool)
    mask_b[1::2, 1::2] = True
    masks = {"R": mask_r, "G": mask_g, "B": mask_b}

    for band in ("R", "G", "B"):
        dy, dx = shifts_px.get(band, (0.0, 0.0))
        band_center = (center[0] + dy, center[1] + dx)
        img_band = integrated_star_profile(
            shape, band_center, profile, params_per_band[band], oversample=oversample
        )
        raw[masks[band]] = img_band[masks[band]]

    return raw
