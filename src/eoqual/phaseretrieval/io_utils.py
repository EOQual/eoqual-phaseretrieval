#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E/S et visualisation communes à toutes les méthodes de reconstruction.

Ce module regroupe les fonctions indépendantes de l'algorithme de
récupération de phase choisie (Gerchberg-Saxton, optimisation directe ou
diversité de phase) : chargement d'images, préparation du masque
pupillaire et diagnostics graphiques Plotly. La mécanique Plotly
générique (heatmap 2D, comparaison de courbes) vit dans ``eoqual-viz`` ;
ce module n'en garde que les enveloppes spécifiques à ce projet
(``plot_mtf_cuts``, ``visualisation_diagnostics``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from eoqual.viz import plot_array
from loguru import logger
from numpy.typing import NDArray
from skimage.transform import resize


def load_image(path: Union[str, Path]) -> NDArray[np.float64]:
    """
    Load an image file (any format supported by ``imageio``).

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    img : ndarray, shape (N, N)
        2-D greyscale image (values as stored in the file, not renormalised).
    """
    import imageio.v3 as iio

    rep_path = Path(path).parent
    if not rep_path.exists():
        raise FileNotFoundError(f"Le répertoire '{rep_path}' n'existe pas.")
    if not rep_path.is_dir():
        raise NotADirectoryError(f"Le chemin '{rep_path}' n'est pas un répertoire.")

    img = np.asarray(iio.imread(path), dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def compute_pupil_footprint_px(
    npix: int,
    wavelength: float,
    focal_length: float,
    pupil_diameter: float,
    pixel_pitch: float,
    oversampling: int = 1,
) -> int:
    """
    Number of ``npix``-grid pixels the aperture must span for the PSF
    produced by :func:`optics.psf_from_phase` to be correctly sampled.

    Shared by :func:`load_and_prepare_mask` and
    :func:`load_and_prepare_ho_wfe`, so a pupil mask and an HO WFE map
    coming from *different* native resolutions still end up registered on
    the exact same footprint within the working ``npix`` grid.

    Parameters
    ----------
    npix : int
        Size (pixels) of the working grid — the PSF array's own size,
        **whatever its native or oversampled resolution actually is**
        (see ``oversampling``).
    wavelength : float
        Central wavelength **in metres**.
    focal_length : float
        Focal length ``f`` **in metres**.
    pupil_diameter : float
        Physical pupil diameter ``D`` **in metres**.
    pixel_pitch : float
        **Native** detector pixel pitch **in metres** — a fixed property
        of the sensor, independent of ``oversampling``.
    oversampling : int, default 1
        Oversampling factor of the PSF/mask/HO-map grid relative to native
        detector sampling — e.g. ``4`` for a PSF reconstructed by
        super-resolution (Shift-and-Add, ePSF, PSFr — see §5.0 of
        ``REFERENCE_TECHNIQUE.md``) at 4x the native pixel pitch. ``1``
        (default) means ``npix`` already counts native detector pixels.
        The *effective* pixel pitch on the working grid is
        ``pixel_pitch / oversampling``.

        Counter-intuitive but correct: when ``npix`` itself scales with
        ``oversampling`` (the normal case — you simply read ``npix`` off
        an oversampled PSF array that already has ``oversampling`` times
        as many samples as its native version), ``N_pup_target`` comes out
        **independent of** ``oversampling`` — the ``k``'s cancel exactly.
        Physically: oversampling a PSF through this single-FFT model means
        zero-padding the pupil array *around* the aperture (more
        background, same aperture resolution in pixels), not resolving
        the aperture itself more finely — exactly the standard Fourier-
        optics zero-padding recipe for a finer output sampling. What
        *does* grow with oversampling is the padding margin,
        ``npix - N_pup_target``.

    Returns
    -------
    int
        ``N_pup_target``, the number of ``npix``-grid pixels the aperture
        spans.

    Notes
    -----
    .. math::
        N_{pup}=\\frac{\\lambda f}{D\\,\\Delta_{det}}\\quad\\text{with}\\quad
        \\Delta_{det}=\\frac{\\text{pixel\\_pitch}}{\\text{oversampling}}

    Raises
    ------
    ValueError
        If the computed footprint exceeds ``npix`` (would need to zero-pad
        the PSF, or increase ``oversampling``/``pixel_pitch`` to shrink it).
    """
    effective_pixel_pitch = pixel_pitch / oversampling
    airy_pixels = (wavelength * focal_length) / (pupil_diameter * effective_pixel_pitch)
    n_pup_target = int(np.round(npix / airy_pixels))

    if n_pup_target > npix:
        raise ValueError(
            f"Required pupil diameter ({n_pup_target} px) exceeds the grid "
            f"size ({npix} px). Pad the PSF, increase oversampling, or "
            "check pixel_pitch/wavelength/focal_length/pupil_diameter."
        )
    return n_pup_target


def _detect_aperture_bbox(
    raw: NDArray[np.float64],
    threshold_frac: float,
) -> Tuple[int, int, int, int]:
    """
    Bounding box (row_min, row_max, col_min, col_max — inclusive) of the
    pixels above ``threshold_frac * raw.max()`` in ``raw``.

    Parameters
    ----------
    raw : NDArray
        Source array.
    threshold_frac : float
        Fraction of ``raw.max()`` above which a pixel counts as
        "illuminated". Small (not exactly 0) on purpose: tolerates the
        near-zero-but-not-quite background of a real file (compression
        artefacts, a graded/apodised mask's own faint tail) without
        being fooled by them into including the whole array.

    Returns
    -------
    tuple of int
        ``(row_min, row_max, col_min, col_max)``.

    Raises
    ------
    ValueError
        If no pixel is above the threshold.
    """
    lit = raw > (threshold_frac * raw.max())
    rows, cols = np.any(lit, axis=1), np.any(lit, axis=0)
    if not rows.any():
        raise ValueError(
            f"Aucun pixel au-dessus de {threshold_frac:.1%} du maximum "
            f"({raw.max():.3g}) — impossible de détecter l'ouverture "
            "automatiquement. Réduire aperture_threshold, ou fournir "
            "explicitement native_aperture_px."
        )
    row_min, row_max = int(np.flatnonzero(rows)[0]), int(np.flatnonzero(rows)[-1])
    col_min, col_max = int(np.flatnonzero(cols)[0]), int(np.flatnonzero(cols)[-1])
    return row_min, row_max, col_min, col_max


def _crop_to_aperture(
    raw: NDArray[np.float64],
    *,
    native_aperture_px: Optional[int],
    aperture_threshold: float,
    native_aperture_center: Optional[Tuple[int, int]] = None,
) -> NDArray[np.float64]:
    """
    Crop ``raw`` down to just its illuminated aperture, stripping any
    zero (or near-zero) margin around it — a no-op when the aperture
    already fills the whole array (the assumption
    :func:`_resample_to_footprint` otherwise makes on its input), so this
    is always safe to call before it.

    Parameters
    ----------
    raw : NDArray
        Source array (mask or HO WFE map), at its own native resolution.
    native_aperture_px : int, optional
        If given, the aperture is known to span exactly this many pixels
        — skips auto-detection entirely (the robust choice when the
        source file's own convention is known, e.g. documented by the
        tool that produced it). ``None`` (default): auto-detect the
        aperture's bounding box from the pixel content (see
        ``aperture_threshold``), then crop a square centred on that
        bounding box's centre, sized to its larger dimension — this path
        already handles an **off-centre** aperture correctly on its own,
        wherever it actually sits in the file.
    aperture_threshold : float
        Only used when ``native_aperture_px is None`` — see
        :func:`_detect_aperture_bbox`.
    native_aperture_center : tuple of (int, int), optional
        Only used when ``native_aperture_px`` is given: ``(row, col)``
        pixel coordinates of the aperture's centre within ``raw``, if it
        is **not** centred in the file (unequal margins on opposite
        sides). ``None`` (default): assume it *is* centred, i.e.
        ``raw.shape[i] // 2`` on each axis — margins equal on opposite
        sides. Has no effect when ``native_aperture_px is None``: the
        auto-detected bounding box already locates an off-centre aperture
        on its own, no need to say where it is.

    Returns
    -------
    NDArray
        The cropped, square aperture sub-image.

    Raises
    ------
    ValueError
        If the requested/detected aperture window does not fit within
        ``raw`` (off-centre content, or ``native_aperture_px`` too large).
    """
    if native_aperture_px is None:
        row_min, row_max, col_min, col_max = _detect_aperture_bbox(raw, aperture_threshold)
        size = max(row_max - row_min + 1, col_max - col_min + 1)
        if size > raw.shape[0] or size > raw.shape[1]:
            raise ValueError(
                f"Ouverture détectée ({size}x{size} px) plus grande que le "
                f"fichier source ({raw.shape[0]}x{raw.shape[1]}) — impossible."
            )
        # Fenêtre carrée de ce côté, centrée sur la boîte englobante
        # détectée, puis **décalée** (jamais déformée) pour rester dans le
        # fichier — au lieu de dériver le coin à partir d'un centre
        # arrondi (piège classique de parité : une ouverture inscrite qui
        # touche déjà les 4 bords, taille paire, donnait un centre décalé
        # d'un demi-pixel et donc une fenêtre hors limites alors que le
        # fichier entier convient tel quel).
        r0 = min(max(row_min - (size - (row_max - row_min + 1)) // 2, 0), raw.shape[0] - size)
        c0 = min(max(col_min - (size - (col_max - col_min + 1)) // 2, 0), raw.shape[1] - size)
    else:
        size = native_aperture_px
        if size > raw.shape[0] or size > raw.shape[1]:
            raise ValueError(
                f"native_aperture_px={size} plus grand que le fichier source "
                f"({raw.shape[0]}x{raw.shape[1]}) — impossible."
            )
        if native_aperture_center is None:
            r0 = (raw.shape[0] - size) // 2
            c0 = (raw.shape[1] - size) // 2
        else:
            centre_row, centre_col = native_aperture_center
            half = size // 2
            r0, c0 = centre_row - half, centre_col - half
            if r0 < 0 or c0 < 0 or r0 + size > raw.shape[0] or c0 + size > raw.shape[1]:
                raise ValueError(
                    f"Ouverture ({size}x{size} px, centrée en "
                    f"{native_aperture_center}) hors des limites du fichier "
                    f"source ({raw.shape[0]}x{raw.shape[1]}) — vérifier "
                    "native_aperture_center (et native_aperture_px)."
                )

    return raw[r0:r0 + size, c0:c0 + size]


def _resample_to_footprint(
    raw: NDArray[np.float64],
    npix: int,
    n_pup_target: int,
    *,
    order: int,
    anti_aliasing: bool,
) -> NDArray[np.float64]:
    """
    Resize ``raw`` (its **whole** array extent taken as the aperture
    footprint, edge to edge — the convention used by
    :class:`pupil.PupilGenerator`, and what ``raw`` has already been
    reduced to by :func:`_crop_to_aperture` upstream in both callers, even
    when the *source file* had a zero margin around the aperture) to
    ``n_pup_target`` x ``n_pup_target``, then zero-pad/centre it into an
    ``npix`` x ``npix`` canvas.

    Shared resampling step for both a pupil mask (see
    ``load_and_prepare_mask``) and an HO WFE map (see
    ``load_and_prepare_ho_wfe``) — only the interpolation ``order`` and
    whether the caller re-binarises afterwards differ between the two.

    Parameters
    ----------
    raw : NDArray
        Source array at its own native resolution.
    npix : int
        Size of the output (working-grid) canvas.
    n_pup_target : int
        Size the aperture footprint must occupy within that canvas — see
        :func:`compute_pupil_footprint_px`.
    order : int
        ``skimage.transform.resize`` interpolation order — ``0``
        (nearest) for a strictly binary mask with no anti-aliasing
        desired, ``1`` (bilinear) for a mildly graded mask, ``3``
        (bicubic) for a smooth, continuous quantity such as a WFE map.
    anti_aliasing : bool
        Passed through to ``skimage.transform.resize`` — should be
        ``True`` whenever downsampling (``raw.shape[0] > n_pup_target``)
        to avoid aliasing fine structure (e.g. spider arms) into noise;
        harmless otherwise.

    Returns
    -------
    NDArray
        ``(npix, npix)`` array, ``float64``.
    """
    if raw.shape == (n_pup_target, n_pup_target):
        resampled = raw.astype(np.float64)
    else:
        resampled = resize(
            raw, output_shape=(n_pup_target, n_pup_target),
            order=order, anti_aliasing=anti_aliasing, preserve_range=True,
        )

    canvas = np.zeros((npix, npix), dtype=np.float64)
    off = (npix - n_pup_target) // 2
    canvas[off:off + n_pup_target, off:off + n_pup_target] = resampled
    return canvas


def load_and_prepare_mask(
    psf_shape: Tuple[int, int],
    mask_path: Union[str, Path],
    wavelength: float,
    focal_length: float,
    pupil_diameter: float,
    pixel_pitch: float,
    oversampling: int = 1,
    *,
    binarize: bool = True,
    native_aperture_px: Optional[int] = None,
    native_aperture_center: Optional[Tuple[int, int]] = None,
    aperture_threshold: float = 0.01,
) -> Tuple[NDArray[np.float64], Optional[int]]:
    """
    Load a pupil mask, resize it to the physical pupil size that
    corresponds to the supplied PSF, and embed it (zero-pad) into an array
    of the same size as the PSF.

    If the raw mask image is *already* the same shape as the PSF, it is
    assumed to be at the same **resolution** (e.g. produced by a
    simulator, or already at the correct oversampled resolution) — no
    physical rescale is applied, this is the normal (and only tested) path
    for masks built alongside their PSF on the same grid. It is *not*,
    however, assumed to already be tightly cropped on its aperture: a zero
    (or near-zero) margin is still auto-detected and stripped/re-centred
    exactly as in the differing-resolution case below (``native_aperture_px``/
    ``aperture_threshold`` apply here too) — same array size as the PSF
    means same resolution, not necessarily "aperture already fills the
    frame". See §12 of ``REFERENCE_TECHNIQUE.md`` for the general
    multi-resolution case (mask/PSF/HO-map each at their own native
    resolution) this function is designed for otherwise.

    Parameters
    ----------
    psf_shape : tuple[int, int]
        Shape of the PSF image (must be square, e.g. ``(N_psf, N_psf)``).
    mask_path : str or pathlib.Path
        Path to the raw mask image (binary or grayscale), at **its own**
        native resolution — unrelated to ``psf_shape``. The aperture need
        *not* fill the whole image edge to edge: a zero (or near-zero)
        margin around it is auto-detected and stripped before the
        physical resize — see ``native_aperture_px``/``aperture_threshold``
        and §12.3 of ``REFERENCE_TECHNIQUE.md``.
    wavelength, focal_length, pupil_diameter, pixel_pitch, oversampling
        See :func:`compute_pupil_footprint_px`.
    native_aperture_px : int, optional
        If the aperture's own pixel span within ``mask_path`` is known
        (documented by whoever produced the file), pass it here for a
        robust, deterministic crop instead of relying on
        ``aperture_threshold`` auto-detection — see
        :func:`_crop_to_aperture`. ``None`` (default): auto-detect.
    native_aperture_center : tuple of (int, int), optional
        Only used when ``native_aperture_px`` is given: ``(row, col)``
        pixel coordinates of the aperture's centre within ``mask_path``,
        if it is **not** centred in the file (unequal margins on opposite
        sides). ``None`` (default): assume it *is* centred — equal
        margins on opposite sides. Auto-detection (``native_aperture_px
        is None``) locates an off-centre aperture on its own already, no
        need for this.
    aperture_threshold : float, default 0.01
        Only used when ``native_aperture_px`` is ``None``: fraction of the
        mask's own maximum value above which a pixel counts as part of
        the aperture, for auto-detecting its bounding box. Irrelevant
        when the aperture already fills the whole file (the box then
        spans the full array — a no-op crop).
    binarize : bool, default True
        If True (default, historical behaviour), threshold the result at
        ``0.5`` to a strict binary mask after resampling. If False, keep
        the graded ``[0, 1]`` transmission produced by the (anti-aliased)
        resize — preserves sub-pixel edge information instead of
        collapsing it to a hard boundary; also the right choice for a
        genuinely apodised (non-binary) pupil. Every method in this
        project accepts a graded mask (:class:`pupil.PupilGenerator`
        already documents this) — binary is a convention, not a
        requirement.

    Returns
    -------
    mask : ndarray, shape (N_psf, N_psf), dtype float64
        Mask centred in the field of view, values in ``{0, 1}`` if
        ``binarize=True`` (default) or ``[0, 1]`` otherwise.
    N_pup_target : int or None
        Number of pixels the aperture actually spans within ``mask`` —
        either after the physical rescale, or after cropping/re-centring a
        margin found despite the mask already being on the working grid.
        ``None`` only when the mask was already on the working grid *and*
        its aperture already fills the whole frame (no margin, nothing to
        correct) — the historical, still most common case.

    Raises
    ------
    ValueError
        If the PSF is not square, or if the required pupil diameter in
        pixels is larger than the PSF matrix — see
        :func:`compute_pupil_footprint_px`.
    """
    N_psf = psf_shape[0]  # On suppose une matrice carrée (N_psf, N_psf)
    if psf_shape[0] != psf_shape[1]:
        raise ValueError(f"PSF must be square – received shape {psf_shape}")

    mask_raw = load_image(mask_path)

    # Le masque simulé/pré-enregistré est déjà sur la grille de travail :
    # aucune mise à l'échelle physique n'est nécessaire. Reste à vérifier
    # qu'il n'a pas, lui non plus, une marge de zéros autour de l'ouverture
    # (même taille de tableau que la PSF n'implique pas que l'ouverture en
    # remplit tout le cadre — §12.3 s'applique aussi ici, pas seulement
    # quand la résolution native diffère).
    if mask_raw.shape == psf_shape:
        mask_aperture = _crop_to_aperture(
            mask_raw, native_aperture_px=native_aperture_px,
            aperture_threshold=aperture_threshold, native_aperture_center=native_aperture_center,
        )
        if mask_aperture.shape == mask_raw.shape:
            # Aucune marge détectée : l'ouverture remplit déjà le cadre —
            # comportement historique inchangé.
            mask = (mask_raw > 0.5).astype(np.float64) if binarize else mask_raw.astype(np.float64)
            logger.warning(
                f"Masque déjà sur la grille de la PSF : ouverture supposée pleine cadre "
                f"({mask_raw.shape[0]}x{mask_raw.shape[1]} px) — mise à l'échelle physique "
                "ignorée, vérifier que c'est bien voulu si le masque vient d'un autre outil "
                "(voir REFERENCE_TECHNIQUE.md §12)."
            )
            return mask, None

        # Marge détectée malgré une taille de tableau déjà égale à celle de
        # la PSF : recadrer/recentrer l'ouverture (aucun changement de
        # résolution par pixel nécessaire, donc pas d'interpolation —
        # _resample_to_footprint le constate lui-même et se contente d'un
        # padding centré), pour que la base de Zernike connaisse la vraie
        # taille de l'ouverture (aperture_px) au lieu de supposer qu'elle
        # remplit tout le cadre (§12.5).
        n_pup_target = mask_aperture.shape[0]
        mask = _resample_to_footprint(mask_aperture, N_psf, n_pup_target, order=1, anti_aliasing=True)
        if binarize:
            mask = (mask > 0.5).astype(np.float64)
        logger.warning(
            f"Masque sur la grille de la PSF ({mask_raw.shape[0]}x{mask_raw.shape[1]}) mais "
            f"ouverture détectée plus petite que le cadre ({n_pup_target}x{n_pup_target} px) : "
            "recadrée/recentrée sans changement de résolution (voir "
            "REFERENCE_TECHNIQUE.md §12.3/§12.5)."
        )
        return mask, n_pup_target

    n_pup_target = compute_pupil_footprint_px(
        N_psf, wavelength, focal_length, pupil_diameter, pixel_pitch, oversampling,
    )
    logger.info(
        f"Ouverture estimée à {n_pup_target}x{n_pup_target} px sur la grille de la PSF "
        f"({N_psf}x{N_psf}) — masque natif {mask_raw.shape[0]}x{mask_raw.shape[1]} px."
    )

    # Marge éventuelle (zéros autour de l'ouverture) retirée avant la mise
    # à l'échelle physique — sinon cette marge serait comprimée avec
    # l'ouverture elle-même dans n_pup_target au lieu d'être ignorée.
    mask_aperture = _crop_to_aperture(
        mask_raw, native_aperture_px=native_aperture_px, aperture_threshold=aperture_threshold,
        native_aperture_center=native_aperture_center,
    )

    # Interpolation bilinéaire + anti-aliasing (plutôt que plus-proche-voisin
    # sans anti-aliasing) : réduit les artefacts d'aliasing sur les
    # structures fines (bras d'araignée) lors d'un sous-échantillonnage,
    # avant un éventuel seuillage binaire.
    mask = _resample_to_footprint(mask_aperture, N_psf, n_pup_target, order=1, anti_aliasing=True)
    if binarize:
        mask = (mask > 0.5).astype(np.float64)

    logger.debug(f"Mask sum (number of illuminated pixels): {float(mask.sum()):.1f}")
    logger.debug(f"Mask shape:   {mask.shape}")
    logger.debug(f"N_pup_target: {n_pup_target}")

    return mask, n_pup_target


def load_and_prepare_ho_wfe(
    psf_shape: Tuple[int, int],
    ho_wfe_path: Union[str, Path],
    wavelength: float,
    focal_length: float,
    pupil_diameter: float,
    pixel_pitch: float,
    oversampling: int = 1,
    *,
    native_aperture_px: Optional[int] = None,
    native_aperture_center: Optional[Tuple[int, int]] = None,
    aperture_threshold: float = 0.01,
) -> NDArray[np.float64]:
    """
    Load a known, fixed high-order (HO) WFE map [nm] at its own native
    resolution, and register it onto the working grid — the same
    footprint computation as :func:`load_and_prepare_mask`, so the two
    end up co-registered even when their source files have different
    native resolutions (e.g. a metrology tool's own grid vs. the pupil
    mask's).

    Unlike a pupil mask, a WFE map is a smooth, continuous physical
    quantity — never binarised, and resampled with cubic interpolation
    (``order=3``) rather than bilinear, for the least lossy reconstruction
    of a smooth surface. See ``known_ho_wfe_nm`` on
    :class:`backends.optimization.OpticalConfig`/
    :class:`backends.phase_diversity.PhaseDiversityConfig` and §11.3/§12 of
    ``REFERENCE_TECHNIQUE.md``.

    Parameters
    ----------
    psf_shape : tuple[int, int]
        Shape of the PSF image (must be square).
    ho_wfe_path : str or pathlib.Path
        Path to the raw WFE map [nm], at its own native resolution. As
        for the mask, need not fill the whole file edge to edge — a zero
        margin around it is stripped the same way, see
        ``native_aperture_px``.
    wavelength, focal_length, pupil_diameter, pixel_pitch, oversampling
        See :func:`compute_pupil_footprint_px`. Must match the values
        used for the accompanying :func:`load_and_prepare_mask` call, so
        both land on the same footprint.
    native_aperture_px : int, optional
        See :func:`load_and_prepare_mask`. Recommended over the default
        auto-detection for a WFE map specifically: unlike a binary mask,
        a smooth WFE surface can legitimately be near-zero within part of
        the aperture too (not just outside it), which auto-detection
        cannot always tell apart from padding.
    native_aperture_center : tuple of (int, int), optional
        See :func:`load_and_prepare_mask`. Only used when
        ``native_aperture_px`` is given.
    aperture_threshold : float, default 0.01
        See :func:`load_and_prepare_mask`. Only used when
        ``native_aperture_px`` is ``None``.

    Returns
    -------
    NDArray
        WFE map [nm], shape ``psf_shape``, registered on the working grid
        (zero outside the aperture footprint).

    Raises
    ------
    ValueError
        If the PSF is not square, or if the required footprint exceeds
        the grid size — see :func:`compute_pupil_footprint_px`.
    """
    N_psf = psf_shape[0]
    if psf_shape[0] != psf_shape[1]:
        raise ValueError(f"PSF must be square – received shape {psf_shape}")

    ho_wfe_raw = load_image(ho_wfe_path)

    if ho_wfe_raw.shape == psf_shape:
        logger.warning(
            f"Carte HO déjà sur la grille de la PSF : ouverture supposée pleine cadre "
            f"({ho_wfe_raw.shape[0]}x{ho_wfe_raw.shape[1]} px) — mise à l'échelle physique "
            "ignorée, vérifier que c'est bien voulu (voir REFERENCE_TECHNIQUE.md §12)."
        )
        return ho_wfe_raw.astype(np.float64)

    n_pup_target = compute_pupil_footprint_px(
        N_psf, wavelength, focal_length, pupil_diameter, pixel_pitch, oversampling,
    )
    logger.info(
        f"Carte HO : ouverture estimée à {n_pup_target}x{n_pup_target} px sur la grille de "
        f"la PSF ({N_psf}x{N_psf}) — fichier natif {ho_wfe_raw.shape[0]}x{ho_wfe_raw.shape[1]} px."
    )
    ho_wfe_aperture = _crop_to_aperture(
        ho_wfe_raw, native_aperture_px=native_aperture_px, aperture_threshold=aperture_threshold,
        native_aperture_center=native_aperture_center,
    )
    return _resample_to_footprint(ho_wfe_aperture, N_psf, n_pup_target, order=3, anti_aliasing=True)


def plot_mtf_cuts(
    mtf_estimated: NDArray[np.float64],
    *,
    mtf_reference: Optional[NDArray[np.float64]] = None,
    fc: Optional[float] = None,
    title: str = "FTM — coupes H/V",
) -> go.Figure:
    """
    Compare les coupes horizontale et verticale d'une FTM estimée (et,
    si disponible, d'une FTM de référence), avec un axe de fréquence en
    fraction de ``fe`` (fréquence d'échantillonnage du détecteur — pas
    ``fe/2`` : voir :func:`diagnostics.mtf_frequency_axis`). Repères
    verticaux : ``fe/2`` (Nyquist, toujours à ``0,5`` dans cette
    convention) en bleu, ``fc`` (fréquence de coupure optique, si
    fournie) en rouge — voir :mod:`diagnostics` (``compute_mtf``,
    ``mtf_frequency_axis``, ``mtf_cuts``, ``mtf_cutoff_frequency``).

    Fine enveloppe spécifique FTM (extraction des coupes, étiquettes,
    fréquence de coupure) au-dessus de
    :func:`eoqual.viz.plot_curve_grid`, qui porte la mécanique Plotly
    générique (voir ``eoqual-viz``, extrait de ce module).

    Seules les fréquences positives sont tracées (convention usuelle pour
    une FTM — ``MTF(-f) == MTF(f)`` pour une PSF réelle, §9 de
    ``REFERENCE_TECHNIQUE.md``).

    Parameters
    ----------
    mtf_estimated : NDArray
        FTM 2D estimée (ex. depuis une PSF reconstruite à partir des
        coefficients de Zernike récupérés) — voir
        :func:`diagnostics.compute_mtf`.
    mtf_reference : NDArray, optional
        FTM 2D de référence (ex. depuis une PSF connue/simulée), même
        forme que ``mtf_estimated``. ``None`` (défaut) : seule l'estimée
        est tracée (pas de vérité terrain disponible).
    fc : float, optional
        Fréquence de coupure optique, en fraction de ``fe`` — voir
        :func:`diagnostics.mtf_cutoff_frequency`. Tracée en repère
        vertical rouge sur les deux coupes si fournie. ``None`` (défaut) :
        paramètres physiques non disponibles, pas de repère tracé.
    title : str
        Titre de la figure.

    Returns
    -------
    go.Figure
        Deux sous-graphes (coupe horizontale, coupe verticale), prêts
        pour ``.show()``.
    """
    from eoqual.viz import plot_curve_grid

    from .diagnostics import mtf_cuts, mtf_frequency_axis

    npix = mtf_estimated.shape[0]
    freq = mtf_frequency_axis(npix)
    positive = freq >= 0  # FTM symétrique : ne tracer que f >= 0 (convention usuelle)

    cut_h_est, cut_v_est = mtf_cuts(mtf_estimated)

    references = None
    if mtf_reference is not None:
        cut_h_ref, cut_v_ref = mtf_cuts(mtf_reference)
        references = [cut_h_ref[positive], cut_v_ref[positive]]

    nyquist = 0.5  # fe/2, toujours à cette position dans la convention "fraction de fe"
    x_max = 1.05 * nyquist if fc is None else max(1.05 * nyquist, 1.05 * fc)
    vlines = [(nyquist, "fe/2", "blue")]
    if fc is not None:
        vlines.append((fc, "fc", "red"))

    return plot_curve_grid(
        freq[positive],
        [("Coupe horizontale", cut_h_est[positive]), ("Coupe verticale", cut_v_est[positive])],
        references=references,
        vlines=vlines,
        xlabel="Fréquence réduite (fraction de fe)",
        ylabel="FTM",
        title=title,
        x_range=(0, x_max),
        y_range=(0, 1.05),
        width=900,
        height=380,
    )


def visualisation_diagnostics(
    psf: np.ndarray,
    mask: np.ndarray,
    phase_fit_nm: np.ndarray,
    rad_to_nm: float,
    *,
    phase_nm: Optional[np.ndarray] = None,
) -> None:
    """
    Produit les visualisations de diagnostics (PSF mesuré, masque, cartes de
    phase, PSF prédit) en utilisant Plotly.

    Parameters
    ----------
    psf : np.ndarray
        PSF mesurée (2-D).
    mask : np.ndarray
        Masque pupillaire binaire (2-D).
    phase_fit_nm : np.ndarray
        Phase reconstruite à partir des coefficients de Zernike, en nm.
    rad_to_nm : float
        Facteur de conversion radian → nanomètre (utilisé pour le calcul du
        PSF prédit).
    phase_nm : np.ndarray, optional
        Phase "mesurée" indépendamment (issue par exemple de la récupération
        de phase Gerchberg-Saxton), en nm. Absente pour les méthodes qui
        estiment directement les coefficients de Zernike (optimisation,
        diversité de phase) : dans ce cas seule la carte reconstruite est
        affichée.
    """
    # ------------------------------------------------------------------
    # 1. PSF mesuré (log) et masque binaire
    # ------------------------------------------------------------------
    log_psf_meas = np.log10(psf + 1e-12)
    vmin_psf, vmax_psf = log_psf_meas.min(), log_psf_meas.max()

    plot_array(
        psf, title="Measured PSF (log)", log=True, vmin=vmin_psf, vmax=vmax_psf,
    ).show()

    plot_array(mask, title="Binary pupil mask").show()

    # ------------------------------------------------------------------
    # 2. Cartes de phase (en nm) – échelle symétrique
    # ------------------------------------------------------------------
    ref_for_scale = phase_nm if phase_nm is not None else phase_fit_nm
    dyn_phase_nm = max(abs(ref_for_scale.min()), abs(ref_for_scale.max()))

    if phase_nm is not None:
        plot_array(
            phase_nm, title="Retrieved phase (nm)", cmap="RdBu",
            vmin=-dyn_phase_nm, vmax=+dyn_phase_nm, cbar_label="WFE (nm)",
        ).show()

    plot_array(
        phase_fit_nm, title="Zernike-reconstructed phase (nm)", cmap="RdBu",
        vmin=-dyn_phase_nm, vmax=+dyn_phase_nm, cbar_label="WFE (nm)",
    ).show()

    # ------------------------------------------------------------------
    # 3. PSF prédit à partir du fit Zernike
    # ------------------------------------------------------------------
    phase_fit_rad = phase_fit_nm / rad_to_nm
    pupil_complex = mask * np.exp(1j * phase_fit_rad)

    psf_pred = np.abs(
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_complex)))
    ) ** 2
    psf_pred /= np.sum(psf_pred)

    plot_array(
        psf_pred, title="Predicted PSF from Zernike fit (log)", log=True,
        vmin=vmin_psf, vmax=vmax_psf,
    ).show()
