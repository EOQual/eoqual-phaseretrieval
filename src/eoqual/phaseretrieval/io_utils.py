#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E/S et visualisation communes à toutes les méthodes de reconstruction.

Ce module regroupe les fonctions indépendantes de l'algorithme de
récupération de phase choisie (Gerchberg-Saxton, optimisation directe ou
diversité de phase) : chargement d'images, préparation du masque
pupillaire et diagnostics graphiques Plotly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
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


def load_and_prepare_mask(
    psf_shape: Tuple[int, int],
    mask_path: Union[str, Path],
    wavelength: float,
    focal_length: float,
    pupil_diameter: float,
    detector_size: float,
) -> Tuple[NDArray[np.uint8], Optional[int]]:
    """
    Load a binary pupil mask, resize it to the physical pupil size that
    corresponds to the supplied PSF, and embed it (zero-pad) into an array
    of the same size as the PSF.

    If the raw mask image is *already* the same shape as the PSF, it is
    assumed to be pre-registered on the PSF grid (e.g. produced by a
    simulator) and is used as-is, skipping the physical resize step.

    Parameters
    ----------
    psf_shape : tuple[int, int]
        Shape of the PSF image (must be square, e.g. ``(N_psf, N_psf)``).
    mask_path : str or pathlib.Path
        Path to the raw mask image (binary or grayscale).
    wavelength : float
        Central wavelength **in metres** (e.g. ``550e-9``).
    focal_length : float
        Focal length ``f`` of the imaging system **in metres**.
    pupil_diameter : float
        Physical pupil diameter ``D`` **in metres**.
    detector_size : float
        Physical size of the detector (pixel-pitch x num-pixels) **in metres**.

    Returns
    -------
    mask : ndarray, shape (N_psf, N_psf), dtype uint8
        Binary mask centred in the field of view. Pixels are ``0``
        (outside pupil) or ``1`` (inside pupil).
    N_pup_target : int or None
        Number of pixels the pupil spans after the geometric scaling, or
        ``None`` when the mask was already on the PSF grid (no scaling
        was applied).

    Notes
    -----
    The scaling comes from the well-known relationship between the Airy
    radius in pixels and the physical parameters of the system:

    .. math::
        N_{pup}=\\frac{\\lambda f}{D\\,\\Delta_{det}}

    where ``Delta_det`` is the detector pixel size (``detector_size / N_psf``).

    Raises
    ------
    ValueError
        If the PSF is not square, or if the required pupil diameter in
        pixels is larger than the PSF matrix (you would need to zero-pad
        the PSF beforehand).
    """
    N_psf = psf_shape[0]  # On suppose une matrice carrée (N_psf, N_psf)
    if psf_shape[0] != psf_shape[1]:
        raise ValueError(f"PSF must be square – received shape {psf_shape}")

    mask_raw = load_image(mask_path)

    # Le masque simulé/pré-enregistré est déjà à la résolution de la PSF :
    # aucune mise à l'échelle physique n'est nécessaire.
    if mask_raw.shape == psf_shape:
        mask = (mask_raw > 0.5).astype(np.uint8)
        logger.debug("Masque déjà sur la grille de la PSF, resize ignoré.")
        return mask, None

    # ---------------------------------------------------------------
    # Nombre de pixels que doit occuper la pupille physique
    # ---------------------------------------------------------------
    # Airy-disk size in pixels (≈ λ f / (D Δdet))
    airy_pixels = (wavelength * focal_length) / (pupil_diameter * detector_size)
    N_pup_target = int(np.round(N_psf / airy_pixels))

    if N_pup_target > N_psf:
        raise ValueError(
            f"Required pupil diameter ({N_pup_target} px) exceeds PSF size "
            f"({N_psf} px). Pad the PSF (or use a larger detector) before "
            "calling this function."
        )

    # ---------------------------------------------------------------
    # Redimensionnement du masque à sa taille physique
    # ---------------------------------------------------------------
    mask_resized = resize(
        mask_raw,
        output_shape=(N_pup_target, N_pup_target),
        order=0,  # nearest-neighbour → conserve les valeurs binaires
        anti_aliasing=False,
        preserve_range=True,
    )

    # ---------------------------------------------------------------
    # Zero-pad / centrage du masque redimensionné
    # ---------------------------------------------------------------
    mask = np.zeros((N_psf, N_psf), dtype=np.float64)
    r_off = (N_psf - N_pup_target) // 2
    c_off = (N_psf - N_pup_target) // 2
    mask[r_off:r_off + N_pup_target, c_off:c_off + N_pup_target] = mask_resized

    mask = (mask > 0.5).astype(np.uint8)

    logger.debug(f"Mask sum (number of illuminated pixels): {int(mask.sum())}")
    logger.debug(f"Mask shape:   {mask.shape}")
    logger.debug(f"N_pup_target: {N_pup_target}")

    return mask, N_pup_target


def plot_array_plotly(
    arr: np.ndarray,
    *,
    title: str = "",
    cmap: str = "viridis",
    log: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: str = "",
) -> go.Figure:
    """
    Visualisation interactive d'un tableau 2-D avec Plotly.

    Parameters
    ----------
    arr : np.ndarray
        Tableau à afficher (2-D).
    title : str, optional
        Titre du graphique.
    cmap : str, default "viridis"
        Nom du colormap Plotly (voir https://plotly.com/python/builtin-colorscales/).
    log : bool, default False
        Si ``True`` on trace ``log10(arr)`` – pratique pour les PSF.
    vmin, vmax : float, optional
        Limites de l'échelle de couleur.
    cbar_label : str, optional
        Légende de la barre de couleur.

    Returns
    -------
    go.Figure
        Figure Plotly prête à être affichée avec ``fig.show()``.
    """
    data = np.log10(arr + 1e-12) if log else arr

    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            colorscale=cmap,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title=cbar_label) if cbar_label else None,
            showscale=True,
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            constrain="domain",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        width=500,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


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

    plot_array_plotly(
        psf, title="Measured PSF (log)", log=True, vmin=vmin_psf, vmax=vmax_psf,
    ).show()

    plot_array_plotly(mask, title="Binary pupil mask").show()

    # ------------------------------------------------------------------
    # 2. Cartes de phase (en nm) – échelle symétrique
    # ------------------------------------------------------------------
    ref_for_scale = phase_nm if phase_nm is not None else phase_fit_nm
    dyn_phase_nm = max(abs(ref_for_scale.min()), abs(ref_for_scale.max()))

    if phase_nm is not None:
        plot_array_plotly(
            phase_nm, title="Retrieved phase (nm)", cmap="RdBu",
            vmin=-dyn_phase_nm, vmax=+dyn_phase_nm, cbar_label="WFE (nm)",
        ).show()

    plot_array_plotly(
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

    plot_array_plotly(
        psf_pred, title="Predicted PSF from Zernike fit (log)", log=True,
        vmin=vmin_psf, vmax=vmax_psf,
    ).show()
