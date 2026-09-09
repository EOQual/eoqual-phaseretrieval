#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération d'une PSF à vérité de Zernike connue.

Code de **simulation**, volontairement séparé des méthodes de retrouvé de
phase (``backends/``) : il sert à fabriquer des cas de test (vérité
terrain connue) pour valider n'importe laquelle des trois méthodes, pas à
en implémenter une. S'appuie sur le même modèle direct que les méthodes
par optimisation (:mod:`eoqual.phaseretrieval.optics`), pour que la PSF
simulée soit rigoureusement cohérente avec ce que ``fit_zernike_from_psf``
/ ``fit_phase_diversity`` essaient de reconstruire.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..optics import build_zernike_basis, phase_from_coefficients, psf_from_phase
from ..pupil import PupilGenerator
from .noise import DetectorNoiseConfig, apply_detector_noise

CoeffsInput = Union[Dict[int, float], NDArray[np.float64]]


def coeffs_to_array(coeffs_nm: CoeffsInput, zernike_order: int) -> NDArray[np.float64]:
    """
    Normalise des coefficients de Zernike vers un tableau dense.

    Parameters
    ----------
    coeffs_nm : dict[int, float] or NDArray
        Soit un dictionnaire ``{indice de Noll: valeur [nm WFE]}`` (ne
        renseigner que les modes non nuls — le plus lisible pour un cas de
        test), soit un tableau déjà dense (indexé à partir de Z1).
    zernike_order : int
        Nombre de modes de la base cible (ordre de Noll maximal).

    Returns
    -------
    NDArray
        Tableau dense de longueur ``zernike_order``, indice 0 = Z1.

    Raises
    ------
    ValueError
        Si un indice de Noll (dict) ou la longueur (array) dépasse
        ``zernike_order``.
    """
    if isinstance(coeffs_nm, dict):
        arr = np.zeros(zernike_order, dtype=np.float64)
        for noll_index, value_nm in coeffs_nm.items():
            if not (1 <= noll_index <= zernike_order):
                raise ValueError(
                    f"Indice de Noll {noll_index} hors de la plage "
                    f"1..{zernike_order} (augmenter zernike_order)."
                )
            arr[noll_index - 1] = value_nm
        return arr

    arr = np.asarray(coeffs_nm, dtype=np.float64)
    if arr.shape[0] > zernike_order:
        raise ValueError(
            f"coeffs_nm a {arr.shape[0]} termes > zernike_order={zernike_order}."
        )
    if arr.shape[0] < zernike_order:
        arr = np.pad(arr, (0, zernike_order - arr.shape[0]))
    return arr


def simulate_psf_from_zernike(
    coeffs_nm: CoeffsInput,
    *,
    wavelength: float = 550e-9,
    npix: int = 256,
    zernike_order: int = 15,
    mask: Optional[NDArray[np.float64]] = None,
    noise: Optional[DetectorNoiseConfig] = None,
) -> Dict[str, object]:
    """
    Simule une PSF diffraction-limitée avec une aberration de Zernike
    connue — la vérité terrain nécessaire pour tester une méthode de
    retrouvé de phase (§6.4 de ``REFERENCE_TECHNIQUE.md``).

    Parameters
    ----------
    coeffs_nm : dict[int, float] or NDArray
        Coefficients de Zernike (Noll), en nm WFE — voir
        :func:`coeffs_to_array`. Exemple : ``{4: 55.0, 5: 27.5}`` pour un
        défocus + astigmatisme connus, tous les autres modes à zéro.
    wavelength : float, default 550e-9
        Longueur d'onde [m].
    npix : int, default 256
        Taille (pixels) de la grille PSF/pupille, carrée.
    zernike_order : int, default 15
        Nombre de modes de Zernike (ordre de Noll) de la base utilisée.
    mask : NDArray, optional
        Masque pupillaire binaire ``(npix, npix)``. Par défaut, un disque
        plein non obstrué (:class:`eoqual.phaseretrieval.pupil.PupilGenerator`).
    noise : DetectorNoiseConfig, optional
        Si fourni, un bruit détecteur réaliste (Poisson + lecture) est
        appliqué (voir :mod:`.noise`). Par défaut ``None`` : seule la PSF
        idéale est renvoyée (clé ``"psf"`` == ``"psf_truth"``).

    Returns
    -------
    dict
        - ``"psf"`` : PSF bruitée (ou idéale si ``noise=None``), énergie
          normalisée — directement utilisable comme entrée de
          ``psf_to_wfe`` / ``fit_zernike_from_psf`` / ``gerchberg_saxton``.
        - ``"psf_truth"`` : PSF idéale (sans bruit), énergie normalisée.
        - ``"mask"`` : masque pupillaire utilisé.
        - ``"coeffs_nm"`` : coefficients de Zernike réellement appliqués
          (tableau dense, longueur ``zernike_order``) — la vérité terrain
          à comparer aux coefficients retrouvés.
        - ``"wavelength"`` : longueur d'onde utilisée [m].
    """
    if mask is None:
        mask = PupilGenerator(npix).simple_disk()

    coeffs_arr_nm = coeffs_to_array(coeffs_nm, zernike_order)
    rad_to_nm = wavelength * 1e9 / (2.0 * np.pi)

    basis = build_zernike_basis(npix=npix, nterms=zernike_order)
    phase_rad = phase_from_coefficients(coeffs_arr_nm / rad_to_nm, basis, mask)
    psf_truth = psf_from_phase(phase_rad, mask)

    psf = apply_detector_noise(psf_truth, noise) if noise is not None else psf_truth

    return {
        "psf": psf,
        "psf_truth": psf_truth,
        "mask": mask,
        "coeffs_nm": coeffs_arr_nm,
        "wavelength": wavelength,
    }
