#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modèle optique direct : base de Zernike, synthèse de phase et PSF.

Fonctions purement numériques, partagées par les méthodes de récupération
de phase basées sur une optimisation (``backends.optimization``,
``backends.phase_diversity``).
"""

from __future__ import annotations

import numpy as np
import poppy.zernike as zern
from numpy.typing import NDArray


def build_zernike_basis(npix: int, nterms: int) -> NDArray[np.float64]:
    """
    Generate a Zernike basis (Noll ordering).

    Parameters
    ----------
    npix : int
        Size (in pixels) of the square array on which each mode is sampled.
    nterms : int
        Number of Zernike modes to generate.

    Returns
    -------
    NDArray
        Zernike basis, shape ``(nterms, npix, npix)``.
    """
    return zern.zernike_basis(nterms=nterms, npix=npix, outside=0.0)


def phase_from_coefficients(
    coeffs_rad: NDArray[np.float64],
    basis: NDArray[np.float64],
    mask: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Build a pupil phase map from Zernike coefficients.

    Parameters
    ----------
    coeffs_rad : NDArray
        Zernike coefficients [rad], shape ``(nterms,)``.
    basis : NDArray
        Zernike basis, shape ``(nterms, npix, npix)``.
    mask : NDArray
        Binary pupil mask, shape ``(npix, npix)``.

    Returns
    -------
    NDArray
        Phase map [rad], restricted to the pupil.
    """
    phase = np.tensordot(coeffs_rad, basis, axes=(0, 0))
    return phase * mask


def psf_from_phase(
    phase_rad: NDArray[np.float64],
    mask: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the (energy-normalised) focal-plane PSF for a given pupil phase.

    Parameters
    ----------
    phase_rad : NDArray
        Pupil phase map [rad].
    mask : NDArray
        Binary pupil mask.

    Returns
    -------
    NDArray
        PSF intensity, normalised so that its sum equals 1.
    """
    pupil = mask * np.exp(1j * phase_rad)
    focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))

    psf = np.abs(focal) ** 2
    psf /= psf.sum()

    return psf
