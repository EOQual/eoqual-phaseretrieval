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


def wfe_map_to_phase(
    wfe_nm: NDArray[np.float64],
    wavelength: float,
    mask: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert a WFE map [nm] to a pupil phase map [rad], restricted to the
    pupil.

    Used to inject a **known, fixed** wavefront-error contribution into the
    direct model — typically a high-order (HO) map beyond the fitted
    ``zernike_order``, measured independently (see ``known_ho_wfe_nm`` on
    :class:`backends.optimization.OpticalConfig` and
    :class:`backends.phase_diversity.PhaseDiversityConfig`, and §11.3 of
    ``REFERENCE_TECHNIQUE.md``). Unlike :func:`phase_from_coefficients`,
    the input here is already a pixel map, not a set of modal coefficients
    — no Zernike basis involved, so any spatial pattern (not just a finite
    sum of Zernike modes) can be supplied.

    Parameters
    ----------
    wfe_nm : NDArray
        Wavefront-error map [nm], same shape as ``mask``.
    wavelength : float
        Wavelength [m], used for the nm -> rad conversion.
    mask : NDArray
        Binary pupil mask, shape ``(npix, npix)``.

    Returns
    -------
    NDArray
        Phase map [rad], restricted to the pupil.

    Raises
    ------
    ValueError
        If ``wfe_nm`` and ``mask`` shapes differ.
    """
    if wfe_nm.shape != mask.shape:
        raise ValueError(
            f"wfe_nm shape {wfe_nm.shape} != mask shape {mask.shape}"
        )
    rad_per_nm = (2.0 * np.pi) / (wavelength * 1e9)
    return wfe_nm * rad_per_nm * mask


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
