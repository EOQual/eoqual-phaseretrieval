#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Récupération de phase itérative par l'algorithme de Gerchberg-Saxton (GS).

Ne nécessite qu'une seule PSF et le masque pupillaire binaire. La phase
pupillaire est récupérée par projections successives entre plan pupille et
plan focal ; elle doit ensuite être décomposée sur la base de Zernike
(voir :mod:`zernike_fit`) pour obtenir des coefficients.

Cette version correspond à la variante la plus aboutie (« V2 ») : la phase
finale est directement extraite via ``np.angle`` (sans déroulement de
phase), ce qui s'est révélé plus stable que le déroulement de phase testé
dans une variante antérieure (« V1 », abandonnée).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class RetrievalConfig:
    """
    Parameters steering the Gerchberg-Saxton (GS) iteration.

    Parameters
    ----------
    max_iter : int
        Maximum number of GS iterations.
    beta : float
        Relaxation factor, reserved for a future Hybrid Input-Output (HIO)
        variant; unused by the plain GS loop implemented here.
    verbose : bool
        Display progress information.
    seed : int or np.random.Generator, optional
        Seed (or generator) for the random pupil-phase initialisation.
        ``None`` (default) reproduces the original behaviour: a fresh,
        non-reproducible draw on every call. Pass a fixed seed for
        reproducible runs (e.g. in tests or notebook validation), since
        plain GS is known to occasionally stagnate depending on its
        random start.
    """
    max_iter: int = 200
    beta: float = 0.9
    verbose: bool = True
    seed: Optional[Union[int, np.random.Generator]] = None


def gerchberg_saxton(
    psf: NDArray[np.float64],
    pupil_mask: NDArray[np.float64],
    config: RetrievalConfig,
) -> NDArray[np.float64]:
    """
    Classical Gerchberg-Saxton phase retrieval.

    Parameters
    ----------
    psf : NDArray
        Focal-plane intensity.
    pupil_mask : NDArray
        Binary aperture mask.
    config : RetrievalConfig
        Must contain ``max_iter``.

    Returns
    -------
    phase : NDArray
        Retrieved pupil phase [rad], restricted to the pupil.
    """
    # --- Vérifications ---
    if np.sum(psf) <= 0:
        raise ValueError("PSF must contain non-zero energy.")
    if np.sum(pupil_mask) <= 0:
        raise ValueError("Pupil mask must contain at least one non-zero pixel.")

    # ------------------------------------------------------------------
    # Amplitude cible dans le plan focal
    # ------------------------------------------------------------------
    focal_amp_target = np.sqrt(np.maximum(psf, 0.0))
    focal_amp_target /= np.linalg.norm(focal_amp_target)

    # ------------------------------------------------------------------
    # Initialisation aléatoire de la phase
    # ------------------------------------------------------------------
    rng = np.random.default_rng(config.seed)
    pupil_phase = rng.uniform(-np.pi, np.pi, pupil_mask.shape)

    # Amplitude imposée dans la pupille
    pupil_amp_target = pupil_mask.astype(np.float64)

    # Champ initial
    pupil_field = pupil_amp_target * np.exp(1j * pupil_phase)

    # ------------------------------------------------------------------
    # Boucle Gerchberg-Saxton
    # ------------------------------------------------------------------
    for _ in range(config.max_iter):
        # --- Pupille -> Focal ---
        focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_field)))

        # Contrainte amplitude dans le plan focal
        focal_amp_current = np.maximum(np.abs(focal_field), 1e-12)
        focal_field = focal_amp_target * focal_field / focal_amp_current

        # --- Focal -> Pupille ---
        pupil_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(focal_field)))

        # Contrainte amplitude dans la pupille
        pupil_amp_current = np.maximum(np.abs(pupil_field), 1e-12)
        pupil_field = pupil_amp_target * pupil_field / pupil_amp_current

    # ------------------------------------------------------------------
    # Phase finale (pas de déroulement : voir docstring du module)
    # ------------------------------------------------------------------
    phase = np.angle(pupil_field)
    phase = phase * pupil_mask

    return phase
