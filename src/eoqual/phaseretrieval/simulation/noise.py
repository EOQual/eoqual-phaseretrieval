#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bruit détecteur réaliste, appliqué à une PSF simulée "vérité terrain".

Deux composantes, indépendantes du modèle optique lui-même (§9 de
``REFERENCE_TECHNIQUE.md`` en discute la pertinence pour la diversité de
phase en télédétection) :

- **Bruit de photon (Poisson)** — irréductible, dépend du flux total
  reçu (``photon_count``, en nombre de photons/électrons intégrés sur
  toute la tache).
- **Bruit de lecture (Gaussien additif)** — électronique du détecteur,
  indépendant du signal (``read_noise_e``, écart-type en électrons).

Un fond de ciel/détecteur constant (``background_e``) peut être ajouté
avant le tirage Poisson (lui aussi bruité par construction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class DetectorNoiseConfig:
    """
    Paramètres du modèle de bruit détecteur.

    Parameters
    ----------
    photon_count : float
        Flux total (photons/électrons) intégré sur toute la PSF — fixe le
        régime de bruit de photon (plus il est faible, plus le SNR est
        dégradé). ``1e6`` correspond à une pose courte mais confortable ;
        descendre vers ``1e3``-``1e4`` simule une acquisition faible flux
        (cf. §8.3/§3.6 de ``REFERENCE_TECHNIQUE.md`` sur la pondération
        Poisson).
    read_noise_e : float
        Écart-type du bruit de lecture [électrons], indépendant du signal.
    background_e : float
        Niveau de fond constant [électrons/pixel], ajouté avant le tirage
        Poisson (donc lui-même bruité).
    seed : int, optional
        Graine du générateur aléatoire, pour des tirages reproductibles.
    """

    photon_count: float = 1e6
    read_noise_e: float = 5.0
    background_e: float = 0.0
    seed: Optional[int] = None


def apply_detector_noise(
    psf: NDArray[np.float64],
    cfg: DetectorNoiseConfig,
    *,
    seed_offset: int = 0,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """
    Applique un bruit détecteur réaliste (Poisson + lecture) à une PSF
    "vérité terrain" (normalisée, ``sum(psf) == 1``).

    Parameters
    ----------
    psf : NDArray
        PSF idéale, énergie normalisée à 1.
    cfg : DetectorNoiseConfig
        Paramètres du bruit.
    seed_offset : int, default 0
        Décalage ajouté à ``cfg.seed`` avant de construire le générateur —
        utile pour tirer deux images corrélées en flux/lecture mais avec
        des réalisations de bruit indépendantes (ex. focus/défocus d'une
        même paire de diversité de phase : ``seed_offset=0`` puis ``=1``).
    normalize : bool, default True
        Renormalise l'image bruitée à ``sum() == 1`` avant de la renvoyer,
        pour reproduire exactement ce que fait
        :func:`eoqual.phaseretrieval.pipeline.psf_to_wfe` juste après
        ``io_utils.load_image`` sur une vraie acquisition — l'image
        renvoyée est donc directement utilisable par n'importe laquelle
        des trois méthodes de retrouvé de phase, sans étape intermédiaire.

    Returns
    -------
    NDArray
        Image bruitée, même forme que ``psf``.
    """
    seed = None if cfg.seed is None else cfg.seed + seed_offset
    rng = np.random.default_rng(seed)

    signal_e = psf * cfg.photon_count + cfg.background_e
    shot_noisy = rng.poisson(np.clip(signal_e, 0.0, None)).astype(np.float64)
    noisy = shot_noisy + rng.normal(0.0, cfg.read_noise_e, size=psf.shape)

    if normalize:
        noisy = noisy / noisy.sum()

    return noisy
