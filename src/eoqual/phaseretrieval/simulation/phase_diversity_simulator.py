#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération d'une paire focus/défocus (diversité de phase) à vérité de
Zernike connue, avec bruit détecteur réaliste.

Simule la donnée d'entrée du backend ``phase_diversity``
(:mod:`eoqual.phaseretrieval.backends.phase_diversity`) : deux images
d'étoile, une à focus nominal et une avec un défocus connu et
volontairement introduit — exactement le protocole de mesure décrit au
§5 de ``REFERENCE_TECHNIQUE.md``. Réutilise
:func:`eoqual.phaseretrieval.backends.phase_diversity.diversity_phase`
pour construire la phase de diversité, afin que le défocus simulé ici
suive exactement la même convention (unités, signe) que celle attendue
par ``PhaseDiversityConfig.defocus`` côté récupération de phase — un
``defocus`` simulé peut donc être réinjecté tel quel dans
``fit_phase_diversity`` pour une validation de bout en bout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ..backends.phase_diversity import diversity_phase
from ..optics import build_zernike_basis, phase_from_coefficients, psf_from_phase
from ..pupil import PupilGenerator
from .noise import DetectorNoiseConfig, apply_detector_noise
from .psf_simulator import CoeffsInput, coeffs_to_array


@dataclass(slots=True)
class PhaseDiversitySimulationConfig:
    """
    Paramètres de simulation d'une paire focus/défocus.

    Parameters
    ----------
    wavelength : float
        Longueur d'onde [m].
    npix : int
        Taille (pixels) de la grille PSF/pupille, carrée.
    zernike_order : int
        Nombre de modes de Zernike (ordre de Noll) de la base utilisée.
    defocus : float
        Défocus de diversité connu [m] — même convention que
        ``backends.phase_diversity.PhaseDiversityConfig.defocus``.
    noise : DetectorNoiseConfig, optional
        Bruit détecteur (Poisson + lecture) appliqué aux deux images.
        ``None`` (défaut) : paire idéale, sans bruit.
    """

    wavelength: float = 550e-9
    npix: int = 256
    zernike_order: int = 15
    defocus: float = 275e-9
    noise: Optional[DetectorNoiseConfig] = None


def simulate_phase_diversity_pair(
    coeffs_nm: CoeffsInput,
    cfg: PhaseDiversitySimulationConfig,
    *,
    mask: Optional[NDArray[np.float64]] = None,
) -> Dict[str, object]:
    """
    Simule une paire d'images d'étoile focus/défocus, à vérité de Zernike
    et de défocus connues, pour tester ``fit_phase_diversity`` de bout en
    bout (voir §6.4 de ``REFERENCE_TECHNIQUE.md``).

    Parameters
    ----------
    coeffs_nm : dict[int, float] or NDArray
        Coefficients de Zernike (Noll) de l'aberration statique du
        système, en nm WFE — voir
        :func:`eoqual.phaseretrieval.simulation.psf_simulator.coeffs_to_array`.
        N'inclut *pas* le défocus de diversité lui-même, ajouté séparément
        via ``cfg.defocus`` (comme sur un instrument réel : c'est un
        déplacement mécanique connu, pas un mode à estimer).
    cfg : PhaseDiversitySimulationConfig
        Paramètres de simulation (longueur d'onde, défocus, bruit...).
    mask : NDArray, optional
        Masque pupillaire binaire. Par défaut, un disque plein non
        obstrué.

    Returns
    -------
    dict
        - ``"psf_focus"``, ``"psf_defocus"`` : images bruitées (ou idéales
          si ``cfg.noise is None``), directement utilisables par
          ``fit_phase_diversity(psf_focus, psf_defocus, mask, pd_cfg)``
          (avec ``pd_cfg.defocus == cfg.defocus``).
        - ``"psf_focus_truth"``, ``"psf_defocus_truth"`` : versions
          idéales sans bruit.
        - ``"mask"`` : masque pupillaire utilisé.
        - ``"coeffs_nm"`` : coefficients de Zernike réellement appliqués
          (tableau dense, longueur ``zernike_order``) — la vérité terrain
          à comparer aux coefficients récupérés.
        - ``"defocus"``, ``"wavelength"`` : rappel des paramètres utilisés.
    """
    if mask is None:
        mask = PupilGenerator(cfg.npix).simple_disk()

    coeffs_arr_nm = coeffs_to_array(coeffs_nm, cfg.zernike_order)
    rad_to_nm = cfg.wavelength * 1e9 / (2.0 * np.pi)

    basis = build_zernike_basis(npix=cfg.npix, nterms=cfg.zernike_order)
    phase_rad = phase_from_coefficients(coeffs_arr_nm / rad_to_nm, basis, mask)
    diversity_rad = diversity_phase(basis, cfg.defocus, cfg.wavelength)

    psf_focus_truth = psf_from_phase(phase_rad, mask)
    psf_defocus_truth = psf_from_phase(phase_rad + diversity_rad, mask)

    if cfg.noise is not None:
        # seed_offset différent entre les deux images : même flux/bruit de
        # lecture *paramétrés*, mais tirages indépendants (comme deux
        # poses réelles), pas une simple copie du même bruit.
        psf_focus = apply_detector_noise(psf_focus_truth, cfg.noise, seed_offset=0)
        psf_defocus = apply_detector_noise(psf_defocus_truth, cfg.noise, seed_offset=1)
    else:
        psf_focus, psf_defocus = psf_focus_truth, psf_defocus_truth

    return {
        "psf_focus": psf_focus,
        "psf_defocus": psf_defocus,
        "psf_focus_truth": psf_focus_truth,
        "psf_defocus_truth": psf_defocus_truth,
        "mask": mask,
        "coeffs_nm": coeffs_arr_nm,
        "defocus": cfg.defocus,
        "wavelength": cfg.wavelength,
    }
