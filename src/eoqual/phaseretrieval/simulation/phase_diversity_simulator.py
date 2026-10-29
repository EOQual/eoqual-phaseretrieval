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
from ..optics import phase_from_coefficients, psf_from_phase, wfe_map_to_phase
from ..pupil import PupilGenerator
from ..zernike_basis_orthogonal import resolve_zernike_basis
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
    orthogonal_basis : bool, default False
        Si ``True``, les coefficients (aberration statique) sont
        interprétés dans la base ré-orthogonalisée par Gram-Schmidt sur le
        masque (§10 de ``REFERENCE_TECHNIQUE.md``) plutôt que la base de
        Noll standard — à utiliser pour tester
        ``PhaseDiversityConfig(orthogonal_basis=True)`` de façon cohérente
        (même remarque que pour ``simulate_psf_from_zernike``). Le terme
        de diversité (``defocus``) lui-même n'est pas affecté par ce
        réglage : ``diversity_phase`` n'utilise que ``basis[3]``, et sur
        un masque centrosymétrique la composante piston que Gram-Schmidt y
        mélange n'a aucun effet sur la PSF (un piston global n'affecte pas
        une intensité) — seul un masque *non* centrosymétrique (bras
        d'araignée asymétriques) pourrait faire apparaître un léger
        tip-tilt parasite dans ce terme, non pris en compte ici.
    known_ho_wfe_nm : NDArray, optional
        Carte de WFE haut ordre (HO) [nm] ajoutée à la phase bas ordre
        dans **les deux** canaux (focus et défocus) — une contribution
        statique de l'optique, présente indépendamment du terme de
        diversité (voir §11.3 de ``REFERENCE_TECHNIQUE.md``) — pour tester
        ``PhaseDiversityConfig.known_ho_wfe_nm`` de bout en bout. ``None``
        (défaut) : pas de contribution HO.
    aperture_px : int, optional
        Diamètre [px] de l'ouverture à l'intérieur du cadre ``npix x
        npix`` — contrôle du suréchantillonnage, en mémoire (voir
        :class:`eoqual.phaseretrieval.pupil.PupilGenerator` et §12.5 de
        ``REFERENCE_TECHNIQUE.md``). Si ``mask is None`` dans
        :func:`simulate_phase_diversity_pair`, contrôle le masque généré
        par défaut. Si un ``mask`` personnalisé est fourni, ``None``
        (défaut) auto-détecte l'ouverture réelle depuis ce masque
        (:func:`eoqual.phaseretrieval.pupil.detect_aperture_px`).
        Appliqué de façon cohérente au masque généré et à la base de
        Zernike (vérité bas ordre *et* terme de diversité, tous deux
        construits à partir de la même base).
    """

    wavelength: float = 550e-9
    npix: int = 256
    zernike_order: int = 15
    defocus: float = 275e-9
    noise: Optional[DetectorNoiseConfig] = None
    orthogonal_basis: bool = False
    known_ho_wfe_nm: Optional[NDArray[np.float64]] = None
    aperture_px: Optional[int] = None


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
        - ``"coeffs_nm"`` : coefficients réellement appliqués (tableau
          dense, longueur ``zernike_order``, dans la base choisie via
          ``cfg.orthogonal_basis``) — la vérité terrain à comparer aux
          coefficients récupérés (ajustés avec le *même* réglage).
        - ``"defocus"``, ``"wavelength"`` : rappel des paramètres utilisés.
        - ``"orthogonal_basis"`` : rappel du réglage utilisé.
        - ``"aperture_px"`` : rappel du réglage utilisé (ou ``None``).
        - ``"known_ho_wfe_nm"`` : rappel de la carte HO utilisée (ou ``None``).

    Raises
    ------
    ValueError
        Si ``cfg.orthogonal_basis=True`` et qu'au moins un mode d'indice ≤
        ``cfg.zernike_order`` est linéairement dépendant des précédents
        sur ce masque — voir :func:`resolve_zernike_basis`. Ou si
        ``cfg.known_ho_wfe_nm`` n'a pas la forme de ``mask``.
    """
    if mask is None:
        mask = PupilGenerator(cfg.npix, aperture_px=cfg.aperture_px).simple_disk()

    coeffs_arr_nm = coeffs_to_array(coeffs_nm, cfg.zernike_order)
    rad_to_nm = cfg.wavelength * 1e9 / (2.0 * np.pi)

    basis = resolve_zernike_basis(
        npix=cfg.npix, nterms=cfg.zernike_order, mask=mask,
        orthogonalize=cfg.orthogonal_basis, aperture_px=cfg.aperture_px,
    )
    phase_rad = phase_from_coefficients(coeffs_arr_nm / rad_to_nm, basis, mask)
    if cfg.known_ho_wfe_nm is not None:
        phase_rad = phase_rad + wfe_map_to_phase(cfg.known_ho_wfe_nm, cfg.wavelength, mask)
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
        "orthogonal_basis": cfg.orthogonal_basis,
        "aperture_px": cfg.aperture_px,
        "known_ho_wfe_nm": cfg.known_ho_wfe_nm,
    }
