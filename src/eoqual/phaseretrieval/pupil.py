#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération de masques de pupille : disque simple, pupille annulaire
(obstruction centrale) et pupille avec bras d'araignée.

Complète [`zernike_basis_orthogonal.py`](zernike_basis_orthogonal.py) (§10
de ``REFERENCE_TECHNIQUE.md``) : ce module génère les pupilles
non standard sur lesquelles la base de Zernike doit être ré-orthogonalisée,
plutôt que de les construire à la main à chaque fois.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def polar_grid(npix: int, aperture_px: Optional[int] = None) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Grille polaire centrée ``(rho, theta)``, ``rho == 1`` au bord d'une
    ouverture de diamètre ``aperture_px`` à l'intérieur d'un cadre
    ``npix x npix`` — source commune utilisée à la fois par
    :class:`PupilGenerator` (pour délimiter un masque) et par
    :func:`eoqual.phaseretrieval.optics.build_zernike_basis` (pour que la
    base de Zernike soit évaluée à la **même** échelle que le masque,
    voir §12.5 de ``REFERENCE_TECHNIQUE.md``) : les deux doivent
    s'accorder sur où se trouve le bord de la pupille, sous peine de
    décalage d'échelle silencieux entre le masque et la base modale.

    Parameters
    ----------
    npix : int
        Taille du cadre (pixels).
    aperture_px : int, optional
        Diamètre de l'ouverture [px] à l'intérieur du cadre. ``None``
        (défaut) : l'ouverture remplit le cadre (``aperture_px = npix``,
        comportement historique, facteur de suréchantillonnage 1).

    Returns
    -------
    rho, theta : NDArray, NDArray
        Rayon normalisé (1 au bord de l'ouverture) et angle [rad], forme
        ``(npix, npix)``.

    Raises
    ------
    ValueError
        Si ``aperture_px`` dépasse ``npix``, ou est trop petit (< 2) pour
        définir un rayon non nul.
    """
    if aperture_px is None:
        aperture_px = npix
    if aperture_px > npix:
        raise ValueError(f"aperture_px ({aperture_px}) ne peut pas dépasser npix ({npix}).")
    if aperture_px < 2:
        raise ValueError(f"aperture_px ({aperture_px}) doit être >= 2.")

    y, x = np.indices((npix, npix)) - npix // 2
    # Rayon normalisé, volontairement NON clippé à 1.0 : PupilGenerator
    # teste justement `rho <= 1.0` pour délimiter la pupille (bug corrigé —
    # un clip ici rendait ce test toujours vrai, produisant un masque
    # plein-cadre au lieu d'un disque/anneau réel ; voir
    # tests/test_pupil.py::test_simple_disk_excludes_corners).
    rho = np.sqrt(x ** 2 + y ** 2) / (aperture_px // 2)
    theta = np.arctan2(y, x)
    return rho, theta


def detect_aperture_px(mask: NDArray[np.float64]) -> int:
    """
    Détecte le diamètre [px] de l'ouverture réellement illuminée dans
    ``mask``, en mesurant l'étendue (bounding box) de ses pixels non nuls.

    Sert à retrouver automatiquement l'``aperture_px`` d'un masque fourni
    par l'appelant (généré ailleurs que par :class:`PupilGenerator`, ou
    chargé d'une source quelconque) plutôt que de supposer à tort que son
    ouverture remplit tout le cadre ``npix x npix`` — c'est exactement le
    décalage d'échelle silencieux du §12.5 de ``REFERENCE_TECHNIQUE.md``,
    mais pour la base de Zernike construite à partir d'un ``mask``
    personnalisé plutôt que d'un masque chargé depuis un fichier (déjà
    couvert par la détection équivalente d'``io_utils._detect_aperture_bbox``).

    Suppose une ouverture centrée dans le cadre, comme :func:`polar_grid` :
    seule l'étendue (plus grande des deux dimensions) de la bounding box
    importe, pas sa position.

    Parameters
    ----------
    mask : NDArray
        Masque pupillaire ``(npix, npix)``, non nul dans l'ouverture.

    Returns
    -------
    int
        Diamètre détecté [px] (plus grande des étendues en ligne et en
        colonne des pixels non nuls).

    Raises
    ------
    ValueError
        Si ``mask`` ne contient aucun pixel non nul.
    """
    lit = mask > 0
    rows, cols = np.any(lit, axis=1), np.any(lit, axis=0)
    if not rows.any():
        raise ValueError("Masque entièrement nul : aucune ouverture à détecter.")
    row_min, row_max = int(np.flatnonzero(rows)[0]), int(np.flatnonzero(rows)[-1])
    col_min, col_max = int(np.flatnonzero(cols)[0]), int(np.flatnonzero(cols)[-1])
    return max(row_max - row_min + 1, col_max - col_min + 1)


class PupilGenerator:
    """
    Générateur de masques de pupille sur une grille carrée ``(N, N)``.

    Parameters
    ----------
    npix : int
        Taille de la grille (pixels).
    aperture_px : int, optional
        Diamètre de l'ouverture [px] à l'intérieur du cadre ``npix x
        npix`` — contrôle du suréchantillonnage. ``None`` (défaut) :
        l'ouverture remplit le cadre (comportement historique, facteur de
        suréchantillonnage 1). Une valeur < ``npix`` centre une ouverture
        plus petite dans un cadre complété de zéros autour — le même
        padding que ``io_utils._resample_to_footprint`` applique en
        chargeant un masque depuis un fichier avec ``oversampling`` > 1
        (§12.1 de ``REFERENCE_TECHNIQUE.md``), mais ici directement, pour
        une génération de pupille **en mémoire** (module de simulation,
        tests, notebooks) sans passer par un fichier. Facteur de
        suréchantillonnage équivalent : ``npix / aperture_px``.

        **Important** : passer ce même ``aperture_px`` à
        :func:`eoqual.phaseretrieval.optics.build_zernike_basis` (ou à
        ``resolve_zernike_basis``) utilisée avec ce masque — sinon la base
        de Zernike reste évaluée comme si l'ouverture remplissait tout le
        cadre, un décalage d'échelle silencieux (défocus tronqué,
        normalisation RMS fausse) décrit et corrigé en §12.5 de
        ``REFERENCE_TECHNIQUE.md``.
    """

    def __init__(self, npix: int, aperture_px: Optional[int] = None) -> None:
        self.npix = npix
        self.aperture_px = npix if aperture_px is None else aperture_px
        self.rho, self.theta = polar_grid(npix, aperture_px)

    def simple_disk(self) -> NDArray[np.float64]:
        """
        Pupille circulaire pleine, non obstruée.

        Returns
        -------
        NDArray
            Masque binaire (1 dans la pupille, 0 ailleurs).
        """
        return (self.rho <= 1.0).astype(np.float64)

    def annular(self, obscuration: float) -> NDArray[np.float64]:
        """
        Pupille annulaire (obstruction centrale circulaire).

        Parameters
        ----------
        obscuration : float
            Ratio d'obstruction (0 = pas d'obstruction, 0.3 = diamètre de
            l'obstruction = 30 % du diamètre de la pupille).

        Returns
        -------
        NDArray
            Masque binaire.
        """
        return ((self.rho <= 1.0) & (self.rho >= obscuration)).astype(np.float64)

    def with_spiders(
        self,
        obscuration: float,
        n_spiders: int,
        width: float,
    ) -> NDArray[np.float64]:
        """
        Pupille annulaire avec bras d'araignée (spiders) radiaux.

        Parameters
        ----------
        obscuration : float
            Ratio d'obstruction centrale (voir :meth:`annular`).
        n_spiders : int
            Nombre de bras, également répartis en angle (typiquement 3 ou 4).
        width : float
            Largeur d'un bras, en fraction du rayon de la pupille (ex. 0.01
            = 1 % du rayon).

        Returns
        -------
        NDArray
            Masque binaire. C'est le cas d'usage typique du §10 de
            ``REFERENCE_TECHNIQUE.md`` : la base de Zernike
            standard n'est plus orthogonale sur cette pupille (les bras
            brisent la symétrie azimutale) — utiliser
            :class:`zernike_basis_orthogonal.ZernikeGramSchmidt` pour la
            décomposition.
        """
        pupil = self.annular(obscuration)

        for i in range(n_spiders):
            angle = i * (2 * np.pi / n_spiders)
            # self.rho est normalisé à l'ouverture (aperture_px), pas au
            # cadre (npix) : reconvertir en distance pixel exige
            # aperture_px, pas npix (identiques quand aperture_px is None).
            dist_to_line = np.abs(np.sin(self.theta - angle) * self.rho * self.aperture_px / 2)
            pupil[dist_to_line < (width * self.aperture_px / 2)] = 0.0

        return pupil
