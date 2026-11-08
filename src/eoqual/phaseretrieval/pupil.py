#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération de masques de pupille : disque simple, pupille annulaire
(obstruction centrale) et pupille avec bras d'araignée.

Complète [`zernike_basis_orthogonal.py`](zernike_basis_orthogonal.py) (§10
de ``REFERENCE_TECHNIQUE.md``) : ce module génère les pupilles
non standard sur lesquelles la base de Zernike doit être ré-orthogonalisée,
plutôt que de les construire à la main à chaque fois.

La grille polaire (:func:`polar_grid`) et la pupille annulaire viennent
d'``eoqual.core.pupil`` (partagées avec eoqual-deconvolution) ;
``polar_grid`` reste importable d'ici.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from eoqual.core.psf import supersampled_coverage
from eoqual.core.pupil import annular_pupil, polar_grid
from numpy.typing import NDArray




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
        return annular_pupil(self.rho)

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
        return annular_pupil(self.rho, obscuration)

    def with_spiders(
        self,
        obscuration: float,
        n_spiders: int,
        width: float,
        *,
        supersample: int = 8,
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
        supersample : int, default 8
            Facteur de sous-échantillonnage pour adoucir le bord de
            chaque bras (:func:`eoqual.core.psf.supersampled_coverage`) —
            voir "Ce qui ne va pas sans ça" ci-dessous. ``1`` retrouve un
            seuillage binaire pur (comportement historique).

        Returns
        -------
        NDArray
            Masque gradué dans ``[0, 1]`` au bord des bras (binaire
            ailleurs — le contour principal et l'obstruction centrale, eux,
            restent un seuillage binaire pur : voir §10.3 de
            ``REFERENCE_TECHNIQUE.md``). C'est le cas d'usage typique du
            §10 : la base de Zernike standard n'est plus orthogonale sur
            cette pupille (les bras brisent la symétrie azimutale) —
            utiliser :class:`zernike_basis_orthogonal.ZernikeGramSchmidt`
            pour la décomposition.

            **Ce qui ne va pas sans ça** : un bras est typiquement fin
            (largeur de l'ordre du pixel) — un seuillage binaire pur lui
            donne une transmission qui dépend de façon quasi arbitraire de
            l'alignement sous-pixel de la grille (un artefact numérique,
            la pupille physique ne « sait » pas où elle tombe sur la
            grille), jusqu'à 2 % d'écart de PSF et 1,5 % de Strehl rien
            qu'en déplaçant le masque d'une fraction de pixel — réduit à
            0,2 %/0,2 % avec le suréchantillonnage par défaut (voir §10.3).
        """
        pupil = self.annular(obscuration)
        # Même convention de centrage que polar_grid (npix // 2, pas
        # (npix - 1) / 2) : supersampled_coverage ne suppose rien à ce
        # sujet, c'est à l'appelant de rester cohérent.
        center = (self.npix // 2, self.npix // 2)
        half_aperture = self.aperture_px // 2  # convention de polar_grid

        for i in range(n_spiders):
            angle = i * (2 * np.pi / n_spiders)

            def blocked(y, x, angle=angle):
                # Même formule que la version historique (binaire),
                # évaluée ici sur la grille fine plutôt que sur self.rho/
                # self.theta — voir la note sur half_aperture ci-dessus
                # pour pourquoi ce n'est pas simplement sqrt(x^2+y^2).
                rho = np.sqrt(x**2 + y**2) / half_aperture
                theta = np.arctan2(y, x)
                dist_to_line = np.abs(np.sin(theta - angle) * rho * self.aperture_px / 2)
                return dist_to_line < (width * self.aperture_px / 2)

            coverage = supersampled_coverage(blocked, (self.npix, self.npix), center, supersample)
            pupil *= 1.0 - coverage

        return pupil
