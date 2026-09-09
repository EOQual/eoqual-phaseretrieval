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

import numpy as np
from numpy.typing import NDArray


class PupilGenerator:
    """
    Générateur de masques de pupille sur une grille carrée ``(N, N)``.

    Parameters
    ----------
    npix : int
        Taille de la grille (pixels).
    """

    def __init__(self, npix: int) -> None:
        self.npix = npix
        y, x = np.indices((npix, npix)) - npix // 2
        # Rayon normalisé, volontairement NON clippé à 1.0 : les méthodes
        # ci-dessous testent justement `rho <= 1.0` pour délimiter la
        # pupille (bug corrigé — un clip ici rendait ce test toujours vrai,
        # produisant un masque plein-cadre au lieu d'un disque/anneau réel ;
        # voir tests/test_pupil.py::test_simple_disk_excludes_corners).
        self.rho = np.sqrt(x ** 2 + y ** 2) / (npix // 2)
        self.theta = np.arctan2(y, x)

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
            dist_to_line = np.abs(np.sin(self.theta - angle) * self.rho * self.npix / 2)
            pupil[dist_to_line < (width * self.npix / 2)] = 0.0

        return pupil
