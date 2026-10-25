#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base de Zernike ré-orthogonalisée par Gram-Schmidt, pour pupilles réelles.

Les polynômes de Zernike (base ``poppy.zernike.zernike_basis`` utilisée par
:mod:`optics`) ne sont orthogonaux que sur une pupille circulaire **pleine,
non obstruée**. Pour une pupille réelle (obstruction centrale, bras
d'araignée — le cas typique en télédétection, voir
``REFERENCE_TECHNIQUE.md`` §8.1) cette base n'est plus
orthogonale : les modes se corrèlent, ce qui dégrade le conditionnement du
fit (déjà surveillé par ``diagnostics["condition_number"]`` dans
:func:`zernike_fit.fit_zernike`) et brouille l'interprétation physique de
chaque coefficient.

Ce module ré-orthogonalise numériquement la base sur le masque *réel*
fourni, par Gram-Schmidt modifié — cf. §3.7bis de
``REFERENCE_TECHNIQUE.md`` pour le contexte et les alternatives
(polynômes annulaires de Mahajan, modes de Karhunen-Loève).

Références
----------
- V. N. Mahajan, "Zernike annular polynomials for imaging systems with
  annular pupils," J. Opt. Soc. Am. 71(1), 75-85 (1981) — solution
  analytique pour une obstruction circulaire centrée, sans bras d'araignée.
- G. Dai, "Modified Hartmann-Shack wavefront sensing and iterative wavefront
  reconstruction," Opt. Eng. 35(12), 3333-3340 (1996).
- D. J. Schroeder, *Astronomical Optics*, Academic Press (1987).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .optics import build_zernike_basis


class ZernikeGramSchmidt:
    """
    Orthonormalise la base de Zernike (Noll) sur une pupille arbitraire.

    Parameters
    ----------
    pupil_mask : NDArray
        Masque de pupille (binaire ou pondéré dans ``[0, 1]``), forme
        ``(npix, npix)``.
    j_max : int, default 21
        Indice de Noll maximum à orthogonaliser.
    aperture_px : int, optional
        Diamètre [px] de l'ouverture dans ``pupil_mask`` — voir
        ``optics.build_zernike_basis``. ``None`` (défaut) : l'ouverture
        remplit tout le cadre (comportement historique). À renseigner
        quand ``pupil_mask`` vient d'une ouverture plus petite que son
        cadre (``pupil.PupilGenerator(npix, aperture_px=...)``,
        suréchantillonnage — §12.5 de ``REFERENCE_TECHNIQUE.md``) : sinon
        la base brute de Noll utilisée comme point de départ de la
        ré-orthogonalisation est déjà mal échelonnée par rapport à
        l'ouverture réelle, avant même la projection Gram-Schmidt.

    Attributes
    ----------
    orthonormal_modes : dict[int, NDArray]
        Modes orthonormés ``{j: Z_j_ortho}``, restreints au masque.
    orthogonality_error : float
        Erreur d'orthogonalité résiduelle maximale
        (``max|<Z_i|Z_j> - delta_ij|``) — doit être proche de l'epsilon
        machine ; une valeur ``> 1e-6`` signale une instabilité numérique
        (souvent : ``j_max`` trop élevé pour la résolution du masque).

    Notes
    -----
    Algorithme (Gram-Schmidt modifié, pour la stabilité numérique) :

    1. ``Z'_1 = Z_1`` (piston conservé).
    2. Pour ``j = 2, ..., j_max`` : on part du polynôme de Noll standard
       ``Z_j``, on soustrait sa projection sur chacun des modes déjà
       orthonormés ``Z'_1, ..., Z'_{j-1}``, puis on normalise.

    Les modes obtenus ne portent plus nécessairement le nom usuel de leur
    indice de Noll (« défocus », « coma », …) au-delà des tout premiers
    ordres : la ré-orthogonalisation mélange les polynômes standards dès
    que la pupille s'écarte du disque plein. C'est le prix de
    l'orthogonalité exacte sur un masque arbitraire.
    """

    def __init__(
        self,
        pupil_mask: NDArray[np.float64],
        j_max: int = 21,
        *,
        aperture_px: Optional[int] = None,
    ) -> None:
        self.pupil_mask = pupil_mask
        self.j_max = j_max
        self._pupil_area = float(np.sum(pupil_mask))
        if self._pupil_area <= 0:
            raise ValueError("Pupil mask must contain at least one non-zero pixel.")

        self._raw_basis = build_zernike_basis(
            npix=pupil_mask.shape[0], nterms=j_max, aperture_px=aperture_px,
        )

        self.orthonormal_modes: Dict[int, NDArray[np.float64]] = {}
        self._orthogonalize()
        self.orthogonality_error = self._check_orthogonality()

    def _inner_product(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """Produit scalaire sur la pupille : ``<a|b> = sum(a*b*mask) / sum(mask)``."""
        return float(np.sum(a * b * self.pupil_mask) / self._pupil_area)

    def _norm(self, a: NDArray[np.float64]) -> float:
        """Norme L2 sur la pupille."""
        return float(np.sqrt(self._inner_product(a, a)))

    def _orthogonalize(self) -> None:
        for j in range(1, self.j_max + 1):
            z_j = self._raw_basis[j - 1] * self.pupil_mask

            z_ortho = z_j.copy()
            for k in range(1, j):
                if k in self.orthonormal_modes:
                    projection = self._inner_product(z_j, self.orthonormal_modes[k])
                    z_ortho = z_ortho - projection * self.orthonormal_modes[k]

            norm = self._norm(z_ortho)
            if norm < 1e-12:
                # Mode linéairement dépendant des précédents sur ce masque : ignoré.
                continue

            self.orthonormal_modes[j] = z_ortho / norm

    def _check_orthogonality(self) -> float:
        j_list = sorted(self.orthonormal_modes)
        n = len(j_list)
        gram = np.zeros((n, n))
        for i, ji in enumerate(j_list):
            for k, jk in enumerate(j_list):
                gram[i, k] = self._inner_product(
                    self.orthonormal_modes[ji], self.orthonormal_modes[jk]
                )
        return float(np.max(np.abs(gram - np.eye(n))))

    def get_mode(self, j: int) -> NDArray[np.float64]:
        """
        Récupère un mode orthonormé.

        Parameters
        ----------
        j : int
            Indice de Noll.

        Returns
        -------
        NDArray

        Raises
        ------
        KeyError
            Si le mode n'a pas été généré (absent de ``orthonormal_modes``,
            typiquement linéairement dépendant sur ce masque).
        """
        if j not in self.orthonormal_modes:
            raise KeyError(f"Mode Z{j} indisponible (j_max={self.j_max}).")
        return self.orthonormal_modes[j]

    def as_basis_array(self) -> NDArray[np.float64]:
        """
        Renvoie les modes disponibles comme un tableau dense, dans l'ordre
        de Noll croissant — compatible avec les fonctions de
        :mod:`optics` (``phase_from_coefficients``, ``psf_from_phase``) qui
        attendent un tableau ``(nterms, npix, npix)``.

        Returns
        -------
        NDArray
            Shape ``(len(orthonormal_modes), npix, npix)``.
        """
        j_list = sorted(self.orthonormal_modes)
        return np.stack([self.orthonormal_modes[j] for j in j_list], axis=0)

    def decompose(self, wavefront: NDArray[np.float64]) -> Dict[int, float]:
        """
        Décompose un front d'onde sur les modes orthonormés.

        Parameters
        ----------
        wavefront : NDArray
            Carte de phase à décomposer, mêmes dimensions que le masque.

        Returns
        -------
        dict[int, float]
            Coefficients ``{j: a_j}`` avec ``a_j = <wavefront | Z_j_ortho>``.
        """
        return {
            j: self._inner_product(wavefront, mode)
            for j, mode in self.orthonormal_modes.items()
        }

    def reconstruct(self, coeffs: Dict[int, float]) -> NDArray[np.float64]:
        """
        Reconstruit un front d'onde à partir de coefficients.

        Parameters
        ----------
        coeffs : dict[int, float]
            Coefficients ``{j: a_j}``.

        Returns
        -------
        NDArray
            ``wavefront = sum_j a_j * Z_j_ortho``.
        """
        wavefront = np.zeros_like(self.pupil_mask, dtype=np.float64)
        for j, a in coeffs.items():
            if j in self.orthonormal_modes:
                wavefront = wavefront + a * self.orthonormal_modes[j]
        return wavefront


def resolve_zernike_basis(
    npix: int,
    nterms: int,
    mask: NDArray[np.float64],
    *,
    orthogonalize: bool = False,
    aperture_px: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Point d'entrée commun pour choisir la base de Zernike utilisée par les
    trois méthodes de récupération de phase (§6.1) et par ``zernike_fit``.

    Parameters
    ----------
    npix : int
        Taille (pixels) de la grille, carrée.
    nterms : int
        Nombre de modes (ordre de Noll maximal).
    mask : NDArray
        Masque pupillaire réellement utilisé — seul argument qui change
        selon ``orthogonalize``.
    orthogonalize : bool, default False
        ``False`` (comportement historique, inchangé) : base de Noll
        standard (:func:`optics.build_zernike_basis`), orthogonale
        uniquement sur un disque plein non obstrué. ``True`` : base
        ré-orthogonalisée par Gram-Schmidt sur ``mask`` (§10) —
        recommandé dès que la pupille n'est pas un disque plein
        (obstruction centrale, bras d'araignée), cas le plus courant en
        télédétection (§8.1).
    aperture_px : int, optional
        Diamètre [px] de l'ouverture dans ``mask``, si elle ne remplit pas
        tout le cadre ``npix x npix`` — voir ``optics.build_zernike_basis``
        et §12.5 de ``REFERENCE_TECHNIQUE.md``. ``None`` (défaut) :
        l'ouverture remplit le cadre (comportement historique).

    Returns
    -------
    NDArray
        Base ``(nterms, npix, npix)``, directement utilisable par
        ``optics.phase_from_coefficients``/``psf_from_phase`` : même
        forme et même correspondance indice ↔ ordre de Noll dans les deux
        cas (``basis[k]`` = mode de Noll ``k + 1``).

    Raises
    ------
    ValueError
        Si ``orthogonalize=True`` et qu'au moins un mode d'indice ≤
        ``nterms`` est linéairement dépendant des précédents sur ce
        masque (cas dégénéré — masque trop obstrué ou sous-échantillonné
        pour ``nterms``) : ``ZernikeGramSchmidt`` l'ignore silencieusement
        (voir sa docstring), ce qui décalerait la correspondance
        indice ↔ Noll dans le tableau dense renvoyé par
        ``as_basis_array`` — mieux vaut échouer explicitement que
        renvoyer des coefficients dont l'indexation a glissé.
    """
    if not orthogonalize:
        return build_zernike_basis(npix=npix, nterms=nterms, aperture_px=aperture_px)

    gram_schmidt = ZernikeGramSchmidt(mask, j_max=nterms, aperture_px=aperture_px)
    available = sorted(gram_schmidt.orthonormal_modes)
    if available != list(range(1, nterms + 1)):
        missing = sorted(set(range(1, nterms + 1)) - set(available))
        raise ValueError(
            f"Gram-Schmidt a écarté le(s) mode(s) {missing} comme "
            f"linéairement dépendant(s) sur ce masque (orthogonality_error="
            f"{gram_schmidt.orthogonality_error:.2e}) — réduire nterms/"
            "zernike_order, ou vérifier l'échantillonnage du masque, plutôt "
            "que de laisser la correspondance indice/Noll glisser."
        )
    return gram_schmidt.as_basis_array()
