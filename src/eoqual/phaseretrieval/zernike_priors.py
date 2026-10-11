#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A priori individuels par mode de Zernike (généralisation de ``l2_reg``).

**Logique.** Une estimation MAP (maximum a posteriori) maximise
``p(c | mesure) ∝ p(mesure | c) * p(c)``, donc minimise
``-log p(mesure|c) - log p(c)``. Avec un bruit gaussien,
``-log p(mesure|c) = ||modèle - mesure||²`` : c'est le moindres-carrés déjà
utilisé partout dans ce projet. Avec un a priori gaussien par mode
``c_k ~ N(mean_k, sigma_k²)``, ``-log p(c) = Σ_k (c_k - mean_k)² / sigma_k²``
: une **pénalité L2**. Avec un a priori *uniforme* sur un intervalle
``[low_k, high_k]`` (et nul en dehors), ``-log p(c)`` est nul à l'intérieur
et infini en dehors : une **contrainte de boîte**. Un MAP gaussien et un
moindres-carrés pénalisé L2 ne sont donc pas deux méthodes différentes —
c'est la même formule vue de deux vocabulaires (statistique / numérique) ;
de même pour un MAP à prior uniforme et une borne dure.

``OpticalConfig.l2_reg`` (``backends/optimization.py``) est déjà un cas
particulier de ce mécanisme : un a priori gaussien centré sur **zéro**,
avec le **même** poids pour tous les modes. Ce module généralise ce cas
particulier à un a priori individuel (moyenne et poids propres à chaque
mode), sous forme de pénalité (:func:`build_prior_penalty`) ou de borne
dure (:func:`build_prior_bounds`) — voir ``REFERENCE_TECHNIQUE.md`` §6.5bis
pour la discussion complète (pourquoi une option plutôt qu'un nouveau
backend, et la comparaison de performance qui motive ce choix).

Utilisé par les deux méthodes basées optimisation
(:mod:`backends.optimization`, :mod:`backends.phase_diversity`) ;
sans objet pour Gerchberg-Saxton (pas de fonction de coût explicite sur les
coefficients de Zernike à pénaliser).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray


def _validate_indices(
    priors: Dict[int, Tuple[float, float]],
    active_noll: NDArray[np.int_],
    *,
    param_name: str,
) -> None:
    """Erreur explicite plutôt qu'un a priori silencieusement ignoré."""
    invalid = sorted(set(priors) - set(int(i) for i in active_noll))
    if invalid:
        raise ValueError(
            f"{param_name} référence le(s) mode(s) de Noll {invalid}, qui ne "
            f"font pas partie des coefficients réellement ajustés dans cet "
            f"appel ({sorted(int(i) for i in active_noll)}) — vérifier "
            f"zernike_order (et fit_piston_tilt le cas échéant)."
        )


def build_prior_penalty(
    priors_nm: Dict[int, Tuple[float, float]],
    active_noll: NDArray[np.int_],
    wavelength: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Vecteurs (moyenne [rad], poids [1/rad]) alignés sur ``active_noll``, pour une
    pénalité L2 par mode à ajouter au résidu de moindres carrés : le
    pseudo-résidu ``weight * (c - mean)`` (voir usage dans
    ``backends.optimization.fit_zernike_from_psf`` /
    ``backends.phase_diversity.fit_phase_diversity``) contribue
    ``((c_nm - mean_nm) / sigma_nm)²`` au coût total — l'équivalent exact,
    en moindres-carrés, d'un a priori gaussien par mode (voir la
    dérivation en tête de module). Un mode absent de ``priors_nm`` reçoit
    un poids nul (aucune pénalité), comme s'il n'avait pas d'a priori.

    Parameters
    ----------
    priors_nm : dict
        Index de Noll -> ``(valeur attendue [nm], écart-type [nm])``. Un
        écart-type petit = a priori "fort" (poids élevé) ; un écart-type
        grand = a priori "faible". Doit être strictement positif.
    active_noll : NDArray
        Indices de **Noll** (1-based, ex. 4 = défocus) réellement libres dans
        cet ajustement, dans l'ordre où ils apparaissent dans le vecteur de
        paramètres du solveur — **pas** les positions 0-based du tableau de
        coefficients (à décaler de ``+1`` par l'appelant, ex.
        ``active_noll = active_array_positions + 1``).
    wavelength : float
        Longueur d'onde [m], pour la conversion nm <-> rad.

    Returns
    -------
    mean_rad, weight : NDArray, NDArray
        Alignés sur ``active`` ; ``weight[k] == 0`` pour un mode sans a
        priori (le pseudo-résidu correspondant est alors toujours nul).

    Raises
    ------
    ValueError
        Si un index de ``priors_nm`` n'est pas dans ``active_noll``, ou si un
        écart-type n'est pas strictement positif.
    """
    _validate_indices(priors_nm, active_noll, param_name="zernike_priors")
    rad_to_nm = wavelength * 1e9 / (2 * np.pi)

    mean_rad = np.zeros(active_noll.size, dtype=np.float64)
    weight = np.zeros(active_noll.size, dtype=np.float64)
    for pos, noll in enumerate(active_noll):
        entry = priors_nm.get(int(noll))
        if entry is None:
            continue
        mean_nm, sigma_nm = entry
        if sigma_nm <= 0.0:
            raise ValueError(
                f"zernike_priors[{int(noll)}] : écart-type {sigma_nm} nm "
                "doit être strictement positif."
            )
        mean_rad[pos] = mean_nm / rad_to_nm
        weight[pos] = rad_to_nm / sigma_nm

    return mean_rad, weight


def build_prior_bounds(
    bounds_nm: Dict[int, Tuple[float, float]],
    active_noll: NDArray[np.int_],
    wavelength: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Vecteurs (borne basse, borne haute) [rad], alignés sur ``active_noll``, pour
    une contrainte de boîte par mode — l'équivalent d'un a priori
    **uniforme** par mode (borne dure plutôt que pénalité continue, voir la
    dérivation en tête de module). Passés tels quels à
    ``scipy.optimize.least_squares(bounds=...)``. Un mode absent de
    ``bounds_nm`` reste libre (``[-inf, +inf]``).

    Parameters
    ----------
    bounds_nm : dict
        Index de Noll -> ``(borne basse [nm], borne haute [nm])``.
    active_noll : NDArray
        Indices de **Noll** (1-based) réellement libres dans cet ajustement —
        voir la même remise en garde que :func:`build_prior_penalty`.
    wavelength : float
        Longueur d'onde [m], pour la conversion nm <-> rad.

    Returns
    -------
    lower_rad, upper_rad : NDArray, NDArray
        Alignés sur ``active`` ; ``-inf``/``+inf`` pour un mode sans borne.

    Raises
    ------
    ValueError
        Si un index de ``bounds_nm`` n'est pas dans ``active_noll``, ou si
        ``borne basse >= borne haute``.
    """
    _validate_indices(bounds_nm, active_noll, param_name="zernike_bounds")
    rad_to_nm = wavelength * 1e9 / (2 * np.pi)

    lower_rad = np.full(active_noll.size, -np.inf, dtype=np.float64)
    upper_rad = np.full(active_noll.size, np.inf, dtype=np.float64)
    for pos, noll in enumerate(active_noll):
        entry = bounds_nm.get(int(noll))
        if entry is None:
            continue
        low_nm, high_nm = entry
        if low_nm >= high_nm:
            raise ValueError(
                f"zernike_bounds[{int(noll)}] : borne basse {low_nm} nm >= "
                f"borne haute {high_nm} nm."
            )
        lower_rad[pos] = low_nm / rad_to_nm
        upper_rad[pos] = high_nm / rad_to_nm

    return lower_rad, upper_rad


def feasible_start(
    lower_rad: NDArray[np.float64],
    upper_rad: NDArray[np.float64],
    *,
    nudge: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Point de départ à zéro, mais écarté d'une borne finie sur laquelle il
    tomberait exactement.

    ``scipy.optimize.least_squares(method="trf", bounds=...)`` accepte un
    point de départ *sur* une borne (ce n'est pas « infaisable »), mais sa
    mise à l'échelle interne (Coleman-Li) y devient dégénérée : le solveur
    peut alors déclarer une convergence immédiate et fictive dès la
    première évaluation, sans avoir bougé. Observé concrètement avec
    ``zernike_bounds={4: (0.0, 400.0)}`` : partir de ``x0=0``, exactement
    sur la borne basse, empêchait tout déplacement (``nfev=2``,
    "`ftol` termination condition satisfied" trompeur). ``nudge`` (en
    radians, très petit devant l'échelle usuelle d'un coefficient de
    Zernike, elle-même de l'ordre de ``nm / (wavelength/2π)`` soit ~0.01 à
    quelques rad) écarte le départ d'une borne finie sans influencer le
    résultat convergé.

    Parameters
    ----------
    lower_rad, upper_rad : NDArray
        Bornes par mode [rad], typiquement issues de
        :func:`build_prior_bounds` (``-inf``/``+inf`` pour un mode libre).
    nudge : float, default 1e-6
        Écart minimal [rad] à une borne finie.

    Returns
    -------
    NDArray
        Point de départ, même forme que ``lower_rad``.
    """
    x0 = np.clip(np.zeros_like(lower_rad), lower_rad, upper_rad)

    at_lower = np.isfinite(lower_rad) & (x0 <= lower_rad)
    x0 = np.where(at_lower, np.minimum(lower_rad + nudge, upper_rad), x0)

    at_upper = np.isfinite(upper_rad) & (x0 >= upper_rad) & ~at_lower
    x0 = np.where(at_upper, np.maximum(upper_rad - nudge, lower_rad), x0)

    return x0
