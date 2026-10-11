"""
A priori individuels par mode de Zernike (`zernike_priors`/`zernike_bounds`)
— généralisation de `OpticalConfig.l2_reg` à une moyenne et un poids par
mode (voir `eoqual.phaseretrieval.zernike_priors` et
`REFERENCE_TECHNIQUE.md` §6.5bis).
"""

from __future__ import annotations

import numpy as np
import pytest

from eoqual.phaseretrieval.backends.optimization import OpticalConfig, fit_zernike_from_psf
from eoqual.phaseretrieval.backends.phase_diversity import PhaseDiversityConfig, fit_phase_diversity
from eoqual.phaseretrieval.simulation import (
    PhaseDiversitySimulationConfig,
    simulate_phase_diversity_pair,
    simulate_psf_from_zernike,
)
from eoqual.phaseretrieval.zernike_priors import build_prior_bounds, build_prior_penalty

WAVELENGTH = 550e-9
NPIX = 128


# ---------------------------------------------------------------------------
# Unitaires : conversion nm <-> rad, valeurs par défaut, validation.
# ---------------------------------------------------------------------------

def test_build_prior_penalty_defaults_to_zero_weight_for_unset_modes():
    active = np.arange(3, 12)
    mean_rad, weight = build_prior_penalty({4: (30.0, 5.0)}, active, WAVELENGTH)

    assert weight.shape == active.shape
    assert np.count_nonzero(weight) == 1
    pos = int(np.where(active == 4)[0][0])
    assert weight[pos] > 0.0
    assert mean_rad[pos] != 0.0
    # Toutes les autres entrées : aucune pénalité.
    other = np.delete(np.arange(active.size), pos)
    assert np.all(weight[other] == 0.0)
    assert np.all(mean_rad[other] == 0.0)


def test_build_prior_penalty_roundtrips_to_nm():
    active = np.arange(3, 12)
    rad_to_nm = WAVELENGTH * 1e9 / (2 * np.pi)
    mean_rad, weight = build_prior_penalty({4: (30.0, 5.0)}, active, WAVELENGTH)
    pos = int(np.where(active == 4)[0][0])

    assert mean_rad[pos] * rad_to_nm == pytest.approx(30.0)
    # weight = rad_to_nm / sigma_nm -> un déplacement d'un sigma coûte 1 en résidu.
    delta_nm = 5.0
    pseudo_residual = weight[pos] * (delta_nm / rad_to_nm)
    assert pseudo_residual == pytest.approx(1.0)


def test_build_prior_penalty_rejects_non_positive_sigma():
    with pytest.raises(ValueError):
        build_prior_penalty({4: (30.0, 0.0)}, np.arange(3, 12), WAVELENGTH)


def test_build_prior_penalty_rejects_mode_outside_active_set():
    with pytest.raises(ValueError):
        build_prior_penalty({99: (30.0, 5.0)}, np.arange(3, 12), WAVELENGTH)


def test_build_prior_bounds_defaults_to_unbounded():
    active = np.arange(3, 12)
    lower, upper = build_prior_bounds({4: (10.0, 50.0)}, active, WAVELENGTH)

    pos = int(np.where(active == 4)[0][0])
    other = np.delete(np.arange(active.size), pos)
    assert np.all(np.isneginf(lower[other]))
    assert np.all(np.isposinf(upper[other]))
    assert np.isfinite(lower[pos]) and np.isfinite(upper[pos])
    assert lower[pos] < upper[pos]


def test_build_prior_bounds_rejects_inverted_interval():
    with pytest.raises(ValueError):
        build_prior_bounds({4: (50.0, 10.0)}, np.arange(3, 12), WAVELENGTH)


# ---------------------------------------------------------------------------
# Intégration : "optimization" — l2_reg est bien un cas particulier.
# ---------------------------------------------------------------------------

def test_zernike_priors_zero_mean_uniform_weight_matches_l2_reg():
    """
    Un a priori individuel (moyenne nulle, même écart-type sur tous les
    modes actifs) doit donner un résultat numériquement très proche de
    l2_reg avec le poids équivalent (l2_reg = weight**2 quand toutes les
    pénalités sont écrites en radians) — l2_reg n'est pas une méthode à
    part, juste le cas particulier "vers zéro, poids uniforme" de
    zernike_priors (voir REFERENCE_TECHNIQUE.md §6.5bis).
    """
    order = 12
    sim = simulate_psf_from_zernike({4: 40.0, 6: 15.0}, wavelength=WAVELENGTH, npix=NPIX, zernike_order=order)

    l2_reg = 1e-3
    rad_to_nm = WAVELENGTH * 1e9 / (2 * np.pi)
    sigma_nm_equiv = rad_to_nm / np.sqrt(l2_reg)

    priors = {k: (0.0, sigma_nm_equiv) for k in range(4, order + 1)}  # Noll 4..order

    coeffs_l2, _ = fit_zernike_from_psf(
        sim["psf"], sim["mask"],
        OpticalConfig(wavelength=WAVELENGTH, zernike_order=order, verbose=False, l2_reg=l2_reg),
    )
    coeffs_prior, _ = fit_zernike_from_psf(
        sim["psf"], sim["mask"],
        OpticalConfig(wavelength=WAVELENGTH, zernike_order=order, verbose=False, zernike_priors=priors),
    )

    assert coeffs_l2 == pytest.approx(coeffs_prior, abs=1.0)  # nm


def test_zernike_priors_debiases_an_underconstrained_fit():
    """
    À ordre élevé et sans a priori, un ajustement mono-PSF peut converger
    loin de la vérité (ambiguïtés, minima locaux — §3.7). Un a priori
    individuel fort et correct sur les modes concernés doit ramener
    l'estimation près de la vérité — la démonstration numérique de
    §6.5bis (MAP gaussien = pénalité L2) plutôt qu'une simple affirmation.
    """
    order = 12  # cf. les autres tests du projet : npix=128, ordre > 12 sujet
    # à un minimum local isolé pour "optimization" (voir CHANGELOG, correctif
    # PupilGenerator, et tests/test_simulation_roundtrip.py::ZERNIKE_ORDER).
    truth_nm = {4: 60.0, 6: 35.0, 8: -25.0}
    sim = simulate_psf_from_zernike(truth_nm, wavelength=WAVELENGTH, npix=NPIX, zernike_order=order)

    # A priori fort (petit sigma) et correct sur les modes de la vérité.
    priors = {k: (v, 2.0) for k, v in truth_nm.items()}

    coeffs, diag = fit_zernike_from_psf(
        sim["psf"], sim["mask"],
        OpticalConfig(wavelength=WAVELENGTH, zernike_order=order, verbose=False, zernike_priors=priors),
    )

    for noll, expected_nm in truth_nm.items():
        assert coeffs[noll - 1] == pytest.approx(expected_nm, abs=5.0)


def test_zernike_bounds_clamps_the_fit_inside_the_box():
    order = 12
    sim = simulate_psf_from_zernike({4: 80.0}, wavelength=WAVELENGTH, npix=NPIX, zernike_order=order)

    # La vérité (80 nm) est délibérément hors de la boîte -> la solution
    # doit rester sur la borne, pas y échapper.
    coeffs, _ = fit_zernike_from_psf(
        sim["psf"], sim["mask"],
        OpticalConfig(wavelength=WAVELENGTH, zernike_order=order, verbose=False, zernike_bounds={4: (0.0, 20.0)}),
    )

    assert -1e-6 <= coeffs[3] <= 20.0 + 1e-6


def test_zernike_priors_bounds_reject_mode_not_in_active_set_via_config():
    order = 12
    sim = simulate_psf_from_zernike({4: 40.0}, wavelength=WAVELENGTH, npix=NPIX, zernike_order=order)

    with pytest.raises(ValueError):
        fit_zernike_from_psf(
            sim["psf"], sim["mask"],
            OpticalConfig(wavelength=WAVELENGTH, zernike_order=order, verbose=False, zernike_priors={999: (0.0, 1.0)}),
        )


# ---------------------------------------------------------------------------
# Intégration : "phase_diversity" — même mécanisme, même API.
# ---------------------------------------------------------------------------

def test_phase_diversity_accepts_priors_and_bounds():
    order = 12
    truth_nm = {4: 275.0, 6: 20.0}  # Z4 proche de la diversité par défaut
    cfg = PhaseDiversitySimulationConfig(wavelength=WAVELENGTH, npix=NPIX, zernike_order=order, defocus=275e-9)
    sim = simulate_phase_diversity_pair(truth_nm, cfg)

    priors = {6: (20.0, 2.0)}
    bounds = {4: (0.0, 400.0)}

    coeffs, diag = fit_phase_diversity(
        sim["psf_focus"], sim["psf_defocus"], sim["mask"],
        PhaseDiversityConfig(
            wavelength=WAVELENGTH, zernike_order=order, defocus=275e-9, verbose=False,
            # loss="linear" : isole le mécanisme MAP gaussien pur de son
            # interaction avec la perte robuste (cf. le caveat documenté sur
            # PhaseDiversityConfig.zernike_priors — "soft_l1", par défaut,
            # amortit aussi la pénalité d'a priori pour un grand écart).
            loss="linear",
            zernike_priors=priors, zernike_bounds=bounds,
        ),
    )

    assert coeffs[5] == pytest.approx(20.0, abs=5.0)
    assert 0.0 - 1e-6 <= coeffs[3] <= 400.0 + 1e-6
