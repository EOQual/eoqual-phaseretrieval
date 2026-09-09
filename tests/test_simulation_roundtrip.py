"""
Validation de bout en bout : simulation (vérité de Zernike connue) ->
récupération de phase -> comparaison à la vérité terrain.

Ces tests ne dépendent d'aucune donnée externe (fixtures ``.tiff``) :
le module ``eoqual.phaseretrieval.simulation`` génère tout ce qu'il faut,
ce qui est précisément son rôle (voir REFERENCE_TECHNIQUE.md §6.4).
"""

from __future__ import annotations

import numpy as np
import pytest

from eoqual.phaseretrieval.backends.gerchberg_saxton import RetrievalConfig, gerchberg_saxton
from eoqual.phaseretrieval.backends.optimization import OpticalConfig, fit_zernike_from_psf
from eoqual.phaseretrieval.backends.phase_diversity import PhaseDiversityConfig, fit_phase_diversity
from eoqual.phaseretrieval.simulation import (
    DetectorNoiseConfig,
    PhaseDiversitySimulationConfig,
    simulate_phase_diversity_pair,
    simulate_psf_from_zernike,
)
from eoqual.phaseretrieval.zernike_fit import ZernikeConfig, fit_zernike

TRUTH_NM = {4: 55.0, 5: 27.5}  # défocus + astigmatisme, cf. Data/*_simu.tiff historiques
WAVELENGTH = 550e-9
NPIX = 128
ZERNIKE_ORDER = 12  # ordre 15 sujet à un minimum local isolé pour "optimization" à ce npix


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def test_simulate_psf_from_zernike_matches_requested_coefficients():
    sim = simulate_psf_from_zernike(
        TRUTH_NM, wavelength=WAVELENGTH, npix=NPIX, zernike_order=ZERNIKE_ORDER
    )
    assert sim["psf"].shape == (NPIX, NPIX)
    assert sim["psf"] is sim["psf_truth"]  # pas de bruit demandé
    assert sim["coeffs_nm"][3] == pytest.approx(55.0)
    assert sim["coeffs_nm"][4] == pytest.approx(27.5)
    assert np.isclose(sim["psf"].sum(), 1.0)


def test_optimization_recovers_truth_without_noise():
    sim = simulate_psf_from_zernike(
        TRUTH_NM, wavelength=WAVELENGTH, npix=NPIX, zernike_order=ZERNIKE_ORDER
    )
    cfg = OpticalConfig(wavelength=WAVELENGTH, zernike_order=ZERNIKE_ORDER, verbose=False)
    coeffs_nm, diagnostics = fit_zernike_from_psf(sim["psf"], sim["mask"], cfg)

    # Ambiguïté de signe attendue sur une seule PSF (voir §3.7) : on
    # compare en valeur absolue de corrélation.
    assert abs(_corr(coeffs_nm[3:], sim["coeffs_nm"][3:])) > 0.999
    assert diagnostics["cost"] < 1e-15


def test_gerchberg_saxton_runs_and_projects_to_zernike():
    sim = simulate_psf_from_zernike(
        TRUTH_NM, wavelength=WAVELENGTH, npix=NPIX, zernike_order=ZERNIKE_ORDER
    )
    gs_cfg = RetrievalConfig(max_iter=100, verbose=False, seed=0)
    phase_rad = gerchberg_saxton(sim["psf"], sim["mask"], gs_cfg)

    zern_cfg = ZernikeConfig(order=ZERNIKE_ORDER, wavelength=WAVELENGTH)
    coeffs_nm, _phase_fit_nm, diagnostics = fit_zernike(phase_rad, sim["mask"], zern_cfg)

    assert coeffs_nm.shape == (ZERNIKE_ORDER,)
    assert "residual_rms_nm" in diagnostics


def test_phase_diversity_resolves_sign_ambiguity_without_noise():
    pd_sim_cfg = PhaseDiversitySimulationConfig(
        wavelength=WAVELENGTH, npix=NPIX, zernike_order=ZERNIKE_ORDER, defocus=275e-9
    )
    pair = simulate_phase_diversity_pair(TRUTH_NM, pd_sim_cfg)

    pd_cfg = PhaseDiversityConfig(
        wavelength=WAVELENGTH, zernike_order=ZERNIKE_ORDER, defocus=pair["defocus"],
        verbose=False, max_nfev=300,
    )
    coeffs_nm, diagnostics = fit_phase_diversity(
        pair["psf_focus"], pair["psf_defocus"], pair["mask"], pd_cfg
    )

    # Contrairement aux deux méthodes mono-PSF : signe exact, pas juste
    # |corrélation| (c'est précisément ce que la diversité de phase lève).
    assert _corr(coeffs_nm[3:], pair["coeffs_nm"][3:]) > 0.999
    assert diagnostics["psf_focus_rmse"] < 1e-6


def test_phase_diversity_recovers_truth_with_realistic_noise():
    pd_sim_cfg = PhaseDiversitySimulationConfig(
        wavelength=WAVELENGTH, npix=NPIX, zernike_order=ZERNIKE_ORDER, defocus=275e-9,
        noise=DetectorNoiseConfig(photon_count=1e6, read_noise_e=5.0, seed=0),
    )
    pair = simulate_phase_diversity_pair(TRUTH_NM, pd_sim_cfg)

    pd_cfg = PhaseDiversityConfig(
        wavelength=WAVELENGTH, zernike_order=ZERNIKE_ORDER, defocus=pair["defocus"],
        verbose=False, max_nfev=300, poisson_weighting=True,
    )
    coeffs_nm, _diagnostics = fit_phase_diversity(
        pair["psf_focus"], pair["psf_defocus"], pair["mask"], pd_cfg
    )

    # Régime bruité : tolérance plus large qu'en l'absence de bruit, mais
    # toujours le bon signe et une reconstruction qualitativement correcte.
    assert _corr(coeffs_nm[3:], pair["coeffs_nm"][3:]) > 0.8


def test_optimization_runs_on_noisy_psf_with_negative_pixels():
    """
    Régression : le bruit de lecture (gaussien additif,
    ``DetectorNoiseConfig``) peut pousser des pixels sous zéro — une PSF
    "mesurée" réelle après soustraction de fond peut en faire autant.
    ``residuals()`` doit clipper avant la racine carrée (comme
    ``gerchberg_saxton`` le fait déjà pour sa cible d'amplitude) plutôt que
    de planter dans ``least_squares`` ("Residuals are not finite").
    """
    sim = simulate_psf_from_zernike(
        TRUTH_NM, wavelength=WAVELENGTH, npix=NPIX, zernike_order=ZERNIKE_ORDER,
        noise=DetectorNoiseConfig(photon_count=1e6, read_noise_e=5.0, seed=0),
    )
    assert (sim["psf"] < 0).any()  # confirme que ce cas de test est bien piégeux

    cfg = OpticalConfig(wavelength=WAVELENGTH, zernike_order=ZERNIKE_ORDER, verbose=False)
    coeffs_nm, diagnostics = fit_zernike_from_psf(sim["psf"], sim["mask"], cfg)

    assert np.all(np.isfinite(coeffs_nm))
    # |corrélation| : "optimization" sur 1 PSF a l'ambiguïté de signe
    # attendue (§3.7) — le point vérifié ici est l'absence de crash/NaN,
    # pas la résolution du signe.
    assert abs(_corr(coeffs_nm[3:], sim["coeffs_nm"][3:])) > 0.8
