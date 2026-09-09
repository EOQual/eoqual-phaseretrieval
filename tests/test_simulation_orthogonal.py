"""
Extension du module de simulation pour ``orthogonal_basis`` (Gram-Schmidt,
REFERENCE_TECHNIQUE.md §10), et conversion des coefficients ainsi obtenus
vers la base de Zernike standard (``project_orthogonal_to_standard``).
"""

from __future__ import annotations

import numpy as np
import pytest

from eoqual.phaseretrieval.backends.optimization import OpticalConfig, fit_zernike_from_psf
from eoqual.phaseretrieval.backends.phase_diversity import PhaseDiversityConfig, fit_phase_diversity
from eoqual.phaseretrieval.pupil import PupilGenerator
from eoqual.phaseretrieval.simulation import (
    DetectorNoiseConfig,
    PhaseDiversitySimulationConfig,
    simulate_phase_diversity_pair,
    simulate_psf_from_zernike,
)
from eoqual.phaseretrieval.zernike_basis_orthogonal import resolve_zernike_basis
from eoqual.phaseretrieval.zernike_fit import project_orthogonal_to_standard

NPIX = 128
ORDER = 12
WAVELENGTH = 550e-9
TRUTH_NM = {4: 55.0, 5: 27.5}


def _obstructed_mask() -> np.ndarray:
    return PupilGenerator(NPIX).annular(obscuration=0.3)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def test_simulate_psf_from_zernike_orthogonal_basis_matches_gram_schmidt_reconstruction():
    mask = _obstructed_mask()
    basis = resolve_zernike_basis(npix=NPIX, nterms=ORDER, mask=mask, orthogonalize=True)
    rad_to_nm = WAVELENGTH * 1e9 / (2 * np.pi)

    sim = simulate_psf_from_zernike(
        TRUTH_NM, wavelength=WAVELENGTH, npix=NPIX, zernike_order=ORDER,
        mask=mask, orthogonal_basis=True,
    )
    assert sim["orthogonal_basis"] is True

    coeffs_rad = sim["coeffs_nm"] / rad_to_nm
    expected_phase = np.tensordot(coeffs_rad, basis, axes=(0, 0)) * mask
    from eoqual.phaseretrieval.optics import psf_from_phase
    expected_psf = psf_from_phase(expected_phase, mask)
    np.testing.assert_allclose(sim["psf_truth"], expected_psf)


def test_optimization_roundtrip_is_exact_when_simulation_and_fit_share_the_basis():
    """
    Le point relevé précédemment (REFERENCE_TECHNIQUE.md §10) : comparer
    une vérité en base standard à un ajustement en base Gram-Schmidt (ou
    l'inverse) n'est pas une comparaison juste. En simulant *aussi* en
    base Gram-Schmidt, la comparaison redevient directe et quasi exacte —
    contrairement à la corrélation ≈ 0,99 observée quand la vérité était en
    base standard.
    """
    mask = _obstructed_mask()
    sim = simulate_psf_from_zernike(
        TRUTH_NM, wavelength=WAVELENGTH, npix=NPIX, zernike_order=ORDER,
        mask=mask, orthogonal_basis=True,
    )

    cfg = OpticalConfig(wavelength=WAVELENGTH, zernike_order=ORDER, verbose=False, orthogonal_basis=True)
    coeffs_nm, diagnostics = fit_zernike_from_psf(sim["psf"], mask, cfg)

    assert _corr(coeffs_nm[3:], sim["coeffs_nm"][3:]) > 0.9999
    assert diagnostics["cost"] < 1e-15


def test_phase_diversity_roundtrip_with_orthogonal_basis_and_noise():
    mask = _obstructed_mask()
    sim_cfg = PhaseDiversitySimulationConfig(
        wavelength=WAVELENGTH, npix=NPIX, zernike_order=ORDER, defocus=275e-9,
        orthogonal_basis=True, noise=DetectorNoiseConfig(photon_count=1e6, read_noise_e=5.0, seed=0),
    )
    pair = simulate_phase_diversity_pair(TRUTH_NM, sim_cfg, mask=mask)
    assert pair["orthogonal_basis"] is True

    pd_cfg = PhaseDiversityConfig(
        wavelength=WAVELENGTH, zernike_order=ORDER, defocus=pair["defocus"],
        verbose=False, max_nfev=300, orthogonal_basis=True,
    )
    coeffs_nm, _diagnostics = fit_phase_diversity(pair["psf_focus"], pair["psf_defocus"], mask, pd_cfg)

    assert _corr(coeffs_nm[3:], pair["coeffs_nm"][3:]) > 0.8


def test_project_orthogonal_to_standard_recovers_standard_truth_approximately():
    """
    Cas réaliste : la vérité (ou l'aberration réelle) est pensée en base
    standard (Z4 = défocus, Z5 = astigmatisme...), mais l'ajustement est
    fait en base Gram-Schmidt pour son meilleur conditionnement sur une
    pupille obstruée. ``project_orthogonal_to_standard`` doit permettre de
    retrouver des coefficients proches de la vérité standard d'origine —
    à la précision de convergence du solveur près (voir `cost`), la
    reprojection elle-même étant un calcul linéaire exact.
    """
    mask = _obstructed_mask()
    sim = simulate_psf_from_zernike(
        TRUTH_NM, wavelength=WAVELENGTH, npix=NPIX, zernike_order=ORDER,
        mask=mask, orthogonal_basis=False,  # vérité en base standard
    )

    cfg = OpticalConfig(wavelength=WAVELENGTH, zernike_order=ORDER, verbose=False, orthogonal_basis=True)
    coeffs_gs_nm, diagnostics = fit_zernike_from_psf(sim["psf"], mask, cfg)

    basis_gs = resolve_zernike_basis(npix=NPIX, nterms=ORDER, mask=mask, orthogonalize=True)
    coeffs_std_nm, _phase_fit_nm, diag2 = project_orthogonal_to_standard(
        coeffs_gs_nm, basis_gs, mask, wavelength=WAVELENGTH,
    )

    assert diagnostics["cost"] < 1e-3
    assert coeffs_std_nm[3] == pytest.approx(55.0, rel=0.1)
    assert coeffs_std_nm[4] == pytest.approx(27.5, rel=0.1)
    assert diag2["condition_number"] > 1.0  # non trivial sur une pupille obstruée


def test_project_orthogonal_to_standard_is_exact_given_an_already_converged_phase():
    """
    Isole la reprojection elle-même de la convergence du solveur en amont :
    construit directement la phase à partir de coefficients Gram-Schmidt
    connus (pas de ``least_squares`` entre les deux), et vérifie qu'elle
    donne exactement les mêmes coefficients standard qu'un ``fit_zernike``
    direct sur cette phase — l'invariant qui compte réellement : la
    reprojection ne fait rien d'autre que ce que ``fit_zernike`` ferait
    directement sur la même phase.
    """
    from eoqual.phaseretrieval.optics import phase_from_coefficients
    from eoqual.phaseretrieval.zernike_fit import ZernikeConfig, fit_zernike

    mask = _obstructed_mask()
    basis_gs = resolve_zernike_basis(npix=NPIX, nterms=ORDER, mask=mask, orthogonalize=True)
    rad_to_nm = WAVELENGTH * 1e9 / (2 * np.pi)

    coeffs_gs_nm = np.zeros(ORDER)
    coeffs_gs_nm[3] = 55.0
    coeffs_gs_nm[4] = 27.5
    original_phase_rad = phase_from_coefficients(coeffs_gs_nm / rad_to_nm, basis_gs, mask)

    coeffs_std_nm, phase_fit_nm, _diag = project_orthogonal_to_standard(
        coeffs_gs_nm, basis_gs, mask, wavelength=WAVELENGTH,
    )

    zern_cfg = ZernikeConfig(order=ORDER, wavelength=WAVELENGTH, orthogonal_basis=False)
    coeffs_direct_nm, phase_direct_nm, _diag_direct = fit_zernike(original_phase_rad, mask, zern_cfg)

    np.testing.assert_allclose(coeffs_std_nm, coeffs_direct_nm, atol=1e-9)
    np.testing.assert_allclose(phase_fit_nm, phase_direct_nm, atol=1e-9)
    # Z4/Z5 « héritent » un peu de défocus depuis le mélange Gram-Schmidt
    # (mode GS #4 = combinaison de Z1 (piston) et Z4 (défocus) sur ce
    # masque annulaire, cf. REFERENCE_TECHNIQUE.md §10) : valeurs
    # différentes de la vérité 55/27.5 saisie en base Gram-Schmidt, mais
    # cohérentes (mêmes valeurs qu'un fit_zernike direct ci-dessus).
    assert coeffs_std_nm[3] != pytest.approx(55.0, abs=1.0)
