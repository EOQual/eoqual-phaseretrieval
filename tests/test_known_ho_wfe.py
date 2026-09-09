"""
Option ``known_ho_wfe_nm`` : incorporation d'une carte de WFE haut ordre
(HO) connue dans le modèle direct, sans jamais l'ajuster
(REFERENCE_TECHNIQUE.md §11.3/§2.2).
"""

from __future__ import annotations

import numpy as np
import pytest

from eoqual.phaseretrieval.backends.optimization import OpticalConfig, fit_zernike_from_psf
from eoqual.phaseretrieval.backends.phase_diversity import PhaseDiversityConfig, fit_phase_diversity
from eoqual.phaseretrieval.optics import build_zernike_basis, phase_from_coefficients, wfe_map_to_phase
from eoqual.phaseretrieval.pupil import PupilGenerator
from eoqual.phaseretrieval.simulation import (
    PhaseDiversitySimulationConfig,
    simulate_phase_diversity_pair,
    simulate_psf_from_zernike,
)

NPIX = 128
ORDER = 12
WAVELENGTH = 550e-9
TRUTH_NM = {4: 55.0, 5: 27.5}


def _ho_wfe_map(mask: np.ndarray) -> np.ndarray:
    """Un unique mode Zernike élevé (Noll 21), hors de portée de ORDER."""
    rad_to_nm = WAVELENGTH * 1e9 / (2 * np.pi)
    basis = build_zernike_basis(npix=NPIX, nterms=25)
    coeffs = np.zeros(25)
    coeffs[20] = 40.0  # Z21, nm
    return phase_from_coefficients(coeffs / rad_to_nm, basis, mask) * rad_to_nm


def test_wfe_map_to_phase_shape_mismatch_raises():
    mask = PupilGenerator(NPIX).simple_disk()
    with pytest.raises(ValueError):
        wfe_map_to_phase(np.zeros((10, 10)), WAVELENGTH, mask)


def test_optimization_without_known_ho_is_biased_by_it():
    mask = PupilGenerator(NPIX).simple_disk()
    ho_wfe_nm = _ho_wfe_map(mask)
    sim = simulate_psf_from_zernike(
        TRUTH_NM, wavelength=WAVELENGTH, npix=NPIX, zernike_order=ORDER,
        mask=mask, known_ho_wfe_nm=ho_wfe_nm,
    )

    cfg = OpticalConfig(wavelength=WAVELENGTH, zernike_order=ORDER, verbose=False)
    coeffs_nm, diagnostics = fit_zernike_from_psf(sim["psf"], mask, cfg)

    # Sans connaître le HO, le fit bas ordre est visiblement biaisé.
    assert abs(coeffs_nm[3] - 55.0) > 5.0
    assert diagnostics["cost"] > 1e-6


def test_optimization_with_known_ho_recovers_truth_exactly():
    mask = PupilGenerator(NPIX).simple_disk()
    ho_wfe_nm = _ho_wfe_map(mask)
    sim = simulate_psf_from_zernike(
        TRUTH_NM, wavelength=WAVELENGTH, npix=NPIX, zernike_order=ORDER,
        mask=mask, known_ho_wfe_nm=ho_wfe_nm,
    )

    cfg = OpticalConfig(
        wavelength=WAVELENGTH, zernike_order=ORDER, verbose=False,
        known_ho_wfe_nm=ho_wfe_nm,
    )
    coeffs_nm, diagnostics = fit_zernike_from_psf(sim["psf"], mask, cfg)

    assert coeffs_nm[3] == pytest.approx(55.0, abs=1e-3)
    assert coeffs_nm[4] == pytest.approx(27.5, abs=1e-3)
    assert diagnostics["cost"] < 1e-15


def test_phase_diversity_with_known_ho_recovers_truth_exactly():
    mask = PupilGenerator(NPIX).simple_disk()
    ho_wfe_nm = _ho_wfe_map(mask)

    sim_cfg = PhaseDiversitySimulationConfig(
        wavelength=WAVELENGTH, npix=NPIX, zernike_order=ORDER, defocus=275e-9,
        known_ho_wfe_nm=ho_wfe_nm,
    )
    pair = simulate_phase_diversity_pair(TRUTH_NM, sim_cfg, mask=mask)
    assert pair["known_ho_wfe_nm"] is ho_wfe_nm

    pd_cfg_biased = PhaseDiversityConfig(
        wavelength=WAVELENGTH, zernike_order=ORDER, defocus=275e-9,
        verbose=False, max_nfev=300,
    )
    coeffs_biased, diag_biased = fit_phase_diversity(
        pair["psf_focus"], pair["psf_defocus"], mask, pd_cfg_biased
    )
    assert abs(coeffs_biased[3] - 55.0) > 1.0

    pd_cfg_known = PhaseDiversityConfig(
        wavelength=WAVELENGTH, zernike_order=ORDER, defocus=275e-9,
        verbose=False, max_nfev=300, known_ho_wfe_nm=ho_wfe_nm,
    )
    coeffs_known, diag_known = fit_phase_diversity(
        pair["psf_focus"], pair["psf_defocus"], mask, pd_cfg_known
    )
    assert coeffs_known[3] == pytest.approx(55.0, abs=1e-3)
    assert coeffs_known[4] == pytest.approx(27.5, abs=1e-3)
    assert diag_known["cost"] < diag_biased["cost"]
