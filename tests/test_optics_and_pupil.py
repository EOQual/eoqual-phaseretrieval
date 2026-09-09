"""Tests unitaires ciblés sur le modèle direct et la génération de pupille."""

from __future__ import annotations

import numpy as np

from eoqual.phaseretrieval.optics import build_zernike_basis, phase_from_coefficients, psf_from_phase
from eoqual.phaseretrieval.pupil import PupilGenerator


def test_pupil_simple_disk_is_binary_and_centred():
    mask = PupilGenerator(64).simple_disk()
    assert set(np.unique(mask)) <= {0.0, 1.0}
    assert mask[32, 32] == 1.0  # centre toujours illuminé


def test_pupil_annular_removes_centre():
    mask = PupilGenerator(64).annular(obscuration=0.3)
    assert mask[32, 32] == 0.0
    assert mask.sum() < PupilGenerator(64).simple_disk().sum()


def test_zero_phase_gives_diffraction_limited_airy_peak():
    npix = 128
    mask = PupilGenerator(npix).simple_disk()
    basis = build_zernike_basis(npix=npix, nterms=4)
    phase = phase_from_coefficients(np.zeros(4), basis, mask)
    psf = psf_from_phase(phase, mask)

    assert np.isclose(psf.sum(), 1.0)
    # Le maximum de la PSF sans aberration est au centre géométrique.
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    assert (cy, cx) == (npix // 2, npix // 2)


def test_defocus_reduces_peak_strehl_like_ratio():
    npix = 128
    mask = PupilGenerator(npix).simple_disk()
    basis = build_zernike_basis(npix=npix, nterms=4)

    phase0 = phase_from_coefficients(np.zeros(4), basis, mask)
    psf0 = psf_from_phase(phase0, mask)

    coeffs = np.zeros(4)
    coeffs[3] = 2.0  # défocus important [rad]
    phase_defocus = phase_from_coefficients(coeffs, basis, mask)
    psf_defocus = psf_from_phase(phase_defocus, mask)

    # Une aberration étale l'énergie : le pic baisse (à énergie totale égale).
    assert psf_defocus.max() < psf0.max()
