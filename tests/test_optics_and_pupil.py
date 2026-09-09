"""Tests unitaires ciblés sur le modèle direct et la génération de pupille."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_pupil_simple_disk_excludes_corners():
    """
    Régression : ``PupilGenerator.rho`` était clippé à 1.0 avant le test
    ``rho <= 1.0``, rendant ce test toujours vrai — ``simple_disk()``
    renvoyait un masque plein cadre (tous les pixels à 1, y compris les
    coins) plutôt qu'un disque inscrit. Aucun test existant ne comparait
    au bon nombre de pixels (seul le centre était vérifié) ni ne vérifiait
    les coins, donc rien ne l'aurait détecté — trouvé en vérifiant la prise
    en compte de résolutions multiples (REFERENCE_TECHNIQUE.md §12).
    """
    npix = 64
    mask = PupilGenerator(npix).simple_disk()
    assert mask[0, 0] == 0.0
    assert mask[0, -1] == 0.0
    assert mask[-1, 0] == 0.0
    assert mask[-1, -1] == 0.0
    # Surface proche de π r² (disque inscrit de rayon npix//2), pas npix².
    expected_area = np.pi * (npix // 2) ** 2
    assert mask.sum() == pytest.approx(expected_area, rel=0.02)
    assert mask.sum() < npix * npix


def test_pupil_annular_excludes_corners_and_outer_ring():
    npix = 64
    mask = PupilGenerator(npix).annular(obscuration=0.3)
    assert mask[0, 0] == 0.0  # coin : hors du disque inscrit, pas seulement hors obstruction
    assert mask.sum() < npix * npix


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
