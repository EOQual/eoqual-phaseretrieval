"""
Auto-détection de ``aperture_px`` depuis un masque personnalisé (§12.5 de
``REFERENCE_TECHNIQUE.md``) : régression du cas où l'utilisateur fournit
un masque "calculé par ailleurs" (pas via ``PupilGenerator``) à
``simulate_psf_from_zernike``/``fit_zernike_from_psf`` sans préciser
``aperture_px`` lui-même. Avant la correction, ce masque était
silencieusement traité comme s'il remplissait tout le cadre ``npix x
npix`` : les coefficients de Zernike estimés divergeaient de la vérité,
avec un biais qui dépendait de ``npix`` (plus `npix` est grand par
rapport à l'ouverture réelle, plus le décalage d'échelle est sévère).
"""

from __future__ import annotations

import numpy as np
import pytest

from eoqual.phaseretrieval.backends.optimization import OpticalConfig, fit_zernike_from_psf
from eoqual.phaseretrieval.pupil import PupilGenerator, detect_aperture_px
from eoqual.phaseretrieval.simulation import simulate_psf_from_zernike
from eoqual.phaseretrieval.zernike_basis_orthogonal import resolve_zernike_basis
from eoqual.phaseretrieval.zernike_fit import resolve_sign_ambiguity

WAVELENGTH = 550e-9
TRUTH_NM = {4: 55.0, 5: 27.5, 7: -30.0}
ZERNIKE_ORDER = 12


def test_detect_aperture_px_matches_the_generator_setting():
    assert detect_aperture_px(PupilGenerator(128, aperture_px=64).simple_disk()) == 65
    assert detect_aperture_px(PupilGenerator(128).simple_disk()) == 128
    assert detect_aperture_px(PupilGenerator(128).annular(obscuration=0.3)) == 128


def test_detect_aperture_px_rejects_empty_mask():
    with pytest.raises(ValueError):
        detect_aperture_px(np.zeros((32, 32)))


def test_resolve_zernike_basis_autodetects_aperture_px_from_a_custom_mask():
    """
    Sans détection (comportement historique), la base serait construite
    comme si l'ouverture remplissait ``npix`` entier : le mode Z4 n'y
    atteindrait jamais son lobe +1 réel (même défaut qu'une ouverture non
    renseignée en §12.5), ce que confirme la comparaison à la base
    construite avec le bon ``aperture_px`` explicite.
    """
    npix, aperture_px = 128, 64
    mask = PupilGenerator(npix, aperture_px=aperture_px).simple_disk()

    autodetected = resolve_zernike_basis(npix=npix, nterms=4, mask=mask)
    explicit = resolve_zernike_basis(npix=npix, nterms=4, mask=mask, aperture_px=aperture_px)
    np.testing.assert_array_equal(autodetected, explicit)

    full_frame = resolve_zernike_basis(npix=npix, nterms=4, mask=mask, aperture_px=npix)
    assert not np.array_equal(autodetected, full_frame)


@pytest.mark.parametrize("npix", [96, 128, 160, 256])
def test_simulation_round_trip_with_a_custom_mask_does_not_diverge_with_npix(npix):
    """
    Reproduction exacte du rapport utilisateur : un masque "calculé par
    ailleurs" (ici une ouverture physique de taille fixe, générée
    indépendamment de ``npix``) donné à ``simulate_psf_from_zernike`` /
    ``fit_zernike_from_psf`` sans ``aperture_px`` explicite ne doit plus
    donner un biais qui croît avec ``npix`` (c'était le cas avant la
    correction : Z4 estimé passait de 44.8 nm à npix=96 à 19.6 nm à
    npix=256, pour une vérité de 55.0 nm).
    """
    fixed_aperture_px = 64
    mask = PupilGenerator(npix, aperture_px=fixed_aperture_px).simple_disk()

    sim = simulate_psf_from_zernike(
        TRUTH_NM, wavelength=WAVELENGTH, npix=npix, zernike_order=ZERNIKE_ORDER, mask=mask,
    )
    coeffs_nm, diagnostics = fit_zernike_from_psf(
        sim["psf"], mask, OpticalConfig(wavelength=WAVELENGTH, zernike_order=ZERNIKE_ORDER, verbose=False),
    )
    coeffs_nm = resolve_sign_ambiguity(coeffs_nm, reference_index=4, expected_sign=np.sign(TRUTH_NM[4]))

    assert diagnostics["cost"] < 1e-6
    assert coeffs_nm[3] == pytest.approx(TRUTH_NM[4], abs=1.0)
    assert coeffs_nm[4] == pytest.approx(TRUTH_NM[5], abs=1.0)
    assert coeffs_nm[6] == pytest.approx(TRUTH_NM[7], abs=1.0)
