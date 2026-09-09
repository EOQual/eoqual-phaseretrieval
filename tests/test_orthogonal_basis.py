"""
Tests de l'option ``orthogonal_basis`` (Gram-Schmidt sur pupille réelle,
REFERENCE_TECHNIQUE.md §10) : comportement par défaut inchangé, garde-fou
sur mode dégénéré, et récupération correcte sur une pupille obstruée
*quand la vérité terrain est exprimée dans la même base*.
"""

from __future__ import annotations

import numpy as np
import pytest

from eoqual.phaseretrieval.backends.optimization import OpticalConfig, fit_zernike_from_psf
from eoqual.phaseretrieval.optics import build_zernike_basis, psf_from_phase
from eoqual.phaseretrieval.pupil import PupilGenerator
from eoqual.phaseretrieval.zernike_basis_orthogonal import ZernikeGramSchmidt, resolve_zernike_basis

NPIX = 128
ORDER = 12


def test_resolve_zernike_basis_default_matches_standard_basis():
    mask = PupilGenerator(NPIX).simple_disk()
    resolved = resolve_zernike_basis(npix=NPIX, nterms=ORDER, mask=mask, orthogonalize=False)
    standard = build_zernike_basis(npix=NPIX, nterms=ORDER)
    np.testing.assert_array_equal(resolved, standard)


def test_resolve_zernike_basis_orthogonalized_matches_gram_schmidt():
    mask = PupilGenerator(NPIX).annular(obscuration=0.3)
    resolved = resolve_zernike_basis(npix=NPIX, nterms=ORDER, mask=mask, orthogonalize=True)
    gs = ZernikeGramSchmidt(mask, j_max=ORDER)
    np.testing.assert_array_equal(resolved, gs.as_basis_array())
    assert resolved.shape == (ORDER, NPIX, NPIX)


def test_resolve_zernike_basis_raises_on_degenerate_mode_instead_of_shifting_index():
    # Pupille très sous-échantillonnée (peu de pixels utiles) + ordre élevé
    # : moins de degrés de liberté (10 pixels illuminés) que de modes (40)
    # -> au moins un mode devient linéairement dépendant sur ce masque ->
    # doit lever, pas décaler silencieusement la correspondance indice/Noll.
    mask = PupilGenerator(8).annular(obscuration=0.9)
    with pytest.raises(ValueError):
        resolve_zernike_basis(npix=8, nterms=40, mask=mask, orthogonalize=True)


def test_optimization_recovers_truth_expressed_in_the_orthogonalized_basis():
    """
    Comparaison "juste" (voir REFERENCE_TECHNIQUE.md §10) : la vérité
    terrain est construite ici directement dans la base Gram-Schmidt de la
    pupille obstruée, pas dans la base standard — contrairement à
    ``simulation.simulate_psf_from_zernike``, qui utilise toujours la base
    standard et n'est donc pas le bon outil pour valider cette option.
    ``fit_zernike_from_psf`` reporte ses coefficients dans la même base que
    celle utilisée pour l'ajustement (ici Gram-Schmidt) : pas de
    décomposition supplémentaire nécessaire pour comparer à la vérité.
    """
    mask = PupilGenerator(NPIX).annular(obscuration=0.3)
    gs = ZernikeGramSchmidt(mask, j_max=ORDER)
    wavelength = 550e-9
    rad_to_nm = wavelength * 1e9 / (2 * np.pi)

    truth_nm = {4: 55.0, 5: 27.5}  # coefficients [nm WFE], base Gram-Schmidt
    phase = gs.reconstruct({j: v / rad_to_nm for j, v in truth_nm.items()})
    psf = psf_from_phase(phase, mask)

    cfg = OpticalConfig(wavelength=wavelength, zernike_order=ORDER, verbose=False, orthogonal_basis=True)
    coeffs_nm, diagnostics = fit_zernike_from_psf(psf, mask, cfg)

    assert coeffs_nm[3] == pytest.approx(55.0, abs=1.0)  # Noll j=4 -> index 3
    assert coeffs_nm[4] == pytest.approx(27.5, abs=1.0)  # Noll j=5 -> index 4
    assert diagnostics["cost"] < 1e-6
