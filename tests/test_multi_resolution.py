"""
Masque, carte de WFE haut ordre (HO) et PSF chacun à leur propre
résolution — y compris une PSF suréchantillonnée (super-résolution) —
REFERENCE_TECHNIQUE.md §12.
"""

from __future__ import annotations

import numpy as np
import pytest
import imageio.v3 as iio

from eoqual.phaseretrieval.backends.optimization import OpticalConfig, fit_zernike_from_psf
from eoqual.phaseretrieval.io_utils import (
    compute_pupil_footprint_px,
    load_and_prepare_ho_wfe,
    load_and_prepare_mask,
)
from eoqual.phaseretrieval.optics import build_zernike_basis, phase_from_coefficients, psf_from_phase
from eoqual.phaseretrieval.pipeline import psf_to_wfe
from eoqual.phaseretrieval.pupil import PupilGenerator

WAVELENGTH = 550e-9
FOCAL_LENGTH = 4.0
PUPIL_DIAMETER = 0.5
PIXEL_PITCH = 4e-6


def test_pupil_footprint_invariant_under_consistent_oversampling():
    """
    Quand ``npix`` grandit proportionnellement à ``oversampling`` (le cas
    réaliste : on lit npix directement sur une PSF suréchantillonnée),
    N_pup_target reste **constant** — le suréchantillonnage revient à
    zero-padder la pupille autour de l'ouverture, pas à résoudre
    l'ouverture elle-même plus finement (voir la docstring de la fonction).
    """
    npix_native = 64
    values = [
        compute_pupil_footprint_px(
            npix_native * k, WAVELENGTH, FOCAL_LENGTH, PUPIL_DIAMETER, PIXEL_PITCH, oversampling=k,
        )
        for k in (1, 2, 3, 4)
    ]
    assert len(set(values)) == 1


def test_pupil_footprint_raises_when_too_large():
    with pytest.raises(ValueError):
        compute_pupil_footprint_px(
            npix=8, wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH,
            pupil_diameter=PUPIL_DIAMETER, pixel_pitch=1e-3,  # pitch absurdement grand -> footprint géant
        )


def _write_tiff(tmp_path, name, array):
    path = tmp_path / name
    iio.imwrite(path, array.astype(np.float32))
    return path


def test_load_and_prepare_mask_rescales_from_a_different_native_resolution(tmp_path):
    """Masque natif à 512 px, PSF/grille de travail à 256 px : doit être
    physiquement re-projeté, pas simplement redimensionné à l'aveugle."""
    npix_psf = 256
    native_mask = PupilGenerator(512).simple_disk()  # résolution propre, indépendante de npix_psf
    mask_path = _write_tiff(tmp_path, "mask_native.tiff", native_mask)

    mask, n_pup_target = load_and_prepare_mask(
        psf_shape=(npix_psf, npix_psf), mask_path=mask_path,
        wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH, pupil_diameter=PUPIL_DIAMETER,
        pixel_pitch=PIXEL_PITCH,
    )
    assert mask.shape == (npix_psf, npix_psf)
    assert n_pup_target == compute_pupil_footprint_px(
        npix_psf, WAVELENGTH, FOCAL_LENGTH, PUPIL_DIAMETER, PIXEL_PITCH,
    )
    # Aire proche de π (N_pup/2)^2, pas de la pupille native (512 px) ni du cadre entier.
    expected_area = np.pi * (n_pup_target / 2) ** 2
    assert mask.sum() == pytest.approx(expected_area, rel=0.05)
    assert set(np.unique(mask)) <= {0.0, 1.0}  # binarize=True par défaut


def test_load_and_prepare_mask_binarize_false_keeps_graded_edge(tmp_path):
    npix_psf = 256
    native_mask = PupilGenerator(512).simple_disk()
    mask_path = _write_tiff(tmp_path, "mask_native.tiff", native_mask)

    mask, _ = load_and_prepare_mask(
        psf_shape=(npix_psf, npix_psf), mask_path=mask_path,
        wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH, pupil_diameter=PUPIL_DIAMETER,
        pixel_pitch=PIXEL_PITCH, binarize=False,
    )
    # Un rééchantillonnage anti-aliasé produit forcément des valeurs
    # intermédiaires au bord du disque (contrairement à binarize=True).
    assert ((mask > 0.0) & (mask < 1.0)).any()


def test_load_and_prepare_ho_wfe_rescales_independently_of_mask_resolution(tmp_path):
    """
    Carte HO native à 128 px, masque natif à 512 px, PSF/grille de travail
    à 256 px : les trois résolutions diffèrent, mais mask et carte HO
    doivent atterrir co-enregistrés sur le même N_pup_target.
    """
    npix_psf = 256

    native_mask = PupilGenerator(512).simple_disk()
    mask_path = _write_tiff(tmp_path, "mask_native.tiff", native_mask)

    ho_basis = build_zernike_basis(npix=128, nterms=25)
    ho_coeffs = np.zeros(25)
    ho_coeffs[20] = 40.0
    rad_to_nm = WAVELENGTH * 1e9 / (2 * np.pi)
    ho_native_mask = PupilGenerator(128).simple_disk()
    ho_wfe_native_nm = (
        phase_from_coefficients(ho_coeffs / rad_to_nm, ho_basis, ho_native_mask) * rad_to_nm
    )
    ho_path = _write_tiff(tmp_path, "ho_native.tiff", ho_wfe_native_nm)

    mask, n_pup_mask = load_and_prepare_mask(
        psf_shape=(npix_psf, npix_psf), mask_path=mask_path,
        wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH, pupil_diameter=PUPIL_DIAMETER,
        pixel_pitch=PIXEL_PITCH,
    )
    ho_wfe_nm = load_and_prepare_ho_wfe(
        psf_shape=(npix_psf, npix_psf), ho_wfe_path=ho_path,
        wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH, pupil_diameter=PUPIL_DIAMETER,
        pixel_pitch=PIXEL_PITCH,
    )

    assert ho_wfe_nm.shape == mask.shape == (npix_psf, npix_psf)
    n_pup_ho = compute_pupil_footprint_px(npix_psf, WAVELENGTH, FOCAL_LENGTH, PUPIL_DIAMETER, PIXEL_PITCH)
    assert n_pup_mask == n_pup_ho  # même footprint, dérivé indépendamment mais de façon cohérente
    # La carte HO n'est jamais binarisée, contrairement au masque.
    assert not np.array_equal(np.unique(ho_wfe_nm), [0.0, np.unique(ho_wfe_nm).max()])


def test_end_to_end_mask_and_ho_wfe_at_different_native_resolutions_still_recover_truth(tmp_path):
    """
    Le scénario concret posé en question : masque et carte HO chacun à
    leur résolution propre (différentes entre elles et de la PSF), PSF
    "suréchantillonnée" simulée directement à 256 px (oversampling=2 par
    rapport à un pitch natif deux fois plus grand). Malgré ça, le fit doit
    recouvrer la vérité de Zernike bas ordre grâce à known_ho_wfe_nm.

    La PSF "vérité terrain" est construite à partir du masque et de la
    carte HO tels que ``load_and_prepare_mask``/``load_and_prepare_ho_wfe``
    les enregistrent réellement (mêmes fonctions que ``psf_to_wfe`` utilise
    en interne) plutôt qu'une reconstruction indépendante à la main : le
    test isole ainsi la question posée (le multi-résolution est-il
    correctement pris en compte ?) de la sensibilité du solveur à
    l'initialisation, déjà couverte ailleurs (§3.7).
    """
    npix_psf = 256
    zernike_order = 12
    truth_nm = {4: 55.0, 5: 27.5}
    native_pixel_pitch = PIXEL_PITCH  # pitch natif du capteur
    oversampling = 2  # la PSF fournie est 2x le pitch natif

    # Masque "outil CAO" à sa propre résolution (1024 px, rien à voir avec npix_psf)
    native_mask = PupilGenerator(1024).simple_disk()
    mask_path = _write_tiff(tmp_path, "mask_cad.tiff", native_mask)

    working_mask, n_pup = load_and_prepare_mask(
        psf_shape=(npix_psf, npix_psf), mask_path=mask_path,
        wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH, pupil_diameter=PUPIL_DIAMETER,
        pixel_pitch=native_pixel_pitch, oversampling=oversampling,
    )

    # Carte HO à SA résolution propre (512 px, différente du masque et de la PSF)
    ho_native_res = 512
    ho_native_mask = PupilGenerator(ho_native_res).simple_disk()
    ho_basis_native = build_zernike_basis(npix=ho_native_res, nterms=25)
    ho_coeffs = np.zeros(25)
    ho_coeffs[20] = 40.0
    rad_to_nm = WAVELENGTH * 1e9 / (2 * np.pi)
    ho_wfe_native_nm = (
        phase_from_coefficients(ho_coeffs / rad_to_nm, ho_basis_native, ho_native_mask) * rad_to_nm
    )
    ho_path = _write_tiff(tmp_path, "ho_native.tiff", ho_wfe_native_nm)

    ho_wfe_registered_nm = load_and_prepare_ho_wfe(
        psf_shape=(npix_psf, npix_psf), ho_wfe_path=ho_path,
        wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH, pupil_diameter=PUPIL_DIAMETER,
        pixel_pitch=native_pixel_pitch, oversampling=oversampling,
    )

    # Vérité bas ordre + carte HO (déjà co-enregistrée), simulées sur la
    # grille de travail avec le masque tel que la bibliothèque le voit.
    basis = build_zernike_basis(npix=npix_psf, nterms=25)
    low_coeffs = np.zeros(25)
    low_coeffs[3], low_coeffs[4] = truth_nm[4], truth_nm[5]

    # La carte HO est déjà une phase par pixel (pas des coefficients
    # modaux) : s'ajoute directement, comme le fait wfe_map_to_phase.
    total_phase_rad = (
        phase_from_coefficients(low_coeffs / rad_to_nm, basis, working_mask)
        + (ho_wfe_registered_nm / rad_to_nm) * working_mask
    )
    psf = psf_from_phase(total_phase_rad, working_mask)
    psf_path = _write_tiff(tmp_path, "psf.tiff", psf)

    coeffs_nm, phase_fit_nm, diagnostics = psf_to_wfe(
        psf_path=psf_path, mask_path=mask_path,
        wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH, pupil_diameter=PUPIL_DIAMETER,
        pixel_pitch=native_pixel_pitch, oversampling=oversampling,
        method="optimization", zernike_order=zernike_order,
        known_ho_wfe_path=ho_path, verbose=False,
        sign_prior=(4, 1.0),  # une seule PSF : ambiguïté de signe attendue (§3.7), levée ici par a priori
    )

    # Tolérance réaliste (pas machine-epsilon) : le masque et la carte HO
    # traversent chacun un aller-retour fichier (float32) + un
    # rééchantillonnage indépendant (bilinéaire / cubique) depuis leur
    # propre résolution native — une petite perte est attendue, pas un
    # échec de la prise en compte du multi-résolution en tant que telle
    # (déjà vérifiée exactement par les tests ci-dessus, sans solveur).
    assert coeffs_nm[3] == pytest.approx(55.0, abs=3.0)
    assert coeffs_nm[4] == pytest.approx(27.5, abs=3.0)
    assert diagnostics["cost"] < 1e-3
