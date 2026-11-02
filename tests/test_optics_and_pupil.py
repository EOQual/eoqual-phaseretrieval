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


# ----------------------------------------------------------------------
# aperture_px : ouverture plus petite que le cadre (suréchantillonnage en
# mémoire, §12.5 de REFERENCE_TECHNIQUE.md).
# ----------------------------------------------------------------------

def test_pupil_aperture_px_none_matches_full_frame_default():
    """``aperture_px=None`` reproduit exactement le comportement historique."""
    npix = 64
    mask_default = PupilGenerator(npix).simple_disk()
    mask_explicit = PupilGenerator(npix, aperture_px=npix).simple_disk()
    assert np.array_equal(mask_default, mask_explicit)


def test_pupil_aperture_px_smaller_than_npix_pads_with_zeros():
    npix, aperture_px = 128, 64
    mask = PupilGenerator(npix, aperture_px=aperture_px).simple_disk()

    # Surface proche de pi*r^2 pour r=aperture_px//2, pas r=npix//2.
    expected_area = np.pi * (aperture_px // 2) ** 2
    assert mask.sum() == pytest.approx(expected_area, rel=0.02)
    # Zéro loin du centre : la marge de suréchantillonnage est bien vide.
    assert mask[0, 0] == 0.0
    assert mask[npix // 2, 0] == 0.0


def test_pupil_aperture_px_rejects_larger_than_npix():
    with pytest.raises(ValueError):
        PupilGenerator(64, aperture_px=128)


def test_pupil_with_spiders_scales_to_aperture_px_not_npix():
    """
    Les bras d'araignée doivent balayer l'ouverture réelle (aperture_px),
    pas tout le cadre (npix) — sinon ``width`` (fraction du rayon de la
    pupille) serait mal mis à l'échelle dès que aperture_px < npix.
    """
    npix, aperture_px = 128, 64
    gen = PupilGenerator(npix, aperture_px=aperture_px)
    pupil = gen.with_spiders(obscuration=0.3, n_spiders=4, width=0.05)

    # Un bras doit couper l'ouverture (retirer des pixels par rapport à
    # l'annulaire pur), mais rester localisé près du centre (dans
    # aperture_px), pas s'étendre jusqu'au bord du cadre npix.
    annular_only = gen.annular(obscuration=0.3)
    assert pupil.sum() < annular_only.sum()
    assert pupil[npix // 2, npix // 2 + aperture_px // 2 + 10] == 0.0  # hors ouverture : déjà à 0


def test_pupil_with_spiders_supersample_one_matches_historical_binary_behaviour():
    """``supersample=1`` retrouve exactement le seuillage binaire d'avant §10.3."""
    npix, aperture_px = 128, 64
    gen = PupilGenerator(npix, aperture_px=aperture_px)
    pupil_ss1 = gen.with_spiders(obscuration=0.3, n_spiders=4, width=0.05, supersample=1)
    assert set(np.unique(pupil_ss1)) <= {0.0, 1.0}


def test_pupil_with_spiders_default_supersample_grades_the_spider_edge():
    """Par défaut (``supersample=8``), les pixels traversés par un bras
    reçoivent une valeur intermédiaire plutôt qu'un 0/1 arbitraire."""
    npix, aperture_px = 128, 64
    gen = PupilGenerator(npix, aperture_px=aperture_px)
    pupil = gen.with_spiders(obscuration=0.3, n_spiders=4, width=0.02)
    assert np.any((pupil > 0.0) & (pupil < 1.0))


def test_pupil_with_spiders_supersampling_stabilises_psf_under_subpixel_shift():
    """
    Régression chiffrée (§10.3 de REFERENCE_TECHNIQUE.md) : sans
    suréchantillonnage, la PSF d'une pupille à bras fins change de façon
    purement numérique selon l'alignement sous-pixel de la grille (la
    pupille physique ne dépend pas d'où elle tombe sur la grille) --
    jusqu'à ~2 % d'écart mesuré. Le suréchantillonnage par défaut doit
    réduire nettement cet artefact.
    """
    from eoqual.core.psf import supersampled_coverage

    npix, aperture_px, obscuration, n_spiders, width = 256, 128, 0.2, 4, 0.01
    half_aperture = aperture_px // 2
    phase0 = np.zeros((npix, npix))

    def pupil_at_shift(shift, supersample):
        y, x = np.indices((npix, npix), dtype=float) - npix // 2
        x, y = x - shift[0], y - shift[1]
        rho = np.sqrt(x**2 + y**2) / half_aperture
        theta = np.arctan2(y, x)
        pupil = ((rho <= 1.0) & (rho >= obscuration)).astype(float)
        if supersample == 1:
            for i in range(n_spiders):
                angle = i * (2 * np.pi / n_spiders)
                dist = np.abs(np.sin(theta - angle) * rho * aperture_px / 2)
                pupil[dist < (width * aperture_px / 2)] = 0.0
            return pupil
        for i in range(n_spiders):
            angle = i * (2 * np.pi / n_spiders)

            def blocked(yy, xx, angle=angle):
                rr = np.sqrt((xx - shift[0]) ** 2 + (yy - shift[1]) ** 2) / half_aperture
                tt = np.arctan2(yy - shift[1], xx - shift[0])
                return np.abs(np.sin(tt - angle) * rr * aperture_px / 2) < (width * aperture_px / 2)

            coverage = supersampled_coverage(blocked, (npix, npix), (npix // 2, npix // 2), supersample)
            pupil = pupil * (1.0 - coverage)
        return pupil

    def psf_at_shift(shift, supersample):
        return psf_from_phase(phase0, pupil_at_shift(shift, supersample))

    shifts = [(0.5, 0.5), (0.1, 0.4), (0.3, 0.7)]
    ref_bin = psf_at_shift((0, 0), 1)
    ref_ss = psf_at_shift((0, 0), 8)

    rel_bin = [np.linalg.norm(psf_at_shift(s, 1) - ref_bin) / np.linalg.norm(ref_bin) for s in shifts]
    rel_ss = [np.linalg.norm(psf_at_shift(s, 8) - ref_ss) / np.linalg.norm(ref_ss) for s in shifts]

    assert max(rel_bin) > 0.01  # artefact non négligeable sans traitement
    assert max(rel_ss) < max(rel_bin) / 2  # suréchantillonnage : au moins 2x plus stable


def test_build_zernike_basis_aperture_px_reaches_full_dynamic_range_inside_aperture():
    """
    Régression du bug corrigé (§12.5) : sans ``aperture_px``, un mode
    (ex. Z4, défocus) évalué sur toute l'ouverture réelle plus petite que
    le cadre ne parcourt qu'une fraction tronquée de sa dynamique propre
    (jamais +1), et sa RMS sur l'ouverture n'est pas 1 comme l'exige la
    normalisation de Noll — alors qu'avec ``aperture_px`` correctement
    renseigné, les deux sont corrects.
    """
    npix, aperture_px = 128, 64
    mask = PupilGenerator(npix, aperture_px=aperture_px).simple_disk()
    inside = mask > 0

    z4_wrong = build_zernike_basis(npix=npix, nterms=4)[3]
    z4_right = build_zernike_basis(npix=npix, nterms=4, aperture_px=aperture_px)[3]

    def rms_inside(z):
        return float(np.sqrt(np.mean(z[inside] ** 2)))

    # Mauvaise échelle : la moitié positive du défocus (rho > ~0.7 en
    # échelle correcte) n'est jamais atteinte dans l'ouverture réelle.
    assert z4_wrong[inside].max() < 0.0
    assert rms_inside(z4_wrong) == pytest.approx(1.32, abs=0.02)

    # Bonne échelle : dynamique complète (+-sqrt(3)) et RMS de Noll (= 1).
    assert z4_right[inside].max() == pytest.approx(np.sqrt(3), rel=1e-6)
    assert z4_right[inside].min() == pytest.approx(-np.sqrt(3), rel=1e-6)
    assert rms_inside(z4_right) == pytest.approx(1.0, abs=0.02)


def test_build_zernike_basis_aperture_px_none_matches_historical_default():
    npix = 64
    basis_default = build_zernike_basis(npix=npix, nterms=6)
    basis_explicit_full = build_zernike_basis(npix=npix, nterms=6, aperture_px=npix)
    assert np.array_equal(basis_default, basis_explicit_full)
