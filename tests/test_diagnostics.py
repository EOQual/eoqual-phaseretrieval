"""
Diagnostics FTM : axe de fréquence réduite (fraction de fe), fréquence
de coupure optique fc, coupes H/V, et leur visualisation Plotly — voir
REFERENCE_TECHNIQUE.md §9 et §12.5 (aperture_px, dont fc dépend).
"""

from __future__ import annotations

import numpy as np
import pytest

from eoqual.phaseretrieval.diagnostics import (
    compute_mtf,
    encircled_energy,
    fwhm,
    mtf_cutoff_frequency,
    mtf_cuts,
    mtf_frequency_axis,
    mtf_integral,
    mtf_value_at,
)
from eoqual.phaseretrieval.io_utils import plot_mtf_cuts
from eoqual.phaseretrieval.optics import psf_from_phase
from eoqual.phaseretrieval.pupil import PupilGenerator


def test_mtf_frequency_axis_edges_reach_nyquist():
    npix = 128
    freq = mtf_frequency_axis(npix)
    assert freq.shape == (npix,)
    # Bord négatif exactement à -0.5 (fe/2) ; bord positif très proche de
    # +0.5 (à 1/npix près, artefact standard d'une FFT de taille paire).
    assert freq.min() == pytest.approx(-0.5)
    assert freq.max() == pytest.approx(0.5, abs=1.0 / npix)


def test_mtf_cutoff_frequency_known_values():
    npix = 128
    assert mtf_cutoff_frequency(npix) == pytest.approx(1.0)  # None -> aperture = npix
    assert mtf_cutoff_frequency(npix, aperture_px=npix) == pytest.approx(1.0)
    assert mtf_cutoff_frequency(npix, aperture_px=npix // 2) == pytest.approx(0.5)  # Nyquist exact (Q=2)
    assert mtf_cutoff_frequency(npix, aperture_px=npix // 4) == pytest.approx(0.25)


def test_mtf_cutoff_frequency_matches_empirical_mtf_support():
    """
    Vérification indépendante : la FTM d'une pupille de diamètre
    ``aperture_px`` doit devenir numériquement nulle exactement à la
    fréquence ``fc`` prédite par la formule — pas une coïncidence pour
    une seule valeur (voir aussi le calcul qui a mené à la formule).
    """
    npix, aperture_px = 128, 32
    mask = PupilGenerator(npix, aperture_px=aperture_px).simple_disk()
    psf = psf_from_phase(np.zeros((npix, npix)), mask)
    mtf = compute_mtf(psf)
    cut_h, _ = mtf_cuts(mtf)
    freq = mtf_frequency_axis(npix)

    fc = mtf_cutoff_frequency(npix, aperture_px)
    nonzero = freq[cut_h > 1e-6]
    # Le support non nul s'arrête à fc (à une bin de discrétisation près).
    assert nonzero.max() == pytest.approx(fc, abs=2.0 / npix)
    assert nonzero.min() == pytest.approx(-fc, abs=2.0 / npix)
    # Au-delà de fc, la FTM est bien nulle (pas juste "petite").
    beyond = np.abs(freq) > fc + 4.0 / npix
    assert np.all(cut_h[beyond] < 1e-9)


def test_mtf_cuts_match_central_row_and_column():
    npix = 64
    mask = PupilGenerator(npix).simple_disk()
    psf = psf_from_phase(np.zeros((npix, npix)), mask)
    mtf = compute_mtf(psf)
    cut_h, cut_v = mtf_cuts(mtf)
    np.testing.assert_array_equal(cut_h, mtf[npix // 2, :])
    np.testing.assert_array_equal(cut_v, mtf[:, npix // 2])


def test_plot_mtf_cuts_estimated_only():
    npix = 64
    mask = PupilGenerator(npix).simple_disk()
    psf = psf_from_phase(np.zeros((npix, npix)), mask)
    mtf = compute_mtf(psf)

    fig = plot_mtf_cuts(mtf)
    # 2 traces (coupe H, coupe V) : pas de référence, pas de fc.
    assert len(fig.data) == 2
    for trace in fig.data:
        assert np.all(np.asarray(trace.x) >= 0)  # seulement les fréquences positives


def test_plot_mtf_cuts_with_reference_and_fc():
    npix, aperture_px = 128, 64
    mask = PupilGenerator(npix, aperture_px=aperture_px).simple_disk()
    psf = psf_from_phase(np.zeros((npix, npix)), mask)
    mtf = compute_mtf(psf)
    fc = mtf_cutoff_frequency(npix, aperture_px)

    fig = plot_mtf_cuts(mtf, mtf_reference=mtf, fc=fc)
    # 4 traces (estimée H/V + référence H/V).
    assert len(fig.data) == 4
    # Repères verticaux fc (rouge) et fe/2 (bleu), chacun sur les deux
    # sous-graphes -> 4 au total.
    vlines = [shape for shape in fig.layout.shapes if shape.type == "line"]
    assert len(vlines) == 4
    fc_lines = [s for s in vlines if s.line.color == "red"]
    nyquist_lines = [s for s in vlines if s.line.color == "blue"]
    assert len(fc_lines) == 2 and all(s.x0 == pytest.approx(fc) for s in fc_lines)
    assert len(nyquist_lines) == 2 and all(s.x0 == pytest.approx(0.5) for s in nyquist_lines)


def test_mtf_value_at_matches_known_points():
    freq = np.array([0.0, 0.25, 0.5])
    cut = np.array([1.0, 0.6, 0.0])
    assert mtf_value_at(freq, cut, 0.25) == pytest.approx(0.6)
    assert mtf_value_at(freq, cut, 0.125) == pytest.approx(0.8)  # interpolé


def test_mtf_integral_of_constant_profile():
    freq = np.linspace(-0.5, 0.5, 201)
    cut = np.ones_like(freq)
    # Intégrale de 1 entre 0 et 0.5 (f_max par défaut) = 0.5.
    assert mtf_integral(freq, cut) == pytest.approx(0.5, abs=1e-3)
    assert mtf_integral(freq, cut, f_max=0.25) == pytest.approx(0.25, abs=1e-3)


def test_encircled_energy_reaches_one():
    npix = 32
    mask = PupilGenerator(npix).simple_disk()
    psf = psf_from_phase(np.zeros((npix, npix)), mask)
    radii, ee = encircled_energy(psf)
    assert ee[0] < ee[-1]
    assert ee[-1] == pytest.approx(1.0, abs=0.05)


def test_fwhm_of_a_simple_peak():
    x = np.linspace(-10, 10, 201)
    sigma = 2.0
    gaussian = np.exp(-0.5 * (x / sigma) ** 2)
    expected_fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
    assert fwhm(x, gaussian) == pytest.approx(expected_fwhm, rel=1e-2)
