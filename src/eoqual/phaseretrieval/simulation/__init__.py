"""
eoqual.phaseretrieval.simulation — génération de cas de test à vérité de
Zernike connue.

Volontairement **séparé** des méthodes de retrouvé de phase
(``eoqual.phaseretrieval.backends``) : ce sous-paquet ne fait que fabriquer
des données d'entrée (PSF simple, ou paire focus/défocus bruitée), il n'en
retrouve aucune. Sert à valider n'importe laquelle des trois méthodes de
bout en bout, en comparant les coefficients retrouvés à la vérité terrain
renvoyée dans ``coeffs_nm``.

Examples
--------
>>> from eoqual.phaseretrieval.simulation import simulate_psf_from_zernike
>>> sim = simulate_psf_from_zernike({4: 55.0, 5: 27.5}, wavelength=550e-9, npix=256, zernike_order=15)
>>> sim["psf"], sim["coeffs_nm"]

>>> from eoqual.phaseretrieval.simulation import (
...     DetectorNoiseConfig, PhaseDiversitySimulationConfig, simulate_phase_diversity_pair,
... )
>>> cfg = PhaseDiversitySimulationConfig(defocus=275e-9, noise=DetectorNoiseConfig(photon_count=1e5))
>>> pair = simulate_phase_diversity_pair({4: 55.0, 5: 27.5}, cfg)
>>> pair["psf_focus"], pair["psf_defocus"], pair["coeffs_nm"]
"""

from .noise import DetectorNoiseConfig, apply_detector_noise  # noqa: F401
from .psf_simulator import coeffs_to_array, simulate_psf_from_zernike  # noqa: F401
from .phase_diversity_simulator import (  # noqa: F401
    PhaseDiversitySimulationConfig,
    simulate_phase_diversity_pair,
)
from . import star_profiles  # noqa: F401

__all__ = [
    "DetectorNoiseConfig",
    "apply_detector_noise",
    "coeffs_to_array",
    "simulate_psf_from_zernike",
    "PhaseDiversitySimulationConfig",
    "simulate_phase_diversity_pair",
    "star_profiles",
]
