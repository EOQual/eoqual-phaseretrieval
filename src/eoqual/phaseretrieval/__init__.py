"""
eoqual.phaseretrieval — PSF -> WFE -> Zernike, par récupération de phase et
diversité de phase.

Convertit une (ou deux) PSF mesurée(s) en coefficients de Zernike (ordre
de Noll), via trois méthodes interchangeables derrière un point d'entrée
unique. Fournit aussi un module de simulation séparé pour générer des
PSF (et des paires focus/défocus bruitées) à vérité de Zernike connue,
utile pour valider n'importe laquelle des méthodes.

Public API
----------
>>> from eoqual.phaseretrieval import psf_to_wfe
>>> coeffs_nm, phase_fit_nm, diagnostics = psf_to_wfe(
...     psf_path, mask_path, wavelength, focal_length, pupil_diameter,
...     pixel_pitch, method="phase_diversity", diversity_path=..., defocus=...,
... )

>>> from eoqual.phaseretrieval.runner import list_methods
>>> list_methods()

>>> from eoqual.phaseretrieval.simulation import simulate_psf_from_zernike, simulate_phase_diversity_pair
>>> psf, truth = simulate_psf_from_zernike({4: 55.0, 5: 27.5}, wavelength=550e-9, npix=256, zernike_order=15)

Voir ``README.md`` pour l'installation et le catalogue, et
``REFERENCE_TECHNIQUE.md`` pour le détail théorique et d'implémentation
de chaque méthode.
"""

__version__ = "0.1.0"
__title__ = "eoqual.phaseretrieval"
__description__ = "PSF -> WFE -> Zernike coefficients, by phase retrieval and phase diversity"
__url__ = "https://github.com/EOQual/eoqual-phaseretrieval"
__uri__ = __url__
__doc__ = __description__ + " <" + __uri__ + ">"
__author__ = "Olivier Amram"
__email__ = "olivier.amram@free.fr"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2026 Olivier Amram"

from .config import METHODS_CONFIGS  # noqa: F401

from .optics import (  # noqa: F401
    build_zernike_basis,
    phase_from_coefficients,
    psf_from_phase,
    wfe_map_to_phase,
)
from .zernike_fit import (  # noqa: F401
    ZernikeConfig,
    fit_zernike,
    project_orthogonal_to_standard,
    resolve_sign_ambiguity,
)
from .zernike_basis_orthogonal import ZernikeGramSchmidt, resolve_zernike_basis  # noqa: F401
from .pupil import PupilGenerator  # noqa: F401
from .diagnostics import (  # noqa: F401
    PSFMetrics,
    airy_psf,
    compute_metrics,
    compute_mtf,
    core_ellipticity,
    diagnose_aberration,
    strehl_ratio,
    wfe_rms_from_strehl,
)
from .io_utils import load_and_prepare_mask, load_image  # noqa: F401

from .backends.gerchberg_saxton import RetrievalConfig, gerchberg_saxton  # noqa: F401
from .backends.optimization import OpticalConfig, fit_zernike_from_psf  # noqa: F401
from .backends.phase_diversity import PhaseDiversityConfig, fit_phase_diversity  # noqa: F401

from .pipeline import psf_to_wfe  # noqa: F401
from .runner import list_methods  # noqa: F401

__all__ = [
    "METHODS_CONFIGS",
    "build_zernike_basis",
    "phase_from_coefficients",
    "psf_from_phase",
    "wfe_map_to_phase",
    "ZernikeConfig",
    "fit_zernike",
    "project_orthogonal_to_standard",
    "resolve_sign_ambiguity",
    "ZernikeGramSchmidt",
    "resolve_zernike_basis",
    "PupilGenerator",
    "PSFMetrics",
    "airy_psf",
    "compute_metrics",
    "compute_mtf",
    "core_ellipticity",
    "diagnose_aberration",
    "strehl_ratio",
    "wfe_rms_from_strehl",
    "load_and_prepare_mask",
    "load_image",
    "RetrievalConfig",
    "gerchberg_saxton",
    "OpticalConfig",
    "fit_zernike_from_psf",
    "PhaseDiversityConfig",
    "fit_phase_diversity",
    "psf_to_wfe",
    "list_methods",
]
