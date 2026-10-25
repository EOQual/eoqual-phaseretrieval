#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point d'entrée unifié : PSF (+ masque pupillaire) -> coefficients de Zernike.

Ce module orchestre les trois méthodes de récupération de phase implémentées
dans ce projet, sélectionnables via le paramètre ``method`` de
:func:`psf_to_wfe` :

============================  =====================================  ================
``method``                    Algorithme                              Entrées
============================  =====================================  ================
``"gerchberg_saxton"``         Gerchberg-Saxton itératif (FFT)         1 PSF
``"optimization"``             Moindres carrés non-linéaires directs   1 PSF
``"phase_diversity"``          Diversité de phase modale               2 PSF
============================  =====================================  ================

Les deux premières méthodes partagent la même signature d'entrée (une
seule PSF) : les fusionner derrière un simple sélecteur ``method`` est
direct. La diversité de phase, elle, a besoin d'une **seconde** PSF
(``diversity_path``) et d'un défocus connu (``defocus``) : ce n'est pas
juste une variante d'algorithme, c'est un protocole de mesure différent.
Elle est donc bien orchestrée ici (un seul point d'entrée, un seul CLI,
diagnostics communs) mais son coeur numérique reste dans son propre module
(:mod:`backends.phase_diversity`), avec sa propre configuration ; le
paramètre ``diversity_path`` est explicitement optionnel et ignoré (avec
avertissement) par les deux autres méthodes.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import commentjson
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from .io_utils import load_and_prepare_ho_wfe, load_and_prepare_mask, load_image, visualisation_diagnostics
from .optics import phase_from_coefficients
from .zernike_basis_orthogonal import resolve_zernike_basis
from .backends.gerchberg_saxton import RetrievalConfig, gerchberg_saxton
from .backends.optimization import OpticalConfig, fit_zernike_from_psf
from .backends.phase_diversity import PhaseDiversityConfig, fit_phase_diversity
from .zernike_fit import ZernikeConfig, fit_zernike, resolve_sign_ambiguity

Method = Literal["gerchberg_saxton", "optimization", "phase_diversity"]
METHODS: Tuple[Method, ...] = ("gerchberg_saxton", "optimization", "phase_diversity")


def psf_to_wfe(
    psf_path: Union[str, Path],
    mask_path: Union[str, Path],
    wavelength: float,
    focal_length: float,
    pupil_diameter: float,
    pixel_pitch: float,
    method: Method = "gerchberg_saxton",
    oversampling: int = 1,
    zernike_order: int = 36,
    orthogonal_basis: bool = False,
    mask_binarize: bool = True,
    mask_native_aperture_px: Optional[int] = None,
    ho_native_aperture_px: Optional[int] = None,
    mask_native_aperture_center: Optional[Tuple[int, int]] = None,
    ho_native_aperture_center: Optional[Tuple[int, int]] = None,
    aperture_threshold: float = 0.01,
    verbose: bool = True,
    # --- Gerchberg-Saxton ---
    max_iter: int = 200,
    beta: float = 0.9,
    seed: Optional[int] = None,
    # --- Optimisation directe ---
    fit_piston_tilt: bool = False,
    max_nfev: Optional[int] = None,
    # --- Diversité de phase (2 PSF) ---
    diversity_path: Optional[Union[str, Path]] = None,
    defocus: Optional[float] = None,
    roi_radius: int = 32,
    loss: str = "soft_l1",
    poisson_weighting: bool = True,
    # --- Correction d'ambiguïté de signe (gerchberg_saxton / optimization) ---
    sign_prior: Optional[Tuple[int, float]] = None,
    # --- WFE haut ordre connue (optimization / phase_diversity) ---
    known_ho_wfe_path: Optional[Union[str, Path]] = None,
    # --- A priori individuels par mode (optimization / phase_diversity) ---
    zernike_priors: Optional[Dict[int, Tuple[float, float]]] = None,
    zernike_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[str, float]]:
    """
    End-to-end conversion **PSF -> Zernike (Noll) coefficients**.

    Parameters
    ----------
    psf_path, mask_path : str or Path
        File paths to the PSF intensity image and the pupil mask — each at
        its **own** native resolution, not necessarily matching
        ``psf_path``'s (see ``oversampling`` and §12 of
        ``REFERENCE_TECHNIQUE.md`` for the general multi-resolution case:
        a super-resolved PSF, a mask from a separate CAD/metrology tool).
    wavelength : float
        Central wavelength in metres (e.g. ``550e-9`` for 550 nm).
    focal_length : float
        Focal length [m], used to physically scale the pupil mask.
    pupil_diameter : float
        Pupil diameter [m].
    pixel_pitch : float
        **Native** detector pixel pitch [m] — a fixed property of the
        sensor, independent of ``oversampling``. See
        :func:`io_utils.compute_pupil_footprint_px`.
    method : {"gerchberg_saxton", "optimization", "phase_diversity"}
        Phase-retrieval algorithm to use.
    oversampling : int, default 1
        Oversampling factor of ``psf_path`` (and ``diversity_path``)
        relative to native detector sampling — e.g. ``4`` for a PSF
        reconstructed by super-resolution (Shift-and-Add, ePSF, PSFr —
        §5.0) at 4x the native pixel pitch. ``1`` (default): the PSF is
        already at native detector resolution. The effective pixel pitch
        used for the mask/HO-map physical registration is
        ``pixel_pitch / oversampling``.
    zernike_order : int, default 36
        Number of Zernike modes (Noll order) to fit.
    orthogonal_basis : bool, default False
        Fit on a Gram-Schmidt basis orthogonalised on the actual pupil
        mask (see :class:`zernike_fit.ZernikeConfig` and §10 of
        ``REFERENCE_TECHNIQUE.md``) instead of the standard Noll basis —
        recommended whenever the pupil is not a full, unobstructed disc
        (central obstruction, spiders). ``False`` (default) keeps the
        historical behaviour, for all three methods.
    mask_binarize : bool, default True
        Threshold the loaded mask to strict ``{0, 1}`` (default, historical
        behaviour) or keep it as a graded ``[0, 1]`` transmission — see
        ``binarize`` on :func:`io_utils.load_and_prepare_mask`. Only
        matters when the mask is *not* already on the PSF's own grid (the
        physical-rescale path); a mask already pre-registered is used
        as-is either way.
    mask_native_aperture_px, ho_native_aperture_px : int, optional
        The aperture's own pixel span *within* ``mask_path``/
        ``known_ho_wfe_path``, when the source file has a zero (or
        near-zero) margin around the aperture rather than filling the
        frame edge to edge — e.g. a mask exported at a fixed canvas size
        larger than the pupil. ``None`` (default): auto-detected from the
        pixel content (see ``aperture_threshold``) — a no-op when the
        aperture already fills the file. Only relevant on the
        physical-rescale path (source shape different from the PSF's);
        see §12.3 of ``REFERENCE_TECHNIQUE.md``.
    mask_native_aperture_center, ho_native_aperture_center : tuple of (int, int), optional
        Only used when the matching ``*_native_aperture_px`` is given:
        ``(row, col)`` pixel coordinates of the aperture's centre within
        the source file, if it is **not** centred there (unequal margins
        on opposite sides). ``None`` (default): assume it *is* centred.
        Auto-detection (``*_native_aperture_px=None``) locates an
        off-centre aperture on its own already, no need for this.
    aperture_threshold : float, default 0.01
        Fraction of the source file's own maximum value above which a
        pixel counts as part of the aperture, for the auto-detection
        above. Ignored wherever the corresponding ``*_native_aperture_px``
        is given.
    verbose : bool, default True
        Print progress / solver traces and display Plotly diagnostics.
    max_iter : int
        [gerchberg_saxton] Number of GS iterations.
    beta : float
        [gerchberg_saxton] Reserved relaxation factor (see
        :class:`backends.gerchberg_saxton.RetrievalConfig`).
    seed : int, optional
        [gerchberg_saxton] Seed for the random pupil-phase initialisation
        (reproducible runs). ``None`` draws a fresh random start each call.
    fit_piston_tilt : bool
        [optimization] If False, Z1-Z3 are held at zero.
    max_nfev : int, optional
        [optimization, phase_diversity] Cap on the number of solver
        function evaluations.
    diversity_path : str or Path, optional
        [phase_diversity] Path to the second (defocused) PSF. **Required**
        when ``method="phase_diversity"``; ignored (with a warning
        otherwise) for the other methods.
    defocus : float, optional
        [phase_diversity] Known diversity defocus [m]. Required when
        ``method="phase_diversity"``.
    roi_radius : int
        [phase_diversity] Half-width of the region of interest [px].
    loss : str
        [phase_diversity] Robust loss passed to ``least_squares``.
    poisson_weighting : bool
        [phase_diversity] Weight residuals by shot-noise.
    sign_prior : tuple of (int, float), optional
        [gerchberg_saxton, optimization] ``(reference_index, expected_sign)``:
        a known sign a priori on one even-azimuthal-order Zernike mode
        (typically ``(4, +1.0)`` for a known defocus direction), used to
        resolve the sign ambiguity inherent to single-PSF phase retrieval
        (see :func:`zernike_fit.resolve_sign_ambiguity` for the theory and
        ``REFERENCE_TECHNIQUE.md`` §3.7). Ignored (with a warning)
        for ``"phase_diversity"``, which is already unambiguous.
    known_ho_wfe_path : str or Path, optional
        [optimization, phase_diversity] Path to a known, fixed high-order
        (HO) WFE map [nm], at its **own** native resolution (registered
        onto the working grid the same way as ``mask_path``, via
        :func:`io_utils.load_and_prepare_ho_wfe` — need not match
        ``mask_path``'s native resolution either) — measured independently
        (interferogram, as-built optical model, ground calibration). Added
        to the modelled phase without ever being optimised, so it stops
        silently biasing the fitted low-order coefficients (see §11.3 of
        ``REFERENCE_TECHNIQUE.md``). Ignored (with a warning) for
        ``"gerchberg_saxton"``, whose iteration has no low/high-order
        split to attach a fixed term to.
    zernike_priors : dict, optional
        [optimization, phase_diversity] Per-mode informative Bayesian
        prior, ``{noll_index: (mean_nm, sigma_nm)}`` — added to the
        residual vector as a per-mode L2 penalty (Gaussian MAP; a strict
        generalisation of ``OpticalConfig.l2_reg``, see
        :mod:`eoqual.phaseretrieval.zernike_priors` and §6.5bis of
        ``REFERENCE_TECHNIQUE.md`` for the derivation and the important
        caveat that a *decreasing-with-radial-order* prior, common in
        ground-based astronomy (Kolmogorov turbulence statistics), is
        **not** generally justified for a satellite instrument's own WFE
        — priors here should come from an as-built model, a prior
        calibration or an error budget, not from that generic assumption).
        Ignored (with a warning) for ``"gerchberg_saxton"``. ``None``
        (default) disables it.
    zernike_bounds : dict, optional
        [optimization, phase_diversity] Per-mode hard box constraint,
        ``{noll_index: (low_nm, high_nm)}`` — the uniform-prior analogue of
        ``zernike_priors`` (see same reference). Can be combined with
        ``zernike_priors``. Ignored (with a warning) for
        ``"gerchberg_saxton"``. ``None`` (default) leaves every mode
        unbounded.

    Returns
    -------
    coeffs_nm : NDArray
        Zernike coefficients (Noll order), in nanometres of WFE.
    phase_fit_nm : NDArray
        Zernike-reconstructed pupil phase map, in nanometres of WFE.
    diagnostics : dict
        Method-specific fit/optimisation diagnostics, plus the ``method``
        name itself.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or if ``method="phase_diversity"`` is
        requested without ``diversity_path``/``defocus``.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")

    if diversity_path is not None and method != "phase_diversity":
        logger.warning(
            f"diversity_path is ignored by method={method!r} "
            "(only 'phase_diversity' uses a second PSF)."
        )

    if known_ho_wfe_path is not None and method == "gerchberg_saxton":
        logger.warning(
            "known_ho_wfe_path is ignored by method='gerchberg_saxton' "
            "(no low/high-order split to attach a fixed term to — see "
            "REFERENCE_TECHNIQUE.md §11.3/§2.2)."
        )

    if (zernike_priors or zernike_bounds) and method == "gerchberg_saxton":
        logger.warning(
            "zernike_priors/zernike_bounds is ignored by "
            "method='gerchberg_saxton' (no explicit cost function on the "
            "Zernike coefficients to penalise/bound — see "
            "REFERENCE_TECHNIQUE.md §6.5bis)."
        )

    # ------------------------------------------------------------------
    # Chargement PSF + masque (commun à toutes les méthodes)
    # ------------------------------------------------------------------
    psf = load_image(psf_path)
    N_psf = psf.shape[0]  # On suppose une matrice carrée (N_psf, N_psf)
    psf = psf / np.sum(psf)

    # n_pup_target : nombre de pixels que l'ouverture occupe réellement
    # dans le cadre (N_psf, N_psf) — None si le masque était déjà sur la
    # grille de la PSF (pas de mise à l'échelle physique, l'ouverture est
    # alors supposée remplir le cadre). Indispensable à la base de Zernike
    # (ci-dessous, aperture_px=...) pour rester à la même échelle que le
    # masque dès que oversampling > 1 introduit une marge — voir §12.5 de
    # REFERENCE_TECHNIQUE.md.
    mask, n_pup_target = load_and_prepare_mask(
        psf_shape=(N_psf, N_psf),
        mask_path=mask_path,
        wavelength=wavelength,
        focal_length=focal_length,
        pupil_diameter=pupil_diameter,
        pixel_pitch=pixel_pitch,
        oversampling=oversampling,
        binarize=mask_binarize,
        native_aperture_px=mask_native_aperture_px,
        native_aperture_center=mask_native_aperture_center,
        aperture_threshold=aperture_threshold,
    )

    known_ho_wfe_nm: Optional[NDArray[np.float64]] = None
    if known_ho_wfe_path is not None and method != "gerchberg_saxton":
        known_ho_wfe_nm = load_and_prepare_ho_wfe(
            psf_shape=(N_psf, N_psf),
            ho_wfe_path=known_ho_wfe_path,
            wavelength=wavelength,
            focal_length=focal_length,
            pupil_diameter=pupil_diameter,
            pixel_pitch=pixel_pitch,
            oversampling=oversampling,
            native_aperture_px=ho_native_aperture_px,
            native_aperture_center=ho_native_aperture_center,
            aperture_threshold=aperture_threshold,
        )

    rad_to_nm = wavelength * 1e9 / (2.0 * np.pi)
    phase_nm: Optional[NDArray[np.float64]] = None

    # ------------------------------------------------------------------
    # Dispatch method
    # ------------------------------------------------------------------
    if method == "gerchberg_saxton":
        retrieval_cfg = RetrievalConfig(max_iter=max_iter, beta=beta, verbose=verbose, seed=seed)
        phase_rad = gerchberg_saxton(psf, mask, retrieval_cfg)
        phase_nm = phase_rad * rad_to_nm

        zern_cfg = ZernikeConfig(
            order=zernike_order, wavelength=wavelength, orthogonal_basis=orthogonal_basis,
            aperture_px=n_pup_target,
        )
        coeffs_nm, phase_fit_nm, diagnostics = fit_zernike(phase_rad, mask, zern_cfg)

    elif method == "optimization":
        opt_cfg = OpticalConfig(
            wavelength=wavelength,
            zernike_order=zernike_order,
            fit_piston_tilt=fit_piston_tilt,
            max_nfev=max_nfev,
            verbose=verbose,
            orthogonal_basis=orthogonal_basis,
            aperture_px=n_pup_target,
            known_ho_wfe_nm=known_ho_wfe_nm,
            zernike_priors=zernike_priors,
            zernike_bounds=zernike_bounds,
        )
        coeffs_nm, diagnostics = fit_zernike_from_psf(psf, mask, opt_cfg)

        basis = resolve_zernike_basis(
            npix=N_psf, nterms=zernike_order, mask=mask, orthogonalize=orthogonal_basis,
            aperture_px=n_pup_target,
        )
        phase_fit_nm = phase_from_coefficients(coeffs_nm / rad_to_nm, basis, mask) * rad_to_nm

    else:  # method == "phase_diversity"
        if diversity_path is None:
            raise ValueError("method='phase_diversity' requires diversity_path (second PSF).")
        if defocus is None:
            raise ValueError("method='phase_diversity' requires a known 'defocus' [m].")

        psf_defocus = load_image(diversity_path)
        psf_defocus = psf_defocus / np.sum(psf_defocus)

        pd_cfg = PhaseDiversityConfig(
            wavelength=wavelength,
            zernike_order=zernike_order,
            defocus=defocus,
            roi_radius=roi_radius,
            loss=loss,
            poisson_weighting=poisson_weighting,
            max_nfev=max_nfev,
            verbose=verbose,
            orthogonal_basis=orthogonal_basis,
            aperture_px=n_pup_target,
            known_ho_wfe_nm=known_ho_wfe_nm,
            zernike_priors=zernike_priors,
            zernike_bounds=zernike_bounds,
        )
        coeffs_nm, diagnostics = fit_phase_diversity(psf, psf_defocus, mask, pd_cfg)

        basis = resolve_zernike_basis(
            npix=N_psf, nterms=zernike_order, mask=mask, orthogonalize=orthogonal_basis,
            aperture_px=n_pup_target,
        )
        phase_fit_nm = phase_from_coefficients(coeffs_nm / rad_to_nm, basis, mask) * rad_to_nm

    # ------------------------------------------------------------------
    # Correction optionnelle de l'ambiguïté de signe (1 PSF uniquement)
    # ------------------------------------------------------------------
    if sign_prior is not None:
        if method == "phase_diversity":
            logger.warning(
                "sign_prior is ignored by method='phase_diversity' (already unambiguous)."
            )
        else:
            reference_index, expected_sign = sign_prior
            coeffs_nm = resolve_sign_ambiguity(
                coeffs_nm, reference_index=reference_index, expected_sign=expected_sign
            )
            basis = resolve_zernike_basis(
                npix=N_psf, nterms=zernike_order, mask=mask, orthogonalize=orthogonal_basis,
            )
            phase_fit_nm = phase_from_coefficients(coeffs_nm / rad_to_nm, basis, mask) * rad_to_nm

    diagnostics = {"method": method, **diagnostics}

    # ------------------------------------------------------------------
    # Diagnostic visualisation (optional)
    # ------------------------------------------------------------------
    if verbose:
        visualisation_diagnostics(
            psf=psf,
            mask=mask,
            phase_fit_nm=phase_fit_nm,
            rad_to_nm=rad_to_nm,
            phase_nm=phase_nm,
        )

    return coeffs_nm, phase_fit_nm, diagnostics


# ======================================================================
# CLI
# ======================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """
    Construit le parseur d'arguments en ligne de commande.

    Returns
    -------
    argparse.ArgumentParser
        Parseur configuré avec les options du script.
    """
    parser = argparse.ArgumentParser(
        description="Recover Zernike (Noll) coefficients from one or two PSF using a binary pupil mask."
    )
    parser.add_argument("--psf", type=Path, required=True, help="Path to the PSF image")
    parser.add_argument("--mask", type=Path, required=True, help="Path to the binary pupil mask image")
    parser.add_argument("--output", type=Path, required=True, help="Répertoire de sortie des résultats.")
    parser.add_argument(
        "--method", choices=METHODS, default="gerchberg_saxton",
        help="Phase-retrieval algorithm (default gerchberg_saxton).",
    )
    parser.add_argument("--wavelength", type=float, default=550e-9, help="Wavelength in metres (default 550 nm)")
    parser.add_argument("--focal_length", type=float, default=4.0, help="Focal length in metres")
    parser.add_argument("--pupil_diameter", type=float, default=0.5, help="Pupil diameter in metres")
    parser.add_argument("--pixel_pitch", type=float, default=4e-6, help="Native detector pixel pitch in metres")
    parser.add_argument(
        "--oversampling", type=int, default=1,
        help="Oversampling factor of --psf (and --diversity) relative to native "
             "detector sampling, e.g. 4 for a super-resolved PSF (default 1: "
             "already at native resolution).",
    )
    parser.add_argument("--zernike_order", type=int, default=36, help="Maximum Zernike Noll order")
    parser.add_argument(
        "--mask_no_binarize", action="store_false", dest="mask_binarize",
        help="Keep the mask as a graded [0,1] transmission after resampling "
             "instead of thresholding it to strict {0,1} (default: binarize).",
    )
    parser.add_argument(
        "--mask_native_aperture_px", type=int, default=None,
        help="Aperture's own pixel span within --mask, if it has a zero margin "
             "around it rather than filling the file edge to edge (default: "
             "auto-detected from the pixel content, see --aperture_threshold).",
    )
    parser.add_argument(
        "--ho_native_aperture_px", type=int, default=None,
        help="Same as --mask_native_aperture_px, for --known_ho_wfe.",
    )
    parser.add_argument(
        "--mask_native_aperture_center", type=int, nargs=2, default=None,
        metavar=("ROW", "COL"),
        help="Pixel coordinates of the aperture's centre within --mask, only "
             "if --mask_native_aperture_px is given and the aperture is NOT "
             "centred in the file (default: assume it is centred).",
    )
    parser.add_argument(
        "--ho_native_aperture_center", type=int, nargs=2, default=None,
        metavar=("ROW", "COL"),
        help="Same as --mask_native_aperture_center, for --known_ho_wfe / "
             "--ho_native_aperture_px.",
    )
    parser.add_argument(
        "--aperture_threshold", type=float, default=0.01,
        help="Fraction of the mask's/HO map's own maximum value above which a "
             "pixel counts as part of the aperture, for auto-detecting its "
             "margin. Ignored wherever *_native_aperture_px is given.",
    )
    parser.add_argument(
        "--orthogonal_basis", action="store_true",
        help="Fit on a Gram-Schmidt basis re-orthogonalised on the actual "
             "pupil mask instead of the standard Noll basis (recommended "
             "for an obstructed pupil: central obstruction, spiders).",
    )
    parser.add_argument("--max_iter", type=int, default=200, help="[gerchberg_saxton] Number of GS iterations")
    parser.add_argument("--beta", type=float, default=0.9, help="[gerchberg_saxton] Relaxation factor")
    parser.add_argument("--seed", type=int, default=None, help="[gerchberg_saxton] RNG seed (reproducible runs)")
    parser.add_argument(
        "--fit_piston_tilt", action="store_true",
        help="[optimization] Also fit piston/tip/tilt (Z1-Z3).",
    )
    parser.add_argument(
        "--diversity", type=Path, default=None,
        help="[phase_diversity] Path to the second (defocused) PSF.",
    )
    parser.add_argument("--defocus", type=float, default=None, help="[phase_diversity] Known defocus in metres")
    parser.add_argument("--roi_radius", type=int, default=32, help="[phase_diversity] ROI half-width in pixels")
    parser.add_argument("--loss", type=str, default="soft_l1", help="[phase_diversity] Robust loss function")
    parser.add_argument(
        "--poisson_weighting", action="store_true", default=True,
        help="[phase_diversity] Weight residuals by shot-noise (default on).",
    )
    parser.add_argument(
        "--sign_prior_index", type=int, default=None,
        help="[gerchberg_saxton, optimization] Noll index of an even-m mode with a known sign (e.g. 4 for defocus).",
    )
    parser.add_argument(
        "--sign_prior_sign", type=float, default=1.0,
        help="[gerchberg_saxton, optimization] Expected sign (+1 or -1) of --sign_prior_index.",
    )
    parser.add_argument(
        "--known_ho_wfe", type=Path, default=None,
        help="[optimization, phase_diversity] Path to a known, fixed "
             "high-order WFE map [nm], at its own native resolution, added "
             "to the model without being optimised. Ignored for "
             "gerchberg_saxton.",
    )
    parser.add_argument("--config_json", type=Path, default=None, help="Optionnel: fichier JSON de config")

    return parser


def _setup_logger(out_root: Path, prefix: str) -> Path:
    """Configure loguru sinks (console + file) and return the log file path."""
    now = datetime.now()
    log_file_path = out_root / f"{prefix}_{now.strftime('%Y%m%d_%Hh%Mm%S')}.log"

    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} |"
        "{level: <8} |"
        "{name: ^22} |"
        "{function: ^20} |"
        "{line: >4} |"
        "{message}"
    )
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", format=fmt, colorize=True, backtrace=True, diagnose=True)
    logger.add(log_file_path, format=fmt, level="DEBUG")
    logger.add(lambda _: sys.exit(1), level="ERROR")

    return log_file_path


def main() -> None:
    """Command-line entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.config_json and args.config_json.exists():
        with open(args.config_json, "r") as file:
            proc_param = commentjson.load(file)
    else:
        proc_param = {
            "psf": str(args.psf),
            "mask": str(args.mask),
            "output": str(args.output),
            "method": args.method,
            "wavelength": args.wavelength,
            "focal_length": args.focal_length,
            "pupil_diameter": args.pupil_diameter,
            "pixel_pitch": args.pixel_pitch,
            "oversampling": args.oversampling,
            "zernike_order": args.zernike_order,
            "orthogonal_basis": args.orthogonal_basis,
            "mask_binarize": args.mask_binarize,
            "mask_native_aperture_px": args.mask_native_aperture_px,
            "ho_native_aperture_px": args.ho_native_aperture_px,
            "mask_native_aperture_center": (
                tuple(args.mask_native_aperture_center) if args.mask_native_aperture_center else None
            ),
            "ho_native_aperture_center": (
                tuple(args.ho_native_aperture_center) if args.ho_native_aperture_center else None
            ),
            "aperture_threshold": args.aperture_threshold,
            "max_iter": args.max_iter,
            "beta": args.beta,
            "seed": args.seed,
            "fit_piston_tilt": args.fit_piston_tilt,
            "diversity": str(args.diversity) if args.diversity else None,
            "defocus": args.defocus,
            "roi_radius": args.roi_radius,
            "loss": args.loss,
            "poisson_weighting": args.poisson_weighting,
            "sign_prior_index": args.sign_prior_index,
            "sign_prior_sign": args.sign_prior_sign,
            "known_ho_wfe": str(args.known_ho_wfe) if args.known_ho_wfe else None,
        }

    out_root = Path(proc_param["output"])
    out_root.mkdir(parents=True, exist_ok=True)
    log_file_path = _setup_logger(out_root, "recale")

    logger.info("=================================")
    logger.info("RECALAGE")
    logger.info("=================================")
    logger.info(f"Fichier de logs: {log_file_path.name}")
    logger.info(f"Paramètres d'initialisation : \n{proc_param}")

    coeffs, _phase, diag = psf_to_wfe(
        psf_path=proc_param["psf"],
        mask_path=proc_param["mask"],
        wavelength=proc_param["wavelength"],
        focal_length=proc_param["focal_length"],
        pupil_diameter=proc_param["pupil_diameter"],
        pixel_pitch=proc_param["pixel_pitch"],
        oversampling=proc_param["oversampling"],
        method=proc_param["method"],
        zernike_order=proc_param["zernike_order"],
        orthogonal_basis=proc_param["orthogonal_basis"],
        mask_binarize=proc_param["mask_binarize"],
        mask_native_aperture_px=proc_param["mask_native_aperture_px"],
        ho_native_aperture_px=proc_param["ho_native_aperture_px"],
        mask_native_aperture_center=proc_param["mask_native_aperture_center"],
        ho_native_aperture_center=proc_param["ho_native_aperture_center"],
        aperture_threshold=proc_param["aperture_threshold"],
        max_iter=proc_param["max_iter"],
        beta=proc_param["beta"],
        seed=proc_param["seed"],
        fit_piston_tilt=proc_param["fit_piston_tilt"],
        diversity_path=proc_param["diversity"],
        defocus=proc_param["defocus"],
        roi_radius=proc_param["roi_radius"],
        loss=proc_param["loss"],
        poisson_weighting=proc_param["poisson_weighting"],
        sign_prior=(
            (proc_param["sign_prior_index"], proc_param["sign_prior_sign"])
            if proc_param["sign_prior_index"] is not None else None
        ),
        known_ho_wfe_path=proc_param["known_ho_wfe"],
        verbose=True,
    )

    logger.info("--- Zernike coefficients — Noll order (Calibrated in nm WFE) ---")
    for i, c in enumerate(coeffs, start=1):
        logger.info(f"   > Z{i:2d}: {c:9.3f} nm")

    logger.info("--- Diagnostics: ---")
    for k, v in diag.items():
        logger.info(f"   > {k}: {v}")

    logger.success("Fin du traitement !")


if __name__ == "__main__":
    main()
