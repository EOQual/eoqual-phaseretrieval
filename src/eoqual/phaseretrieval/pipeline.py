#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point d'entrée unifié : PSF (+ masque pupillaire) -> coefficients de Zernike.

Ce module orchestre les trois méthodes de retrouvé de phase implémentées
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
(:mod:`retrieval_phase_diversity`), avec sa propre configuration ; le
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

from .io_utils import load_and_prepare_mask, load_image, visualisation_diagnostics
from .optics import build_zernike_basis, phase_from_coefficients
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
    detector_size: float,
    method: Method = "gerchberg_saxton",
    zernike_order: int = 36,
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
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[str, float]]:
    """
    End-to-end conversion **PSF -> Zernike (Noll) coefficients**.

    Parameters
    ----------
    psf_path, mask_path : str or Path
        File paths to the PSF intensity image and the binary pupil mask.
    wavelength : float
        Central wavelength in metres (e.g. ``550e-9`` for 550 nm).
    focal_length : float
        Focal length [m], used to physically scale the pupil mask.
    pupil_diameter : float
        Pupil diameter [m].
    detector_size : float
        Detector size [m].
    method : {"gerchberg_saxton", "optimization", "phase_diversity"}
        Phase-retrieval algorithm to use.
    zernike_order : int, default 36
        Number of Zernike modes (Noll order) to fit.
    verbose : bool, default True
        Print progress / solver traces and display Plotly diagnostics.
    max_iter : int
        [gerchberg_saxton] Number of GS iterations.
    beta : float
        [gerchberg_saxton] Reserved relaxation factor (see
        :class:`retrieval_gs.RetrievalConfig`).
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
        ``phase_retrieval_and_diversity.md`` §3.7). Ignored (with a warning)
        for ``"phase_diversity"``, which is already unambiguous.

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

    # ------------------------------------------------------------------
    # Chargement PSF + masque (commun à toutes les méthodes)
    # ------------------------------------------------------------------
    psf = load_image(psf_path)
    N_psf = psf.shape[0]  # On suppose une matrice carrée (N_psf, N_psf)
    psf = psf / np.sum(psf)

    mask, _ = load_and_prepare_mask(
        psf_shape=(N_psf, N_psf),
        mask_path=mask_path,
        wavelength=wavelength,
        focal_length=focal_length,
        pupil_diameter=pupil_diameter,
        detector_size=detector_size,
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

        zern_cfg = ZernikeConfig(order=zernike_order, wavelength=wavelength)
        coeffs_nm, phase_fit_nm, diagnostics = fit_zernike(phase_rad, mask, zern_cfg)

    elif method == "optimization":
        opt_cfg = OpticalConfig(
            wavelength=wavelength,
            zernike_order=zernike_order,
            fit_piston_tilt=fit_piston_tilt,
            max_nfev=max_nfev,
            verbose=verbose,
        )
        coeffs_nm, diagnostics = fit_zernike_from_psf(psf, mask, opt_cfg)

        basis = build_zernike_basis(npix=N_psf, nterms=zernike_order)
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
        )
        coeffs_nm, diagnostics = fit_phase_diversity(psf, psf_defocus, mask, pd_cfg)

        basis = build_zernike_basis(npix=N_psf, nterms=zernike_order)
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
            basis = build_zernike_basis(npix=N_psf, nterms=zernike_order)
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
    parser.add_argument("--detector_size", type=float, default=4e-6, help="Detector size in metres")
    parser.add_argument("--zernike_order", type=int, default=36, help="Maximum Zernike Noll order")
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
            "detector_size": args.detector_size,
            "zernike_order": args.zernike_order,
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
        detector_size=proc_param["detector_size"],
        method=proc_param["method"],
        zernike_order=proc_param["zernike_order"],
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
