#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Décomposition d'une carte de phase sur la base de Zernike (ordre de Noll).

Utilisé par la méthode Gerchberg-Saxton (``backends.gerchberg_saxton``) pour convertir
la phase pupillaire récupérée itérativement en coefficients de Zernike.

Ce module contient aussi :func:`resolve_sign_ambiguity`, un correctif de
l'ambiguïté de signe inhérente à la récupération de phase à partir d'une seule
PSF (voir ``REFERENCE_TECHNIQUE.md``, §2 et §3.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import poppy.zernike as zern
from numpy.typing import NDArray

from .zernike_basis_orthogonal import resolve_zernike_basis


@dataclass(slots=True)
class ZernikeConfig:
    """
    Parameters for the Zernike fit.

    Parameters
    ----------
    order : int
        Number of Zernike modes to fit (Noll ordering), default 36.
    wavelength : float
        Central wavelength in metres, used for the rad → nm conversion.
    orthogonal_basis : bool, default False
        If True, project onto a basis re-orthogonalised (Gram-Schmidt) on
        the actual pupil mask (see
        :func:`zernike_basis_orthogonal.resolve_zernike_basis`, §10 of
        ``REFERENCE_TECHNIQUE.md``) instead of the standard Noll basis.
        Recommended whenever the pupil is not a full, unobstructed disc
        (central obstruction, spiders) — the standard basis is not
        orthogonal there, which degrades ``condition_number`` and mixes
        the physical meaning of each coefficient. ``False`` (default)
        keeps the historical behaviour unchanged.
    aperture_px : int, optional
        Diameter [px] of the aperture within ``pupil_mask``, when it does
        not fill the whole array — see
        ``zernike_basis_orthogonal.resolve_zernike_basis`` and §12.5 of
        ``REFERENCE_TECHNIQUE.md``. ``None`` (default): the aperture fills
        the array (historical behaviour).
    """
    order: int = 36
    wavelength: float = 550e-9
    orthogonal_basis: bool = False
    aperture_px: Optional[int] = None


def fit_zernike(
    phase_rad: NDArray[np.float64],
    pupil_mask: NDArray[np.float64],
    zern_cfg: ZernikeConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[str, float]]:
    """
    Fit Zernike modes (Noll ordering) to a phase map by linear least-squares.

    Parameters
    ----------
    phase_rad : NDArray
        Phase map [rad].
    pupil_mask : NDArray
        Binary pupil mask.
    zern_cfg : ZernikeConfig
        Fit configuration (mode order, wavelength).

    Returns
    -------
    coeffs_nm : NDArray
        Zernike coefficients (Noll order), in nanometres of WFE.
    phase_fit_nm : NDArray
        Reconstructed phase map, in nanometres of WFE.
    diagnostics : dict
        Fit quality metrics (matrix rank, condition number, RMS, residuals),
        all in nanometres.
    """
    if phase_rad.shape != pupil_mask.shape:
        raise ValueError(
            f"phase shape {phase_rad.shape} != pupil_mask shape {pupil_mask.shape}"
        )

    inside = pupil_mask > 0
    if not np.any(inside):
        raise ValueError("Pupil mask contains no valid pixels")

    npix = phase_rad.shape[0]

    phase_rad = np.asarray(phase_rad, dtype=np.float64)
    pupil_mask = np.asarray(pupil_mask, dtype=np.float64)

    # Ensure phase is only defined inside pupil
    phase_rad = phase_rad * pupil_mask

    # Build Zernike basis (Noll ordering, or Gram-Schmidt-orthogonalised on
    # this mask if zern_cfg.orthogonal_basis — see ZernikeConfig docstring).
    basis = resolve_zernike_basis(
        npix=npix, nterms=zern_cfg.order, mask=pupil_mask,
        orthogonalize=zern_cfg.orthogonal_basis, aperture_px=zern_cfg.aperture_px,
    )

    # Design matrix
    A = np.column_stack([mode[inside] for mode in basis])
    y = phase_rad[inside]
    rank = np.linalg.matrix_rank(A)
    if rank < A.shape[1]:
        raise np.linalg.LinAlgError(
            f"Zernike matrix rank deficient ({rank}/{A.shape[1]})"
        )

    # Least-squares fit
    coeffs_rad, *_ = np.linalg.lstsq(A, y, rcond=None)

    # Remove piston and tilts
    if coeffs_rad.size >= 3:
        coeffs_rad[:3] = 0.0

    # Reconstruction
    phase_fit_rad = np.tensordot(coeffs_rad, basis, axes=(0, 0))
    phase_fit_rad *= pupil_mask

    # Conversion rad --> nm
    rad_to_nm = zern_cfg.wavelength * 1e9 / (2.0 * np.pi)
    coeffs_nm = coeffs_rad * rad_to_nm
    phase_fit_nm = phase_fit_rad * rad_to_nm

    # Diagnostics
    phase_rms_rad = np.std(phase_rad[inside])
    fit_rms_rad = np.std(phase_fit_rad[inside])
    residual_rad = (phase_rad - phase_fit_rad)[inside]
    res_rms_rad = np.sqrt(np.mean(residual_rad ** 2))
    res_std_rad = np.std(residual_rad)

    diagnostics = {
        "matrix_rank": int(rank),
        "condition_number": float(np.linalg.cond(A)),
        "phase_rms_nm": float(phase_rms_rad * rad_to_nm),
        "fit_rms_nm": float(fit_rms_rad * rad_to_nm),
        "residual_rms_nm": float(res_rms_rad * rad_to_nm),
        "residual_std_nm": float(res_std_rad * rad_to_nm),
    }

    return coeffs_nm, phase_fit_nm, diagnostics


def even_azimuthal_mask(order: int) -> NDArray[np.bool_]:
    """
    Flag the Zernike modes (Noll ordering) with an **even** azimuthal
    frequency ``m`` — the modes affected by the sign ambiguity of
    single-PSF phase retrieval (see :func:`resolve_sign_ambiguity`).

    Parameters
    ----------
    order : int
        Number of Zernike modes (Noll indices ``1..order``).

    Returns
    -------
    NDArray
        Boolean array, shape ``(order,)``; ``True`` at even-``m`` modes
        (piston, defocus, astigmatism, spherical, secondary astigmatism,
        quadrafoil, …).
    """
    m = np.array([zern.noll_indices(j)[1] for j in range(1, order + 1)])
    return (m % 2 == 0)


def resolve_sign_ambiguity(
    coeffs_nm: NDArray[np.float64],
    reference_index: int = 4,
    expected_sign: float = 1.0,
) -> NDArray[np.float64]:
    """
    Resolve the single-PSF sign ambiguity using one a priori sign.

    Background
    ----------
    For a real, centrosymmetric pupil amplitude (any circular/annular mask
    without an orientation-breaking feature), a pupil phase
    :math:`\\phi(\\xi,\\eta)` and its "twin" :math:`-\\phi(-\\xi,-\\eta)`
    produce **exactly** the same PSF (a direct consequence of the Fourier
    conjugation identity :math:`|\\mathrm{FT}\\{g^*(-\\cdot)\\}| = |\\mathrm{FT}\\{g\\}|`).
    Expanded on the Zernike (Noll) basis, this twin has coefficients
    :math:`c_j' = (-1)^{m_j+1} c_j`, where :math:`m_j` is the azimuthal
    frequency of mode :math:`j`: modes with **even** :math:`m` (piston,
    defocus, astigmatism, spherical, …) flip sign **together**, while
    modes with **odd** :math:`m` (tip/tilt, coma, trefoil, …) keep their
    correct sign in *both* branches — they are never ambiguous from a
    single PSF.

    Consequently, a single sign a priori on **any one** even-``m`` mode
    (typically :math:`Z_4`, whose sign is often known from the direction
    of a deliberately introduced focus offset) is enough to disambiguate
    *all* even-``m`` modes at once; nothing needs to be done for the
    odd-``m`` modes.

    This only corrects the physical twin-image ambiguity. It does **not**
    fix a solver that failed to converge to either exact branch (check
    ``diagnostics["cost"]`` / ``"residual_rms_nm"`` first — see the
    "Gerchberg-Saxton" caveat in ``REFERENCE_TECHNIQUE.md``, §3.7).

    Parameters
    ----------
    coeffs_nm : NDArray
        Zernike coefficients (Noll order), as returned by
        :func:`fit_zernike`, :func:`backends.optimization.fit_zernike_from_psf`
        or (for reference only — see caveat above)
        :func:`backends.gerchberg_saxton.gerchberg_saxton` + :func:`fit_zernike`.
    reference_index : int, default 4
        Noll index (1-based) of the even-``m`` mode whose sign is known a
        priori. Must itself have an even azimuthal frequency (the default,
        :math:`Z_4` / defocus, always does).
    expected_sign : float, default 1.0
        Known sign of ``coeffs_nm[reference_index - 1]`` (only its sign is
        used).

    Returns
    -------
    NDArray
        Corrected coefficients: unchanged if the reference mode already
        has ``expected_sign``, otherwise all even-``m`` modes negated.

    Raises
    ------
    ValueError
        If ``reference_index`` does not correspond to an even-``m`` mode.
    """
    order = coeffs_nm.size
    even = even_azimuthal_mask(order)

    if not even[reference_index - 1]:
        n, m = zern.noll_indices(reference_index)
        raise ValueError(
            f"Z{reference_index} has an odd azimuthal frequency (n={n}, m={m}); "
            "it is never sign-ambiguous, so it cannot be used as the reference "
            "for this correction. Pick an even-m mode (e.g. Z4, the default)."
        )

    corrected = coeffs_nm.copy()
    if np.sign(corrected[reference_index - 1]) != np.sign(expected_sign):
        corrected[even] *= -1.0

    return corrected


def project_orthogonal_to_standard(
    coeffs_orthogonal_nm: NDArray[np.float64],
    orthogonal_basis: NDArray[np.float64],
    mask: NDArray[np.float64],
    wavelength: float,
    *,
    order: Optional[int] = None,
    aperture_px: Optional[int] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[str, float]]:
    """
    Re-express Gram-Schmidt-basis coefficients on the standard Noll basis.

    Any of the three retrieval methods can be run with
    ``orthogonal_basis=True`` (§10 of ``REFERENCE_TECHNIQUE.md``) for a
    well-conditioned fit on an obstructed pupil, but the resulting
    coefficients are *not* directly labelled "defocus", "astigmatism",
    etc. beyond the first couple of modes — the Gram-Schmidt basis mixes
    standard modes together. This function converts back: it rebuilds the
    phase map from ``coeffs_orthogonal_nm``/``orthogonal_basis`` and
    re-decomposes it on the standard basis via :func:`fit_zernike`, which
    is exact (up to floating-point/conditioning error) whenever the two
    bases span the same subspace — i.e. no mode was dropped as degenerate
    when ``orthogonal_basis`` was built (see
    :func:`zernike_basis_orthogonal.resolve_zernike_basis`). This is
    possible precisely because the Gram-Schmidt basis is, by
    construction, a set of linear combinations of the standard Zernike
    modes restricted to the mask (§10): the phase it represents lies
    exactly in the span of the standard basis up to the same order, so a
    linear re-fit onto that basis recovers it without approximation
    (setting conditioning aside).

    Parameters
    ----------
    coeffs_orthogonal_nm : NDArray
        Coefficients in the orthogonal (Gram-Schmidt) basis, in nm WFE —
        as returned by :func:`backends.optimization.fit_zernike_from_psf`,
        :func:`backends.phase_diversity.fit_phase_diversity` or
        :func:`fit_zernike`, called with ``orthogonal_basis=True``.
    orthogonal_basis : NDArray
        The actual basis array used for that fit — from
        ``resolve_zernike_basis(npix, nterms, mask, orthogonalize=True)``
        (same ``mask`` and ``nterms`` as the fit).
    mask : NDArray
        Binary pupil mask (must match the one used to build
        ``orthogonal_basis``).
    wavelength : float
        Wavelength [m], for the rad -> nm conversion.
    order : int, optional
        Number of standard Zernike modes to re-fit. Defaults to
        ``orthogonal_basis.shape[0]`` (same order as the original fit) —
        the natural choice, since that is exactly the subspace the phase
        lies in; a higher value cannot recover information that was never
        fitted, and would only add unconstrained standard modes.
    aperture_px : int, optional
        Diameter [px] of the aperture within ``mask``, if it does not fill
        the whole array — must match the ``aperture_px`` used to build
        ``orthogonal_basis`` itself. ``None`` (default): the aperture
        fills the array (historical behaviour).

    Returns
    -------
    coeffs_standard_nm : NDArray
        Coefficients on the standard Noll basis, in nm WFE — the familiar
        naming (:math:`Z_4` = defocus, :math:`Z_5, Z_6` = astigmatism, …).
    phase_fit_nm : NDArray
        Reconstructed phase map, in nm WFE (identical whichever basis it
        is expressed in — only the coefficients change).
    diagnostics : dict
        Same fields as :func:`fit_zernike` — in particular
        ``condition_number``, which quantifies how numerically fragile
        this reprojection is on this mask (large on a heavily obstructed
        pupil, where the standard basis is far from orthogonal — see §10).
    """
    rad_to_nm = wavelength * 1e9 / (2.0 * np.pi)
    coeffs_rad = coeffs_orthogonal_nm / rad_to_nm
    phase_rad = np.tensordot(coeffs_rad, orthogonal_basis, axes=(0, 0)) * mask

    zern_cfg = ZernikeConfig(
        order=order or orthogonal_basis.shape[0], wavelength=wavelength, orthogonal_basis=False,
        aperture_px=aperture_px,
    )
    return fit_zernike(phase_rad, mask, zern_cfg)
