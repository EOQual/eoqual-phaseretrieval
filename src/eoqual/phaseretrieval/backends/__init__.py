"""Une méthode de retrouvé de phase = un module de ce paquet.

Chaque backend expose sa propre dataclass de configuration et une fonction
``fit_...(psf(s), mask, cfg) -> (coeffs_nm, diagnostics)``. Voir
``REFERENCE_TECHNIQUE.md`` §6.5 pour la marche à suivre pour en ajouter un.
"""
