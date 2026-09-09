"""
Utilitaires transverses : liste des méthodes disponibles.

Le point d'entrée unique pour appeler une méthode par son nom est déjà
:func:`eoqual.phaseretrieval.pipeline.psf_to_wfe` (paramètre ``method=``) —
ce module n'ajoute donc qu'un utilitaire de découverte, dans le même
esprit que ``list_methods()`` des autres modules EOQual.

Exemple
-------
>>> from eoqual.phaseretrieval.runner import list_methods
>>> list_methods()
>>> from eoqual.phaseretrieval.pipeline import psf_to_wfe
>>> coeffs_nm, phase_nm, diag = psf_to_wfe(..., method="phase_diversity")
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .config import METHODS_CONFIGS

__all__ = ["list_methods"]


def list_methods() -> None:
    """Affiche toutes les méthodes de récupération de phase disponibles."""
    console = Console()
    table = Table(
        title="Méthodes disponibles dans eoqual.phaseretrieval",
        header_style="bold magenta",
    )
    table.add_column("Méthode", style="bold cyan", no_wrap=True)
    table.add_column("PSF requises", justify="center")
    table.add_column("Licence")
    table.add_column("Référence")

    for name, cfg in METHODS_CONFIGS.items():
        table.add_row(name, str(cfg["n_psf"]), cfg["license"], cfg["reference"])

    console.print(table)
