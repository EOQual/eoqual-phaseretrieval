"""
Registre central des méthodes de retrouvé de phase disponibles dans
``eoqual.phaseretrieval``.

Chaque entrée décrit : le nombre de PSF requises, la licence de
l'implémentation (toutes réimplémentées ici à partir des articles cités,
aucun portage de code tiers) et sa référence bibliographique principale.

Voir ``README.md`` (catalogue) et ``REFERENCE_TECHNIQUE.md`` (principe,
implémentation, limites de chaque méthode) pour le détail.
"""

from collections import OrderedDict

METHODS_CONFIGS: OrderedDict = OrderedDict(
    {
        "gerchberg_saxton": {
            "n_psf": 1,
            "license": "MIT (implémentation propre)",
            "reference": "Gerchberg & Saxton (1972), Optik 35, 237-246",
        },
        "optimization": {
            "n_psf": 1,
            "license": "MIT (implémentation propre)",
            "reference": (
                "Branch, Coleman & Li (1999), SIAM J. Sci. Comput. 21(1), 1-23 "
                "(algorithme Trust-Region-Reflective) ; Fienup (1993), "
                "Applied Optics 32(10), 1737-1746"
            ),
        },
        "phase_diversity": {
            "n_psf": 2,
            "license": "MIT (implémentation propre)",
            "reference": (
                "Gonsalves (1982), Optical Engineering 21(5), 829-832 ; "
                "Paxman, Schulz & Fienup (1992), JOSA A 9(7), 1072-1085"
            ),
        },
    }
)
