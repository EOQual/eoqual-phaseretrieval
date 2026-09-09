# Change Log

## Version History

0.1.0 (2026-09-08)

:   -   Première version publique.
    -   **3 méthodes** dans le socle (MIT, réimplémentations propres à
        partir des articles cités — aucun portage de code tiers) :
        `gerchberg_saxton`, `optimization`, `phase_diversity`, toutes
        derrière un point d'entrée unique
        `eoqual.phaseretrieval.pipeline.psf_to_wfe(method=...)`.
    -   Registre central (`config.py::METHODS_CONFIGS`) et utilitaire de
        découverte `eoqual.phaseretrieval.runner.list_methods()`.
    -   Décomposition en coefficients de Zernike (ordre de Noll) commune
        aux trois méthodes (`zernike_fit.py`), y compris la résolution de
        l'ambiguïté de signe par a priori (`resolve_sign_ambiguity`) et
        une base Zernike ré-orthogonalisée pour pupilles obstruées
        (`zernike_basis_orthogonal.ZernikeGramSchmidt`).
    -   Diagnostics de qualité optique indépendants du fit Zernike
        (`diagnostics.py`) : Strehl, WFE RMS (Maréchal), MTF, ellipticité.
    -   **Module de simulation séparé**
        (`eoqual.phaseretrieval.simulation`) : génération d'une PSF (ou
        d'une paire focus/défocus) à vérité de Zernike connue, avec bruit
        détecteur réaliste optionnel (Poisson + lecture,
        `simulation.noise.DetectorNoiseConfig`) — sert de banc de test
        pour les trois méthodes, sans dépendre de données figées
        (`tests/test_simulation_roundtrip.py`).
    -   Documentation : `README.md` (installation, catalogue, licence),
        `REFERENCE_TECHNIQUE.md` (théorie, implémentation, limites,
        cas d'usage télédétection), `THIRD_PARTY_LICENSES.md`,
        `CONTRIBUTING.md`.
    -   Architecture en paquet `src/eoqual/phaseretrieval` : une méthode
        de retrouvé de phase = un module de `backends/`.

## Évolutions futures

Voir `REFERENCE_TECHNIQUE.md` §11 (modèle système complet : détecteur,
WFE haut ordre, filée, polychromatisme) et §6.5 (ajouter une méthode).
