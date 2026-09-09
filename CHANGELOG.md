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
        de récupération de phase = un module de `backends/`.
    -   Option `orthogonal_basis` (`ZernikeConfig`, `OpticalConfig`,
        `PhaseDiversityConfig`, `psf_to_wfe(...)`/`--orthogonal_basis` en
        CLI) : bascule les trois méthodes sur une base de Zernike
        ré-orthogonalisée par Gram-Schmidt sur le masque réel
        (`zernike_basis_orthogonal.ZernikeGramSchmidt`, déjà présent mais
        jusqu'ici non branché) plutôt que la base de Noll standard —
        recommandé pour une pupille obstruée. `False` par défaut,
        comportement historique inchangé.
    -   `eoqual.phaseretrieval.simulation` étendu à `orthogonal_basis`
        (`simulate_psf_from_zernike`, `PhaseDiversitySimulationConfig`) :
        génère une vérité terrain directement dans la base Gram-Schmidt du
        masque, pour valider `orthogonal_basis=True` de bout en bout sans
        le biais de changement de base identifié à l'introduction de cette
        option (`tests/test_simulation_orthogonal.py`).
    -   `zernike_fit.project_orthogonal_to_standard` : reconvertit des
        coefficients ajustés en base Gram-Schmidt vers la base de Noll
        standard (Z4 = défocus, …) pour l'interprétation — conversion
        linéaire exacte (aux erreurs de conditionnement/convergence en
        amont près), voir REFERENCE_TECHNIQUE.md §10.2.
    -   Option `known_ho_wfe_nm` (`OpticalConfig`, `PhaseDiversityConfig`,
        `psf_to_wfe(..., known_ho_wfe_path=...)`/`--known_ho_wfe` en CLI) :
        incorpore une carte de WFE haut ordre (HO) connue, mesurée
        indépendamment, au modèle direct — additive en phase (nouvelle
        fonction `optics.wfe_map_to_phase`), jamais optimisée, dans les
        deux méthodes basées optimisation (pour `"phase_diversity"`,
        ajoutée aux deux canaux focus/défocus). Débiaise les coefficients
        bas ordre estimés d'une contribution HO non modélisée par ailleurs
        (voir REFERENCE_TECHNIQUE.md §11.3) — vérifié empiriquement
        (`tests/test_known_ho_wfe.py`) : sans la carte, un $Z_{21}$ connu
        de 40 nm biaise $Z_4$ de plus de 20 nm ; avec, la vérité est
        recouvrée exactement. Ignorée (avec avertissement) pour
        `"gerchberg_saxton"`. `eoqual.phaseretrieval.simulation` accepte le
        même paramètre pour générer des cas de test correspondants.
    -   Correctif de robustesse : `backends.optimization.residuals` clippe
        désormais `measured_psf` à des valeurs non négatives avant la
        racine carrée (comme le fait déjà `gerchberg_saxton` pour sa cible
        d'amplitude) — une PSF bruitée réaliste (bruit de lecture gaussien,
        `simulation.noise.DetectorNoiseConfig`) peut contenir des pixels
        légèrement négatifs, ce qui faisait auparavant échouer
        `least_squares` ("Residuals are not finite"). Découvert en
        exécutant `examples/psf2zernike_demo.ipynb` de bout en bout ;
        régression couverte par
        `tests/test_simulation_roundtrip.py::test_optimization_runs_on_noisy_psf_with_negative_pixels`.
    -   `examples/psf2zernike_demo.ipynb` : références obsolètes corrigées
        (chemins de module `retrieval_*.py`/`pipeline.py` pré-réorganisation
        en paquet, restées dans des cellules de prose depuis le transfert
        initial) et nouvelle section « Fonctionnalités avancées »
        démontrant le module de simulation en mémoire, `orthogonal_basis` +
        `project_orthogonal_to_standard`, et `known_ho_wfe_nm` — les trois
        n'étaient jusqu'ici couverts par aucun exemple exécutable. Les deux
        notebooks (`examples/*.ipynb`) sont ré-exécutés de bout en bout
        (aucune erreur) pour valider l'ensemble.

## Évolutions futures

Voir `REFERENCE_TECHNIQUE.md` §11 (modèle système complet : détecteur,
WFE haut ordre, filée, polychromatisme) et §6.5 (ajouter une méthode).
