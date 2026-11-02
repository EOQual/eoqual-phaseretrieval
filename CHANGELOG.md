# Change Log

## 0.1.0 (2026-09-08)

- Première version publique. 3 méthodes de récupération de phase
  (réimplémentations propres, MIT) : `gerchberg_saxton`, `optimization`,
  `phase_diversity`, derrière un point d'entrée unique
  `pipeline.psf_to_wfe(method=...)`. Registre `config.METHODS_CONFIGS` +
  `runner.list_methods()`.
- Décomposition en coefficients de Zernike (ordre de Noll) commune aux
  trois méthodes (`zernike_fit.py`), avec résolution de l'ambiguïté de
  signe par a priori (`resolve_sign_ambiguity`, §3.7) et base Gram-Schmidt
  pour pupilles obstruées (`zernike_basis_orthogonal.ZernikeGramSchmidt`).
- Diagnostics optiques indépendants du fit Zernike (`diagnostics.py`) :
  Strehl, WFE RMS (Maréchal), MTF, ellipticité.
- Module de simulation séparé (`eoqual.phaseretrieval.simulation`) : PSF
  (ou paire focus/défocus) à vérité de Zernike connue, bruit détecteur
  optionnel (Poisson + lecture) — banc de test pour les trois méthodes
  (`tests/test_simulation_roundtrip.py`).
- Documentation : `README.md`, `REFERENCE_TECHNIQUE.md`,
  `THIRD_PARTY_LICENSES.md`, `CONTRIBUTING.md`.
- Option `orthogonal_basis` (`ZernikeConfig`/`OpticalConfig`/
  `PhaseDiversityConfig`/`psf_to_wfe`/`--orthogonal_basis`) : base de
  Zernike ré-orthogonalisée par Gram-Schmidt sur le masque réel,
  recommandée pour une pupille obstruée (`False` par défaut). Étendue au
  module de simulation (`tests/test_simulation_orthogonal.py`).
- `zernike_fit.project_orthogonal_to_standard` : reconvertit des
  coefficients Gram-Schmidt vers la base de Noll standard pour
  l'interprétation (§10.2).
- Option `known_ho_wfe_nm` (`OpticalConfig`/`PhaseDiversityConfig`/
  `psf_to_wfe(..., known_ho_wfe_path=...)`/`--known_ho_wfe`) : incorpore
  une carte de WFE haut ordre connue au modèle direct (additive, jamais
  optimisée) — débiaise les coefficients bas ordre (§11.3,
  `tests/test_known_ho_wfe.py`). Ignorée pour `"gerchberg_saxton"`.
- Correctif : `backends.optimization.residuals` clippe `measured_psf` à
  des valeurs non négatives avant la racine carrée (une PSF bruitée
  réaliste peut être légèrement négative) — évitait un crash
  `least_squares`.
- `examples/psf2zernike_demo.ipynb` : références obsolètes corrigées ;
  section « Fonctionnalités avancées » (simulation en mémoire,
  `orthogonal_basis`, `known_ho_wfe_nm`).
- **Correctif (bug significatif)** : `pupil.PupilGenerator` clippait son
  rayon à 1.0 avant le test de délimitation de la pupille, renvoyant un
  masque plein cadre au lieu d'un disque/anneau réel — non détecté car
  les tests round-trip étaient auto-cohérents (même masque biaisé utilisé
  pour simuler *et* ajuster). Régression couverte
  (`tests/test_optics_and_pupil.py`) ; exemples numériques du §10/§10.2
  recalculés.
- **Résolutions multiples** (masque/carte HO/PSF à des résolutions
  natives indépendantes, §12) : `detector_size` remplacé par
  `pixel_pitch` + `oversampling` (changement cassant) ;
  `io_utils.compute_pupil_footprint_px`/`load_and_prepare_ho_wfe` ;
  interpolation bilinéaire + anti-aliasing pour le masque, option
  `binarize` (`tests/test_multi_resolution.py`).
- **Masque/carte HO plus grands que l'ouverture** (marge de zéros, §12.3) :
  `mask_native_aperture_px`/`ho_native_aperture_px` (+ auto-détection) et
  `*_native_aperture_center` pour une ouverture décentrée dans le fichier
  source.
- **A priori individuels par mode** (`zernike_priors`/`zernike_bounds`,
  nouveau module `zernike_priors.py`) sur `OpticalConfig`/
  `PhaseDiversityConfig`/`psf_to_wfe` : généralisation de `l2_reg` à une
  moyenne/poids propre à chaque mode (pénalité L2 = MAP gaussien, borne
  dure = MAP uniforme ; dérivation en §6.5bis). Ignoré pour
  `"gerchberg_saxton"`. Correctif associé (`zernike_priors.feasible_start`) :
  démarrer exactement sur une borne finie faisait déclarer à
  `least_squares` une convergence fictive.
- `license.txt` : titulaire des droits (Olivier Amram) au lieu du nom du
  module.

## 0.2.0 (2026-10-25)

- **Correctif (§10.3)** : `PupilGenerator.with_spiders` seuillait chaque
  bras en binaire pur — pour un bras fin (largeur de l'ordre du pixel),
  la transmission dépendait de façon quasi arbitraire de l'alignement
  sous-pixel de la grille (artefact numérique, pas physique), mesuré
  jusqu'à 2 % d'écart de PSF et 1,5 % de Strehl rien qu'en déplaçant le
  masque d'une fraction de pixel. Corrigé par un nouveau paramètre
  `supersample` (défaut `8`, `1` retrouve le comportement historique) :
  la bande de chaque bras est calculée sur une grille interne plus fine
  puis moyennée (`eoqual.core.psf.supersampled_coverage`, nouveau dans
  eoqual-core 0.4.0), réduisant l'artefact d'un facteur ~10. Étude menée
  avant implémentation pour éviter de complexifier sans gain avéré : le
  contour principal d'une pupille (disque, obstruction centrale) n'a
  volontairement **pas** été touché, son erreur de discrétisation étant
  déjà < 0,3 % dès `aperture_px ≥ 64` (le régime normal de ce code) —
  voir REFERENCE_TECHNIQUE.md §10.3 pour les deux études chiffrées.
  Dépendance `eoqual-core` relevée de 0.1.0 à 0.4.0 (purement additif,
  voir son CHANGELOG). 3 nouveaux tests
  (`tests/test_optics_and_pupil.py`) ; suite complète d'eoqual-phaseretrieval
  (82 tests), d'eoqual-deconvolution (111 tests) et d'eoqual-core
  (45 tests) revérifiées sans régression.
- **Correctif (bug significatif, §12.5)** : la base de Zernike était
  toujours évaluée en supposant que l'ouverture remplit tout le cadre,
  alors que `oversampling > 1` (§12.1) la zero-padde dans un cadre plus
  grand — faussait l'échelle de tous les modes dès que l'ouverture ne
  remplissait pas le cadre (défocus tronqué, RMS ≠ 1 de Noll). `pipeline.py`
  calculait déjà `N_pup_target` mais le jetait sans le transmettre à la
  base (`mask, _ = load_and_prepare_mask(...)`). Non détecté pour la même
  raison que le bug `PupilGenerator` ci-dessus : les tests de bout en bout
  du §12 construisaient leur propre vérité terrain avec le même appel non
  paramétré que celui utilisé en interne par `psf_to_wfe` — les deux
  fausses de la même façon, donc cohérentes entre elles, sans jamais être
  comparées à une géométrie de référence correcte. Corrigé par un nouveau
  paramètre `aperture_px`, propagé de `pupil.polar_grid`/`PupilGenerator`/
  `build_zernike_basis` jusqu'à `pipeline.psf_to_wfe`. Aucun changement
  d'API publique ni de comportement par défaut. Les deux tests concernés
  corrigés pour construire leur vérité avec le même `aperture_px`
  (`tests/test_multi_resolution.py`).
- `PupilGenerator(npix, aperture_px=...)` et
  `simulation.simulate_psf_from_zernike`/
  `PhaseDiversitySimulationConfig(aperture_px=...)` : suréchantillonnage
  en mémoire, sans fichier, cohérent avec la base de Zernike.
- **Correctif** : `load_and_prepare_mask`/`load_and_prepare_ho_wfe`
  ignoraient toute marge de zéros autour de l'ouverture dès que le
  fichier source faisait déjà la même taille de tableau que la PSF
  (branche "pas de mise à l'échelle physique") — seule la branche
  "résolution différente" détectait une telle marge (§12.3). Même taille
  de tableau que la PSF ne veut dire que ça : même résolution, pas
  "ouverture déjà cadrée sur le fichier". Corrigé : la détection/recadrage
  (`_crop_to_aperture`) s'applique maintenant aussi dans ce cas,
  recentrant l'ouverture sans changer sa résolution quand une marge est
  trouvée (`aperture_px` renseigné en conséquence) ; comportement
  inchangé quand l'ouverture remplit effectivement le cadre (le cas le
  plus courant). 5 tests dédiés (`tests/test_multi_resolution.py`).
- La taille de l'ouverture estimée est maintenant toujours visible dans
  les logs (`logger.info`/`logger.warning` selon le cas) — auparavant
  seulement en `logger.debug` sur le chemin "résolution différente", et
  absente du tout sur les deux chemins "déjà sur la grille de la PSF".
- **Correctif (§12.6)** : un masque personnalisé ("calculé par ailleurs",
  pas via `PupilGenerator`) passé à `simulate_psf_from_zernike`/
  `fit_zernike_from_psf`/`fit_phase_diversity` sans `aperture_px`
  explicite était silencieusement traité comme si son ouverture
  remplissait tout le cadre `npix` — même décalage d'échelle que §12.5,
  par un chemin différent (masque fourni, pas généré). Le biais sur les
  coefficients de Zernike estimés dépendait de `npix` (rapporté par
  l'utilisateur : la divergence apparaissait en changeant la taille de
  PSF demandée). Corrigé : `zernike_basis_orthogonal.resolve_zernike_basis`
  auto-détecte désormais `aperture_px` depuis le `mask` fourni quand il
  n'est pas donné explicitement (nouvelle fonction
  `pupil.detect_aperture_px`, bounding box des pixels non nuls) ; aucun
  changement de comportement quand le masque remplit déjà le cadre (cas
  par défaut). Nouveaux tests, dont la reproduction exacte du rapport
  (`tests/test_aperture_px_autodetect.py`).
- Documentation (§12.3, §12.7 nouveau) et `examples/psf2zernike_demo.ipynb` :
  exemple de bout en bout avec un masque à résolution native différente
  de la PSF (comme un fichier `mask.tif` réel, avec marge de zéros
  autour de l'ouverture), illustrant (a) un piège distinct — hors bug de
  bibliothèque — consistant à construire soi-même une PSF de « vérité »
  via `build_zernike_basis` en oubliant de propager le `N_pup_target`
  renvoyé par `load_and_prepare_mask` (coût d'optimisation faible, mais
  coefficients de Zernike récupérés totalement faux, §12.7), et (b) une
  vérification visuelle que `_crop_to_aperture` a bien lieu avant le
  changement d'échelle (masque correctement reconstruit comparé à un
  resize naïf sans recadrage préalable, §12.3).
- **FTM : coupes H/V, axe de fréquence réduite, fréquence de coupure**
  (§9.5) : `diagnostics.mtf_frequency_axis`/`mtf_cutoff_frequency`/
  `mtf_cuts` (fonctions numériques pures — restées ici plutôt que dans
  `eoqual-core`, dont la règle de gouvernance n'admet une brique qu'à
  partir de deux modules consommateurs) et `io_utils.plot_mtf_cuts`
  (figure Plotly : coupes H/V, référence optionnelle, repère `fc`).
  Démontré dans `examples/psf2zernike_demo.ipynb`.
- **Fonctions graphiques génériques extraites vers
  [`eoqual-viz`](https://github.com/EOQual/eoqual-viz)**, nouveau paquet
  public : `io_utils.plot_array_plotly` (renommé `eoqual.viz.plot_array`)
  et la mécanique Plotly de `io_utils.plot_mtf_cuts` (généralisée en
  `eoqual.viz.plot_curve_grid`, comparaison de courbes/coupes avec
  repères verticaux, pas spécifique à une FTM). `io_utils.plot_mtf_cuts`
  reste ici comme fine enveloppe spécifique (assemble les coupes/fc puis
  appelle `eoqual.viz.plot_curve_grid`) ; `visualisation_diagnostics`
  appelle désormais `eoqual.viz.plot_array`. Nouvelle dépendance
  `eoqual-viz` (voir §6.1 de `REFERENCE_TECHNIQUE.md`). `examples/*.ipynb`
  mis à jour et ré-exécutés (aucune erreur).

- **Rupture d'unité (§9.5)** : l'axe de fréquence réduite de la FTM
  (`mtf_frequency_axis`, `mtf_cutoff_frequency`, `io_utils.plot_mtf_cuts`)
  est maintenant exprimé en fraction de `fe` (fréquence d'échantillonnage)
  plutôt que de `fe/2` (Nyquist) — `fe/2` vaut donc `0,5` au lieu de
  `1,0`. `plot_mtf_cuts` trace désormais systématiquement un repère `fe/2`
  en bleu, en plus de `fc` en rouge si fourni (nécessite `eoqual-viz` ≥
  la version avec `vlines=(valeur, étiquette, couleur)`, rupture d'API
  mineure côté `eoqual-viz` elle-même).
- `diagnostics.mtf_value_at`/`mtf_integral` : valeur interpolée de la FTM
  à une fréquence donnée, intégrale entre `0` et une fréquence max (règle
  des trapèzes) — pour résumer une coupe de FTM en quelques chiffres.
- `diagnostics.encircled_energy`/`fwhm` : délèguent respectivement à
  `eoqual.core.psf.encircled_energy` et `eoqual.core.edge.fwhm` (déjà
  présentes dans `eoqual-core`, utilisées par d'autres modules — aucune
  nouvelle brique à y faire accepter). `fwhm`, conçue pour une LSF, est
  purement générique et s'applique directement à une coupe de PSF.
- `examples/psf2zernike_demo.ipynb` : les trois exemples de la section
  « Fonctionnalités avancées » (suréchantillonnage en mémoire, lecture
  d'une PSF, tirage aléatoire de Zernike) affichent désormais, en plus de
  la FTM : un tableau `pandas` des valeurs de FTM à `fe/4`/`fe/2` et de
  son intégrale `[0, fe/2]`, et un tableau `pandas` de métriques PSF
  (Strehl, FWHM H/V, énergie encerclée à 2 et 5 px) comparant la PSF
  simulée/mesurée à la PSF récupérée/reconstruite.
- **« Interface CLI unifiée » et « Organisation du code » déplacées du
  notebook vers `README.md`** (nouvelles sections, avec ancre dans le
  Sommaire) : plus à leur place dans une référence stable (CLI,
  arborescence du paquet) que dans un notebook de démonstration. Le
  Sommaire de `examples/psf2zernike_demo.ipynb` renvoie désormais vers
  `README.md` pour ces deux sujets.

## Évolutions futures

Voir `REFERENCE_TECHNIQUE.md` §11 (modèle système complet : détecteur,
WFE haut ordre, filée, polychromatisme) et §6.5 (ajouter une méthode).
La comparaison de performance `torch`/autograd vs. `scipy` (jacobien
analytique ou backend autodiff optionnel) reste documentée en §6.5bis
comme décision séparée, à ne prendre que si un besoin réel s'installe —
voir les déclencheurs concrets qui y sont listés.
