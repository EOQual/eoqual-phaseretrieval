# Méthodes retenues et implémentation

Les références numérotées [n] renvoient à la [bibliographie](bibliographie.md).

## 4. Méthodes retenues dans ce projet

Sur les sept familles ci-dessus, trois sont implémentées — les deux
premières lignes du panorama (Error‑Reduction/HIO pur) n'ont pas été
retenues séparément : notre implémentation GS correspond à l'algorithme
originel de Gerchberg & Saxton (le paramètre `beta` de
[`RetrievalConfig`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/backends/gerchberg_saxton.py) est réservé à une future extension
HIO). La pondération de bruit Poisson et les pertes robustes ne sont pas
des méthodes à part entière mais des options de la diversité de phase.

| `method=` (code) | Technique | Référence principale | Module |
|---|---|---|---|
| `"gerchberg_saxton"` | Gerchberg‑Saxton | [1] | [`backends/gerchberg_saxton.py`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/backends/gerchberg_saxton.py) |
| `"optimization"` | Ajustement direct par moindres carrés non linéaires | [6, 7] | [`backends/optimization.py`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/backends/optimization.py) |
| `"phase_diversity"` | Diversité de phase modale, pondération Poisson optionnelle, perte robuste optionnelle | [8, 9, 10, 11] | [`backends/phase_diversity.py`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/backends/phase_diversity.py) |

La décomposition finale en coefficients de Zernike (ordre de Noll) est
commune aux trois méthodes : linéaire par moindres carrés pour GS (la phase
est d'abord récupérée pixel à pixel, puis projetée — voir
[`zernike_fit.py`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/zernike_fit.py)), directe pour les deux autres (les
coefficients *sont* les paramètres de l'optimisation).

## 6. Implémentation technique

### 6.1 Architecture des modules

```
src/eoqual/phaseretrieval/
    io_utils.py                    # chargement image/masque, enveloppes Plotly spécifiques
    optics.py                      # base de Zernike, modèle direct phase -> PSF
    zernike_fit.py                 # décomposition d'une carte de phase (Noll)
    pipeline.py                    # psf_to_wfe(method=...) + CLI unifiée
    diagnostics.py                 # Strehl, WFE (Maréchal), MTF, ellipticité (§9)
    zernike_basis_orthogonal.py    # base Zernike orthogonalisée (pupille obstruée, §10)
    pupil.py                       # pupilles disque/annulaire/araignées (test, §10)
    config.py                      # registre METHODS_CONFIGS
    runner.py                      # list_methods()
    backends/
        gerchberg_saxton.py        # Gerchberg-Saxton                  (1 PSF)
        optimization.py            # optimisation directe              (1 PSF)
        phase_diversity.py         # diversité de phase modale         (2 PSF)
    simulation/                    # génération de cas de test (vérité connue) — voir §6.7
        psf_simulator.py
        phase_diversity_simulator.py
        noise.py
        star_profiles.py           # simulation étoile empirique (Gauss/Moffat/Airy, Bayer)
examples/
    psf2zernike_demo.ipynb         # démonstration + validation
    simu_psf_mask.ipynb            # simulateur PSF (monochrome + Bayer RGuGlB, Zernike+FFT)
```

`optics.py` et `zernike_fit.py` sont les briques partagées : les deux
méthodes par optimisation (`backends/optimization.py`,
`backends/phase_diversity.py`) s'appuient sur `optics.phase_from_coefficients`
et `optics.psf_from_phase` pour évaluer leur modèle direct à chaque
itération du solveur ; `backends/gerchberg_saxton.py` n'en a pas besoin
(il travaille directement sur la phase pixel à pixel) mais passe par
`zernike_fit.py` *après coup* pour produire des coefficients comparables
aux deux autres
méthodes.

**Fonctions graphiques génériques (heatmap 2D, comparaison de
courbes/coupes avec repères verticaux) externalisées dans
[`eoqual-viz`](https://github.com/EOQual/eoqual-viz)** — elles n'avaient
rien de spécifique à la récupération de phase (ex-`io_utils.plot_array_plotly`,
ex-`io_utils.plot_mtf_cuts`). `io_utils.py` ne garde que les enveloppes
spécifiques à ce projet (`plot_mtf_cuts`, assemblage des coupes FTM +
fréquence de coupure avant de passer les tableaux à
`eoqual.viz.plot_curve_grid`) et `diagnostics.py` les fonctions
purement numériques (`mtf_frequency_axis`, `mtf_cutoff_frequency`,
`mtf_cuts`, `mtf_value_at`, `mtf_integral`, [§9.5](qualite-optique.md#95-ftm--coupes-hv-axe-de-fréquence-réduite-fréquence-de-coupure)) — ces dernières sont
restées ici plutôt que dans `eoqual-core` : sa propre règle de
gouvernance n'y admet une brique que si **au moins deux modules**
l'utilisent (voir `eoqual-core/README.md` §Règles), ce qui n'est pas
encore le cas. `encircled_energy` et `fwhm` ([§9.5](qualite-optique.md#95-ftm--coupes-hv-axe-de-fréquence-réduite-fréquence-de-coupure)), à l'inverse,
délèguent bien à `eoqual-core` (`psf.encircled_energy`, `edge.fwhm`) :
ces fonctions-là y existaient déjà, utilisées par d'autres modules —
aucune nouvelle brique à y faire accepter.

### 6.2 Point d'entrée unifié

```python
coeffs_nm, phase_fit_nm, diagnostics = psf_to_wfe(
    psf_path, mask_path, wavelength, focal_length, pupil_diameter, pixel_pitch,
    method="gerchberg_saxton" | "optimization" | "phase_diversity",
    oversampling=1,  # §12 : >1 pour une PSF suréchantillonnée (super-résolution)
    zernike_order=36,
    # ... paramètres spécifiques à chaque méthode (voir pipeline.py) ...
)
```

`psf_to_wfe` ([`pipeline.py`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/pipeline.py)) charge la PSF et le masque une
seule fois (`io_utils.load_image`, `io_utils.load_and_prepare_mask`), puis
dispatch vers le module correspondant à `method`. `diversity_path` et
`defocus` ne sont utilisés (et requis) que par `"phase_diversity"` ; ils
sont ignorés — avec un avertissement `loguru` — pour les deux autres
méthodes. C'est le choix architectural documenté en tête de ce fichier :
un point d'entrée commun, mais aucune logique de méthode mélangée entre les
modules.

### 6.3 Détail par méthode

| | GS | Optimisation | Diversité de phase |
|---|---|---|---|
| Config | `RetrievalConfig` | `OpticalConfig` | `PhaseDiversityConfig` |
| Variables ajustées | phase $\phi$ (par pixel) | coefficients $c_4 \dots c_N$ | coefficients $c_4 \dots c_N$ |
| Fonction de coût | contraintes d'amplitude alternées (pas de coût explicite) | `backends.optimization.residuals` | `backends.phase_diversity.residuals` (deux canaux concaténés) |
| Solveur | boucle FFT/IFFT (`max_iter` itérations) | `scipy.optimize.least_squares(method="trf")` | idem, + `loss=` robuste et pondération Poisson optionnelle |
| Étape Zernike | `zernike_fit.fit_zernike` (a posteriori, moindres carrés linéaires) | intégrée à l'optimisation | intégrée à l'optimisation |
| Diagnostics | rang, conditionnement, RMS résiduel (nm) | coût, `nfev`, RMSE PSF | coût, `nfev`, RMSE PSF (focus + défocus) |

`sign_prior=(reference_index, expected_sign)` — ex. `(4, +1.0)` — résout
l'ambiguïté de signe des modes pairs pour `"gerchberg_saxton"` et
`"optimization"` (voir [§3.7](principes.md#37-lever-lambiguïté-de-signe-par-un-a-priori) pour la théorie et ses limites) ; ignoré (avec
avertissement) pour `"phase_diversity"`, déjà non ambiguë.

`roi_radius` (diversité de phase uniquement) restreint le résidu de
`least_squares` à une fenêtre carrée centrée de demi-largeur `roi_radius`
px — avec la valeur par défaut (32) sur nos images 256×256
(`examples/data/psf_simu.tiff`), c'est une fenêtre de 64×64 px, soit environ 6 % des
pixels. Ce n'est pas cosmétique : la quasi-totalité de l'énergie utile
d'une PSF (tache d'Airy + anneaux) est concentrée près du centre, alors que
les pixels de bord ne contiennent presque que du bruit de fond (voire des
artefacts — repliement FFT, autre source dans le champ, pixel chaud) ; les
inclure dans le résidu dilue le signal utile sans rien apporter, et
ralentit le solveur pour rien. Le compromis à régler : trop petit, on coupe
une partie du signal qui contraint les modes de Zernike élevés (qui
étalent la tache plus loin du centre) ; trop grand, on réintroduit du bruit
de fond inutile. `roi_radius=32` convient à nos images 256×256 avec
`zernike_order≈15`, mais est à ajuster si la taille réelle de la tache
change (autre `zernike_order`, niveau d'aberration, échantillonnage
pupille/détecteur différent).

`core_weight_radius` / `l2_reg` (méthode `"optimization"` uniquement,
`OpticalConfig`, désactivés par défaut = comportement inchangé) apportent à
l'optimisation directe le même principe que `roi_radius` sans en payer le
coût — pas de fenêtre dure qui coupe le signal, mais une pondération douce
(taper gaussien) qui priorise le cœur à haut SNR sur les ailes bruitées
(`core_weight_radius`), et une régularisation de Tikhonov qui stabilise le
fit quand il est sous-contraint (`l2_reg`, ordre élevé + PSF peu
lumineuse/bruitée).

### 6.4 Validation

La validation numérique utilise le module de simulation (§6.7,
`eoqual.phaseretrieval.simulation`), qui génère une vérité de Zernike
connue — `tests/test_simulation_roundtrip.py` est la suite
de référence (simulation → récupération de phase → comparaison à la vérité
terrain), rejouée à chaque exécution des tests plutôt que figée dans des
fixtures. Les résultats de fond sont résumés en [§2](principes.md#2-limites-dune-récupération-à-partir-dune-seule-psf) ; ils confirment sur
des données contrôlées les ambiguïtés théoriques du [§2](principes.md#2-limites-dune-récupération-à-partir-dune-seule-psf) et [§3](principes.md#3-panorama-des-méthodes-de-récupération-de-phase), et donnent
une reconstruction quasi exacte (RMS ≈ 0 nm) pour la diversité de phase
en l'absence de bruit — voir aussi `examples/psf2zernike_demo.ipynb` pour
une démonstration interactive équivalente, à base des données
`examples/data/*_simu.tiff`.

### 6.5 Étendre le code (ajouter une 4ᵉ méthode)

1. Créer `backends/<nom>.py` avec sa propre dataclass de configuration et
   sa fonction `fit_...(psf(s), mask, cfg) -> (coeffs_nm, diagnostics)`,
   en réutilisant `optics.py` si la méthode passe par un modèle direct
   PSF(coeffs).
2. Ajouter la valeur correspondante à `pipeline.METHODS` (et une entrée à
   `config.METHODS_CONFIGS`) et une branche de dispatch dans `psf_to_wfe`.
3. Ajouter les options CLI correspondantes dans `build_arg_parser`.
4. Ajouter un test dans `tests/test_simulation_roundtrip.py` (simulation
   → nouvelle méthode → comparaison à `coeffs_nm` de vérité) et, si utile,
   une section de démonstration dans `examples/psf2zernike_demo.ipynb`.

### 6.5bis A priori individuels par mode (`zernike_priors` / `zernike_bounds`)

À la suite d'une revue de scripts externes (non intégrés au projet)
implémentant des a priori individuels par mode via PyTorch/`torchmin`
(minimisation à contraintes de boîte ou pénalité L2 autour d'une valeur
attendue par mode), deux voies étaient envisageables : (a) une **option**
sur les méthodes existantes, ou (b) un **4ᵉ backend**. Choix retenu et
implémenté : **(a), une option** (module
[`zernike_priors.py`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/zernike_priors.py), branché
sur `OpticalConfig`/`PhaseDiversityConfig`/`psf_to_wfe`) — pour deux
raisons indépendantes :

- **Le modèle direct ne change pas.** Un a priori individuel par mode
  n'est qu'un terme de pénalité $\sum_k w_k\,(c_k-\bar c_k)^2$ ajouté au
  résidu (ou une borne par coefficient), au‑dessus du même
  $\text{PSF}_{\text{modèle}}(\mathbf c)$ déjà utilisé par `"optimization"`
  et `"phase_diversity"`. Ce n'est pas un changement de méthode au sens de
  §6.5 (qui suppose un modèle direct ou une stratégie de résolution
  différente) : c'est un terme de coût supplémentaire, paramétrable par
  une nouvelle entrée de configuration (ex. `zernike_priors:
  Dict[int, Tuple[float, float]]`, mode Noll → (valeur attendue nm,
  écart‑type nm)) sur `OpticalConfig`/`PhaseDiversityConfig`, dans la même
  veine que `sign_prior` ([§3.7](principes.md#37-lever-lambiguïté-de-signe-par-un-a-priori)) déjà présent. Une pénalité L2 s'exprime
  nativement comme des lignes supplémentaires du vecteur de résidus de
  `scipy.optimize.least_squares` — aucune dépendance à un cadre
  différentiable n'est nécessaire pour cette seule extension.
- **Le gain de performance de l'implémentation PyTorch de référence n'est
  pas lié à l'a priori, mais au jacobien.** Comparaison chronométrée
  (`fit_zernike_from_psf` vs. un portage direct de `psf_from_phase` en
  PyTorch + `torch.optim.LBFGS`, même pupille, même bruit, CPU) :

  | Ordre de Zernike | scipy (`least_squares`, jacobien par différences finies) | PyTorch (rétropropagation) |
  |---|---|---|
  | 12 | 0,09 s | 0,05 s |
  | 36 | 3,6 s | 0,05 s |
  | 66 | 17–42 s | 0,11 s |

  L'écart croît avec l'ordre (×70 à 36, ×270 à 66) pour une raison
  structurelle : `least_squares` évalue le modèle direct par différences
  finies pour construire son jacobien (coût par itération proportionnel au
  nombre de coefficients libres), alors que la rétropropagation calcule un
  gradient exact en une seule passe, à coût quasi indépendant du nombre de
  paramètres. **Ce n'est donc pas une différence négligeable propre à ce
  projet** (ordre ≤36 en pratique courante) : dès l'ordre 36 l'écart est
  déjà d'un ordre de grandeur. Mais ce gain vient du calcul du gradient,
  pas de la formulation bayésienne de l'a priori — ajouter un a priori
  individuel ne le crée ni ne le supprime. Un futur passage à un jacobien
  analytique (le modèle `psf_from_phase` est dérivable en forme close) ou
  à l'autodiff (JAX/PyTorch) resterait donc une décision **séparée**, à
  ne prendre que si un besoin réel s'installe (ordres bien supérieurs à
  36, ajustements répétés en volume, ou GPU) — pas un prérequis pour
  exploiter l'idée d'a priori individuels.

**La logique, en détail (pourquoi ce n'est pas une nouvelle "méthode").**
Une estimation MAP (maximum a posteriori) maximise
$p(\mathbf c\mid\text{mesure})\propto p(\text{mesure}\mid\mathbf c)\,p(\mathbf c)$,
donc minimise $-\log p(\text{mesure}\mid\mathbf c)-\log p(\mathbf c)$. Avec un
bruit gaussien, $-\log p(\text{mesure}\mid\mathbf c)=\lVert\text{modèle}-\text{mesure}\rVert^2$
— le moindres‑carrés déjà utilisé partout dans ce projet. Avec un a priori
**gaussien** par mode $c_k\sim\mathcal N(\bar c_k,\sigma_k^2)$,
$-\log p(\mathbf c)=\sum_k(c_k-\bar c_k)^2/\sigma_k^2$ — une **pénalité L2**.
Avec un a priori **uniforme** sur $[\text{low}_k,\text{high}_k]$ (nul en
dehors), $-\log p(\mathbf c)$ vaut $0$ à l'intérieur et $+\infty$ en dehors
— une **contrainte de boîte**. Un MAP gaussien et un moindres‑carrés
pénalisé L2 ne sont donc pas deux méthodes différentes : c'est **la même
formule**, écrite dans deux vocabulaires (statistique / numérique) ; de
même pour un MAP à prior uniforme et une borne dure.

`OpticalConfig.l2_reg` (§6.3 plus haut) est déjà un **cas particulier** de
ce mécanisme : un a priori gaussien centré sur **zéro**, avec le **même**
poids pour tous les modes. `zernike_priors`/`zernike_bounds` généralisent
ce cas particulier à un a priori **individuel** (moyenne et poids propres
à chaque mode), sous forme de pénalité ou de borne dure. Concrètement,
`zernike_priors`/`zernike_bounds` ajoutent des lignes supplémentaires au
vecteur de résidus de `scipy.optimize.least_squares` (même mécanisme que
`l2_reg`) et, pour les bornes, un `bounds=` natif à `least_squares`
(inutilisé jusqu'ici) — voir
[`zernike_priors.build_prior_penalty`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/zernike_priors.py)/
[`build_prior_bounds`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/zernike_priors.py) pour le
détail des conversions nm ↔ rad et
[`feasible_start`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/zernike_priors.py) pour un
piège évité (démarrer *exactement* sur une borne finie, ex.
`(0.0, 400.0)`, dégénère la mise à l'échelle interne de `trf` et peut faire
déclarer au solveur une convergence immédiate et fictive : le point de
départ est donc légèrement écarté d'une borne sur laquelle il
tomberait).

**Exemple d'usage** — un a priori individuel de laboratoire (mesure
interférométrique préalable, incertaine) sur le sphérique ($Z_{11}$), et
une borne physique large sur le défocus ($Z_4$, jamais négatif dans ce
montage) :

```python
from eoqual.phaseretrieval import OpticalConfig, fit_zernike_from_psf

cfg = OpticalConfig(
    wavelength=550e-9,
    zernike_order=36,
    zernike_priors={11: (8.0, 3.0)},   # Z11 = 8 nm ± 3 nm (a priori labo)
    zernike_bounds={4: (0.0, 500.0)},  # Z4 physiquement dans [0, 500] nm
)
coeffs_nm, diagnostics = fit_zernike_from_psf(psf, mask, cfg)
```

Même API sur `PhaseDiversityConfig`/`psf_to_wfe(zernike_priors=...,
zernike_bounds=...)`. Un mode absent des deux dictionnaires reste libre,
sans pénalité ; c'est le comportement par défaut (`None`).
Voir `examples/psf2zernike_demo.ipynb`, section « Fonctionnalités
avancées », pour une démonstration exécutable (y compris le cas où un a
priori fort désambiguïse/débiaise un ajustement autrement mal contraint).

**Mise en garde — d'où doit venir un a priori, dans ce contexte.** Rien
n'empêche techniquement d'utiliser `zernike_priors` pour encoder une
hypothèse générique du type « les modes de haut ordre radial ont une
amplitude décroissante » (courante en optique adaptative sol, où elle vient
de la statistique de turbulence de Kolmogorov — la variance du coefficient
de Noll $j$ décroît comme une loi de puissance de l'ordre radial, résultat
de Noll 1976). **Cette hypothèse n'est en général pas justifiée pour la
WFE propre d'un instrument d'observation de la Terre** ([§8](observation-terre.md#8-cas-dusage--observation-de-la-terre)) : contrairement
à la turbulence atmosphérique, la WFE statique d'un instrument spatial
(fabrication, assemblage, thermo‑élastique en orbite) n'a aucune raison de
suivre un spectre de Kolmogorov — elle peut être dominée par quelques modes
bas ordre (défauts d'alignement), ou porter une contribution haut ordre
significative et localisée (print‑through d'un miroir allégé, réplique de
polissage) sans décroissance monotone. Un a priori raisonnable dans ce
contexte doit venir d'une source **spécifique à l'instrument** — modèle
optique as‑built, budget d'erreur, calibration sol/vol antérieure ([§8.2](observation-terre.md#82-recommandation)) —
et non d'une décroissance générique empruntée à la littérature de
turbulence atmosphérique ; à défaut d'une telle source, préférer
`l2_reg` (a priori faible et neutre, sans favoriser arbitrairement une
forme de spectre) à un `zernike_priors` inventé.

**Déclencheurs concrets** (pour ne plus avoir à rouvrir la question à
chaque fois) — passer à un jacobien analytique ou à l'autodiff dès que
**l'un** de ces critères devient vrai en usage réel (pas en exploration) :

- `zernike_order` routinier > ~40 (le point où l'écart chronométré ci‑dessus
  dépasse ~×70, tableau ci‑dessus) ;
- volume : ajustements répétés en série (monitoring image par image,
  campagne de calibration multi‑poses) où le temps cumulé devient un facteur
  opérationnel, même à ordre modéré ;
- un GPU est déjà disponible dans l'environnement cible (le gain
  d'autodiff se cumule alors avec le parallélisme matériel).

**Ce qu'il faudrait concrètement**, par ordre de coût croissant :

1. **Jacobien analytique dans `scipy`, sans nouvelle dépendance** — à
   tenter en premier. `psf_from_phase` est une fonction analytique fermée
   des coefficients (FFT d'une exponentielle complexe) ; le jacobien de
   $\sqrt{\text{PSF}_{\text{modèle}}}$ par rapport à $\mathbf c$ se dérive
   en forme close et se passe à `least_squares(..., jac=...)` — supprime
   l'évaluation par différences finies sans quitter numpy/scipy. Récupère
   une bonne partie du gain observé sans ajouter de dépendance ; à évaluer
   avant l'option suivante.
2. **Backend autodiff optionnel** (PyTorch ou JAX) — seulement si (1) ne
   suffit pas. Éléments nécessaires : un port différentiable de
   `phase_from_coefficients`/`psf_from_phase` (FFT complexe, déjà supportée
   nativement par l'autodiff des deux bibliothèques) ; un nouveau backend
   `backends/<nom>.py` suivant le patron de §6.5 (même signature
   `fit_...(psf, mask, cfg) -> (coeffs_nm, diagnostics)`, pour rester
   interchangeable via `psf_to_wfe(method=...)`) ; la dépendance ajoutée en
   **extra optionnel** de `pyproject.toml` (ex. `poetry install -E torch`),
   pas en dépendance de base, pour ne pas alourdir l'installation par
   défaut (§6.5bis, comparatif de dépendances) ; un test de non‑régression
   comparant les coefficients récupérés à ceux de `"optimization"` sur un
   même cas simulé (`tests/test_simulation_roundtrip.py`), pour garantir
   l'équivalence numérique avant d'en faire une alternative de confiance ;
   et un test de performance chronométré (le tableau ci‑dessus, codifié en
   test plutôt qu'en observation ponctuelle) pour objectiver le
   déclencheur plus haut au lieu de le réargumenter qualitativement.

### 6.6 Interface en ligne de commande

```bash
python -m eoqual.phaseretrieval.pipeline \
    --method {gerchberg_saxton,optimization,phase_diversity} \
    --psf PSF --mask MASK --output OUTPUT [--diversity DIVERSITY --defocus DEFOCUS] ...
```

Voir `python -m eoqual.phaseretrieval.pipeline --help` (ou la section CLI
de `examples/psf2zernike_demo.ipynb`) pour la liste complète des options,
groupées par méthode.

### 6.7 Module de simulation (données à vérité connue)

`eoqual.phaseretrieval.simulation` est volontairement **séparé** des
backends de reconstruction de phase : il ne fait que fabriquer des cas de
test, avec un défocus/une aberration connus par construction, pas les
récupérer. Deux générateurs :

- `simulate_psf_from_zernike(coeffs_nm, ...)` — une PSF unique, pour les
  méthodes `"gerchberg_saxton"`/`"optimization"`. Réutilise directement
  `optics.phase_from_coefficients`/`optics.psf_from_phase` (le même
  modèle direct que les backends par optimisation), donc la PSF simulée
  est rigoureusement cohérente avec ce que ces méthodes essaient de
  reconstruire.
- `simulate_phase_diversity_pair(coeffs_nm, cfg)` — une paire
  focus/défocus, pour `"phase_diversity"`. Construit la phase de
  diversité en réutilisant directement
  `backends.phase_diversity.diversity_phase` (même convention d'unité et
  de signe que `PhaseDiversityConfig.defocus`) : le `defocus` renvoyé par
  le simulateur se réinjecte tel quel dans `fit_phase_diversity`, sans
  conversion.

`coeffs_nm` s'exprime en dictionnaire `{indice de Noll: valeur en nm
WFE}` (ex. `{4: 55.0, 5: 27.5}`) — plus lisible qu'un tableau dense pour
un cas de test qui n'active que quelques modes ; converti en interne par
`coeffs_to_array`.

**Bruit détecteur** (`simulation.noise.DetectorNoiseConfig`, optionnel,
`None` par défaut = PSF idéale) : Poisson (bruit de photon, piloté par
`photon_count`, le flux total intégré sur la tache) + gaussien additif
(bruit de lecture, `read_noise_e`) + fond constant optionnel
(`background_e`). Pour une paire focus/défocus, les deux images sont
tirées avec des réalisations de bruit indépendantes mais le même budget
de photons (`seed_offset` différent, mêmes paramètres) — comme deux poses
réelles consécutives, pas une copie du même bruit. C'est le même régime
de bruit que celui pour lequel `poisson_weighting=True`
(`PhaseDiversityConfig`, §6.3) a été conçu — voir [§9](qualite-optique.md#9-repères-de-qualité-optique--strehl-wfe-diagnostics-de-psf) pour la discussion
de sa pertinence en télédétection.
