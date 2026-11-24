# Repères de qualité optique

Les références numérotées [n] renvoient à la [bibliographie](bibliographie.md).

## 9. Repères de qualité optique : Strehl, WFE, diagnostics de PSF

Complément indépendant du fit Zernike : le module [`diagnostics.py`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/diagnostics.py)
fournit des métriques de qualité optique calculables **avant** ou **sans**
reconstruction de phase — utiles pour un premier diagnostic rapide, ou pour
comparer plusieurs méthodes entre elles sur un pied d'égalité.

### 9.1 Rapport de Strehl

$$
\text{Strehl} = \frac{\max(\text{PSF}_{\text{mesurée}})}{\max(\text{PSF}_{\text{diffraction-limitée}})}
$$

**Conditions strictes de validité** (sinon le Strehl calculé est faux) :
les deux PSF doivent être normalisées identiquement (`∫PSF = 1`), avoir le
même échantillonnage (`pixel_scale`) et être dans la même bande spectrale ;
la référence diffraction-limitée doit correspondre à la même ouverture et
la même longueur d'onde effective. `diagnostics.strehl_ratio` **avertit**
(`logger.warning`) si le résultat dépasse 1 — signe quasi certain que l'une
de ces conditions est violée plutôt qu'une optique meilleure que parfaite.

Deux façons d'obtenir la PSF de référence :
- **Absolue** : `diagnostics.airy_psf(...)`, la tache d'Airy théorique
  ($D$, $\lambda$, `pixel_scale` réels de l'instrument) — la définition
  canonique, mais qui suppose un `pixel_scale` réellement calibré
  (`p/f`, voir `diagnostics.pixel_scale_from_pixel_pitch`). **Notre propre
  simulateur** (`examples/simu_psf_mask.ipynb`, `optics.psf_from_phase`) fait une
  simple FFT du masque sans modèle physique focale/détecteur explicite :
  son `pixel_scale` absolu n'est donc pas rigoureusement défini, et
  comparer sa sortie à une tache d'Airy calculée à partir d'un
  `pixel_scale` deviné produira typiquement ce garde-fou (`Strehl > 1`) —
  ce n'est pas un bug, c'est le garde-fou qui fonctionne.
- **Relative** (recommandée en interne, sur données simulées) : utiliser
  la meilleure PSF disponible comme référence *relative* plutôt qu'une
  tache d'Airy théorique — par exemple `psf_simu.tiff` (la mesure) comme
  référence pour la PSF reconstruite à partir du fit Zernike. Le Strehl
  n'est alors plus absolu mais reste parfaitement comparable entre
  méthodes. Vérifié sur nos données (`method="optimization"`,
  `zernike_order=15`) : Strehl relatif fit-vs-mesuré ≈ 1.0003 — cohérent
  avec le résidu quasi nul déjà observé en [§6.4](implementation.md#64-validation).

### 9.2 WFE RMS depuis le Strehl (approximation de Maréchal)

$$
S \approx \exp\!\left[-\left(\frac{2\pi\sigma}{\lambda}\right)^2\right]
\quad\Longleftrightarrow\quad
\sigma = \frac{\lambda}{2\pi}\sqrt{-\ln S}
$$

(`diagnostics.wfe_rms_from_strehl`) — déjà mentionnée en [§1](principes.md#1-pourquoi-le-phaseretrieval-) [4], valable
pour $S \gtrsim 0.1$–0.3. Comparer ce WFE global à celui obtenu directement
du fit Zernike (`diagnostics["residual_rms_nm"]`/`"fit_rms_nm"` de
`zernike_fit.fit_zernike`) est un bon test de cohérence croisée.

### 9.3 Tables de repère

| Strehl | Qualité optique (critère de Maréchal) |
|---|---|
| > 0,8 | quasi diffraction-limitée |
| 0,6 – 0,8 | très bonne |
| 0,4 – 0,6 | correcte |
| 0,2 – 0,4 | dégradée |
| < 0,2 | forte aberration |

| WFE RMS (nm @ 550 nm) | Qualité |
|---|---|
| < 40 nm | quasi parfaite |
| 40–80 nm | excellente |
| 80–120 nm | bonne |
| 120–200 nm | moyenne |
| > 200 nm | très dégradée |

### 9.4 Diagnostics complémentaires (sans fit Zernike)

`diagnostics.py` fournit aussi : `radial_profile` (profil radial moyenné
azimutalement), `compute_mtf` (MTF par simple FFT de la PSF) et
`core_ellipticity` (ellipticité du cœur par moments quadratiques pondérés
— signature du coma/astigmatisme). `compute_metrics` + `diagnose_aberration`
combinent ces éléments en un diagnostic qualitatif rapide (« défocus
dominant », « coma/astigmatisme dominant », …) — **heuristique**, à
confirmer systématiquement par un vrai fit Zernike avant toute conclusion.

### 9.5 FTM : coupes H/V, axe de fréquence réduite, fréquence de coupure

Plusieurs fonctions complètent `compute_mtf` pour une lecture physique de
la FTM plutôt qu'un simple tableau 2D. **Convention d'axe : fraction de
la fréquence d'échantillonnage `fe` du détecteur, pas de Nyquist `fe/2`**
— `fe/2` vaut donc `0,5` dans cette unité, pas `1,0` (choisi pour que
l'axe des fréquences coïncide avec la convention `np.fft.fftfreq`, sans
reconversion mentale).

- `mtf_frequency_axis(npix)` : axe de fréquence réduite (fraction de
  `fe`) — `0` au centre, `±0,5` exactement au bord du tableau (par
  construction du modèle PSF = `|FFT(pupille)|²`, le bord correspond
  toujours à `fe/2`, quelle que soit l'ouverture — mais ``0,5`` dans
  cette unité, pas ``1,0``).
- `mtf_cutoff_frequency(npix, aperture_px=None)` : fréquence de coupure
  optique `fc` (diffraction), dans la même unité — `fc = aperture_px /
  npix` (voir §12.5 pour `aperture_px`). Avec `aperture_px=None`
  (ouverture pleine cadre) : `fc = 1,0`, *hors* de l'intervalle `[0, 0,5]`
  représentable par une FFT — ce modèle brut, sans marge de
  suréchantillonnage, échantillonne en-deçà du critère de Nyquist pour
  la diffraction (`Q = 1`). Avec `aperture_px = npix/2` (voir
  [§12.1](resolutions.md#121-le-principe--pixel_pitch--oversampling-pas-detector_size)/[§12.5](resolutions.md#125-la-base-de-zernike-doit-partager-le-même-aperture_px-que-le-masque)) : `fc = 0,5`, l'échantillonnage de Nyquist exact
  (`Q = 2`) — `fc` coïncide alors pile avec `fe/2`.
- `mtf_cuts(mtf)` : coupes horizontale et verticale (lignes/colonnes
  centrales), même convention de centrage que les deux fonctions
  ci-dessus. Fonctionne sur n'importe quel tableau 2D (pas spécifique à
  une FTM) — s'applique tout aussi bien à une coupe de PSF.
- `mtf_value_at(freq, cut, frequency)` : valeur interpolée (linéaire) de
  la FTM à une fréquence réduite donnée — ex. `mtf_value_at(freq, cut,
  0.25)` pour la valeur à `fe/4`.
- `mtf_integral(freq, cut, f_max=0.5)` : intégrale (trapèzes) de la FTM
  entre `0` et `f_max` — résume "combien de contraste la FTM conserve en
  moyenne" sur une bande (`[0, fe/2]` par défaut, tout le domaine
  représentable par le détecteur).

`io_utils.plot_mtf_cuts(mtf_estimated, *, mtf_reference=None, fc=None,
title=...)` assemble les trois premières en une figure Plotly (deux
sous-graphes, fréquences positives seulement — `MTF(-f) == MTF(f)` pour
une PSF réelle) : la référence, quand fournie, se superpose à l'estimée ;
deux repères verticaux toujours tracés — `fe/2` en **bleu** (fixe, à
`0,5`), et `fc` en **rouge** si fourni. Démontré dans
`examples/psf2zernike_demo.ipynb`, avec le tableau `mtf_value_at`/
`mtf_integral` à l'appui (sections « Suréchantillonnage en mémoire » —
cas `fc = 0,5`, confondu avec `fe/2` — et « lire une PSF et la décomposer »
— cas `fc = 1,0`, bien au-delà). La mécanique Plotly elle-même
(`eoqual.viz.plot_curve_grid`) est générique — voir la note du [§6.1](implementation.md#61-architecture-des-modules).

**Métriques PSF complémentaires** (Strehl, FWHM, énergie encerclée),
toutes deux simulée/vérité et récupérée/restituée, pour comparaison
directe — `diagnostics.encircled_energy` (délègue à
`eoqual.core.psf.encircled_energy`) et `diagnostics.fwhm` (délègue à
`eoqual.core.edge.fwhm`, conçue pour une LSF mais purement générique :
s'applique directement à une coupe de PSF, comme `mtf_cuts` pour la FTM).
`strehl_ratio` existait déjà (§9.1) ; seule différence ici, calculé à la
fois sur la PSF simulée et sur la PSF récupérée, contre la même référence
diffraction-limitée (même masque, phase nulle). Démontré dans les trois
exemples de `examples/psf2zernike_demo.ipynb` (section « Fonctionnalités
avancées »).
