# Vers un modèle système complet

## 11. Vers un modèle système complet (extensions futures)

Le modèle actuellement implémenté ([§4](implementation.md#4-méthodes-retenues-dans-ce-projet), [§6](implementation.md#6-implémentation-technique)) est volontairement simplifié :
pupille + phase Zernike + FFT, sans détecteur, sans bruit explicite (hors
pondération Poisson optionnelle de la diversité de phase), sans WFE haut
ordre au-delà de `zernike_order`, sans filée (jitter) ni intégration
polychromatique réelle (notre simulateur Bayer, [§6](implementation.md#6-implémentation-technique) du notebook,
approxime chaque bande par une PSF monochromatique séparée plutôt que par
une intégrale sur le spectre du filtre). Cette section synthétise, à titre
de feuille de route, un modèle plus complet identifié lors de la revue de
notes internes sur la diversité de phase.

### 11.1 Décomposition multiplicative en OTF

$$
\text{OTF}_k(\mathbf f) =
\text{OTF}_{\text{opt},k}(\mathbf f)\cdot
\text{OTF}_{\text{HO}}(\mathbf f)\cdot
\text{OTF}_{\text{det}}(\mathbf f)\cdot
\text{OTF}_{\text{jitter},k}(\mathbf f)\cdot
\text{OTF}_{\text{poly},k}(\mathbf f)
$$

où, pour l'image $k$ (diversité $Z_4=\alpha_k$) :
- $\text{OTF}_{\text{opt},k}$ — la seule partie non linéaire (notre modèle
  actuel, [§4](implementation.md#4-méthodes-retenues-dans-ce-projet)) ;
- $\text{OTF}_{\text{HO}}$ — le WFE haut ordre (au-delà de `zernike_order`),
  traité comme un terme **mesuré et figé** plutôt qu'estimé, ou par son
  effet statistique $\exp[-\tfrac12 D_\phi(\lambda f)]$ ($D_\phi$ = fonction
  de structure du front d'onde) si seule sa PSD est connue ;
- $\text{OTF}_{\text{det}} = \text{sinc}(\pi p f_x)\,\text{sinc}(\pi p f_y) \cdot \text{OTF}_{\text{emp}}$
  — MTF pixel (pas $p$) éventuellement complétée d'une MTF empirique
  mesurée en laboratoire ;
- $\text{OTF}_{\text{jitter},k} = \exp(-2\pi^2\sigma_k^2 f^2)$ — filée
  gaussienne isotrope de RMS $\sigma_k$, mesurée ou bornée par image ;
- $\text{OTF}_{\text{poly},k}(\mathbf f) = \dfrac{\int S(\lambda)\,\text{OTF}_{\text{opt},k}(\lambda f/\lambda_0;\lambda)\,d\lambda}{\int S(\lambda)\,d\lambda}$
  — intégration sur le spectre du filtre $S(\lambda)$ (5–9 longueurs
  d'onde suffisent en pratique ; même $\phi_{\text{HO}}$, phase $\propto 1/\lambda$).

**Ce qui change concrètement pour passer du monochromatique actuel à ce
terme $\text{OTF}_{\text{poly}}$.** Le modèle actuel (`optics.psf_from_phase`)
suppose une seule longueur d'onde $\lambda_0$ pour tout le calcul : les
coefficients de Zernike sont convertis une fois en radians via
`rad_to_nm = wavelength * 1e9 / (2π)`, puis une seule FFT donne la PSF.
L'intégration polychromatique demanderait, à chaque évaluation du modèle
direct (donc à *chaque itération du solveur* pour `"optimization"`/
`"phase_diversity"`) :

1. Reconvertir la **même** carte de phase (en mètres d'OPD, indépendante
   de $\lambda$) en radians pour *chaque* $\lambda_i$ du sous-échantillonnage
   spectral (`phase_rad_i = opd_m * 2π / λ_i`, phase $\propto 1/\lambda$) ;
2. Calculer la PSF pour chaque $\lambda_i$ — mais **pas** par un simple
   `psf_from_phase` sur la même grille : la taille de la tache de
   diffraction (en pixels) dépend de $\lambda$, donc chaque PSF
   monochromatique doit être calculée sur une grille rééchantillonnée
   avant de revenir sur la grille commune, exactement la technique déjà
   implémentée dans
   [`simulation/star_profiles.py`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/simulation/star_profiles.py)
   (`psf_from_pupil_field_wavelength_scaled`, utilisée pour le simulateur
   Bayer multi-bande) — à réutiliser telle quelle plutôt qu'à réécrire ;
3. Sommer les PSF pondérées par $S(\lambda_i)$ et normaliser.

**Coût** : $N_\lambda$ FFT (+ un ré-échantillonnage) au lieu d'une seule
par évaluation du modèle — pour `"optimization"`/`"phase_diversity"`, qui
évaluent le modèle à chaque itération du solveur (`least_squares`, souvent
des dizaines à quelques centaines d'évaluations), un facteur $N_\lambda$
(5–9 dans la pratique citée en [§8.3](observation-terre.md#83-la-piste-poisson--maximum-de-vraisemblance-atelle-été-testée-)) sur le temps de calcul total.
Gerchberg-Saxton n'a pas d'équivalent naturel : son itération alterne des
contraintes d'amplitude *exactes* dans les deux plans, ce qui ne se
généralise pas directement à une somme incohérente sur plusieurs
longueurs d'onde ; l'extension polychromatique concerne donc surtout les
deux méthodes basées optimisation. Un $S(\lambda)$ mal connu ou approximé
introduit en outre une erreur de modèle difficile à distinguer, en
pratique, du WFE haut ordre ou du détecteur (voir §11.2 juste après).

### 11.2 Le problème d'identifiabilité

Point le plus important, à retenir avant d'implémenter quoi que ce soit de
cette liste : **détecteur, WFE haut ordre, filée et polychromatisme
produisent la même signature globale** — une perte de MTF aux hautes
fréquences. On ne peut pas tous les estimer simultanément avec le seul
Zernike bas ordre ; il faut en **figer** certains (mesurés indépendamment)
pour que l'estimation reste bien conditionnée. Dans les chaînes
« instrument-grade » (AO, spatial, solaire), la diversité de phase
n'estime que le bas ordre ; le reste est mesuré à part ou traité comme un
bruit multiplicatif fixé — l'objectif n'est pas de tout reconstruire, mais
de ne pas biaiser $Z_4, Z_5, Z_6$.

### 11.3 Statut

Deux cas bien différents selon ce dont on dispose réellement sur le WFE
haut ordre (HO) — seul le premier est implémenté à ce stade :

- **Carte de phase HO connue** (mesurée indépendamment — interférogramme,
  as-built optique, calibration sol) : **implémenté**, via
  `known_ho_wfe_nm` (une carte de WFE `[nm]`, même forme que le masque)
  sur `OpticalConfig`/`PhaseDiversityConfig`, ou `psf_to_wfe(...,
  known_ho_wfe_path=...)`/`--known_ho_wfe` en CLI (chemin vers une image,
  chargée comme la PSF/le masque). L'intégration est **directe et
  déterministe**, sans passer par l'OTF : la carte connue s'ajoute
  simplement, point à point en radians (conversion via
  `optics.wfe_map_to_phase`), à la phase bas ordre *avant* le calcul de la
  PSF — exactement le même principe que
  `backends.phase_diversity.diversity_phase`, qui ajoute déjà un terme de
  phase connu (le défocus de diversité) sans le faire dépendre du
  solveur :

  ```python
  phase_low_order = phase_from_coefficients(coeffs_rad, basis, mask)  # paramètres libres
  phase_total = phase_low_order + wfe_map_to_phase(known_ho_wfe_nm, wavelength, mask)  # terme fixe
  model_psf = psf_from_phase(phase_total, mask)
  ```

  Pour `"phase_diversity"`, la carte HO est ajoutée dans **les deux**
  canaux (focus et défocus) : c'est une contribution statique de
  l'optique, présente indépendamment du terme de diversité lui-même.
  Dans les deux méthodes, seuls les bas ordres restent des paramètres
  libres — c'est précisément ce qui limite le biais évoqué en §11.2
  (« ne pas biaiser $Z_4, Z_5, Z_6$ »), vérifié empiriquement
  (`tests/test_known_ho_wfe.py`) : sans la carte HO, un mode $Z_{21}$
  connu de 40 nm biaise $Z_4$ de plus de 20 nm (55 nm de vérité mesuré à
  ~26–48 nm selon la méthode) ; avec la carte fournie, la vérité est
  recouvrée exactement (coût du solveur ≈ 0). Non branché pour
  `"gerchberg_saxton"` (avertissement, valeur ignorée) : l'algorithme ne
  paramètre pas sa phase par une base modale — il faudrait plutôt
  soustraire l'effet de la carte HO connue de l'amplitude mesurée avant
  la boucle GS, une transformation plus invasive, non implémentée ici.
  Le module de simulation (`simulate_psf_from_zernike`,
  `PhaseDiversitySimulationConfig`) accepte le même paramètre pour générer
  des cas de test réalistes.
- **Seule la statistique (PSD) du HO est connue**, pas sa carte exacte
  (cas le plus courant en pratique — turbulence résiduelle, tolérances de
  fabrication non mesurées individuellement) : **non implémenté**. C'est
  le cas visé par la formule
  $\text{OTF}_{\text{HO}}\approx\exp[-\tfrac12 D_\phi(\lambda f)]$
  de §11.1. Ici, pas de carte à ajouter en phase : il faudrait multiplier,
  point à point en fréquence, l'OTF optique déjà calculée par
  `optics.psf_from_phase` (en passant par sa transformée de Fourier) par
  ce terme statistique, avant de revenir en PSF par FFT inverse — sans
  toucher au reste du modèle d'optimisation ([§6.3](implementation.md#63-détail-par-méthode)). Cette approche est
  plus approximative (un effet *moyen*, pas la vraie carte) mais
  s'appliquerait quand la carte exacte ci-dessus n'est pas disponible.
