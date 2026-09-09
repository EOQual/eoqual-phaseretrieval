# eoqual-phaseretrieval — Référence technique

> Document de référence théorique **et** technique de ce module. Il couvre
> la famille complète des méthodes de récupération de phase implémentées ici — y
> compris la diversité de phase, qui en est une variante à deux images plutôt
> qu'une méthode indépendante — d'où le nom du module (« phase retrieval »
> chapeaute « phase diversity », pas l'inverse). Les trois méthodes partagent la même
> base physique (§1‑2) et la même interface de code (§6).

---

## 1. Pourquoi le phase‑retrieval ?

- La PSF est l'intensité du champ focal :

$$
\text{PSF}(x,y)=\bigl| \mathcal{F}\{P(\xi,\eta)\,e^{i\phi(\xi,\eta)}\}\bigr|^{2}
$$

où $P$ est la fonction pupille (amplitude, ex. le masque binaire) et $\phi$
le terme de phase que l'on cherche (la WFE — *Wave-Front Error*).
- En connaissant $P$ (forme de l'ouverture, éventuelles obstructions, etc.)
et la **longueur d'onde** utilisée, la PSF contient indirectement
l'information de $\phi$.
- Récupérer $\phi$ à partir d'une mesure d'intensité uniquement est
exactement le problème de *phase retrieval* [1, 2].
- Une fois $\phi$ connue, elle est usuellement décomposée sur la base des
polynômes de Zernike, en **ordre de Noll** [3], ce qui donne des
coefficients directement interprétables (défocus, astigmatisme, coma, …) et
un critère de qualité optique compact : le **Strehl ratio** peut être
approché par $S \approx \exp\!\bigl[-(2\pi\,\sigma_{\text{WFE}}/\lambda)^2\bigr]$
pour de petites aberrations ($\sigma_{\text{WFE}} \lesssim \lambda/14$) [4].

---

## 2. Limites d'une récupération à partir d'une **seule** PSF

| Ambiguïté | Description |
|-----------|-------------|
| **Piston** | Un terme constant $\phi_0$ ne change pas la PSF (on ne le récupère pas). |
| **Symétrie de signe** | $\phi$ et $-\phi$ produisent la même PSF (ambiguïté de signe / conjugaison). |
| **Tip‑tilt** | Un décalage linéaire du front d'onde se traduit par un déplacement du PSF ; si le recadrage n'est pas connu, tip‑tilt reste indéterminé. |
| **Conjugaison complexe** | La transformation de Fourier étant symétrique, il y a parfois deux solutions conjuguées de la pupille elle-même. |

**Conclusion :** avec une PSF seule, on ne peut récupérer que la **forme**
de la WFE modulo ces degrés de liberté [1, 5]. C'est précisément ce que la
*diversité de phase* (§3.4) est conçue pour lever, en ajoutant une seconde
mesure dans laquelle l'ambiguïté ne se manifeste pas de la même façon.

### 2.1 Amplitude et centrage : que récupèrent réellement ces méthodes ?

Question fréquente, à préciser avant tout usage sur des données réelles :
les trois méthodes ne récupèrent **que la phase** $\phi$, jamais
l'amplitude de la pupille $P$, qui est toujours prise comme une donnée
d'entrée fixe (le masque binaire fourni à `psf_to_wfe`) — pas comme un
paramètre ajusté. Une pupille réellement apodisée (transmission non
uniforme, vignettage) et modélisée ici par un masque purement binaire
introduit donc un biais de modèle qui n'est jamais détecté ni corrigé par
l'algorithme lui-même.

Le centrage (décalage $dx, dy$ du pic de la PSF, physiquement équivalent
au tip-tilt $Z_2, Z_3$ — voir la ligne « Tip-tilt » ci-dessus) est traité
différemment selon la méthode, et **aucune des trois ne l'estime par
défaut** :

| Méthode | Tip-tilt / décalage | Détail |
|---|---|---|
| `"optimization"` | Estimable en option | `OpticalConfig(fit_piston_tilt=True)` ajoute $Z_2, Z_3$ aux paramètres libres — physiquement équivalent à un décalage sous-pixel dans l'approximation paraxiale. `False` par défaut : $Z_1$–$Z_3$ figés à 0. |
| `"gerchberg_saxton"` | Non restituable | La phase brute récupérée pixel à pixel peut absorber un décalage réel, mais `zernike_fit.fit_zernike` **remet systématiquement $Z_1$–$Z_3$ à zéro** après projection (`# Remove piston and tilts`) : même si GS l'a implicitement « vu », l'information est perdue à l'étape de décomposition en Zernike. |
| `"phase_diversity"` | Non implémenté | $Z_1$–$Z_3$ sont figés à 0 en dur (`coeffs[3:] = active_coeffs`), sans option pour les libérer — un vrai décalage entre les deux PSF biaiserait donc l'ensemble du fit conjoint. |

**Implication pratique :** l'implémentation actuelle suppose que la (les)
PSF fournie(s) en entrée sont déjà recentrées sur la grille (pic au
centre géométrique du tableau) — c'est typiquement le rôle d'une étape de
recalage en amont (voir §5.0 : Shift-and-Add, ePSF, PSFr) plutôt que du
récupération de phase elle-même. Sur une PSF mal centrée, `"optimization"` avec
`fit_piston_tilt=True` est la seule option qui peut absorber un léger
décalage résiduel ; les deux autres méthodes en hériteraient un biais sur
tous les coefficients (le modèle, toujours centré par construction —
`optics.psf_from_phase` — ne peut pas reproduire un pic décalé autrement
qu'en dégradant l'ajustement des modes réels).

### Illustration empirique

La validation menée dans [`examples/psf2zernike_demo.ipynb`](examples/psf2zernike_demo.ipynb) sur les
données du simulateur (`examples/data/*_simu.tiff`, vérité terrain
$Z_4=55{,}0$ nm, $Z_5=27{,}5$ nm) rend ces ambiguïtés très concrètes :

| Méthode | Corrélation avec la vérité | Interprétation |
|---|---|---|
| Optimisation directe (1 PSF) | **−1,00** | Ambiguïté de signe exacte — l'amplitude est juste, le signe est inversé. |
| Gerchberg‑Saxton (1 PSF) | −1,00 à +0,83 selon l'initialisation | Même ambiguïté, plus une sensibilité à l'initialisation aléatoire. |
| Diversité de phase (2 PSF) | **+1,00** | Ambiguïté levée par la seconde image. |

---

## 3. Panorama des méthodes de récupération de phase

| Méthode | Principe | Points forts / faibles | Référence(s) |
|---|---|---|---|
| **Gerchberg‑Saxton (GS)** | Alternance entre le plan pupille (imposer l'amplitude $P$, laisser la phase libre) et le plan image (imposer l'amplitude $\sqrt{\text{PSF}}$). | Simple à implémenter ; convergence non garantie, sensible aux minima locaux et à l'initialisation. | Gerchberg & Saxton (1972) [1] |
| **Error‑Reduction / Hybrid Input‑Output (HIO)** | Variantes de GS ; HIO introduit un terme de rétroaction $g_{k+1}=g_k-\beta g_k'$ hors du support pour éviter les stagnations. | Meilleure robustesse que GS pur ; paramètre de relaxation $\beta$ à régler (typ. 0,7–0,9). | Fienup (1978, 1982) [2, 5] |
| **Ajustement direct par optimisation non linéaire** (*model‑based fitting*) | Les coefficients de Zernike sont les paramètres libres d'un modèle direct $\text{PSF}_{\text{modèle}}(\mathbf{c})$ ; on minimise $\lVert\sqrt{\text{PSF}_{\text{modèle}}}-\sqrt{\text{PSF}_{\text{mes}}}\rVert^2$ par moindres carrés (ex. Trust‑Region‑Reflective). | Déterministe, rapide à converger, paramétrisation de dimension réduite (quelques dizaines de coefficients au lieu d'un pixel par pixel) ; hérite des mêmes ambiguïtés qu'une seule PSF (piston/tilt/signe). | Branch, Coleman & Li (1999) pour TRF [6] ; formulation proche de Fienup (1993) [7] |
| **Diversité de phase (*phase diversity*)** | On mesure **deux** (ou plus) PSF : une focalisée et une avec une aberration connue et volontairement introduite (typ. défocus). L'algorithme ajuste conjointement les deux images. | Résout l'ambiguïté de signe et de piston/tilt, améliore la stabilité ; nécessite un second cliché et de connaître précisément la diversité introduite. | Gonsalves (1982) [8] ; Paxman, Schulz & Fienup (1992) [9] |
| **Maximum de vraisemblance / bayésien, pondération de bruit** | Formulation statistique qui pondère le résidu par le modèle de bruit (Poisson pour un détecteur à comptage de photons) et peut intégrer des a priori (ex. parcimonie sur les coefficients de Zernike). | Optimal statistiquement pour des PSF bruitées ; plus coûteux, nécessite de connaître le modèle de bruit. | Löfdahl & Scharmer (1994) [10] |
| **Pertes robustes (M‑estimateurs)** | Remplace le résidu quadratique par une fonction moins sensible aux valeurs aberrantes (`soft_l1`, `huber`) dans l'optimiseur. | Complément aux deux méthodes basées optimisation, utile en présence de pixels chauds / saturation / mauvais pixels. | Huber (1964) [11] |
| **Approches linéarisées (OTF)** | On calcule l'OTF $= \mathcal{F}\{\text{PSF}\}$ ; le module de la pupille complexe est lié à $\sqrt{\text{OTF}}$, la phase se récupère par des méthodes linéaires (filtrage de Wiener, déconvolution). | Rapide, mais valable seulement pour de petites aberrations (approximation linéaire). | Goodman (2005), ch. 6 [4] |

### 3.6 Historiques ou toujours pertinentes ?

Aucune de ces méthodes n'est vraiment obsolète : chacune reste le meilleur
choix dans un régime particulier.

- **GS / HIO / Error‑Reduction** — nées les premières (1972–1982), elles
  restent **le standard actuel** dès que la phase n'est *pas* bien décrite
  par une poignée de modes Zernike (pupille segmentée, imagerie
  cohérente à rayons X/EUV — *coherent diffractive imaging*, lithographie).
  Là où notre paramétrisation modale s'applique (une pupille simple,
  quelques dizaines de coefficients suffisent), l'ajustement direct
  (§3, ligne 3) les rend souvent inutiles : il converge en quelques
  dizaines d'évaluations, de façon déterministe, contre des centaines
  d'itérations FFT/IFFT pour un résultat plus incertain (§3.7, §6.4).
- **Ajustement direct par optimisation non linéaire** — pas « historique »
  au sens où GS l'est ; c'est l'approche par défaut dès que le nombre de
  degrés de liberté est réduit (base de Zernike, base de modes de
  segments…), largement utilisée en métrologie optique actuelle. Sensible
  au choix de l'initialisation sur des cas difficiles (§3.7).
- **Diversité de phase** — **toujours activement utilisée**, en particulier
  en astronomie : c'est la technique employée pour la mise au point
  optique et le suivi de front d'onde du télescope spatial JWST en orbite,
  ainsi que dans plusieurs instruments solaires au sol. Rien d'historique
  ici.
- **Maximum de vraisemblance / pondération de bruit (Poisson)** — c'est au
  contraire le raffinement **le plus moderne** des méthodes précédentes,
  de plus en plus pertinent avec les détecteurs à comptage de photons
  (imagerie faible flux, exoplanètes en imagerie directe). Notre option
  `poisson_weighting=True` de la diversité de phase (§6.3) *est* cette
  approche.
- **Pertes robustes (`soft_l1`, `huber`)** — un bon réflexe d'ingénierie
  plutôt qu'une "méthode" : à activer par défaut dès que des données
  réelles (pixels chauds, saturation) sont utilisées à la place de
  simulations propres.
- **Approches linéarisées (OTF)** — les plus datées en pratique : valables
  seulement pour de petites aberrations, elles restent utiles comme
  estimation initiale rapide (ex. boucle d'optique adaptative en régime
  déjà corrigé, où le résidu est petit par construction) mais sont
  supplantées par les méthodes non linéaires dès que l'aberration est
  significative.

### 3.7 Lever l'ambiguïté de signe par un a priori

**Peut-on récupérer le bon signe avec un a priori sur $Z_4$ (et éventuellement
d'autres modes) ?** Oui — et il suffit d'**un seul** a priori, sur **un
seul** mode, pour résoudre le signe de tous les modes concernés à la fois.
Ce résultat se démontre exactement, pas seulement empiriquement.

**Démonstration.** Pour une pupille $P$ réelle et centrosymétrique
($P(-\xi,-\eta)=P(\xi,\eta)$ — vrai pour toute ouverture circulaire ou
annulaire sans détail brisant la symétrie), l'identité de Fourier
$\lvert\mathcal{F}\{g^*(-\cdot)\}\rvert = \lvert\mathcal{F}\{g\}\rvert$
appliquée à $g=P\,e^{i\phi}$ montre que $(P,\phi)$ et
$(P,\,-\phi(-\xi,-\eta))$ produisent **exactement** la même PSF — c'est
l'ambiguïté « image jumelle », cataloguée par Millane (1990) [12] parmi les
ambiguïtés triviales de la reconstruction de phase. En décomposant $\phi$ sur la
base de Zernike (ordre de Noll $j$, fréquence azimutale $m_j$), l'inversion
de coordonnées donne $Z_j(\rho,\theta+\pi)=(-1)^{m_j}Z_j(\rho,\theta)$, d'où
les coefficients de la « jumelle » :

$$
c_j' = (-1)^{m_j+1}\,c_j
\qquad\Longrightarrow\qquad
\begin{cases}
c_j' = -c_j & \text{si } m_j \text{ pair (défocus } Z_4\text{, astigmatisme } Z_5,Z_6\text{, sphérique } Z_{11}, \dots) \\
c_j' = +c_j & \text{si } m_j \text{ impair (tip/tilt, coma } Z_7,Z_8\text{, trèfle } Z_9,Z_{10}, \dots)
\end{cases}
$$

**Conséquence pratique :** les modes de fréquence azimutale **impaire**
(coma, trèfle, coma secondaire, …) sont récupérés avec le **bon signe dès
la première PSF**, sans aucune ambiguïté. Seuls les modes de fréquence
**paire** (défocus, astigmatisme, sphérique, astigmatisme secondaire,
quadrafoil, …) peuvent être inversés — mais **ensemble**, comme un seul
bloc. Un unique a priori de signe sur *n'importe lequel* d'entre eux
(typiquement $Z_4$, dont le signe est souvent connu — sens d'un défocus
volontairement introduit) suffit donc à lever l'ambiguïté pour tous les
modes pairs simultanément ; rien à faire pour les modes impairs.

C'est implémenté dans [`zernike_fit.resolve_sign_ambiguity`](src/eoqual/phaseretrieval/zernike_fit.py)
et exposé via `psf_to_wfe(..., sign_prior=(4, +1.0))` (§6.3).

**Validation empirique** (phases synthétiques multi-modes, $j=4\dots13$ et
$j=4\dots15$, coefficients aléatoires, masque `examples/data/mask_simu.tiff`,
`examples/psf2zernike_demo.ipynb`) :

| Méthode | Sans correction | Avec correction (a priori sur $Z_4$) |
|---|---|---|
| Optimisation directe, solveur bien convergé (`cost≈0`) | corrélation aléatoire selon la branche atteinte (souvent $-1$) | **corrélation ≈ +1, RMS < 2 nm** (8/10 tirages directement ; 10/10 avec quelques redémarrages aléatoires en cas de minimum local) |
| Gerchberg‑Saxton | très variable | **améliore nettement certains tirages** (jusqu'à corrélation ≈ +0,999) **mais dégrade parfois les autres** |

Le correctif lève exactement l'ambiguïté théorique quand le solveur a
convergé vers l'une des deux branches exactes (coût quasi nul — à vérifier
via `diagnostics["cost"]`) ; il ne corrige pas une non‑convergence générique
(minimum local sans lien avec l'ambiguïté — un simple redémarrage
multi‑départ suffit à y remédier pour `"optimization"`, cf. §6.3). Pour
Gerchberg‑Saxton, la correction reste **peu fiable** : l'algorithme ne
converge presque jamais exactement vers l'une des deux branches théoriques,
si bien que le signe du mode de référence ($Z_4$) utilisé pour décider du
correctif est lui‑même parfois erroné — le correctif peut alors dégrader le
résultat au lieu de l'améliorer. `"phase_diversity"` reste la seule méthode
qui lève l'ambiguïté de façon fiable, sans hypothèse a priori (§3.4).

---

## 4. Méthodes retenues dans ce projet

Sur les sept familles ci-dessus, trois sont implémentées — les deux
premières lignes du panorama (Error‑Reduction/HIO pur) n'ont pas été
retenues séparément : notre implémentation GS correspond à l'algorithme
originel de Gerchberg & Saxton (le paramètre `beta` de
[`RetrievalConfig`](src/eoqual/phaseretrieval/backends/gerchberg_saxton.py) est réservé à une future extension
HIO). La pondération de bruit Poisson et les pertes robustes ne sont pas
des méthodes à part entière mais des options de la diversité de phase.

| `method=` (code) | Technique | Référence principale | Module |
|---|---|---|---|
| `"gerchberg_saxton"` | Gerchberg‑Saxton | [1] | [`backends/gerchberg_saxton.py`](src/eoqual/phaseretrieval/backends/gerchberg_saxton.py) |
| `"optimization"` | Ajustement direct par moindres carrés non linéaires | [6, 7] | [`backends/optimization.py`](src/eoqual/phaseretrieval/backends/optimization.py) |
| `"phase_diversity"` | Diversité de phase modale, pondération Poisson optionnelle, perte robuste optionnelle | [8, 9, 10, 11] | [`backends/phase_diversity.py`](src/eoqual/phaseretrieval/backends/phase_diversity.py) |

La décomposition finale en coefficients de Zernike (ordre de Noll) est
commune aux trois méthodes : linéaire par moindres carrés pour GS (la phase
est d'abord récupérée pixel à pixel, puis projetée — voir
[`zernike_fit.py`](src/eoqual/phaseretrieval/zernike_fit.py)), directe pour les deux autres (les
coefficients *sont* les paramètres de l'optimisation).

---

## 5. Workflow typique (exemple avec diversité de phase)

### 5.0 D'où vient la PSF d'entrée ?

`psf_to_wfe` (§6) part d'une PSF déjà formée — mais sur un instrument réel,
cette PSF unique est rarement une simple exposition brute : elle est le
plus souvent **reconstruite à partir d'un burst** de plusieurs images
légèrement décalées (dithering, agilité du satellite, turbulence
moyennée), pour dépasser le sous-échantillonnage du détecteur. Trois
familles de méthodes, par complexité croissante :

- **Shift-and-Add / Drizzle** — recalage sous-pixel (souvent par
  corrélation de phase) puis projection/empilement sur une grille
  sur-échantillonnée [21].
- **ePSF (Effective PSF)** — méthode empirique itérative d'Anderson & King
  [22], devenue le standard pour HST/JWST ; implémentée par
  `photutils.psf.EPSFBuilder`.
- **PSFr** — variante itérative par résidus, alignement/recentrage répété
  à chaque itération.

C'est exactement ce que fait le projet séparé `starpsfsr` (mêmes trois
méthodes, sur données Bayer multi-bandes — voir aussi §8.1) : sa sortie
(une PSF par bande) est le point d'entrée naturel de ce pipeline-ci. Aucune
des trois méthodes ne fait partie de ce dépôt.

1. **Acquisition des images**
   - PSF focalisée (à focus nominal).
   - PSF avec une diversité connue (ex. défocus de $\Delta z$ introduit
     volontairement).

2. **Modélisation du système**
   - Fonction pupille $P(\xi,\eta)$ (circulaire, annulaire, obstruction
     centrale, araignées) → masque binaire dans ce projet.
   - Longueur d'onde $\lambda$.
   - Paramètre de diversité (défocus $\Delta z$ → phase quadratique connue,
     mode de Noll $Z_4$).

3. **Algorithme de récupération**
   ```text
   initialiser φ (ou les coefficients de Zernike c)
   répéter jusqu'à convergence :
       - passer au plan image (FFT) → comparer à sqrt(PSF_foc) et sqrt(PSF_div)
       - mise à jour de φ (ou de c) selon l'algorithme choisi (GS, TRF, …)
   ```
   La convergence est surveillée via une métrique d'erreur, par exemple le
   coût des moindres carrés $\varepsilon = \lVert\sqrt{I_{\text{mod}}}-\sqrt{I_{\text{mes}}}\rVert_2$
   pour les méthodes par optimisation, ou la corrélation phase mesurée vs.
   phase déroulée pour GS.

4. **Post‑traitement**
   - Décomposition de la phase $\phi(\xi,\eta)$ en coefficients Zernike
     (ordre de Noll).
   - Suppression du piston, tip‑tilt ($Z_1$–$Z_3$, non observables sans
     référence externe).
   - Visualisation : carte de WFE (nm), PSF prédite vs. mesurée.

5. **Validation**
   - Re‑simulation de la PSF à partir de la WFE estimée et comparaison avec
     les mesures (RMSE dans `diagnostics["psf_rmse"]` / `"psf_*_rmse"`).
   - Comparaison à une vérité terrain simulée (`examples/data/phase_simu.tiff`) ou,
     idéalement, à une mesure indépendante (Shack‑Hartmann, interférogramme).

---

## 6. Implémentation technique

### 6.1 Architecture des modules

```
src/eoqual/phaseretrieval/
    io_utils.py                    # chargement image/masque, diagnostics Plotly
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

### 6.2 Point d'entrée unifié

```python
coeffs_nm, phase_fit_nm, diagnostics = psf_to_wfe(
    psf_path, mask_path, wavelength, focal_length, pupil_diameter, detector_size,
    method="gerchberg_saxton" | "optimization" | "phase_diversity",
    zernike_order=36,
    # ... paramètres spécifiques à chaque méthode (voir pipeline.py) ...
)
```

`psf_to_wfe` ([`pipeline.py`](src/eoqual/phaseretrieval/pipeline.py)) charge la PSF et le masque une
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
`"optimization"` (voir §3.7 pour la théorie et ses limites) ; ignoré (avec
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
connue à la demande — `tests/test_simulation_roundtrip.py` est la suite
de référence (simulation → récupération de phase → comparaison à la vérité
terrain), rejouée à chaque exécution des tests plutôt que figée dans des
fixtures. Les résultats de fond sont résumés en §2 ; ils confirment sur
des données contrôlées les ambiguïtés théoriques du §2 et §3, et donnent
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
(`PhaseDiversityConfig`, §6.3) a été conçu — voir §9 pour la discussion
de sa pertinence en télédétection.

---

## 7. Checklist opérationnelle

1. **Définir les données d'entrée**
   - Une PSF, ou une paire focus/défocus ? Format (tiff, fits…) ?
   - Longueur d'onde, diamètre de pupille, longueur focale, taille détecteur
     connus ? (nécessaires pour `load_and_prepare_mask`, sauf si le masque
     est déjà à la résolution de la PSF).

2. **Choisir la méthode** (`method=` de `psf_to_wfe`)
   - Une seule PSF, tolérance à l'ambiguïté de signe → `"optimization"`
     (déterministe, rapide) ou `"gerchberg_saxton"` (si l'on veut la phase
     pixel à pixel plutôt qu'une base modale).
   - Une seule PSF, mais signe connu d'au moins un mode pair (ex. sens du
     défocus) → idem + `sign_prior=(4, +1.0)` (§3.7 ; fiable avec
     `"optimization"`, pas avec `"gerchberg_saxton"`).
   - Deux PSF (focus + défocus connu) → `"phase_diversity"`, seule méthode
     qui lève l'ambiguïté de signe sans a priori.

3. **Paramétrer l'algorithme**
   - `zernike_order` : 15–36 selon la précision souhaitée.
   - GS : `max_iter` (typ. 200–500), `seed` pour la reproductibilité.
   - Optimisation / diversité : `max_nfev` si besoin de plafonner le coût
     de calcul ; `loss`/`poisson_weighting` pour la diversité en présence
     de bruit de photon.

4. **Vérifier la convergence**
   - `diagnostics["cost"]`, `"nfev"`, `"message"` (optimisation/diversité).
   - `diagnostics["residual_rms_nm"]` (GS, après projection Zernike).
   - Visualiser la PSF prédite vs. mesurée (`psf_to_wfe(..., verbose=True)`).

5. **Extraire et interpréter la WFE**
   - Coefficients de Zernike en nm WFE (ordre de Noll), piston/tip/tilt mis
     à zéro par convention (non observables sans référence externe).
   - RMS WFE global, Strehl estimé (§1).

---

## 8. Cas d'usage : observation de la Terre

Ce qui suit répond à une question fréquente sur ce projet : quelles
méthodes privilégier pour un instrument d'observation de la Terre (défilant
ou agile), et la piste « maximum de vraisemblance / pondération Poisson »
(§3, §3.6) a‑t‑elle déjà été éprouvée dans ce contexte précis ?

### 8.1 Contexte spécifique à la télédétection

- **Cible et acquisition.** Contrairement à un télescope astronomique, le
  diagnostic se fait généralement lors de campagnes dédiées (visée sur une
  étoile, manœuvre de défocalisation contrôlée) plutôt qu'en continu ; le
  traitement est fait au sol, pas en temps réel — le budget de calcul est
  donc rarement la contrainte dominante.
- **Le signe compte opérationnellement.** Se tromper de signe sur $Z_4$
  revient à commander le mécanisme de mise au point (typiquement un
  actionneur thermique sur le miroir secondaire) **dans le mauvais sens** —
  contrairement à un contexte purement diagnostique, l'erreur est
  immédiatement coûteuse et vérifiable. C'est précisément le cas d'usage
  visé par le satellite Pléiades (CNES) : le défocus est estimé à partir
  d'acquisitions d'étoiles puis corrigé par actionnement thermique du
  miroir M2 [13].
- **Pupille réelle.** Un télescope de télédétection est généralement
  obstrué (miroir secondaire, araignées) ; l'hypothèse de centrosymétrie
  utilisée en §3.7 pour garantir que l'ambiguïté de signe est un bloc
  global propre n'est qu'approximative si les araignées ne sont pas
  symétriques — à vérifier au cas par cas plutôt qu'à supposer.
- **Objet inconnu (imagerie de paysage).** En dehors des campagnes dédiées
  sur étoiles, l'essentiel des acquisitions se fait sur des scènes
  terrestres étendues, dont le contenu est *lui‑même* inconnu : c'est le
  régime de la diversité de phase « myope » (estimation **conjointe** de
  l'objet et de l'aberration), formalisée par Paxman, Schulz & Fienup [9]
  et explicitement testée sur des images de paysage par Ikoma, Delvit,
  Latry, Thiébaut & Morel (CNES, 2015) [15]. **Ce n'est pas ce
  qu'implémente actuellement `backends/phase_diversity.py`** : notre module suppose
  l'objet ponctuel (source du type étoile), valide pour les campagnes
  dédiées mais pas pour un suivi de focus à partir de l'imagerie
  opérationnelle courante — voir §6.5 si cette extension est souhaitée.

### 8.2 Recommandation

1. **Campagne dédiée avec manœuvre de défocalisation (étoiles)** →
   `"phase_diversity"`, sans hésitation : c'est la méthode déjà validée et
   utilisée opérationnellement dans ce contexte précis (Pléiades [13],
   étude de sensibilité pour les futurs télescopes d'observation de la
   Terre [13, 14]), et c'est la seule des trois qui lève l'ambiguïté de
   signe de façon fiable — critique ici (§8.1).
2. **Suivi courant, une seule acquisition disponible, sens du défocus déjà
   connu** (mécanisme de mise au point à position nominale connue) →
   `"optimization"` + `sign_prior=(4, ...)` (§3.7) en secours léger : rapide,
   déterministe, mais à valider (§3.7 : la garantie théorique suppose la
   pupille centrosymétrique, §8.1).
3. **Gerchberg‑Saxton** → non recommandée comme diagnostic principal ici
   (instabilité quantifiée en §3.7, §6.4) ; tout au plus comme vérification
   rapide ou initialisation.
4. **Suivi à partir de scènes terrestres courantes (pas d'étoile dédiée)**
   → hors du périmètre actuel du code (§8.1) ; nécessiterait un module de
   diversité de phase *myope* (objet + aberration conjoints), sur le modèle
   de [9, 15].

### 8.3 La piste Poisson / maximum de vraisemblance a‑t‑elle été testée ?

**Oui, la question a été spécifiquement étudiée par la communauté qui a
développé la diversité de phase pour la télédétection française** (ONERA,
qui a formalisé la méthode statistiquement, et CNES, qui l'exploite en
vol) :

- Le cadre du maximum de vraisemblance conjoint — avec **modèle de bruit
  gaussien et poissonien explicitement traités tous les deux** — remonte à
  l'article fondateur de Paxman, Schulz & Fienup (1992) [9], déjà notre
  référence pour la diversité de phase (§3, §4).
- Idier, Mugnier & Blanc, "Statistical behavior of joint least-squares
  estimation in the phase diversity context," *IEEE Transactions on Image
  Processing*, 14(12), 2107–2116 (2005) [16], analysent précisément le
  **biais statistique** de l'estimateur par moindres carrés conjoints (donc
  non pondéré) utilisé en diversité de phase — exactement la question
  qu'une pondération de bruit (Poisson ou autre) cherche à corriger — sur
  des données incluant de l'imagerie satellite (SPOT3, CNES).
- Blanc, Mugnier & Idier, "Marginal estimation of aberrations and image
  restoration by use of phase diversity," *JOSA A*, 20(6), 1035–1045
  (2003) [17], proposent un **estimateur marginal** (bayésien) alternatif
  au moindres carrés conjoints justement parce que ce dernier est
  statistiquement sous‑optimal — un signal supplémentaire que le
  traitement rigoureux du bruit est pris au sérieux dans ce contexte.

**Aucune publiation trouvée** (recherches bibliographiques ciblées, non
exhaustives) : une publication qui teste et quantifie explicitement une
pondération **Poisson** (par opposition à gaussienne/non pondérée) sur des
données réelles de télédétection avec chiffres à l'appui — plutôt des
travaux qui traitent la statistique du bruit comme un problème général
(gaussien et poissonien) et l'appliquent aux deux régimes (§9 de [9]
notamment).

**Est‑elle adaptée au contexte ?** Physiquement, oui, dans les mêmes
proportions qu'en astronomie : les acquisitions d'étoiles pour campagne de
calibration sont des expositions courtes sur une source relativement
faible comparée à une scène terrestre — un régime où le bruit de photon
n'est pas négligeable, sans être nécessairement le régime extrême du
comptage de photons uniques. La pondération Poisson (déjà implémentée,
`poisson_weighting=True` par défaut dans `backends/phase_diversity.py`)
y est donc une amélioration raisonnable et peu coûteuse par rapport à un
résidu non pondéré — mais, dans ce contexte spécifiquement, l'essentiel de
l'effort de recherche cité ci‑dessus porte moins sur le *poids* du bruit
que sur l'estimateur lui‑même (conjoint vs. marginal) et, pour l'imagerie
de paysage, sur l'objet inconnu (§8.1, §8.2‑4) — c'est probablement là que
se trouve le gain le plus important pour un usage opérationnel en
télédétection, plus que dans le raffinement du modèle de bruit seul.

---

## 9. Repères de qualité optique : Strehl, WFE, diagnostics de PSF

Complément indépendant du fit Zernike : le module [`diagnostics.py`](src/eoqual/phaseretrieval/diagnostics.py)
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
  avec le résidu quasi nul déjà observé en §6.4.

### 9.2 WFE RMS depuis le Strehl (approximation de Maréchal)

$$
S \approx \exp\!\left[-\left(\frac{2\pi\sigma}{\lambda}\right)^2\right]
\quad\Longleftrightarrow\quad
\sigma = \frac{\lambda}{2\pi}\sqrt{-\ln S}
$$

(`diagnostics.wfe_rms_from_strehl`) — déjà mentionnée en §1 [4], valable
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

---

## 10. Pupilles réelles : au-delà du disque plein

La base de Zernike (`poppy.zernike.zernike_basis`, utilisée par
`optics.build_zernike_basis`) n'est **orthogonale que sur une pupille
circulaire pleine, non obstruée**. Toutes nos validations (§6.4, §8) ont
utilisé un masque simulé de ce type. Un télescope réel — en particulier en
télédétection (§8.1) — a presque toujours une **obstruction centrale**
et/ou des **bras d'araignée** : sur cette pupille, la base standard n'est
plus orthogonale, les modes se corrèlent, et `condition_number` (déjà
calculé par `zernike_fit.fit_zernike`) s'en dégrade.

Trois solutions existent, par ordre de généralité croissante et
d'interprétabilité physique décroissante :

| Solution | Cas d'usage | Limites |
|---|---|---|
| Polynômes annulaires de Mahajan [18] | Obstruction circulaire centrée, **sans** bras d'araignée | Solution analytique élégante, mais suppose la symétrie azimutale parfaite |
| Gram-Schmidt numérique | Bras d'araignée ou masque de forme quelconque | Seule solution générale ; les modes au-delà des tout premiers ordres ne portent plus le nom usuel (« coma », « trèfle »…) |
| Modes de Karhunen-Loève | Optique adaptative, analyse statistique | Base optimale au sens de la variance, mais perd l'interprétation physique ; nécessite les statistiques d'aberration du système |

Ce projet fournit la solution générale : [`zernike_basis_orthogonal.py`](src/eoqual/phaseretrieval/zernike_basis_orthogonal.py)
(`ZernikeGramSchmidt`), qui ré-orthogonalise numériquement la base sur le
masque *réel* fourni (Gram-Schmidt modifié [19, 20], réutilise
`optics.build_zernike_basis` comme point de départ). Vérifié
numériquement : orthogonalité quasi machine-epsilon (`< 1e-14`) aussi bien
sur notre masque plein que sur une pupille annulaire synthétique
(obstruction centrale). C'est un utilitaire **autonome** (`decompose`/
`reconstruct`/`as_basis_array`), et un point d'entrée commun
(`resolve_zernike_basis`) le branche en **option** (`orthogonal_basis:
bool = False`, sur `ZernikeConfig`, `OpticalConfig`, `PhaseDiversityConfig`
et `psf_to_wfe(..., orthogonal_basis=True)` / `--orthogonal_basis` en CLI)
dans les trois méthodes de récupération de phase — désactivée par défaut
(comportement historique inchangé), à activer dès qu'un masque non
standard (obstrué, avec araignées) est en jeu. Un garde-fou explicite
(`ValueError`) empêche silencieusement le décalage d'index qui se
produirait si Gram-Schmidt écartait un mode dégénéré (masque trop
obstrué/sous-échantillonné pour `zernike_order`) : voir la docstring de
`resolve_zernike_basis`.

**Mise en garde pour la comparaison à une vérité terrain.** Les
coefficients obtenus avec `orthogonal_basis=True` ne sont *pas*
directement comparables, indice par indice, à des coefficients de
référence exprimés sur la base de Noll standard (ex. la vérité simulée
par `eoqual.phaseretrieval.simulation`, qui utilise toujours la base
standard) : même sous-espace de fonctions, mais une base différente au-delà
des tout premiers modes (§ ci-dessus). Vérifié empiriquement sur une
pupille annulaire à 30 % d'obstruction : la même PSF simulée, ajustée
avec `orthogonal_basis=True`, donne une corrélation ≈ 0,99 (au lieu de
1,00 en base standard) avec la vérité *exprimée en base standard* — pas
une régression de la méthode, un effet de changement de base (et de
convergence du solveur, le coût final y étant aussi moins bon). Pour un
test de bout en bout rigoureux avec cette option, décomposer la vérité
terrain dans la *même* base orthogonalisée (`ZernikeGramSchmidt.decompose`)
avant de comparer, plutôt que dans la base standard.

---

## 11. Vers un modèle système complet (extensions futures)

Le modèle actuellement implémenté (§4, §6) est volontairement simplifié :
pupille + phase Zernike + FFT, sans détecteur, sans bruit explicite (hors
pondération Poisson optionnelle de la diversité de phase), sans WFE haut
ordre au-delà de `zernike_order`, sans filée (jitter) ni intégration
polychromatique réelle (notre simulateur Bayer, §6 du notebook,
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
  actuel, §4) ;
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
   [`simulation/star_profiles.py`](src/eoqual/phaseretrieval/simulation/star_profiles.py)
   (`psf_from_pupil_field_wavelength_scaled`, utilisée pour le simulateur
   Bayer multi-bande) — à réutiliser telle quelle plutôt qu'à réécrire ;
3. Sommer les PSF pondérées par $S(\lambda_i)$ et normaliser.

**Coût** : $N_\lambda$ FFT (+ un ré-échantillonnage) au lieu d'une seule
par évaluation du modèle — pour `"optimization"`/`"phase_diversity"`, qui
évaluent le modèle à chaque itération du solveur (`least_squares`, souvent
des dizaines à quelques centaines d'évaluations), un facteur $N_\lambda$
(5–9 dans la pratique citée en §8.3) sur le temps de calcul total.
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

Rien de cette section n'est implémenté dans ce projet à ce stade — c'est
une piste d'extension, pas une capacité actuelle. Deux cas bien différents
selon ce dont on dispose réellement sur le WFE haut ordre (HO) :

- **Carte de phase HO connue** (mesurée indépendamment — interférogramme,
  as-built optique, calibration sol) : l'intégration est **directe et
  déterministe**, sans passer par l'OTF. La carte HO connue s'ajoute
  simplement, point à point en radians, à la phase bas ordre *avant* le
  calcul de la PSF — exactement le même principe que
  `backends.phase_diversity.diversity_phase`, qui ajoute déjà un terme de
  phase connu (le défocus de diversité) sans le faire dépendre du
  solveur :

  ```python
  phase_low_order = phase_from_coefficients(coeffs_rad, basis, mask)  # paramètres libres
  phase_total = phase_low_order + phase_ho_known_rad                  # terme fixe, non ajusté
  model_psf = psf_from_phase(phase_total, mask)
  ```

  Pour `"optimization"`/`"phase_diversity"`, c'est un ajout ponctuel dans
  `residuals()`/`fit_...()` (un paramètre `known_phase_rad: NDArray` de
  plus, jamais optimisé) — la structure de l'optimisation ne change pas,
  seuls les bas ordres restent des paramètres libres, ce qui est
  précisément ce qui limite le biais évoqué en §11.2 (« ne pas biaiser
  $Z_4, Z_5, Z_6$ »). Moins naturel pour `"gerchberg_saxton"`, qui ne
  paramètre pas sa phase par une base modale — il faudrait plutôt
  soustraire l'effet de la carte HO connue de l'amplitude mesurée avant
  la boucle GS, une transformation plus invasive et non implémentée ici.
- **Seule la statistique (PSD) du HO est connue**, pas sa carte exacte
  (cas le plus courant en pratique — turbulence résiduelle, tolérances de
  fabrication non mesurées individuellement) : c'est le cas visé par la
  formule $\text{OTF}_{\text{HO}}\approx\exp[-\tfrac12 D_\phi(\lambda f)]$
  de §11.1. Ici, pas de carte à ajouter en phase : on multiplie,
  point à point en fréquence, l'OTF optique déjà calculée par
  `optics.psf_from_phase` (en passant par sa transformée de Fourier) par
  ce terme statistique, avant de revenir en PSF par FFT inverse — sans
  toucher au reste du modèle d'optimisation (§6.3). Cette approche est
  plus approximative (un effet *moyen*, pas la vraie carte) mais
  s'applique quand la seconde option ci-dessus n'est pas disponible.

---

## Références

1. R. W. Gerchberg and W. O. Saxton, "A practical algorithm for the
   determination of phase from image and diffraction plane pictures,"
   *Optik*, 35, 237–246 (1972).
2. J. R. Fienup, "Reconstruction of an object from the modulus of its
   Fourier transform," *Optics Letters*, 3(1), 27–29 (1978).
3. R. J. Noll, "Zernike polynomials and atmospheric turbulence," *Journal
   of the Optical Society of America*, 66(3), 207–211 (1976).
4. J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts &
   Company Publishers (2005).
5. J. R. Fienup, "Phase retrieval algorithms: a comparison," *Applied
   Optics*, 21(15), 2758–2769 (1982).
6. M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior, and
   Conjugate Gradient Method for Large-Scale Bound-Constrained Minimization
   Problems," *SIAM Journal on Scientific Computing*, 21(1), 1–23 (1999).
   (algorithme *Trust Region Reflective*, utilisé par
   `scipy.optimize.least_squares(method="trf")`)
7. J. R. Fienup, "Phase-retrieval algorithms for a complicated optical
   system," *Applied Optics*, 32(10), 1737–1746 (1993).
8. R. A. Gonsalves, "Phase retrieval and diversity in adaptive optics,"
   *Optical Engineering*, 21(5), 829–832 (1982).
9. R. G. Paxman, T. J. Schulz, and J. R. Fienup, "Joint estimation of
   object and aberrations by using phase diversity," *Journal of the
   Optical Society of America A*, 9(7), 1072–1085 (1992).
10. M. G. Löfdahl and G. B. Scharmer, "Wavefront sensing and image
    restoration from focused and defocused solar images," *Astronomy and
    Astrophysics Supplement Series*, 107, 243–264 (1994).
11. P. J. Huber, "Robust Estimation of a Location Parameter," *Annals of
    Mathematical Statistics*, 35(1), 73–101 (1964).
12. R. P. Millane, "Phase retrieval in crystallography and optics,"
    *Journal of the Optical Society of America A*, 7(3), 394–411 (1990).
    (catalogue les ambiguïtés triviales de la récupération de phase, dont
    l'ambiguïté « image jumelle » de conjugaison/inversion utilisée en §3.7)
13. Ch. Latry, J.-M. Delvit, and C. Thiébaut, "Sensitivity analysis of
    phase diversity technique for high resolution earth observing
    telescopes," *Proc. SPIE* 10562, International Conference on Space
    Optics — ICSO 2016, 105623X (2017). doi:10.1117/12.2296051. (CNES —
    diversité de phase pour la mise au point en vol de Pléiades‑HR)
14. "Wave front sensing for next generation earth observation telescope,"
    *Proc. SPIE* 10562, ICSO 2016, 105625H (2017). doi:10.1117/12.2296162.
15. H. Ikoma, J.-M. Delvit, C. Latry, C. Thiébaut, and J.-M. Morel, "Phase
    Diversity: Estimation of an Optical Wavefront from Landscape Images,"
    *Applied Inverse Problems Conference*, Helsinki (2015). (CNES —
    diversité de phase myope, objet+aberration conjoints, sur imagerie de
    paysage plutôt que sur étoile)
16. J. Idier, L. M. Mugnier, and A. Blanc, "Statistical behavior of joint
    least-squares estimation in the phase diversity context," *IEEE
    Transactions on Image Processing*, 14(12), 2107–2116 (2005). (ONERA)
17. A. Blanc, L. M. Mugnier, and J. Idier, "Marginal estimation of
    aberrations and image restoration by use of phase diversity," *Journal
    of the Optical Society of America A*, 20(6), 1035–1045 (2003). (ONERA)
18. V. N. Mahajan, "Zernike annular polynomials for imaging systems with
    annular pupils," *Journal of the Optical Society of America*, 71(1),
    75–85 (1981). (polynômes orthogonaux pour pupille annulaire, §10)
19. G. Dai, "Modified Hartmann-Shack wavefront sensing and iterative
    wavefront reconstruction," *Optical Engineering*, 35(12), 3333–3340
    (1996). (§10, Gram-Schmidt)
20. D. J. Schroeder, *Astronomical Optics*, Academic Press (1987). (§10,
    Gram-Schmidt)
21. A. S. Fruchter and R. N. Hook, "Drizzle: A Method for the Linear
    Reconstruction of Undersampled Images," *Publications of the
    Astronomical Society of the Pacific*, 114, 144–152 (2002). (§5.0)
22. J. Anderson and I. R. King, "Toward High-Precision Astrometry with
    WFPC2: I. Deriving an Accurate Point-Spread Function," *Publications
    of the Astronomical Society of the Pacific*, 112, 1360–1382 (2000).
    (§5.0, méthode ePSF)

> Note de vérification : les références [13]–[17], [21]–[22] ont été
> identifiées par recherche bibliographique ciblée (titres, auteurs,
> revues/actes et années confirmés) plutôt que par lecture intégrale des
> articles. Les références [18]–[20] proviennent de la docstring d'un
> prototype externe au dépôt (`Star/star_PSF/psf_star_V2d.py`) et n'ont pas été re-vérifiées indépendamment cette fois-ci.
> Dans tous les cas, les détails quantitatifs qu'elles contiennent restent
> à vérifier directement dans les textes avant citation dans un document
> externe à ce projet.
