# Principes et panorama des méthodes

Les références numérotées [n] renvoient à la [bibliographie](bibliographie.md).

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
recalage en amont (voir [§5.0](guide.md#50-doù-vient-la-psf-dentrée-) : Shift-and-Add, ePSF, PSFr) plutôt que de la
récupération de phase elle-même. Sur une PSF mal centrée, `"optimization"` avec
`fit_piston_tilt=True` est la seule option qui peut absorber un léger
décalage résiduel ; les deux autres méthodes en hériteraient un biais sur
tous les coefficients (le modèle, toujours centré par construction —
`optics.psf_from_phase` — ne peut pas reproduire un pic décalé autrement
qu'en dégradant l'ajustement des modes réels).

### 2.2 Modifications envisageables pour les récupérer

Deux problèmes de nature très différente — l'un tractable avec les outils
déjà en place, l'autre une refonte plus profonde.

**Centrage (tip-tilt $Z_2, Z_3$) — tractable.**

- `"phase_diversity"` est le vrai manque : contrairement à `"optimization"`,
  aucune option n'existe pour libérer $Z_2, Z_3$. Ajout naturel : un
  `fit_tip_tilt: bool = False` sur `PhaseDiversityConfig`, qui étendrait
  `active_coeffs` à `coeffs[1:]` (Z2 et au-delà) au lieu de `coeffs[3:]`
  dans `residuals`/`fit_phase_diversity` — même mécanique que
  `OpticalConfig.fit_piston_tilt`, sans toucher au reste du modèle
  (`diversity_phase` reste inchangée, elle ne dépend que de `basis[3]`).
- Nuance sur `"optimization"` : `fit_piston_tilt=True` libère *aussi* le
  piston ($Z_1$), un paramètre strictement inerte (une constante de phase
  n'affecte jamais $|\cdot|^2$ dans `psf_from_phase`) — `least_squares` le
  laisse à 0 pendant tout l'ajustement, sans erreur, mais c'est un
  paramètre gaspillé. Un nom plus précis, `fit_tip_tilt`, qui ne libère
  que $Z_2, Z_3$, serait plus correct (à garder rétro-compatible via un
  alias, ou à documenter tel quel — piston inerte, sans conséquence).
- `"gerchberg_saxton"` : ajouter un `zero_tip_tilt: bool = True` sur
  `ZernikeConfig`/`fit_zernike` permettrait de ne plus jeter le tip-tilt
  que GS a implicitement absorbé dans sa phase pixel à pixel (contrairement
  au piston $Z_1$, dont la valeur GS est arbitraire — un vrai artefact de
  l'initialisation aléatoire, pas une mesure — et qui doit rester filtré).
- **Alternative recommandée en amont plutôt qu'en aval** : un recalage
  sous-pixel par corrélation de phase (déjà cité [§5.0](guide.md#50-doù-vient-la-psf-dentrée-), Shift-and-Add)
  avant même d'appeler `psf_to_wfe`, est la solution la plus robuste — un
  tip-tilt Zernike n'est une bonne approximation d'un décalage réel que
  dans le régime paraxial (petits décalages), alors qu'une corrélation de
  phase reste valide à n'importe quelle amplitude de décalage. Les options
  ci-dessus sont utiles en secours (décalage résiduel sous-pixel après un
  recalage imparfait), pas comme substitut à un recalage correct en amont.

**Amplitude de la pupille — refonte plus profonde, et d'identifiabilité
incertaine avec la donnée actuelle.**

- Un premier obstacle est réglable sans toucher aux algorithmes : `mask`
  n'a *besoin* d'être binaire nulle part dans `optics.psf_from_phase`
  (`pupil = mask * exp(i·phase)` accepte déjà n'importe quelle amplitude
  réelle dans `[0, 1]`) — c'est
  `io_utils.load_and_prepare_mask` qui **force** la binarisation
  (`(mask_raw > 0.5).astype(uint8)`). Une pupille apodisée *mesurée
  indépendamment* (calibration labo, modèle as-built) pourrait donc déjà
  être utilisée telle quelle, comme amplitude fixe, en assouplissant
  uniquement ce chargeur — sans aucune modification des trois méthodes de
  récupération de phase, puisqu'aucune ne suppose l'amplitude binaire nulle
  part ailleurs dans le calcul.
- **Ajuster** l'amplitude conjointement à la phase (plutôt que la mesurer
  à part) est une tout autre ampleur de changement :
  - Il faudrait un modèle paramétrique de transmission (ex. une expansion
    radiale à quelques coefficients, $P=\text{mask}\cdot(1+\sum_k a_k R_k(\rho))$)
    plutôt qu'une amplitude libre pixel par pixel (bien trop de paramètres
    pour rester bien posé, à l'inverse de la philosophie « quelques
    dizaines de coefficients » de ce projet, §1).
  - Pour `"optimization"`/`"phase_diversity"` : ajouter les $a_k$ au
    vecteur de paramètres libres de `least_squares`, et modifier
    `psf_from_phase` (ou une variante) pour prendre un `pupil` complet
    plutôt qu'un masque fixe — mécaniquement direct, mais avec un vrai
    risque d'identifiabilité (voir [§11.2](modele-systeme.md#112-le-problème-didentifiabilité) : plusieurs effets peuvent
    produire une signature MTF similaire) qu'il faudrait vérifier
    empiriquement avant de faire confiance au résultat, surtout avec une
    seule PSF (`"optimization"`) — `"phase_diversity"`, avec ses deux
    images à aberration connue et différente, est structurellement mieux
    posée pour espérer séparer amplitude et phase.
  - `"gerchberg_saxton"` n'est pas un bon candidat : l'algorithme repose
    sur l'imposition d'une amplitude **exacte et connue** dans le plan
    pupille à chaque itération — rendre cette amplitude elle-même
    inconnue change la nature du problème (retrouvé conjoint
    amplitude+phase), qui sort du cadre de l'algorithme GS classique
    implémenté ici.
  - Non implémenté à ce stade : à traiter comme une piste de recherche
    plutôt qu'une simple option à ajouter, contrairement au centrage
    ci-dessus.

### Illustration empirique

La validation menée dans [`examples/psf2zernike_demo.ipynb`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/examples/psf2zernike_demo.ipynb) sur les
données du simulateur (`examples/data/*_simu.tiff`, vérité terrain
$Z_4=55{,}0$ nm, $Z_5=27{,}5$ nm) rend ces ambiguïtés très concrètes :

| Méthode | Corrélation avec la vérité | Interprétation |
|---|---|---|
| Optimisation directe (1 PSF) | **−1,00** | Ambiguïté de signe exacte — l'amplitude est juste, le signe est inversé. |
| Gerchberg‑Saxton (1 PSF) | −1,00 à +0,83 selon l'initialisation | Même ambiguïté, plus une sensibilité à l'initialisation aléatoire. |
| Diversité de phase (2 PSF) | **+1,00** | Ambiguïté levée par la seconde image. |

## 3. Panorama des méthodes de récupération de phase

| Méthode | Principe | Points forts / faibles | Référence(s) |
|---|---|---|---|
| **Gerchberg‑Saxton (GS)** | Alternance entre le plan pupille (imposer l'amplitude $P$, laisser la phase libre) et le plan image (imposer l'amplitude $\sqrt{\text{PSF}}$). | Simple à implémenter ; convergence non garantie, sensible aux minima locaux et à l'initialisation. | Gerchberg & Saxton (1972) [1] |
| **Error‑Reduction / Hybrid Input‑Output (HIO)** | Variantes de GS ; HIO introduit un terme de rétroaction $g_{k+1}=g_k-\beta g_k'$ hors du support pour éviter les stagnations. | Meilleure robustesse que GS pur ; paramètre de relaxation $\beta$ à régler (typ. 0,7–0,9). | Fienup (1978, 1982) [2, 5] |
| **Ajustement direct par optimisation non linéaire** (*model‑based fitting*) | Les coefficients de Zernike sont les paramètres libres d'un modèle direct $\text{PSF}_{\text{modèle}}(\mathbf{c})$ ; on minimise $\lVert\sqrt{\text{PSF}_{\text{modèle}}}-\sqrt{\text{PSF}_{\text{mes}}}\rVert^2$ par moindres carrés (ex. Trust‑Region‑Reflective). | Déterministe, rapide à converger, paramétrisation de dimension réduite (quelques dizaines de coefficients au lieu d'un pixel par pixel) ; hérite des mêmes ambiguïtés qu'une seule PSF (piston/tilt/signe). | Branch, Coleman & Li (1999) pour TRF [6] ; formulation proche de Fienup (1993) [7] |
| **Diversité de phase (*phase diversity*)** | On mesure **deux** (ou plus) PSF : une focalisée et une avec une aberration connue et volontairement introduite (typ. défocus). L'algorithme ajuste conjointement les deux images. | Résout l'ambiguïté de signe et de piston/tilt, améliore la stabilité ; nécessite un second cliché et de connaître précisément la diversité introduite. | Gonsalves (1982) [8] ; Paxman, Schulz & Fienup (1992) [9] |
| **Maximum de vraisemblance / bayésien, pondération de bruit et a priori individuels** | Formulation statistique qui pondère le résidu par le modèle de bruit (Poisson pour un détecteur à comptage de photons) et peut intégrer des a priori — globaux (parcimonie) ou **individuels, un par mode de Zernike** (valeur attendue ± incertitude, ex. issue d'une mesure de laboratoire ou d'un budget d'erreur), sous forme de pénalité L2 pondérée ou de contrainte de boîte. | Optimal statistiquement pour des PSF bruitées ; un a priori individuel bien choisi peut se substituer à une partie du rôle désambiguïsant d'une seconde image (§3.7), au prix d'un biais si l'a priori est faux. Plus coûteux, nécessite de connaître le modèle de bruit (et, pour la variante à a priori individuels, une incertitude par mode). | Löfdahl & Scharmer (1994) [10] |
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
  d'itérations FFT/IFFT pour un résultat plus incertain (§3.7, [§6.4](implementation.md#64-validation)).
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
- **Maximum de vraisemblance / pondération de bruit (Poisson) et a priori
  individuels** — c'est au contraire le raffinement **le plus moderne** des
  méthodes précédentes, de plus en plus pertinent avec les détecteurs à
  comptage de photons (imagerie faible flux, exoplanètes en imagerie
  directe). Notre option `poisson_weighting=True` de la diversité de phase
  ([§6.3](implementation.md#63-détail-par-méthode)) *est* cette approche. **Précision (v. la remarque plus bas) :**
  cette ligne n'est pas une évolution de l'ajustement direct par
  optimisation (ligne du dessus) — c'est une évolution de la **diversité
  de phase elle‑même** et de son traitement statistique du bruit, comme le
  montrent ses propres références historiques (Paxman, Schulz & Fienup
  1992 [9] ; Löfdahl & Scharmer 1994 [10] ; Idier, Mugnier & Blanc 2005
  [16] ; Blanc, Mugnier & Idier 2003 [17], tous cités en diversité de
  phase, [§8.3](observation-terre.md#83-la-piste-poisson--maximum-de-vraisemblance-atelle-été-testée-)) : le point de départ est toujours la fonction de coût
  moindres carrés conjointe focus/défocus (§3, [§4](implementation.md#4-méthodes-retenues-dans-ce-projet)), à laquelle on
  substitue une pondération statistiquement correcte du bruit, puis un a
  priori. L'extension à un **a priori individuel par mode** (valeur
  attendue ± incertitude, plutôt qu'une simple parcimonie globale) en est
  la généralisation naturelle : elle prolonge à la fois ce traitement
  statistique du bruit ([§8.3](observation-terre.md#83-la-piste-poisson--maximum-de-vraisemblance-atelle-été-testée-)) et le mécanisme de désambiguïsation par a
  priori déjà présent dans le projet sous une forme simplifiée
  (`sign_prior`, un signe sur un seul mode, §3.7) — vers un a priori
  continu sur plusieurs modes à la fois. Poussée à l'extrême (a priori
  individuel fort sur chaque mode, cf. l'usage `individual_constraint` de
  la référence externe discutée en [§6.5](implementation.md#65-étendre-le-code-ajouter-une-4ᵉ-méthode)), cette généralisation peut même
  se substituer à la seconde image de diversité pour la désambiguïsation —
  ce qui en fait, dans ce cas limite, une manière de retrouver une partie
  du bénéfice de la diversité de phase **à partir d'une seule PSF**, en
  échangeant la mesure supplémentaire contre une connaissance a priori
  externe (risque de biais si cet a priori est faux, à la différence d'une
  vraie seconde mesure).
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

C'est implémenté dans [`zernike_fit.resolve_sign_ambiguity`](https://github.com/EOQual/eoqual-phaseretrieval/blob/main/src/eoqual/phaseretrieval/zernike_fit.py)
et exposé via `psf_to_wfe(..., sign_prior=(4, +1.0))` ([§6.3](implementation.md#63-détail-par-méthode)).

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
multi‑départ suffit à y remédier pour `"optimization"`, cf. [§6.3](implementation.md#63-détail-par-méthode)). Pour
Gerchberg‑Saxton, la correction reste **peu fiable** : l'algorithme ne
converge presque jamais exactement vers l'une des deux branches théoriques,
si bien que le signe du mode de référence ($Z_4$) utilisé pour décider du
correctif est lui‑même parfois erroné — le correctif peut alors dégrader le
résultat au lieu de l'améliorer. `"phase_diversity"` reste la seule méthode
qui lève l'ambiguïté de façon fiable, sans hypothèse a priori (§3.4).
