# Guide d'utilisation

Les références numérotées [n] renvoient à la [bibliographie](bibliographie.md).

## 5. Workflow typique (exemple avec diversité de phase)

### 5.0 D'où vient la PSF d'entrée ?

`psf_to_wfe` ([§6](implementation.md#6-implémentation-technique)) part d'une PSF déjà formée — mais sur un instrument réel,
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
méthodes, sur données Bayer multi-bandes — voir aussi [§8.1](observation-terre.md#81-contexte-spécifique-à-la-télédétection)) : sa sortie
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
     défocus) → idem + `sign_prior=(4, +1.0)` ([§3.7](principes.md#37-lever-lambiguïté-de-signe-par-un-a-priori) ; fiable avec
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
   - RMS WFE global, Strehl estimé ([§1](principes.md#1-pourquoi-le-phaseretrieval-)).
