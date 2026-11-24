# Observation de la Terre

Les références numérotées [n] renvoient à la [bibliographie](bibliographie.md).

## 8. Cas d'usage : observation de la Terre

Ce qui suit répond à une question fréquente sur ce projet : quelles
méthodes privilégier pour un instrument d'observation de la Terre (défilant
ou agile), et la piste « maximum de vraisemblance / pondération Poisson »
([§3](principes.md#3-panorama-des-méthodes-de-récupération-de-phase), [§3.6](principes.md#36-historiques-ou-toujours-pertinentes-)) a‑t‑elle déjà été éprouvée dans ce contexte précis ?

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
  utilisée en [§3.7](principes.md#37-lever-lambiguïté-de-signe-par-un-a-priori) pour garantir que l'ambiguïté de signe est un bloc
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
  opérationnelle courante — voir [§6.5](implementation.md#65-étendre-le-code-ajouter-une-4ᵉ-méthode) si cette extension est souhaitée.

### 8.2 Recommandation

1. **Campagne dédiée avec manœuvre de défocalisation (étoiles)** →
   `"phase_diversity"`, sans hésitation : c'est la méthode déjà validée et
   utilisée opérationnellement dans ce contexte précis (Pléiades [13],
   étude de sensibilité pour les futurs télescopes d'observation de la
   Terre [13, 14]), et c'est la seule des trois qui lève l'ambiguïté de
   signe de façon fiable — critique ici (§8.1).
2. **Suivi courant, une seule acquisition disponible, sens du défocus déjà
   connu** (mécanisme de mise au point à position nominale connue) →
   `"optimization"` + `sign_prior=(4, ...)` ([§3.7](principes.md#37-lever-lambiguïté-de-signe-par-un-a-priori)) en secours léger : rapide,
   déterministe, mais à valider ([§3.7](principes.md#37-lever-lambiguïté-de-signe-par-un-a-priori) : la garantie théorique suppose la
   pupille centrosymétrique, §8.1).
3. **Gerchberg‑Saxton** → non recommandée comme diagnostic principal ici
   (instabilité quantifiée en [§3.7](principes.md#37-lever-lambiguïté-de-signe-par-un-a-priori), [§6.4](implementation.md#64-validation)) ; tout au plus comme vérification
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
  référence pour la diversité de phase ([§3](principes.md#3-panorama-des-méthodes-de-récupération-de-phase), [§4](implementation.md#4-méthodes-retenues-dans-ce-projet)).
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
(gaussien et poissonien) et l'appliquent aux deux régimes ([§9](qualite-optique.md#9-repères-de-qualité-optique--strehl-wfe-diagnostics-de-psf) de [9]
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
