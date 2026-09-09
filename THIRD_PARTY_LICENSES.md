# Audit des licences tierces

> Toute correction constatée après coup sera appliquée, implémentation
> retirée du socle si nécessaire — voir `CONTRIBUTING.md` pour la
> politique de licence appliquée aux futures contributions.

## 1. Principe

Les trois méthodes de retrouvé de phase (`gerchberg_saxton`,
`optimization`, `phase_diversity`) sont des **réimplémentations propres**
à partir des articles cités dans `REFERENCE_TECHNIQUE.md` §3-4 — aucun
portage de code tiers, donc aucune question de licence les concernant
directement.

## 2. Dépendances externes

| Dépendance | Usage | Licence | Vérifiée via |
|---|---|---|---|
| [poppy](https://github.com/spacetelescope/poppy) | Base de Zernike, ordre de Noll (`optics.build_zernike_basis`, `zernike_fit.py`) | BSD-3-Clause | `LICENSE.md` du dépôt (STScI) |
| numpy, scipy, scikit-image | Calcul numérique, optimisation (`least_squares`), redimensionnement de masque | BSD-3-Clause | Notoriété / PyPI |
| plotly | Diagnostics visuels interactifs (`io_utils.py`) | MIT | PyPI |
| loguru | Logs | MIT | PyPI |
| imageio | Chargement d'images (TIFF...) | BSD-2-Clause | PyPI |
| commentjson | Config JSON avec commentaires (CLI) | MIT | PyPI |
| rich | Affichage console (`runner.list_methods`) | MIT | PyPI |

Toutes les dépendances du socle sont sous licence permissive
(BSD/MIT) — aucun extra pip, aucune dépendance copyleft à ce jour.

## 3. Historique

`0.1.0` : audit initial. `hcipy`, envisagé un temps, a été écarté :
jamais importé par le code, dépendance inutile.
