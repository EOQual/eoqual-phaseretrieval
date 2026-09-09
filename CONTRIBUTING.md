# Contribuer à eoqual-phaseretrieval

Les contributions sont bienvenues, sous toutes leurs formes : signalement
de bug, correction, proposition de nouvelle méthode, implémentation d'une
méthode déjà proposée, amélioration de la documentation.

## Signaler un bug

Ouvrez une issue avec :
- la méthode concernée (`list_methods()` pour voir le registre complet
  dans votre environnement) ;
- un cas de test minimal reproduisant le problème — idéalement via
  `eoqual.phaseretrieval.simulation` (vérité de Zernike connue) plutôt
  qu'une image externe, pour que le problème soit reproductible sans
  partager de données ;
- le résultat obtenu vs. attendu.

## Proposer une méthode

Ouvrez une issue décrivant la méthode (nom, référence bibliographique, ce
qu'elle apporte par rapport aux méthodes déjà cataloguées — voir
`REFERENCE_TECHNIQUE.md` §3 pour le panorama complet, y compris les
familles *non* retenues à ce stade). Pas besoin de savoir l'implémenter
pour la proposer.

## Implémenter une méthode

Convention du projet (voir `REFERENCE_TECHNIQUE.md` §6.5 pour le détail) :
une méthode = un module `src/eoqual/phaseretrieval/backends/<nom>.py`,
avec sa propre dataclass de configuration et une fonction
`fit_...(psf(s), mask, cfg) -> (coeffs_nm, diagnostics)`, réutilisant
`optics.py` si la méthode passe par un modèle direct PSF(coeffs). Puis :
déclarer la méthode dans `config.py::METHODS_CONFIGS`, ajouter la branche
de dispatch dans `pipeline.psf_to_wfe` et les options CLI correspondantes
dans `build_arg_parser`.

**Licence — condition de fusion, pas une formalité.** Toute implémentation
reprenant ou s'inspirant de près d'un code publié ailleurs (mêmes noms de
fonctions, même structure, portage quasi littéral) doit citer sa source
et la licence *vérifiée* de cette source dans le module (vérification
directe du fichier LICENSE ou du champ `license` de l'API GitHub/PyPI,
pas une supposition). Une implémentation dont la licence d'origine n'est
pas identifiée, ou qui s'avère copyleft/non permissive, **ne sera pas
acceptée dans le socle**. Voir `THIRD_PARTY_LICENSES.md` pour l'audit de
l'existant (à ce jour : uniquement `poppy`, BSD-3-Clause, pour la base de
Zernike — aucun autre portage de code tiers).

## Tests

Toute nouvelle méthode doit être accompagnée d'un test dans
`tests/test_simulation_roundtrip.py` : simulation d'un cas à vérité de
Zernike connue (`eoqual.phaseretrieval.simulation`, avec et sans bruit
détecteur) → appel de la méthode → comparaison des coefficients retrouvés
à la vérité terrain. Le module de simulation étant lui-même séparé des
méthodes (`simulation/` n'importe aucun `backends/*`, sauf
`diversity_phase` réutilisé pour garantir la même convention de défocus),
un test peut être écrit sans dépendre d'aucune donnée externe.

## Documentation

Toute méthode nouvelle ou modifiée doit rester reflétée dans `README.md`
(catalogue) et `REFERENCE_TECHNIQUE.md` (principe, implémentation,
limites, cas d'usage) — les docstrings seules ne suffisent pas, ces deux
documents sont la carte d'ensemble du projet.
