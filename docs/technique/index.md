# Documents techniques

Documentation théorique **et** technique de ce module. Elle couvre
la famille complète des méthodes de récupération de phase implémentées ici — y
compris la diversité de phase, qui en est une variante à deux images plutôt
qu'une méthode indépendante — d'où le nom du module (« phase retrieval »
chapeaute « phase diversity », pas l'inverse). Les trois méthodes partagent la même
base physique ([principes](principes.md), §1-2) et la même interface de
code ([implémentation](implementation.md), §6).

Les chapitres gardent une numérotation unique d'un document à l'autre : un
renvoi « §n » désigne toujours le même chapitre. Ces documents décrivent
l'état présent du code ; ce qui change d'une version à l'autre est dans les
[notes de version](../changelog.md).

| Document | Contenu |
|---|---|
| [Principes et panorama des méthodes](principes.md) | Pourquoi le phase retrieval, limites d'une seule PSF, panorama des méthodes, ambiguïté de signe |
| [Méthodes retenues et implémentation](implementation.md) | Méthodes du module, architecture, point d'entrée, détail par méthode, validation, a priori par mode, CLI, simulation |
| [Guide d'utilisation](guide.md) | Workflow typique (diversité de phase), origine de la PSF d'entrée, checklist opérationnelle |
| [Observation de la Terre](observation-terre.md) | Contexte de la télédétection, recommandation, pondération Poisson |
| [Repères de qualité optique](qualite-optique.md) | Strehl, WFE RMS (Maréchal), tables de repère, diagnostics, FTM |
| [Pupilles réelles](pupilles.md) | Pupilles obstruées et à araignées, base orthogonalisée, bord des bras |
| [Résolutions multiples](resolutions.md) | Masque, carte HO et PSF suréchantillonnée à des résolutions différentes, `aperture_px` |
| [Vers un modèle système complet](modele-systeme.md) | Décomposition en OTF, identifiabilité, WFE haut ordre connue |
| [Bibliographie](bibliographie.md) | Références numérotées [n] citées dans les documents |

```{toctree}
:hidden:
:maxdepth: 2

principes
implementation
guide
observation-terre
qualite-optique
pupilles
resolutions
modele-systeme
bibliographie
```
