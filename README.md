<p align="center">
  <img src="resources/eoqual_rounded.png" alt="EOQual" width="280">
</p>

# eoqual-phaseretrieval

PSF → WFE → coefficients de Zernike : récupération de phase et diversité de
phase, pour l'observation de la Terre — voir
**[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)** pour le détail
théorique et d'implémentation de chaque méthode, et
**[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** pour l'audit des
licences des dépendances tierces.

## Sommaire

1. [Installation](#installation)
2. [Démarrage rapide](#démarrage-rapide)
3. [Catalogue des méthodes](#catalogue-des-méthodes)
4. [Simulation](#simulation)
5. [Licence](#licence)
6. [Contribuer](#contribuer)

---

## Installation

```bash
pip install git+https://github.com/EOQual/eoqual-phaseretrieval.git
```

Toutes les dépendances (numpy, scipy, scikit-image, poppy, plotly,
loguru, commentjson, imageio, rich) s'installent automatiquement — voir
`pyproject.toml`.

## Démarrage rapide

```python
from eoqual.phaseretrieval import psf_to_wfe

coeffs_nm, phase_fit_nm, diagnostics = psf_to_wfe(
    "psf.tiff", "mask.tiff",
    wavelength=550e-9, focal_length=4.0, pupil_diameter=0.5, detector_size=4e-6,
    method="phase_diversity",
    diversity_path="psf_defocus.tiff", defocus=275e-9,
)
```

`method=` sélectionne l'algorithme (`"gerchberg_saxton"`, `"optimization"`
ou `"phase_diversity"`) derrière ce même point d'entrée — voir
`eoqual.phaseretrieval.runner.list_methods()` pour l'afficher en console,
et [REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md) §4-8 pour savoir
laquelle choisir selon le cas d'usage.

Ligne de commande équivalente :

```bash
python -m eoqual.phaseretrieval.pipeline --method phase_diversity \
    --psf psf.tiff --mask mask.tiff --output out/ \
    --diversity psf_defocus.tiff --defocus 275e-9
```

## Catalogue des méthodes

| `method=` | Technique | PSF requises | Référence |
|---|---|---|---|
| `"gerchberg_saxton"` | Gerchberg-Saxton (alternance plan pupille/image) | 1 | Gerchberg & Saxton (1972) |
| `"optimization"` | Ajustement direct des coefficients de Zernike par moindres carrés non linéaires | 1 | Branch, Coleman & Li (1999) ; Fienup (1993) |
| `"phase_diversity"` | Diversité de phase modale (focus + défocus connu), pondération Poisson et pertes robustes optionnelles | 2 | Gonsalves (1982) ; Paxman, Schulz & Fienup (1992) |

Les trois méthodes sont **interchangeables** derrière `psf_to_wfe(method=...)`
et partagent la même décomposition finale en coefficients de Zernike
(ordre de Noll). Voir [REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md) §2-3
pour les ambiguïtés inhérentes à une reconstruction à partir d'une seule
PSF (piston, tip-tilt, signe) et pourquoi la diversité de phase les lève.

**Ajouter une 4ᵉ méthode** : voir
[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md) §6.5 et
[CONTRIBUTING.md](CONTRIBUTING.md) — l'architecture (`backends/<méthode>.py`
+ dispatch dans `pipeline.py`) est conçue pour ça, y compris pour des
approches qui ne partagent pas le même modèle direct (ex. approches
linéarisées OTF).

**WFE haut ordre connue** : `known_ho_wfe_nm` (`OpticalConfig`,
`PhaseDiversityConfig`) ou `psf_to_wfe(..., known_ho_wfe_path=...)` /
`--known_ho_wfe` en CLI ajoutent une carte de WFE mesurée indépendamment
(interférogramme, as-built, calibration sol) au modèle direct, sans jamais
l'ajuster — évite qu'elle ne biaise silencieusement les coefficients bas
ordre estimés. Voir [REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md) §11.3.
Non applicable à `"gerchberg_saxton"` (ignoré, avec avertissement).

## Simulation

`eoqual.phaseretrieval.simulation` est un sous-paquet **indépendant** des
méthodes de récupération de phase : il génère des cas de test à vérité de
Zernike connue (PSF simple, ou paire focus/défocus bruitée), pour valider
n'importe laquelle des trois méthodes de bout en bout — sans dépendre
d'une acquisition réelle.

```python
from eoqual.phaseretrieval.simulation import (
    simulate_psf_from_zernike,
    simulate_phase_diversity_pair,
    PhaseDiversitySimulationConfig,
    DetectorNoiseConfig,
)

# Une PSF connue (défocus + astigmatisme), pour tester GS/optimisation
sim = simulate_psf_from_zernike({4: 55.0, 5: 27.5}, wavelength=550e-9, npix=256, zernike_order=15)
sim["psf"], sim["coeffs_nm"]  # coeffs_nm = vérité terrain

# Une paire focus/défocus bruitée (Poisson + lecture), pour tester la diversité de phase
cfg = PhaseDiversitySimulationConfig(
    defocus=275e-9,
    noise=DetectorNoiseConfig(photon_count=1e6, read_noise_e=5.0, seed=0),
)
pair = simulate_phase_diversity_pair({4: 55.0, 5: 27.5}, cfg)
pair["psf_focus"], pair["psf_defocus"], pair["coeffs_nm"]
```

`orthogonal_basis=True` (voir [REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)
§10) est aussi disponible côté simulation, pour tester de façon cohérente
un ajustement fait avec `orthogonal_basis=True` sur une pupille obstruée —
et `zernike_fit.project_orthogonal_to_standard` reconvertit après coup des
coefficients ajustés en base Gram-Schmidt vers la base standard (Z4 =
défocus, …) pour l'interprétation.

`known_ho_wfe_nm` (même clé côté `simulate_psf_from_zernike`/
`PhaseDiversitySimulationConfig`) permet aussi de simuler une carte de WFE
haut ordre connue, pour valider `known_ho_wfe_nm` côté récupération de
phase de bout en bout (`tests/test_known_ho_wfe.py`).

Voir `tests/test_simulation_roundtrip.py`/`tests/test_simulation_orthogonal.py`/
`tests/test_known_ho_wfe.py` pour des exemples complets (simulation →
récupération de phase → comparaison à la vérité terrain) et
[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md) §6.4/§9 pour la
discussion du modèle de bruit.

## Licence

MIT — voir [license.txt](license.txt). Toutes les méthodes sont des
réimplémentations propres à partir des articles cités (aucun portage de
code tiers) — voir [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
pour l'audit des seules dépendances externes (poppy, BSD-3-Clause).

## Contribuer

Bug, méthode manquante, cas d'usage non couvert : voir
[CONTRIBUTING.md](CONTRIBUTING.md).

## Credits

- [poppy](https://poppy-optics.readthedocs.io/) (STScI) — base de Zernike
- Voir [AUTHORS.md](AUTHORS.md)
