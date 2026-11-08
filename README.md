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
5. [Interface en ligne de commande](#interface-en-ligne-de-commande)
6. [Organisation du code](#organisation-du-code)
7. [Licence](#licence)
8. [Contribuer](#contribuer)

---

## Installation

```bash
pip install git+https://github.com/EOQual/eoqual-phaseretrieval.git
```

Toutes les dépendances (eoqual-core, eoqual-viz, numpy, scipy, scikit-image, plotly,
loguru, commentjson, imageio, rich) s'installent automatiquement — voir
`pyproject.toml`.

## Démarrage rapide

```python
from eoqual.phaseretrieval import psf_to_wfe

coeffs_nm, phase_fit_nm, diagnostics = psf_to_wfe(
    "psf.tiff", "mask.tiff",
    wavelength=550e-9, focal_length=4.0, pupil_diameter=0.5, pixel_pitch=4e-6,
    method="phase_diversity",
    diversity_path="psf_defocus.tiff", defocus=275e-9,
)
```

`method=` sélectionne l'algorithme (`"gerchberg_saxton"`, `"optimization"`
ou `"phase_diversity"`) derrière ce même point d'entrée — voir
`eoqual.phaseretrieval.runner.list_methods()` pour l'afficher en console,
et [REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md) §4-8 pour savoir
laquelle choisir selon le cas d'usage.

Masque, carte de WFE haut ordre (`known_ho_wfe_path`) et PSF peuvent
chacun venir d'une résolution native différente — y compris une PSF
suréchantillonnée par super-résolution (`oversampling=`), ou un fichier
masque/carte plus grand que l'ouverture elle-même (marge de zéros autour,
`mask_native_aperture_px=`/`ho_native_aperture_px=`) : voir
[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md) §12.

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

## Interface en ligne de commande

Les trois méthodes partagent le même point d'entrée en ligne de commande
(`python -m eoqual.phaseretrieval.pipeline ...`), sélectionné par
`--method`.

```text
usage: -c [-h] --psf PSF --mask MASK --output OUTPUT
          [--method {gerchberg_saxton,optimization,phase_diversity}]
          [--wavelength WAVELENGTH] [--focal_length FOCAL_LENGTH]
          [--pupil_diameter PUPIL_DIAMETER] [--pixel_pitch PIXEL_PITCH]
          [--oversampling OVERSAMPLING] [--zernike_order ZERNIKE_ORDER]
          [--mask_no_binarize]
          [--mask_native_aperture_px MASK_NATIVE_APERTURE_PX]
          [--ho_native_aperture_px HO_NATIVE_APERTURE_PX]
          [--mask_native_aperture_center ROW COL]
          [--ho_native_aperture_center ROW COL]
          [--aperture_threshold APERTURE_THRESHOLD] [--orthogonal_basis]
          [--max_iter MAX_ITER] [--beta BETA] [--seed SEED]
          [--fit_piston_tilt] [--diversity DIVERSITY] [--defocus DEFOCUS]
          [--roi_radius ROI_RADIUS] [--loss LOSS] [--poisson_weighting]
          [--sign_prior_index SIGN_PRIOR_INDEX]
          [--sign_prior_sign SIGN_PRIOR_SIGN] [--known_ho_wfe KNOWN_HO_WFE]
          [--config_json CONFIG_JSON]

Recover Zernike (Noll) coefficients from one or two PSF using a binary pupil
mask.

options:
  -h, --help            show this help message and exit
  --psf PSF             Path to the PSF image
  --mask MASK           Path to the binary pupil mask image
  --output OUTPUT       Répertoire de sortie des résultats.
  --method {gerchberg_saxton,optimization,phase_diversity}
                        Phase-retrieval algorithm (default gerchberg_saxton).
  --wavelength WAVELENGTH
                        Wavelength in metres (default 550 nm)
  --focal_length FOCAL_LENGTH
                        Focal length in metres
  --pupil_diameter PUPIL_DIAMETER
                        Pupil diameter in metres
  --pixel_pitch PIXEL_PITCH
                        Native detector pixel pitch in metres
  --oversampling OVERSAMPLING
                        Oversampling factor of --psf (and --diversity)
                        relative to native detector sampling, e.g. 4 for a
                        super-resolved PSF (default 1: already at native
                        resolution).
  --zernike_order ZERNIKE_ORDER
                        Maximum Zernike Noll order
  --mask_no_binarize    Keep the mask as a graded [0,1] transmission after
                        resampling instead of thresholding it to strict {0,1}
                        (default: binarize).
  --mask_native_aperture_px MASK_NATIVE_APERTURE_PX
                        Aperture's own pixel span within --mask, if it has a
                        zero margin around it rather than filling the file
                        edge to edge (default: auto-detected from the pixel
                        content, see --aperture_threshold).
  --ho_native_aperture_px HO_NATIVE_APERTURE_PX
                        Same as --mask_native_aperture_px, for --known_ho_wfe.
  --mask_native_aperture_center ROW COL
                        Pixel coordinates of the aperture's centre within
                        --mask, only if --mask_native_aperture_px is given and
                        the aperture is NOT centred in the file (default:
                        assume it is centred).
  --ho_native_aperture_center ROW COL
                        Same as --mask_native_aperture_center, for
                        --known_ho_wfe / --ho_native_aperture_px.
  --aperture_threshold APERTURE_THRESHOLD
                        Fraction of the mask's/HO map's own maximum value
                        above which a pixel counts as part of the aperture,
                        for auto-detecting its margin. Ignored wherever
                        *_native_aperture_px is given.
  --orthogonal_basis    Fit on a Gram-Schmidt basis re-orthogonalised on the
                        actual pupil mask instead of the standard Noll basis
                        (recommended for an obstructed pupil: central
                        obstruction, spiders).
  --max_iter MAX_ITER   [gerchberg_saxton] Number of GS iterations
  --beta BETA           [gerchberg_saxton] Relaxation factor
  --seed SEED           [gerchberg_saxton] RNG seed (reproducible runs)
  --fit_piston_tilt     [optimization] Also fit piston/tip/tilt (Z1-Z3).
  --diversity DIVERSITY
                        [phase_diversity] Path to the second (defocused) PSF.
  --defocus DEFOCUS     [phase_diversity] Known defocus in metres
  --roi_radius ROI_RADIUS
                        [phase_diversity] ROI half-width in pixels
  --loss LOSS           [phase_diversity] Robust loss function
  --poisson_weighting   [phase_diversity] Weight residuals by shot-noise
                        (default on).
  --sign_prior_index SIGN_PRIOR_INDEX
                        [gerchberg_saxton, optimization] Noll index of an
                        even-m mode with a known sign (e.g. 4 for defocus).
  --sign_prior_sign SIGN_PRIOR_SIGN
                        [gerchberg_saxton, optimization] Expected sign (+1 or
                        -1) of --sign_prior_index.
  --known_ho_wfe KNOWN_HO_WFE
                        [optimization, phase_diversity] Path to a known, fixed
                        high-order WFE map [nm], at its own native resolution,
                        added to the model without being optimised. Ignored
                        for gerchberg_saxton.
  --config_json CONFIG_JSON
                        Optionnel: fichier JSON de config
```

(Sortie de `python -m eoqual.phaseretrieval.pipeline --help`, figée ici
pour référence — peut légèrement dériver de la version installée ; voir
`CHANGELOG.md` pour l'historique des options ajoutées.)

**Non exposé en CLI, API Python uniquement** : `zernike_priors`/
`zernike_bounds` (a priori individuels par mode, §6.5bis) et
`aperture_px` côté simulation (§12.5) — seulement `psf_to_wfe(...)`/
`OpticalConfig`/`PhaseDiversityConfig`/les fonctions de
`eoqual.phaseretrieval.simulation`.

Exemples d'invocation :

```bash
# Gerchberg-Saxton
python -m eoqual.phaseretrieval.pipeline --method gerchberg_saxton \
    --psf data/psf_simu.tiff --mask data/mask_simu.tiff --output output \
    --zernike_order 15 --max_iter 300 --seed 2

# Optimisation directe
python -m eoqual.phaseretrieval.pipeline --method optimization \
    --psf data/psf_simu.tiff --mask data/mask_simu.tiff --output output \
    --zernike_order 15

# Diversité de phase (2 PSF)
python -m eoqual.phaseretrieval.pipeline --method phase_diversity \
    --psf data/psf_simu.tiff --diversity data/psf_diversity_simu.tiff \
    --mask data/mask_simu.tiff --output output \
    --zernike_order 15 --defocus 275e-9 --roi_radius 32
```

## Organisation du code

```
src/eoqual/phaseretrieval/
    io_utils.py                    # chargement image/masque, enveloppes Plotly spécifiques
    optics.py                      # base de Zernike, phase -> PSF (modèle direct)
    zernike_fit.py                 # décomposition d'une carte de phase (Noll) + project_orthogonal_to_standard
    zernike_priors.py              # a priori individuels par mode (§6.5bis)
    pipeline.py                    # psf_to_wfe(method=...) + CLI
    config.py                      # registre METHODS_CONFIGS
    runner.py                      # list_methods()
    zernike_basis_orthogonal.py    # ZernikeGramSchmidt + resolve_zernike_basis (§10)
    pupil.py                       # pupilles disque/annulaire/araignées, polar_grid (§12.5)
    diagnostics.py                 # Strehl, WFE (Maréchal), FTM (axe fe, fc, coupes), ellipticité ;
                                    # FWHM/énergie encerclée délégués à eoqual-core (§9)
    backends/
        gerchberg_saxton.py        # Gerchberg-Saxton                  (1 PSF)
        optimization.py            # optimisation directe              (1 PSF)
        phase_diversity.py         # diversité de phase modale         (2 PSF)
    simulation/                    # module de simulation (voir "Simulation" plus haut)
        psf_simulator.py
        phase_diversity_simulator.py
        noise.py
examples/
    psf2zernike_demo.ipynb         # démonstration + validation
    simu_psf_mask.ipynb            # génération de data/*_simu.tiff (Bayer multi-bande)
```

`optics.py` et `zernike_fit.py` sont les briques partagées : les deux
méthodes par optimisation (`backends/optimization.py`,
`backends/phase_diversity.py`) s'appuient sur `optics.phase_from_coefficients`
et `optics.psf_from_phase` pour évaluer leur modèle direct à chaque
itération du solveur ; `backends/gerchberg_saxton.py` n'en a pas besoin
(il travaille directement sur la phase pixel à pixel) mais passe par
`zernike_fit.py` *après coup* pour produire des coefficients comparables
aux deux autres méthodes. Les fonctions graphiques génériques (heatmap
2D, comparaison de courbes/coupes) vivent dans
[eoqual-viz](https://github.com/EOQual/eoqual-viz), pas ici — voir
[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md) §6.1.

## Licence

MIT — voir [license.txt](license.txt). Toutes les méthodes sont des
réimplémentations propres à partir des articles cités (aucun portage de
code tiers) — voir [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
pour l'audit des dépendances externes.

## Contribuer

Bug, méthode manquante, cas d'usage non couvert : voir
[CONTRIBUTING.md](CONTRIBUTING.md).

## Credits

- Voir [AUTHORS.md](AUTHORS.md)
