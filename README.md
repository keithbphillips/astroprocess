# astroprocess

Stacking and processing pipeline for deep-sky astrophotography from a **ZWO Seestar S50** and a
**Canon 5D (Mark III) + telephoto lens**, with separate looks for **nebulae**, **galaxies** and **star
clusters**. One command takes a folder of raw subframes to a finished picture:

```
astro run "/run/media/keith/STAGING/Space/MyWorks/NGC 6888-sub"
```

Output goes to `/run/media/keith/STAGING/Space/Processed/<Target>/` and `astro gallery` writes an
`index.html` there for browsing everything processed.

## What it does

| Stage | Tool | Notes |
|---|---|---|
| Find & group frames | `inventory.py` | Seestar FITS by OBJECT + FILTER (LP and IRCUT kept apart, nights merged); DSLR raws split into sessions by time gap / exposure / ISO |
| Hot pixels, debayer | Siril | No darks on the Seestar: CFA cosmetic correction instead. DSLR: master bias/dark/flat when present, synthetic bias at the black level otherwise |
| Register | Siril two-pass global star alignment | Handles the Seestar's alt-az field rotation and DSLR reframing |
| Frame rejection | Siril | Drops frames with bad FWHM, roundness or background (clouds, trees, wind) |
| Stack | Siril | Weighted by wFWHM; rejection algorithm chosen from frame count |
| Crop | coverage map | Exact per-pixel frame coverage from the registered frames; crops the low-coverage rotation wedges |
| Gradient removal | GraXpert AI (GPU) | Division mode for DSLR without flats (vignetting) |
| Colour calibration | Siril SPCC → star white balance | SPCC uses the Seestar's sensor + LP/IR-cut filter curves against Gaia spectra. PCC is available but was too blue on every test |
| Noise reduction | GraXpert AI denoise (GPU) | On linear data |
| Star separation | StarNet2 | Stars and nebula/galaxy are stretched separately |
| Stretch & finish | numpy (`imageops.py`) | arcsinh + midtones stretch, contrast curve, multi-scale detail, local contrast, masked saturation, green removal, colour-noise cleanup, star blue-fringe fix, star reduction |
| Narrowband | | Seestar LP (dual-band) nebulae also get an **HOO** rendition (Ha→red, OIII→teal) |

### Outputs (per target)

| File | What |
|---|---|
| `<name>.jpg`, `<name>_web.jpg` | Full-size and 2048 px JPEGs |
| `<name>.tif` | 16-bit TIFF for further editing |
| `<name>_hoo*.jpg/.tif` | Narrowband HOO version (LP-filter nebulae) |
| `<name>_vs_seestar.jpg` | Side by side with the Seestar's own on-board stack |
| `<name>_linear.fits` | Calibrated, gradient-free, denoised linear master for hand processing in Siril |
| `<name>_stack.fit` | Raw stack |
| `<name>_report.json`, `<name>_profile.yaml` | Frames used, crop, colour method, timings and the exact settings |

## Usage

```bash
astro scan  <folder>                 # what can be stacked (and DSLR field jumps)
astro run   <folder>                 # stack + process everything found
astro run   <folder> --type galaxy   # force a look (nebula | galaxy | cluster)
astro run   <folder> --filter LP     # only one Seestar filter group
astro batch /run/media/keith/STAGING/Space/MyWorks --min-frames 50   # whole library
astro process "<Target>/Stacked_366_NGC 6888_10.0s_LP_....fit" --filter LP   # a Seestar on-board stack
astro gallery                        # rebuild index.html
astro profile --instrument seestar --type nebula --filter LP   # show merged settings
```

Re-running is cheap. The stack and every linear step are cached in `~/AstroWork/work/<target>/` and keyed on
their settings. Changing only the look (`post.*`) re-renders in a few seconds, and `--restack` forces a fresh
stack.

### Tweaking the look

Every setting lives in `profiles/` (`defaults.yaml` ← `instruments/*.yaml` ← `targets/*.yaml`) and can be
overridden for one run:

```bash
astro run "<folder>" --set post.saturation.amount=1.9 --set post.stretch.target_bg=0.10
astro run "<folder>" --set post.stars.strength=0.6          # fainter stars
astro run "<folder>" --set linear.autocrop.min_coverage=0.8  # cleaner edges, smaller field
astro run "<folder>" --out ~/AstroWork/trial/x               # try something without touching the gallery
```

Useful knobs: `post.stretch.target_bg` (brightness of the sky), `post.stretch.asinh_beta` (core protection,
galaxies use 400), `post.detail.gains`, `post.local_contrast.amount`, `post.saturation.amount`,
`post.stars.{strength,shrink,stretch_factor}`, `post.hoo.{oiii_boost,green_ha}`, `linear.denoise.strength`.

### Canon 5D sessions

A memory card dump works directly. Frames are split into sessions by time gap, exposure and ISO, and `scan`
shows where the star field jumps (re-pointing or reframing):

```bash
astro scan /run/media/keith/EOS_DIGITAL/DCIM
astro run  /run/media/keith/EOS_DIGITAL/DCIM --target "M 31" --session 2024-08-06 --first _MG_7669
```

For better results, organise sessions like this and add calibration frames:

```
M31_2024-08-06/
  lights/   *.CR2
  darks/    same ISO/exposure/temperature, lens cap on
  flats/    same focus and orientation, evenly lit
  biases/   fastest shutter, lens cap on
```

`--target` names the object, which is needed for plate solving (the lens reports no focal length, so a ladder
of focal lengths is tried and the match is remembered). The 5D isn't in Siril's SPCC database, so colour comes
from star white balance.

## Installation (already done on this machine)

| Component | Where |
|---|---|
| Siril 1.4 (`siril-cli`) | system package |
| GraXpert 3.0.2 | `~/.local/opt/graxpert/GraXpert-linux/` |
| cuDNN 8 for GraXpert's GPU path | `~/.local/opt/cudnn8/` (`uv pip install --target ... nvidia-cudnn-cu12==8.9.7.29`) |
| StarNet2 2.6.2 CLI | `~/.local/opt/starnet2/` |
| Siril SPCC sensor/filter database | `~/.local/share/siril/siril-spcc-database` (git clone) |
| Gaia catalogues for offline plate solving + SPCC (~12 GB) | `~/.local/share/siril/catalogs` via `scripts/install_catalogs.py` (resumable) |
| Python env | `.venv`: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt` |
| `astro` command | `~/.local/bin/astro` → `./astro` |

Paths can be overridden in `~/.config/astroprocess/config.yaml` (`output_root`, `work_root`, `tools.*`).

## Not covered

Comets (need comet-nucleus alignment), the Moon, the Sun and planets (lucky imaging from video) are skipped by
`run`/`batch`.
