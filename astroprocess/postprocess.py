"""Turn a linear stack into a finished picture.

Linear stage (each result is cached in <work>/linear/ keyed on its settings, so re-running with a different
look only redoes the fast non-linear stage):
    autocrop -> background extraction (GraXpert AI) -> plate solve + colour calibration (Siril SPCC, then PCC,
    then simple neutralisation) -> AI denoise (GraXpert) -> star separation (StarNet2)
Non-linear stage (numpy, profile driven):
    stretch starless (arcsinh + midtones) -> contrast curve -> multi-scale detail -> local contrast ->
    masked saturation -> green removal -> colour noise cleanup -> stretch + shrink stars -> screen stars back
    [-> optional HOO narrowband rendition for dual-band filter data]
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import imageops as ops
from .siril import SirilError, catalogue_settings, platesolve_summary, run_script

# pixel pitch (µm) of cameras the pipeline may meet; used for plate solving DSLR frames
PIXEL_SIZE_UM = {
    "Seestar S50": 2.9, "Seestar S30": 2.9,
    "Canon EOS 5D": 8.2, "Canon EOS 5D Mark II": 6.4, "Canon EOS 5D Mark III": 6.25,
    "Canon EOS 5D Mark IV": 5.36, "Canon EOS 5DS": 4.14, "Canon EOS 5DS R": 4.14,
    "Canon EOS 6D": 6.54, "Canon EOS 6D Mark II": 5.67, "Canon EOS R5": 4.39, "Canon EOS R6": 6.56,
}


class Step:
    """Cache helper: a step's output is reused when its inputs and settings are unchanged."""

    def __init__(self, folder: Path, name: str, params: dict, inputs: list[Path]):
        self.folder = folder
        key = json.dumps(params, sort_keys=True, default=str)
        key += "".join(f"{p}:{p.stat().st_mtime_ns}" for p in inputs if p and p.exists())
        self.digest = hashlib.sha1(key.encode()).hexdigest()[:10]
        self.name = name
        self.marker = folder / f"{name}.done"

    def path(self, suffix: str = ".fits") -> Path:
        return self.folder / f"{self.name}{suffix}"

    def cached(self) -> bool:
        return self.marker.exists() and self.marker.read_text() == self.digest

    def done(self) -> None:
        self.marker.write_text(self.digest)


def _env_with_cuda(cfg: dict) -> dict:
    env = dict(os.environ)
    libs = cfg["tools"].get("cuda_libs", "")
    if libs and os.path.isdir(libs):
        env["LD_LIBRARY_PATH"] = libs + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def _run(cmd: list[str], log_path: Path, env: dict | None = None) -> None:
    with open(log_path, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-10:]
        raise RuntimeError(f"{Path(cmd[0]).name} failed (see {log_path}):\n" + "\n".join(tail))


def graxpert(cfg: dict, cmd: str, src: Path, dst_noext: Path, args: list[str]) -> Path:
    exe = cfg["tools"]["graxpert"]
    _run([exe, "-cli", "-cmd", cmd, "-gpu", "true", "-output", str(dst_noext), *args, str(src)],
         dst_noext.with_suffix(".log"), env=_env_with_cuda(cfg))
    out = dst_noext.with_suffix(".fits")
    if not out.exists():
        raise RuntimeError(f"GraXpert produced no output for {cmd}")
    return out


# ------------------------------------------------------------------------------------- linear stage
def linear_stage(stack: Path, cfg: dict, prof: dict, work: Path, meta: dict, log=print) -> dict:
    """Run the linear steps; returns paths {'linear', 'starless', 'stars'} and fills `meta`."""
    lp = prof["linear"]
    folder = work / "linear"
    folder.mkdir(parents=True, exist_ok=True)

    # 1. crop the low-coverage borders left by field rotation / drift
    st = Step(folder, "01_crop", {**lp.get("autocrop", {}), "manual": lp.get("crop")},
              [stack, stack.parent / "coverage.fits"])
    if not st.cached():
        img, hdr = ops.load_fits(stack)
        ac = lp.get("autocrop", {})
        cov_file = stack.parent / "coverage.fits"
        if ac.get("enabled", True):
            if cov_file.exists():  # exact frame coverage written by the stacking step
                from astropy.io import fits

                cov = fits.getdata(cov_file).astype(np.float32)
                y0, y1, x0, x1 = ops.coverage_crop_box(cov, ac.get("min_coverage", 0.5))
            else:  # e.g. a Seestar on-board stack: estimate from noise
                y0, y1, x0, x1 = ops.autocrop_box(img, tol=ac.get("tolerance", 1.35),
                                                 max_frac=ac.get("max_fraction", 0.25))
            img = img[y0:y1, x0:x1]
        manual = lp.get("crop")
        if manual:  # [left, top, right, bottom] as fractions of the auto-cropped frame
            h, w = img.shape[:2]
            l, t, r, b = manual
            img = img[int(t * h):int(b * h), int(l * w):int(r * w)]
            meta["crop"] = {"from": [int(hdr.get("NAXIS1", 0)), int(hdr.get("NAXIS2", 0))],
                            "box_xywh": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]}
        ops.save_fits(st.path(), img, hdr)
        (folder / "crop.json").write_text(json.dumps(meta.get("crop", {})))
        st.done()
    else:
        meta["crop"] = json.loads((folder / "crop.json").read_text() or "{}")
    cur = st.path()
    log(f"  crop: {meta.get('crop', {}).get('box_xywh', 'none')}")

    # 2. background / gradient removal
    bgp = lp["background"]
    st = Step(folder, "02_background", bgp, [cur])
    if not st.cached():
        t = time.time()
        method = bgp.get("method", "graxpert")
        if method == "graxpert":
            out = graxpert(cfg, "background-extraction", cur, st.path("").with_suffix(""),
                           ["-correction", bgp.get("correction", "Subtraction"),
                            "-smoothing", str(bgp.get("smoothing", 0.1))])
            if out != st.path():
                shutil.move(out, st.path())
        elif method == "siril":
            run_script(cfg["tools"]["siril"], folder, "background",
                       [f"load {cur}", f"subsky -rbf -samples={bgp.get('samples', 20)} "
                        f"-tolerance={bgp.get('tolerance', 1.0)} -smooth={bgp.get('smooth', 0.5)}",
                        f"save {st.path().with_suffix('')}"])
            Path(str(st.path().with_suffix("")) + ".fit").replace(st.path())
        else:
            shutil.copy(cur, st.path())
        meta.setdefault("timings", {})["background_s"] = round(time.time() - t, 1)
        st.done()
    cur = st.path()
    log(f"  background: {bgp.get('method')}")

    # 3. plate solve + colour calibration
    cp = lp["color"]
    st = Step(folder, "03_color", {**cp, "hint": meta.get("solve_hint")}, [cur])
    if not st.cached():
        meta["color"] = _color_calibrate(cur, st.path(), cfg, prof, folder, meta, log)
        (folder / "color.json").write_text(json.dumps({k: meta.get(k) for k in ("color", "platesolve")}))
        st.done()
    else:
        saved = json.loads((folder / "color.json").read_text())
        meta.update({k: v for k, v in saved.items() if v is not None})
    cur = st.path()
    log(f"  colour: {meta.get('color')}")

    # 4. AI denoise (on linear data)
    dp = lp["denoise"]
    st = Step(folder, "04_denoise", dp, [cur])
    if not st.cached():
        t = time.time()
        if dp.get("strength", 0) > 0:
            out = graxpert(cfg, "denoising", cur, st.path("").with_suffix(""),
                           ["-strength", str(dp["strength"])])
            if out != st.path():
                shutil.move(out, st.path())
        else:
            shutil.copy(cur, st.path())
        meta.setdefault("timings", {})["denoise_s"] = round(time.time() - t, 1)
        st.done()
    cur = st.path()
    log(f"  denoise: strength {dp.get('strength', 0)}")
    paths = {"linear": cur}

    # 5. star separation
    sp = lp["starnet"]
    if sp.get("enabled", True):
        st = Step(folder, "05_starnet", sp, [cur])
        if not st.cached():
            t = time.time()
            # StarNet's --linear mode clips at 1.0: rescale so the brightest pixel is 1
            img, hdr = ops.load_fits(cur)
            peak = float(np.percentile(img, 99.999))
            src = folder / "05_in.fits"
            ops.save_fits(src, np.clip(img / max(peak, 1e-6), 0, 1), hdr)
            cmd = [cfg["tools"]["starnet"], "-i", str(src), "-o", str(folder / "05_starless.fits"),
                   "-n", str(folder / "05_stars.fits"), "--linear", "-q", "-s", str(sp.get("stride", 256))]
            if sp.get("upsample"):
                cmd.append("-u")
            _run(cmd, folder / "05_starnet.log")
            (folder / "05_scale.json").write_text(json.dumps({"peak": peak}))
            src.unlink()
            meta.setdefault("timings", {})["starnet_s"] = round(time.time() - t, 1)
            st.done()
        paths["starless"] = folder / "05_starless.fits"
        paths["stars"] = folder / "05_stars.fits"
        paths["star_scale"] = json.loads((folder / "05_scale.json").read_text())["peak"]
        log("  stars separated (StarNet2)")
    return paths


FOCAL_LADDER = [50, 85, 100, 135, 180, 200, 250, 300, 400, 500, 600, 800]  # lenses that report no focal length
WCS_PREFIXES = ("CTYPE", "CRPIX", "CRVAL", "CD1_", "CD2_", "CDELT", "CROTA", "PC1_", "PC2_", "A_", "B_", "AP_",
                "BP_", "LONPOLE", "LATPOLE", "EQUINOX", "RADESYS", "PLTSOLVD")


def _plate_solve(src: Path, cfg: dict, folder: Path, meta: dict, log) -> tuple[Path | None, bool]:
    """Solve `src` with Siril; returns (solved FITS, whether Siril mirrored the rows) or (None, False).

    Without a known focal length (manual lenses report 0) a ladder of common focal lengths is tried; the
    one that works is remembered in <work>/linear/solve.json for the next run."""
    hint = meta.get("solve_hint") or {}
    memo = folder / "solve.json"
    focals = [hint["focal"]] if hint.get("focal") else FOCAL_LADDER
    if memo.exists():
        known = json.loads(memo.read_text()).get("focal")
        focals = [known] + [f for f in focals if f != known]
    if not hint.get("pixel_size") and not hint.get("focal"):
        log("  plate solve skipped: unknown pixel size and focal length")
        return None, False
    out = folder / "solved"
    for focal in focals:
        cmd = "platesolve"
        if hint.get("coords"):
            cmd += f" {hint['coords']} -force"
        cmd += f" -focal={focal}"
        if hint.get("pixel_size"):
            cmd += f" -pixelsize={hint['pixel_size']}"
        try:
            text = run_script(cfg["tools"]["siril"], folder, "platesolve",
                              catalogue_settings(cfg["tools"]) + [f"load {src}", cmd, f"save {out}"],
                              timeout=180)
        except SirilError:
            if len(focals) > 1:
                log(f"  plate solve: no match at {focal} mm")
            continue
        meta["platesolve"] = platesolve_summary(text)
        memo.write_text(json.dumps({"focal": focal, **meta["platesolve"]}))
        log(f"  plate solved: {meta['platesolve'].get('resolution_arcsec_px')}\"/px, "
            f"focal {meta['platesolve'].get('focal_mm')} mm")
        return Path(str(out) + ".fit"), "Flipping image" in text
    return None, False


def _color_calibrate(src: Path, dst: Path, cfg: dict, prof: dict, folder: Path, meta: dict, log) -> str:
    """Plate solve, then SPCC (sensor + filter curves vs Gaia spectra) / PCC / star white balance, in the
    order the profile lists them. Returns the method that worked."""
    cp = prof["linear"]["color"]
    methods = cp.get("methods", ["spcc", "pcc", "starwb"])
    solved, flipped = (None, False)
    if any(m in ("spcc", "pcc") for m in methods):
        solved, flipped = _plate_solve(src, cfg, folder, meta, log)
    for method in methods:
        if method in ("spcc", "pcc") and solved is None:
            continue
        if method == "spcc" and not cp.get("spcc"):
            continue
        if method == "starwb":
            img, hdr = ops.load_fits(src)
            img, k, n = ops.star_white_balance(img)
            ops.save_fits(dst, img, hdr)
            log(f"  star white balance from {n} stars: k={[round(v, 3) for v in k]}")
            return f"starwb ({n} stars)"
        cmd = "pcc"
        if method == "spcc":
            sp = cp["spcc"]
            cmd = "spcc"
            if sp.get("sensor"):
                cmd += f' "-oscsensor={sp["sensor"]}"'
            filt = sp.get("filter_by_name", {}).get(meta.get("filter", ""), sp.get("filter"))
            if filt:
                cmd += f' "-oscfilter={filt}"'
            if sp.get("lpf"):
                cmd += f' "-osclpf={sp["lpf"]}"'
            cmd += f' "-whiteref={sp.get("whiteref", "Average Spiral Galaxy")}"'
        try:
            run_script(cfg["tools"]["siril"], folder, f"color_{method}",
                       catalogue_settings(cfg["tools"]) + [f"load {solved}", cmd, f"save {dst.with_suffix('')}"],
                       timeout=180)
        except SirilError as e:
            log(f"  {method} failed: {str(e).splitlines()[-1]}")
            continue
        Path(str(dst.with_suffix("")) + ".fit").replace(dst)
        if flipped:
            # Siril mirrored the rows to match the astrometric solution; restore the original framing
            img, hdr = ops.load_fits(dst)
            for k in [k for k in hdr if k.startswith(WCS_PREFIXES)]:
                del hdr[k]
            ops.save_fits(dst, img[::-1], hdr)
        return method
    shutil.copy(src, dst)
    return "none (background neutralisation only)"


# --------------------------------------------------------------------------------- non-linear stage
def _stretch_fn(ref: np.ndarray, s: dict):
    return ops.make_stretch(ref, target_bg=s["target_bg"], shadow_sigma=s["shadow_sigma"],
                            asinh_beta=s.get("asinh_beta", 0), color_preserve=s.get("color_preserve", 0.6))


def _finish_starless(img: np.ndarray, pp: dict) -> np.ndarray:
    out = _stretch_fn(img, pp["stretch"])(img)
    out = ops.curves_contrast(out, pp.get("contrast", 0))
    mask = ops.signal_mask(out, soft=pp.get("mask_softness", 0.12))
    d = pp.get("detail", {})
    out = ops.multiscale_detail(out, d.get("scales", []), d.get("gains", []), mask)
    lc = pp.get("local_contrast", {})
    out = ops.local_contrast(out, lc.get("amount", 0), lc.get("tiles", 8), lc.get("clip", 0.01), mask)
    out = ops.scnr_green(out, pp.get("scnr", 0))
    sat = pp.get("saturation", {})
    out = ops.saturate(out, sat.get("amount", 1.0), mask, sat.get("background", 1.0))
    nz = pp.get("noise", {})
    out = ops.chroma_denoise(out, nz.get("chroma_sigma", 0), mask)
    out = ops.lum_denoise(out, nz.get("lum_sigma", 0), mask)
    return out


def _finish_stars(starless: np.ndarray, full: np.ndarray, pp: dict) -> np.ndarray:
    """Star layer: stretch the full and starless images with the same (milder) curve and take the
    screen-difference, so stars look like a normal stretch would render them, just smaller."""
    s = pp["stars"]
    st = dict(pp["stretch"])
    st["target_bg"] = st["target_bg"] * s.get("stretch_factor", 0.7)
    st["asinh_beta"] = s.get("asinh_beta", st.get("asinh_beta", 0))
    st["color_preserve"] = s.get("color_preserve", 0.9)
    f = _stretch_fn(starless, st)
    out = ops.unscreen(f(full), f(starless))
    out = ops.tame_blue_fringe(out, s.get("blue_fringe_limit"))
    out = ops.scnr_green(out, pp.get("scnr", 0))  # no star is green
    out = ops.saturate(out, s.get("saturation", 1.3))
    out = ops.limit_chroma(out, s.get("max_chroma"))
    out = ops.desaturate_halos(out, s.get("halo_desaturate", 0.0))
    out = ops.shrink_stars(out, s.get("shrink", 0.0))
    return np.clip(out * s.get("strength", 1.0), 0, 1).astype(np.float32)


def nonlinear_stage(paths: dict, prof: dict) -> dict[str, np.ndarray]:
    """Returns {'main': image, optional 'hoo': image} (display range, FITS row order)."""
    pp = prof["post"]
    results = {}
    if "starless" in paths:
        starless, _ = ops.load_fits(paths["starless"])
        stars_u, _ = ops.load_fits(paths["stars"])
        full = starless + stars_u * (1 - starless)  # StarNet's documented reconstruction (linear)
        stars = _finish_stars(starless, full, pp)
        main = _finish_starless(starless, pp)
        main = ops.screen(main, stars)
    else:
        lin, _ = ops.load_fits(paths["linear"])
        main = _finish_starless(lin, pp)
        stars = None
    f = pp.get("final", {})
    main = ops.curves_contrast(main, f.get("contrast", 0))
    if f.get("black_target"):
        main = ops.set_black(main, f["black_target"])
    results["main"] = main

    nb = pp.get("hoo", {})
    if nb.get("enabled") and "starless" in paths:
        starless, _ = ops.load_fits(paths["starless"])
        starless, _, _ = ops.neutralize_background(starless)
        ha, oiii = ops.hoo(starless)
        s = pp["stretch"]
        # linked stretch: one curve (from the Ha statistics) for both channels, so a target with little
        # OIII stays red instead of turning grey from stretched OIII noise; OIII gets a linear boost
        ha_bg, ha_sig = ops.bg_stats(ha)
        o_bg = ops.bg_stats(oiii)[0]
        oiii = (oiii - o_bg) * nb.get("oiii_boost", 1.0) + ha_bg
        ref = np.repeat(ha[..., None], 3, -1)
        linked = ops.make_stretch(ref, target_bg=s["target_bg"], shadow_sigma=s["shadow_sigma"],
                                  asinh_beta=nb.get("asinh_beta", 5), color_preserve=0)
        ha_s = linked(ref)[..., 0]
        o_s = linked(np.repeat(oiii[..., None], 3, -1))[..., 0]
        g_mix = nb.get("green_ha", 0.3)
        img = np.stack([ha_s, np.clip(g_mix * ha_s + (1 - g_mix) * o_s, 0, 1), np.clip(o_s, 0, 1)], -1)
        mask = ops.signal_mask(img, soft=pp.get("mask_softness", 0.12))
        d = pp.get("detail", {})
        img = ops.multiscale_detail(img, d.get("scales", []), d.get("gains", []), mask)
        img = ops.saturate(img, nb.get("saturation", 1.3), mask, 1.0)
        img = ops.chroma_denoise(img, pp.get("noise", {}).get("chroma_sigma", 0), mask)
        if stars is not None:
            img = ops.screen(img, ops.saturate(stars, nb.get("star_saturation", 0.6)))
        img = ops.curves_contrast(img, f.get("contrast", 0))
        results["hoo"] = img
    return results


def write_outputs(results: dict[str, np.ndarray], paths: dict, outdir: Path, name: str, prof: dict,
                  seestar_jpgs: list[Path] | None = None) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, img in results.items():
        stem = name if key == "main" else f"{name}_{key}"
        ops.save_tiff16(outdir / f"{stem}.tif", img)
        ops.save_jpeg(outdir / f"{stem}.jpg", img, quality=95)
        ops.save_jpeg(outdir / f"{stem}_web.jpg", img, quality=88, max_dim=2048)
        written += [outdir / f"{stem}.tif", outdir / f"{stem}.jpg", outdir / f"{stem}_web.jpg"]
    # linear, calibrated, denoised master for hand processing in Siril/PixInsight
    shutil.copy(paths["linear"], outdir / f"{name}_linear.fits")
    written.append(outdir / f"{name}_linear.fits")
    if seestar_jpgs:
        cmp = outdir / f"{name}_vs_seestar.jpg"
        _comparison(seestar_jpgs[-1], outdir / f"{name}.jpg", cmp)
        written.append(cmp)
    return written


def _comparison(seestar_jpg: Path, ours_jpg: Path, dst: Path, height: int = 1200) -> None:
    from PIL import Image, ImageDraw

    a, b = Image.open(seestar_jpg).convert("RGB"), Image.open(ours_jpg).convert("RGB")
    a = a.resize((int(a.width * height / a.height), height), Image.LANCZOS)
    b = b.resize((int(b.width * height / b.height), height), Image.LANCZOS)
    canvas = Image.new("RGB", (a.width + b.width + 20, height + 50), (20, 20, 20))
    canvas.paste(a, (0, 50))
    canvas.paste(b, (a.width + 20, 50))
    d = ImageDraw.Draw(canvas)
    d.text((10, 15), "Seestar on-board stack", fill=(220, 220, 220))
    d.text((a.width + 30, 15), "astroprocess", fill=(220, 220, 220))
    canvas.save(dst, quality=90)
