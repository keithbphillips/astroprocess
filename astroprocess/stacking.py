"""Calibrate, register and stack a group of frames with Siril.

Seestar (no calibration frames): CFA cosmetic correction (hot/cold pixels) -> debayer -> two-pass global
registration (handles the alt-az field rotation) -> quality filter on wFWHM / roundness / background ->
weighted winsorized-sigma rejection stack.

DSLR: master bias/dark/flat are built from whatever calibration frames exist. Missing bias is replaced by
a synthetic offset at the camera black level. Missing dark falls back to CFA cosmetic correction.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from .inventory import Frame, Group
from .siril import run_script, stack_summary


def _link_frames(frames: list[Frame], dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for i, fr in enumerate(frames, 1):
        os.symlink(fr.path, dest / f"{i:05d}{fr.path.suffix.lower()}")


def _rejection(n: int, prof: dict) -> str:
    """Pick a rejection algorithm that suits the number of frames."""
    if prof.get("rejection"):
        return prof["rejection"]
    if n < 6:
        return "mean"  # too few frames for statistics: plain average
    if n < 15:
        return "rej p 3 3"  # percentile clipping
    if n < 60:
        return "rej w 3 3"  # winsorized sigma
    return "rej g 0.3 0.05"  # generalized extreme studentized deviate (large stacks)


def _dslr_black_level(frame: Frame) -> int | None:
    try:
        import rawpy

        with rawpy.imread(str(frame.path)) as raw:
            bl = raw.black_level_per_channel
            return int(round(sum(bl) / len(bl))) if bl else None
    except Exception:
        return None


def coverage_map(proc: Path, seq: str):
    """Fraction of stacked frames covering each pixel, from the registered frames' empty (zero) areas.

    Field rotation and drift leave wedge-shaped borders seen by only some frames; they are noisier and get
    cropped away later (see imageops.coverage_crop_box)."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from astropy.io import fits

    files = sorted(proc.glob(f"{seq}_*.fit"))
    if not files:
        return None

    def covered(f):
        with fits.open(f, memmap=False) as h:
            d = h[0].data
            return (d[0] if d.ndim == 3 else d) > 0

    acc = None
    with ThreadPoolExecutor(16) as ex:
        for m in ex.map(covered, files):
            acc = m.astype(np.uint16) if acc is None else acc + m
    return (acc / len(files)).astype(np.float32)


def stack_group(group: Group, cfg: dict, prof: dict, workdir: Path, log=print) -> tuple[Path, dict]:
    """Stack `group` inside `workdir`; returns (path to 32-bit stack FITS, summary dict)."""
    sp = prof["stack"]
    t0 = time.time()
    workdir.mkdir(parents=True, exist_ok=True)
    proc = workdir / "proc"
    if proc.exists():
        shutil.rmtree(proc)
    lights = group.lights
    _link_frames(lights, workdir / "lights")
    seestar = group.instrument == "seestar"
    ext = "fit"
    cmds = ["set32bits" if sp.get("intermediate_32bit") else "set16bits", f"setext {ext}"]

    # ---- calibration masters (DSLR only) ----
    masters: dict[str, str] = {}
    convert = "link" if seestar else "convert"
    for kind, frames in (("bias", group.biases), ("dark", group.darks), ("flat", group.flats)):
        if not frames or seestar:
            continue
        _link_frames(frames, workdir / f"{kind}s")
        cmds += [f"cd {workdir / (kind + 's')}", f"{convert} {kind} -out={proc}", f"cd {proc}"]
        if kind == "flat":
            if "bias" in masters:
                cmds.append(f"calibrate flat -bias={masters['bias']}")
                cmds.append("stack pp_flat rej 3 3 -norm=mul -out=master_flat")
            else:
                cmds.append("stack flat rej 3 3 -norm=mul -out=master_flat")
        else:
            cmds.append(f"stack {kind} rej 3 3 -nonorm -out=master_{kind}")
        masters[kind] = f"{proc}/master_{kind}"

    cmds += [f"cd {workdir / 'lights'}", f"{convert} light -out={proc}", f"cd {proc}"]
    seq = "light"
    drizzle = bool(sp.get("drizzle"))
    if seestar or "dark" not in masters:
        cmds.append(f"seqfind_cosme_cfa {seq} {sp['cosme_sigma'][0]} {sp['cosme_sigma'][1]} -prefix=cc_")
        seq = "cc_" + seq
    cal = []
    if "dark" in masters:
        cal += [f"-dark={masters['dark']}", "-cc=dark 3 3"]
    elif not seestar:
        if "bias" in masters:
            cal.append(f"-bias={masters['bias']}")
        else:
            bl = _dslr_black_level(lights[0])
            if bl:
                cal.append(f'-bias="={bl}"')  # synthetic offset at the camera black level
    if "flat" in masters:
        cal += [f"-flat={masters['flat']}", "-cfa", "-equalize_cfa"]
    if not drizzle:
        cal.append("-debayer")
    cmds.append(f"calibrate {seq} {' '.join(cal)} -prefix=pp_")
    seq = "pp_" + seq

    # ---- registration + quality filtering ----
    reg = [f"register {seq} -2pass"]
    if sp.get("max_stars"):
        reg[0] += f" -maxstars={sp['max_stars']}"
    if sp.get("distortion") and not seestar:
        reg[0] += f" -disto={sp['distortion']}"
    cmds += reg
    flt = " ".join(f"-filter-{k}={v}" for k, v in sp.get("filters", {}).items())
    apply = f"seqapplyreg {seq} {flt} -framing={sp.get('framing', 'cog')}"
    if drizzle:
        apply += f" -drizzle -scale={sp.get('drizzle_scale', 2)} -pixfrac={sp.get('drizzle_pixfrac', 1.0)} -kernel=square"
    cmds.append(apply)
    seq = "r_" + seq

    out = workdir / "stack"
    rej = _rejection(len(lights), sp)
    cmds.append(f"stack {seq} {rej} -norm=addscale -output_norm -rgb_equal -weight={sp.get('weight', 'wfwhm')} -32b -out={out}")

    log(f"  stacking {len(lights)} frames ({rej}{', drizzle' if drizzle else ''})")
    text = run_script(cfg["tools"]["siril"], workdir, "stack", cmds, log)
    summary = stack_summary(text)
    summary.update({
        "lights": len(lights), "darks": len(group.darks), "flats": len(group.flats), "biases": len(group.biases),
        "nights": group.nights, "exposure_s": sorted({f.exptime for f in lights}), "rejection": rej,
        "integration_min": round(sum(f.exptime for f in lights) / 60, 1), "drizzle": drizzle,
        "stack_seconds": round(time.time() - t0, 1),
    })
    if "frames_stacked" in summary:
        used = summary["frames_stacked"]
        kept = [int(re.search(r"(\d+)\.fit$", f.name).group(1)) for f in proc.glob(f"{seq}_*.fit")]
        if len({f.exptime for f in lights}) > 1 and len(kept) == used and max(kept, default=0) <= len(lights):
            # mixed exposures: the registered files keep the input frame numbers, so sum the real exposures
            summary["integration_used_min"] = round(sum(lights[i - 1].exptime for i in kept) / 60, 1)
        else:
            summary["integration_used_min"] = round(used * (lights[0].exptime or 0) / 60, 1)
    cov = coverage_map(proc, seq)
    if cov is not None:
        from astropy.io import fits

        fits.PrimaryHDU(cov).writeto(workdir / "coverage.fits", overwrite=True)
    (workdir / "stack_summary.json").write_text(json.dumps(summary, indent=2))
    if not cfg.get("keep_work"):
        shutil.rmtree(proc, ignore_errors=True)
    return Path(str(out) + ".fit"), summary
