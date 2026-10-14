"""astro: stack and process Seestar / DSLR astrophotography.

    astro scan   <folder>                      list what can be stacked
    astro run    <folder> [--type galaxy]      stack + process every group found in <folder>
    astro process <stack.fit> --name M_81      process an existing stack (yours or the Seestar's Stacked_*.fit)
    astro batch  <root> --min-frames 50        run every target under <root>
    astro gallery                              index.html of everything processed
    astro profile --instrument seestar --type nebula [--filter LP]   show the merged settings
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import yaml

from . import postprocess
from .config import list_profiles, load_config, load_profile, parse_set_args
from .inventory import Group, field_jumps, scan, seestar_stacks_only, slugify
from .stacking import stack_group
from .targets import guess_type, is_solar_system


def log(msg: str) -> None:
    print(msg, flush=True)


def _instrument_profile(group_instrument: str) -> str:
    return "seestar" if group_instrument == "seestar" else "dslr"


def _solve_hint(group: Group | None, prof: dict, header_has_coords: bool, target: str) -> dict:
    hint = {k: v for k, v in (prof.get("solve") or {}).items() if v}
    if group is not None and group.instrument == "dslr":
        fr = group.lights[0]
        if fr.focal and "focal" not in hint:
            hint["focal"] = round(fr.focal, 1)
        if "pixel_size" not in hint and fr.camera in postprocess.PIXEL_SIZE_UM:
            hint["pixel_size"] = postprocess.PIXEL_SIZE_UM[fr.camera]
    if not header_has_coords:
        try:
            from astropy.coordinates import SkyCoord

            c = SkyCoord.from_name(target)
            hint["coords"] = f"{c.ra.deg:.5f},{c.dec.deg:+.5f}"
        except Exception:
            log(f"  could not resolve '{target}' to coordinates; plate solving may fail (use --target)")
    return hint


def process_stack(stack: Path, name: str, target: str, prof: dict, cfg: dict, work: Path, outdir: Path,
                  meta: dict, group: Group | None = None) -> list[Path]:
    from astropy.io import fits

    t0 = time.time()
    hdr = fits.getheader(stack)
    meta["solve_hint"] = _solve_hint(group, prof, hdr.get("RA") is not None, target)
    meta["filter"] = prof.get("filter", "")
    paths = postprocess.linear_stage(stack, cfg, prof, work, meta, log)
    log("  rendering...")
    results = postprocess.nonlinear_stage(paths, prof)
    jpgs = []
    if group is not None:
        jpgs = [p.with_suffix(".jpg") for p in group.seestar_stacks if p.with_suffix(".jpg").exists()]
    written = postprocess.write_outputs(results, paths, outdir, name, prof, jpgs)
    meta["process_seconds"] = round(time.time() - t0, 1)
    (outdir / f"{name}_report.json").write_text(json.dumps(meta, indent=2, default=str))
    (outdir / f"{name}_profile.yaml").write_text(yaml.safe_dump(prof, sort_keys=False))
    return written


def run_group(group: Group, args, cfg: dict) -> list[Path]:
    ttype = args.type or guess_type(group.target)
    inst = _instrument_profile(group.instrument)
    overrides = parse_set_args(args.set)
    if args.drizzle:
        overrides.setdefault("stack", {})["drizzle"] = True
    if group.instrument == "dslr" and not group.flats:
        # no flats: vignetting is multiplicative, so let the background model divide it out
        bg = overrides.setdefault("linear", {}).setdefault("background", {})
        bg.setdefault("correction", "Division")
    prof = load_profile(inst, ttype, group.filter, overrides)
    label = group.label
    name = f"{group.slug}_{label}_{ttype}"
    work = cfg["work_root"] / f"{group.slug}_{label}"
    outdir = Path(args.out).expanduser() if args.out else cfg["output_root"] / group.slug
    log(f"\n== {group.target} [{inst}, {label}, {ttype}] {len(group.lights)} frames -> {outdir}")
    meta = {"target": group.target, "instrument": inst, "camera": group.camera, "filter": group.filter,
            "type": ttype, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    stack = work / "stack.fit"
    summary_file = work / "stack_summary.json"
    if stack.exists() and summary_file.exists() and not args.restack:
        log("  using cached stack (--restack to redo)")
        meta["stack"] = json.loads(summary_file.read_text())
    else:
        stack, meta["stack"] = stack_group(group, cfg, prof, work, log)
    s = meta["stack"]
    log(f"  stacked {s.get('frames_stacked', '?')}/{s.get('lights')} frames "
        f"({s.get('integration_used_min', '?')} min used)")
    outdir.mkdir(parents=True, exist_ok=True)
    written = process_stack(stack, name, group.target, prof, cfg, work, outdir, meta, group)
    shutil.copy(stack, outdir / f"{name}_stack.fit")
    log(f"  done in {meta['process_seconds']} s: {outdir / (name + '.jpg')}")
    return written


def cmd_scan(args) -> int:
    groups = scan(Path(args.path), args.target)
    if not groups:
        log("no stackable frames found")
    for g in groups:
        if len(g.lights) >= args.min_frames:
            extra = "  (solar system: skipped by run)" if is_solar_system(g.target) else ""
            log(f"{g.describe()}  type={args.type or guess_type(g.target)}{extra}")
            if g.instrument == "dslr":
                jumps = field_jumps(g.lights)
                log(f"    {g.lights[0].path.stem} .. {g.lights[-1].path.stem}"
                    + (f"; field jumps (re-pointed/reframed) at: {', '.join(jumps)}" if jumps else ""))
    only = {k: v for k, v in seestar_stacks_only(Path(args.path)).items()
            if not any(g.target == k for g in groups)}
    if only and args.verbose:
        log("\nTargets with only Seestar on-board stacks (use `astro process`):")
        for k, v in only.items():
            log(f"  {k}: {len(v)} file(s)")
    return 0


def _select(groups: list[Group], args) -> list[Group]:
    out = []
    for g in groups:
        if args.filter and g.filter.upper() != args.filter.upper():
            continue
        if args.session and not g.session.startswith(args.session):
            continue
        if args.first or args.last:
            g.lights = [f for f in g.lights if (not args.first or f.path.stem >= args.first)
                        and (not args.last or f.path.stem <= args.last)]
        if len(g.lights) < args.min_frames or is_solar_system(g.target):
            continue
        out.append(g)
    return out


def cmd_run(args) -> int:
    cfg = load_config()
    groups = _select(scan(Path(args.path), args.target), args)
    if not groups:
        log("nothing to do (see `astro scan`)")
        return 1
    failures = 0
    for g in groups:
        try:
            run_group(g, args, cfg)
        except Exception as e:  # keep going in batch mode
            failures += 1
            log(f"  FAILED: {e}")
            if args.debug:
                traceback.print_exc()
    if not args.out:
        from .gallery import build

        build(cfg["output_root"])
    return 1 if failures else 0


def cmd_process(args) -> int:
    cfg = load_config()
    stack = Path(args.stack).expanduser().resolve()
    target = args.target or args.name or stack.stem
    ttype = args.type or guess_type(target)
    filt = args.filter
    if not filt:
        from astropy.io import fits

        filt = str(fits.getheader(stack).get("FILTER", "")).strip()
    prof = load_profile(args.instrument, ttype, filt, parse_set_args(args.set))
    slug = slugify(args.name or target)
    name = f"{slug}_{ttype}"
    work = cfg["work_root"] / f"{slug}_process"
    outdir = Path(args.out).expanduser() if args.out else cfg["output_root"] / slug
    meta = {"target": target, "instrument": args.instrument, "type": ttype, "input": str(stack)}
    log(f"== processing {stack.name} as {ttype} -> {outdir}")
    process_stack(stack, name, target, prof, cfg, work, outdir, meta)
    log(f"  done: {outdir / (name + '.jpg')}")
    return 0


def cmd_gallery(args) -> int:
    from .gallery import build

    root = Path(args.path).expanduser() if args.path else load_config()["output_root"]
    log(f"wrote {build(root)}")
    return 0


def cmd_profile(args) -> int:
    if not args.instrument:
        log("instruments: " + ", ".join(list_profiles("instruments")))
        log("target types: " + ", ".join(list_profiles("targets")))
        return 0
    prof = load_profile(args.instrument, args.type, args.filter or "", parse_set_args(args.set))
    print(yaml.safe_dump(prof, sort_keys=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="astro", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run=True):
        p.add_argument("--type", choices=list_profiles("targets"), help="override target type detection")
        p.add_argument("--target", help="target name (overrides folder/FITS name)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="override a profile value, e.g. post.saturation.amount=1.8 (repeatable)")
        p.add_argument("--out", help="output folder (default: <output_root>/<target>)")
        p.add_argument("--debug", action="store_true")
        if run:
            p.add_argument("--filter", help="only this filter group (e.g. LP or IRCUT)")
            p.add_argument("--session", help="only DSLR sessions starting with this (e.g. 2024-08-06)")
            p.add_argument("--first", help="first frame to use, by file name (e.g. _MG_7669)")
            p.add_argument("--last", help="last frame to use, by file name (e.g. _MG_7737)")
            p.add_argument("--min-frames", type=int, default=5)
            p.add_argument("--restack", action="store_true", help="ignore the cached stack")
            p.add_argument("--drizzle", action="store_true", help="2x bayer drizzle (needs many frames)")

    p = sub.add_parser("scan", help="list stackable groups")
    p.add_argument("path")
    p.add_argument("--target")
    p.add_argument("--type", choices=list_profiles("targets"))
    p.add_argument("--min-frames", type=int, default=1)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_scan)

    for name in ("run", "batch"):
        p = sub.add_parser(name, help="stack and process" if name == "run" else "run every target under a root")
        p.add_argument("path")
        common(p)
        p.set_defaults(fn=cmd_run)

    p = sub.add_parser("process", help="process an existing stacked FITS")
    p.add_argument("stack")
    p.add_argument("--name", help="output name")
    p.add_argument("--instrument", default="seestar", choices=list_profiles("instruments"))
    p.add_argument("--filter", help="filter name (LP enables HOO for nebulae on the Seestar)")
    common(p, run=False)
    p.set_defaults(fn=cmd_process)

    p = sub.add_parser("gallery", help="write an index.html of all processed images")
    p.add_argument("path", nargs="?", help="output root (default from config)")
    p.set_defaults(fn=cmd_gallery)

    p = sub.add_parser("profile", help="show merged profile settings")
    p.add_argument("--instrument", choices=list_profiles("instruments"))
    p.add_argument("--type", default="nebula", choices=list_profiles("targets"))
    p.add_argument("--filter")
    p.add_argument("--set", action="append")
    p.set_defaults(fn=cmd_profile)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
