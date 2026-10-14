"""Find stackable frames and group them.

Two input shapes are understood:

* Seestar exports: ``<Target>-sub/Light_<Target>_<exp>_<FILTER>_<date>.fit`` (raw CFA subframes) next to
  ``<Target>/Stacked_*.fit`` (the Seestar's own live stack, kept for comparison or as a fallback).
* DSLR sessions (Canon 5D etc.): a folder of raw files, optionally split into calibration sub-folders
  ``lights/ darks/ flats/ biases/`` (names are matched loosely: "Light", "Darks", "bias", "offset", "flat"...).
  Without sub-folders every raw file is treated as a light.

Frames are grouped by (instrument, target, filter). Several nights of the same target and filter are
stacked together; Seestar LP (dual-band) and IRCUT (broadband) frames are kept apart.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from astropy.io import fits

RAW_EXT = {".cr2", ".cr3", ".crw", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"}
FITS_EXT = {".fit", ".fits", ".fts"}

KIND_WORDS = {
    "dark": ("dark",),
    "flat": ("flat",),
    "bias": ("bias", "offset"),
    "light": ("light",),
}


@dataclass
class Frame:
    path: Path
    instrument: str  # 'seestar' | 'dslr'
    kind: str = "light"  # light | dark | flat | bias
    target: str = ""
    filter: str = ""
    exptime: float = 0.0
    date: str = ""  # YYYY-MM-DD (local night not resolved; good enough for grouping display)
    camera: str = ""
    iso: int | None = None
    focal: float | None = None
    ra: float | None = None
    dec: float | None = None
    when: datetime | None = None


@dataclass
class Group:
    instrument: str
    target: str
    filter: str
    session: str = ""  # DSLR: one night/exposure run, e.g. 2024-08-06_2251_30s
    lights: list[Frame] = field(default_factory=list)
    darks: list[Frame] = field(default_factory=list)
    flats: list[Frame] = field(default_factory=list)
    biases: list[Frame] = field(default_factory=list)
    seestar_stacks: list[Path] = field(default_factory=list)

    @property
    def integration_s(self) -> float:
        return sum(f.exptime for f in self.lights)

    @property
    def nights(self) -> list[str]:
        return sorted({f.date for f in self.lights if f.date})

    @property
    def camera(self) -> str:
        return self.lights[0].camera if self.lights else ""

    @property
    def slug(self) -> str:
        return slugify(self.target)

    @property
    def label(self) -> str:
        return self.filter or self.session or self.instrument

    def describe(self) -> str:
        cal = []
        for name in ("darks", "flats", "biases"):
            n = len(getattr(self, name))
            if n:
                cal.append(f"{n} {name}")
        mins = self.integration_s / 60
        return (f"{self.target:<28} {self.instrument:<8} {self.label:<6} {len(self.lights):>5} lights "
                f"{mins:7.1f} min  nights={','.join(self.nights) or '-'}"
                + (f"  [{', '.join(cal)}]" if cal else ""))


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return s or "target"


def _kind_from_path(p: Path, root: Path) -> str:
    for part in reversed(p.relative_to(root).parts[:-1]):
        low = part.lower()
        for kind, words in KIND_WORDS.items():
            if any(w in low for w in words):
                return kind
    return "light"


def _seestar_frame(p: Path) -> Frame | None:
    try:
        h = fits.getheader(p)
    except Exception:
        return None
    creator = f"{h.get('CREATOR', '')} {h.get('INSTRUME', '')} {h.get('TELESCOP', '')}"
    if "seestar" not in creator.lower() or h.get("NAXIS") != 2:
        return None
    target = str(h.get("OBJECT", "")).strip() or p.parent.name.removesuffix("-sub")
    return Frame(
        path=p, instrument="seestar", kind=str(h.get("IMAGETYP", "Light")).strip().lower() or "light",
        target=target, filter=str(h.get("FILTER", "")).strip(),
        exptime=float(h.get("EXPTIME", h.get("EXPOSURE", 0)) or 0),
        date=str(h.get("DATE-OBS", ""))[:10], camera=str(h.get("INSTRUME", "Seestar")).strip(),
        focal=float(h.get("FOCALLEN", 250)), ra=h.get("RA"), dec=h.get("DEC"),
    )


def _dslr_frame(p: Path, root: Path, target: str) -> Frame:
    import exifread

    f = Frame(path=p, instrument="dslr", kind=_kind_from_path(p, root), target=target)
    try:
        with open(p, "rb") as fh:
            tags = exifread.process_file(fh, details=False, stop_tag="EXIF FocalLength")
    except Exception:
        tags = {}

    def num(key):
        v = tags.get(key)
        if v is None:
            return None
        try:
            r = v.values[0]
            return float(r.num) / float(r.den) if hasattr(r, "num") else float(r)
        except Exception:
            return None

    f.camera = str(tags.get("Image Model", "")).strip()
    f.exptime = num("EXIF ExposureTime") or 0.0
    iso = num("EXIF ISOSpeedRatings")
    f.iso = int(iso) if iso else None
    f.focal = num("EXIF FocalLength")
    dt = str(tags.get("EXIF DateTimeOriginal", tags.get("Image DateTime", "")))
    f.date = dt[:10].replace(":", "-") if dt else ""
    try:
        f.when = datetime.strptime(dt[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        pass
    return f


GENERIC_FOLDER = re.compile(r"^(\d{3}[A-Z0-9_]{5}|DCIM|EOS_DIGITAL|lights?)$", re.I)


def split_sessions(frames: list[Frame], gap_min: float = 45, exp_ratio: float = 1.5) -> list[list[Frame]]:
    """Split DSLR lights into runs: a new session starts after a time gap or a big exposure or ISO change.
    An odd test exposure in the middle of a run doesn't split it: frames rejoin the latest run they match.
    Re-pointing without a pause is NOT split here (reframing the same target is common and image
    correlation can't tell it from a new target); `scan` reports field jumps and `run --first/--last`
    selects a frame range."""
    frames = sorted(frames, key=lambda f: (f.when or datetime.min, f.path.name))
    sessions: list[list[Frame]] = []

    def fits_run(run: list[Frame], f: Frame) -> bool:
        ref = sorted(x.exptime for x in run)[len(run) // 2] or 1
        last = run[-1]
        gap = (f.when - last.when).total_seconds() / 60 if f.when and last.when else 0
        isos = sorted(x.iso or 0 for x in run)
        iso_ref, iso = isos[len(isos) // 2] or 1, f.iso or isos[len(isos) // 2] or 1
        return (gap <= gap_min and max(f.exptime, ref) / max(min(f.exptime, ref), 1e-3) <= exp_ratio
                and max(iso, iso_ref) / min(iso, iso_ref) <= 2.5)  # ISO 800+1000 mix fine, 25600 test not

    for f in frames:
        for run in reversed(sessions):  # the time-gap test keeps this local
            if fits_run(run, f):
                run.append(f)
                break
        else:
            sessions.append([f])
    return sessions


def field_jumps(frames: list[Frame]) -> list[str]:
    """Frames where the star field jumps (re-pointed / reframed), for `scan` to report."""
    from . import pointing

    jumps, prev = [], None
    for f in frames:
        cur = pointing.signature(f.path)
        if prev is not None and pointing.similarity(prev, cur) < pointing.SAME_FIELD:
            jumps.append(f.path.stem)
        prev = cur
    return jumps


def session_name(frames: list[Frame]) -> str:
    f = frames[0]
    exp = sorted(x.exptime for x in frames)[len(frames) // 2]
    stamp = f.when.strftime("%Y-%m-%d_%H%M") if f.when else (f.date or "session")
    return f"{stamp}_{exp:.0f}s"


def _dslr_target_name(folder: Path, root: Path) -> str:
    """The session folder name, skipping generic calibration folder names."""
    for part in reversed(folder.relative_to(root.parent).parts):
        low = part.lower()
        if not any(w in low for words in KIND_WORDS.values() for w in words) and low not in ("raw", "raws"):
            return part
    return root.name


def scan(root: Path, target_override: str | None = None) -> list[Group]:
    """Scan a folder tree and return stackable groups (largest first)."""
    root = Path(root).expanduser().resolve()
    groups: dict[tuple, Group] = {}
    seestar_stacks: dict[str, list[Path]] = defaultdict(list)
    dslr_cal: dict[Path, list[Frame]] = defaultdict(list)
    dslr_lights: dict[tuple, list[Frame]] = defaultdict(list)

    def group_for(fr: Frame) -> Group:
        key = (fr.instrument, fr.target, fr.filter)
        if key not in groups:
            groups[key] = Group(fr.instrument, fr.target, fr.filter)
        return groups[key]

    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    for p in files:
        ext = p.suffix.lower()
        if ext in FITS_EXT:
            if p.name.startswith("Stacked_"):
                tgt = p.parent.name.removesuffix("-sub")
                seestar_stacks[target_override or tgt].append(p)
                continue
            fr = _seestar_frame(p)
            if fr is None:
                continue
            if target_override:
                fr.target = target_override
            if fr.kind == "light":
                group_for(fr).lights.append(fr)
        elif ext in RAW_EXT:
            tgt = target_override or _dslr_target_name(p.parent, root)
            if GENERIC_FOLDER.match(tgt):
                tgt = "DSLR"  # camera card folder: name with --target
            fr = _dslr_frame(p, root, tgt)
            if fr.kind == "light":
                dslr_lights[(tgt, _session_root(p, root))].append(fr)
            else:
                dslr_cal[_session_root(p, root)].append(fr)

    for (tgt, sroot), frames in dslr_lights.items():
        for run in split_sessions(frames):
            g = Group("dslr", tgt, "", session=session_name(run), lights=run)
            groups[("dslr", tgt, g.session, str(sroot))] = g

    # attach DSLR calibration frames found in the same session folder as the lights
    for g in groups.values():
        if g.instrument != "dslr":
            continue
        for sr in {_session_root(f.path, root) for f in g.lights}:
            for c in dslr_cal.get(sr, []):
                getattr(g, {"dark": "darks", "flat": "flats", "bias": "biases"}[c.kind]).append(c)
    for g in groups.values():
        found = list(seestar_stacks.get(g.target, []))
        if not found and g.instrument == "seestar":
            # run on "<Target>-sub": the Seestar's own stacks live in the sibling "<Target>" folder
            for fr in g.lights[:1]:
                sib = fr.path.parent.with_name(fr.path.parent.name.removesuffix("-sub"))
                if sib != fr.path.parent and sib.is_dir():
                    found = [p for p in sib.glob("Stacked_*.fit") if not p.name.startswith(".")]
        g.seestar_stacks = sorted(found)
    return sorted(groups.values(), key=lambda g: -len(g.lights))


def _session_root(p: Path, root: Path) -> Path:
    """Folder that holds lights/darks/flats siblings (or the file's own folder)."""
    parent = p.parent
    low = parent.name.lower()
    if parent != root and any(w in low for words in KIND_WORDS.values() for w in words):
        return parent.parent
    return parent


def seestar_stacks_only(root: Path) -> dict[str, list[Path]]:
    """Targets that only have Seestar's own Stacked_*.fit (no subframes)."""
    out: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(Path(root).rglob("Stacked_*.fit")):
        if not p.name.startswith("."):
            out[p.parent.name.removesuffix("-sub")].append(p)
    return out
