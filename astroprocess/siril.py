"""Run Siril scripts headlessly and pull useful numbers out of the log."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


class SirilError(RuntimeError):
    pass


def catalogue_settings(tools: dict) -> list[str]:
    """`set` lines pointing Siril at the local Gaia catalogues once fully installed, else at nothing
    (Siril then queries the online services instead of failing on a missing local chunk)."""
    astro, photo = tools.get("gaia_astro", ""), tools.get("gaia_photo", "")
    complete = (Path(astro).parent / "INSTALLED").exists() if astro else False
    missing = "/nonexistent/siril-catalogue"
    return [f'set "core.catalogue_gaia_astro={astro if complete and os.path.exists(astro) else missing}"',
            f'set "core.catalogue_gaia_photo={photo if complete and os.path.isdir(photo) else missing}"']


def run_script(siril: str, workdir: Path, name: str, commands: list[str], log=print,
               timeout: float | None = None) -> str:
    """Write `commands` to <workdir>/<name>.ssf, run it, return the log text. Raises SirilError on failure
    (including `timeout` seconds elapsing, e.g. an online catalogue that never answers)."""
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / f"{name}.ssf"
    script.write_text("requires 1.3.0\n" + "\n".join(commands) + "\n")
    logfile = workdir / f"{name}.log"
    with open(logfile, "w") as fh:
        try:
            proc = subprocess.run([siril, "-d", str(workdir), "-s", str(script)], stdout=fh,
                                  stderr=subprocess.STDOUT, cwd=workdir, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SirilError(f"Siril script {script.name} timed out after {timeout:.0f} s (see {logfile})")
    text = logfile.read_text(errors="replace")
    if proc.returncode != 0 or "Script execution finished successfully" not in text:
        tail = [l for l in text.splitlines() if l.startswith("log:") and "Filter Pattern" not in l][-15:]
        raise SirilError(f"Siril script {script.name} failed (see {logfile}):\n" + "\n".join(tail))
    return text


def stack_summary(text: str) -> dict:
    """Pull frame counts / noise / rejection out of a stacking log."""
    out: dict = {}
    m = re.search(r"Integration of (\d+) images on (\d+) of the sequence", text)
    if m:
        out["frames_stacked"], out["frames_in_sequence"] = int(m.group(1)), int(m.group(2))
    m = re.findall(r"Background noise value \(channel: #(\d)\): ([\d.e+-]+) \(([\d.e+-]+)\)", text)
    if m:
        out["bg_noise"] = [float(x[2]) for x in m]
    m = re.findall(r"Pixel rejection in channel #\d: ([\d.]+)% - ([\d.]+)%", text)
    if m:
        out["rejection_pct"] = [[float(a), float(b)] for a, b in m]
    m = re.findall(r"Total: (\d+) failed, (\d+) registered", text)
    if m:
        out["registration_failed"], out["registered"] = int(m[-1][0]), int(m[-1][1])
    m = re.findall(r"Total: (\d+) failed, (\d+) exported", text)
    if m:
        out["passed_quality_filter"] = int(m[-1][1])
    m = re.findall(r"choosing image (\d+) as new reference", text)
    if m:
        out["reference_frame"] = int(m[-1])
    filters = re.findall(r"Using (star weighted FWHM|star roundness|image background|number of stars)"
                         r" images filter \((below|above) ([\d.e+-]+)\)", text)
    if filters:
        out["quality_limits"] = {f"{n} {d}": float(v) for n, d, v in filters}
    return out


def platesolve_summary(text: str) -> dict:
    out: dict = {}
    for key, pat in {
        "resolution_arcsec_px": r"Resolution:\s+([\d.]+) arcsec/px",
        "focal_mm": r"Focal length:\s+([\d.]+) mm",
        "rotation_deg": r"Up is ([+-]?[\d.]+) deg",
    }.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"Image center: alpha: ([\d ]+\.?\d*), delta: ([+-][\d ]+\.?\d*)", text)
    if m:
        out["center"] = f"{m.group(1).strip()} {m.group(2).strip()}"
    return out
