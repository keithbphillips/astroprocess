#!/usr/bin/env python3
"""Download Siril's local Gaia catalogues (astrometric + SPCC xp-sampled) from Zenodo.

Makes plate solving and SPCC work offline and independent of the ESA Gaia servers.
Resumable: chunks that are already decompressed are skipped. Usage:
    scripts/install_catalogs.py [--dest ~/.local/share/siril/catalogs] [--jobs 4]
Then point Siril at them (the pipeline does this automatically via `set` in each script):
    core.catalogue_gaia_astro = <dest>/siril_cat_healpix8_astro.dat
    core.catalogue_gaia_photo = <dest>/siril_cat1_healpix8_xpsamp
"""
import argparse, bz2, hashlib, json, os, shutil, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

ASTRO_RECORD, SPCC_RECORD = 14692304, 14738271


def files_of(record):
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}") as r:
        return {f["key"]: f for f in json.load(r)["files"]}


def fetch(entry, outdir):
    key = entry["key"]
    final = os.path.join(outdir, key[:-4])  # strip .bz2
    if os.path.exists(final) and os.path.getsize(final) > 0:
        return f"skip {key}"
    tmp = os.path.join(outdir, key + ".part")
    url = entry["links"]["self"]
    h = hashlib.md5()
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    want = entry["checksum"].split(":", 1)[1]
    if h.hexdigest() != want:
        os.remove(tmp)
        raise RuntimeError(f"checksum mismatch for {key}")
    with bz2.open(tmp, "rb") as src, open(final + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 22)
    os.replace(final + ".part", final)
    os.remove(tmp)
    return f"ok   {key}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.expanduser("~/.local/share/siril/catalogs"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-spcc", action="store_true")
    a = ap.parse_args()
    spcc_dir = os.path.join(a.dest, "siril_cat1_healpix8_xpsamp")
    os.makedirs(spcc_dir, exist_ok=True)
    jobs = [(e, a.dest) for k, e in files_of(ASTRO_RECORD).items() if k.endswith(".bz2")]
    if not a.no_spcc:
        jobs += [(e, spcc_dir) for k, e in files_of(SPCC_RECORD).items() if k.endswith(".bz2")]
    failed = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for fut in [ex.submit(fetch, e, d) for e, d in jobs]:
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # keep going, report at the end
                failed += 1
                print(f"FAIL {exc}", flush=True)
    print(f"done: {len(jobs) - failed}/{len(jobs)} files; dest={a.dest}")
    if not failed:  # the pipeline only switches Siril to the local catalogues once this marker exists
        open(os.path.join(a.dest, "INSTALLED"), "w").write("ok\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
