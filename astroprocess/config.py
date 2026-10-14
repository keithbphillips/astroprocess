"""Tool locations, folder defaults and profile loading.

Everything here can be overridden in ~/.config/astroprocess/config.yaml, e.g.

    output_root: /run/media/keith/STAGING/Space/Processed
    work_root: ~/AstroWork/work
    tools:
      graxpert: ~/.local/opt/graxpert/GraXpert-linux/GraXpert
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROFILE_DIR = PROJECT_DIR / "profiles"
USER_CONFIG = Path("~/.config/astroprocess/config.yaml").expanduser()

DEFAULTS = {
    "output_root": "/run/media/keith/STAGING/Space/Processed",
    "work_root": "~/AstroWork/work",
    "keep_work": False,  # delete the (large) registered frames after stacking
    "tools": {
        "siril": "/usr/bin/siril-cli",
        "graxpert": "~/.local/opt/graxpert/GraXpert-linux/GraXpert",
        "starnet": "~/.local/opt/starnet2/starnet2_linux_2.6.2-0242_ORT_x64_cli/starnet2",
        # cuDNN 8 for GraXpert's bundled onnxruntime-gpu (pip wheel nvidia-cudnn-cu12==8.9.7.29)
        "cuda_libs": "~/.local/opt/cudnn8/nvidia/cudnn/lib",
        "gaia_astro": "~/.local/share/siril/catalogs/siril_cat_healpix8_astro.dat",
        "gaia_photo": "~/.local/share/siril/catalogs/siril_cat1_healpix8_xpsamp",
    },
}


def deep_merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config() -> dict:
    cfg = DEFAULTS
    if USER_CONFIG.exists():
        cfg = deep_merge(cfg, yaml.safe_load(USER_CONFIG.read_text()) or {})
    cfg = copy.deepcopy(cfg)
    for k in ("output_root", "work_root"):
        cfg[k] = Path(os.path.expanduser(cfg[k]))
    cfg["tools"] = {k: os.path.expanduser(v) for k, v in cfg["tools"].items()}
    return cfg


def list_profiles(kind: str) -> list[str]:
    return sorted(p.stem for p in (PROFILE_DIR / kind).glob("*.yaml"))


def load_profile(instrument: str, target_type: str, filter_name: str = "",
                 overrides: dict | None = None) -> dict:
    """defaults <- instruments/<instrument> <- targets/<type> <- instrument per_type/per_filter <- overrides."""
    def read(p: Path) -> dict:
        if not p.exists():
            raise SystemExit(f"profile not found: {p}")
        return yaml.safe_load(p.read_text()) or {}

    prof = read(PROFILE_DIR / "defaults.yaml")
    prof = deep_merge(prof, read(PROFILE_DIR / "instruments" / f"{instrument}.yaml"))
    prof = deep_merge(prof, read(PROFILE_DIR / "targets" / f"{target_type}.yaml"))
    # instrument files may carry per-type and per-filter tweaks (e.g. Seestar LP filter + nebula -> HOO)
    per_type = prof.pop("per_type", {})
    per_filter = prof.pop("per_filter", {})
    prof = deep_merge(prof, per_type.get(target_type, {}))
    pf = per_filter.get(filter_name, {})
    prof = deep_merge(prof, {k: v for k, v in pf.items() if k != "per_type"})
    prof = deep_merge(prof, pf.get("per_type", {}).get(target_type, {}))
    prof = deep_merge(prof, overrides or {})
    prof["instrument"], prof["target_type"], prof["filter"] = instrument, target_type, filter_name
    return prof


def parse_set_args(items: list[str] | None) -> dict:
    """Turn ['post.stretch.target_bg=0.12', 'stack.drizzle=true'] into a nested dict."""
    out: dict = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        val = yaml.safe_load(raw)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out
