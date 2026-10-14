"""Detect when a DSLR was re-pointed between frames (reported by `astro scan`).

Each raw's embedded preview is reduced to a small star map; consecutive frames of the same field give a
strong phase-correlation peak (0.5-0.96 measured on 5D Mark III data, tracked mount) while a re-pointed or
reframed field gives ~0.04-0.1. Reframing the same target looks the same as a new target, so this only
informs; it does not split sessions.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

CACHE = Path("~/.cache/astroprocess/pointing").expanduser()
SAME_FIELD = 0.15
SHAPE = (240, 360)


def signature(path: Path) -> np.ndarray | None:
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()
    cached = CACHE / f"{key}.npy"
    if cached.exists():
        return np.load(cached).astype(np.float32)
    try:
        import rawpy
        from PIL import Image
        from scipy import ndimage as ndi

        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format != rawpy.ThumbFormat.JPEG:
            return None
        im = Image.open(io.BytesIO(thumb.data))
        im.draft("L", (SHAPE[1] * 2, SHAPE[0] * 2))
        a = np.asarray(im.convert("L").resize((SHAPE[1], SHAPE[0])), np.float32)
    except Exception:
        return None
    hp = a - ndi.median_filter(a, 7)
    hp = np.clip(hp - 2 * hp.std(), 0, None)  # keep star peaks only
    hp -= hp.mean()
    sig = hp / (np.linalg.norm(hp) + 1e-9)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cached, sig.astype(np.float16))
    return sig


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Phase-correlation peak (0..1) between two star maps; 1.0 if either is unavailable (don't split)."""
    if a is None or b is None:
        return 1.0
    f = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    f /= np.abs(f) + 1e-9
    return float(np.abs(np.fft.ifft2(f)).max())
