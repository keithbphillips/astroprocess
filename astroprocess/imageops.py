"""Numerical image operations used after stacking.

Images are float32 arrays shaped (H, W, 3) in [0, 1]. Row 0 is the top of the picture: Siril writes its
stacks top-down, and the pipeline undoes the flip Siril applies when plate solving (see postprocess), so
arrays are written to TIFF/JPEG as they are.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi

EPS = 1e-7


# ----------------------------------------------------------------------------------------------- I/O
def load_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()
    if data.ndim == 2:
        data = np.repeat(data[None], 3, axis=0)
    img = np.moveaxis(data, 0, -1)
    if img.max() > 1.5:  # integer-scaled FITS
        img = img / (65535.0 if img.max() > 255 else 255.0)
    return np.ascontiguousarray(img), header


def save_fits(path: Path, img: np.ndarray, header: fits.Header | None = None) -> None:
    hdr = fits.Header() if header is None else header.copy()
    for k in ("BZERO", "BSCALE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"):
        hdr.pop(k, None)
    fits.PrimaryHDU(np.moveaxis(img.astype(np.float32), -1, 0), header=hdr).writeto(path, overwrite=True)


def to_display(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 1)


def save_tiff16(path: Path, img: np.ndarray) -> None:
    import tifffile

    tifffile.imwrite(path, (to_display(img) * 65535 + 0.5).astype(np.uint16), photometric="rgb",
                     compression="zlib")


def save_jpeg(path: Path, img: np.ndarray, quality: int = 92, max_dim: int | None = None) -> None:
    from PIL import Image

    im = Image.fromarray((to_display(img) * 255 + 0.5).astype(np.uint8))
    if max_dim and max(im.size) > max_dim:
        im.thumbnail((max_dim, max_dim), Image.LANCZOS)
    im.save(path, quality=quality, optimize=True, progressive=True)


# ----------------------------------------------------------------------------------------- statistics
def _sample(x: np.ndarray, n: int = 400_000) -> np.ndarray:
    flat = x.reshape(-1)
    if flat.size <= n:
        return flat
    step = flat.size // n
    return flat[::step]


def bg_stats(ch: np.ndarray, iters: int = 3) -> tuple[float, float]:
    """Sigma-clipped background median and noise sigma (MAD based) of a single channel."""
    s = _sample(ch)
    s = s[np.isfinite(s)]
    med, sig = float(np.median(s)), float(np.median(np.abs(s - np.median(s))) * 1.4826)
    for _ in range(iters):
        keep = s[(s > med - 3 * sig) & (s < med + 2.5 * sig)]
        if keep.size < 1000:
            break
        med = float(np.median(keep))
        sig = float(np.median(np.abs(keep - med)) * 1.4826)
    return med, max(sig, 1e-7)


def luminance(img: np.ndarray) -> np.ndarray:
    return img.mean(axis=-1)


# --------------------------------------------------------------------------------------------- crop
def coverage_crop_box(cov: np.ndarray, min_cov: float = 0.75, margin: int = 6, step: int = 4):
    """Largest axis-aligned rectangle whose pixels are all covered by >= `min_cov` of the stacked frames.

    Classic maximal-rectangle-in-a-binary-matrix (row histograms + monotonic stack) on a `step`-downsampled
    mask. Returns (y0, y1, x0, x1) in full-resolution pixels.
    """
    good = cov >= min_cov * float(cov.max())
    h, w = good.shape
    hs, ws = h // step, w // step
    small = good[: hs * step, : ws * step].reshape(hs, step, ws, step).all(axis=(1, 3))
    heights = np.zeros(ws, np.int32)
    best = (0, 0, 0, 0, 0)  # area, top, bottom(excl), left, right(excl)
    for r in range(hs):
        heights = np.where(small[r], heights + 1, 0)
        stack: list[int] = []
        for c in range(ws + 1):
            hc = heights[c] if c < ws else 0
            while stack and heights[stack[-1]] >= hc:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = int(heights[top]) * (c - left)
                if area > best[0]:
                    best = (area, r - int(heights[top]) + 1, r + 1, left, c)
            stack.append(c)
    if best[0] == 0:
        return 0, h, 0, w
    _, t, b, l, rr = best
    return t * step + margin, b * step - margin, l * step + margin, rr * step - margin


def autocrop_box(img: np.ndarray, max_frac: float = 0.25, tol: float = 1.35, margin: int = 4):
    """Find the region where stack noise is close to the centre's.

    Field rotation and drift mean the borders of a stack were covered by fewer frames: they are noisier
    (and sometimes empty). Walk in from each edge until local noise drops to `tol` x the central noise.
    Returns (y0, y1, x0, x1).
    """
    L = luminance(img)
    hp = L - ndi.uniform_filter(L, 9)
    h, w = L.shape
    cy, cx = slice(int(h * 0.35), int(h * 0.65)), slice(int(w * 0.35), int(w * 0.65))
    ref = float(np.median(np.abs(hp[cy, cx]))) + 1e-9

    def noise(a: np.ndarray, axis: int) -> np.ndarray:
        return np.median(np.abs(a), axis=axis) / ref

    rows = noise(hp[:, int(w * 0.2):int(w * 0.8)], 1)
    cols = noise(hp[int(h * 0.2):int(h * 0.8), :], 0)
    zero_r = np.mean(L[:, int(w * 0.2):int(w * 0.8)] <= 0, axis=1)
    zero_c = np.mean(L[int(h * 0.2):int(h * 0.8), :] <= 0, axis=0)

    def edge(prof, zeros, n, reverse):
        rng = range(n - 1, n - 1 - int(n * max_frac), -1) if reverse else range(int(n * max_frac))
        for i in rng:
            lo, hi = max(0, i - 3), min(n, i + 4)
            if np.median(prof[lo:hi]) < tol and zeros[i] < 0.01:
                return i
        return (n - 1 - int(n * max_frac)) if reverse else int(n * max_frac)

    y0, y1 = edge(rows, zero_r, h, False), edge(rows, zero_r, h, True)
    x0, x1 = edge(cols, zero_c, w, False), edge(cols, zero_c, w, True)
    return y0 + margin, y1 - margin + 1, x0 + margin, x1 - margin + 1


# ------------------------------------------------------------------------------------------- stretch
def mtf(m: float, x: np.ndarray) -> np.ndarray:
    """Midtones transfer function (PixInsight/Siril definition)."""
    return ((m - 1.0) * x) / (((2.0 * m - 1.0) * x) - m)


def mtf_for(x: float, y: float) -> float:
    """Midtones balance m so that mtf(m, x) == y."""
    x = float(np.clip(x, 1e-6, 0.999999))
    return x * (y - 1.0) / (2.0 * x * y - x - y)


def neutralize_background(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Equalise per-channel backgrounds; returns (img, common bg, noise sigma)."""
    stats = [bg_stats(img[..., c]) for c in range(3)]
    bgs = np.array([s[0] for s in stats], np.float32)
    sig = float(np.mean([s[1] for s in stats]))
    common = float(bgs.mean())
    return img - bgs + common, common, sig


def make_stretch(ref: np.ndarray, target_bg: float = 0.12, shadow_sigma: float = 2.8, asinh_beta: float = 0.0,
                 color_preserve: float = 0.6):
    """Build an autostretch from the statistics of `ref` (a linear RGB image); returns f(img) -> img.

    1. per-channel background offsets are equalised (neutral sky)
    2. black point at background - `shadow_sigma` * noise
    3. optional arcsinh pre-stretch (`asinh_beta` > 0): compresses highlights so galaxy cores and bright
       nebula centres keep their detail and colour
    4. midtones transfer so the background lands on `target_bg`
    Colour: a blend of a hue-preserving (luminance ratio) stretch and a per-channel stretch;
    `color_preserve` = 1 keeps star/nebula colour ratios exactly, 0 behaves like a classic linked stretch.
    The same function can be applied to several images (e.g. starless and full) so they stay consistent.
    """
    stats = [bg_stats(ref[..., c]) for c in range(3)]
    bgs = np.array([s[0] for s in stats], np.float32)
    bg = float(bgs.mean())
    sig = float(np.mean([s[1] for s in stats]))
    offsets = bgs - bg
    bp = max(bg - shadow_sigma * sig, 0.0)
    bgn = (bg - bp) / (1.0 - bp)

    def curve(v):
        if asinh_beta > 0:
            v = np.arcsinh(asinh_beta * v) / np.arcsinh(asinh_beta)
        return v

    m = mtf_for(float(curve(np.float32(bgn))), target_bg)

    def f(v):
        return np.clip(mtf(m, curve(v)), 0, 1)

    def apply(img: np.ndarray) -> np.ndarray:
        x = np.clip((img - offsets - bp) / (1.0 - bp), 0.0, 1.0)
        per_channel = f(x)
        if color_preserve <= 0:
            return per_channel.astype(np.float32)
        L = luminance(x)
        ratio = f(L) / np.maximum(L, EPS)
        hue = x * ratio[..., None]
        peak = hue.max(axis=-1, keepdims=True)
        hue = np.where(peak > 1.0, hue / np.maximum(peak, EPS), hue)  # keep hue when a channel would clip
        return (color_preserve * hue + (1 - color_preserve) * per_channel).astype(np.float32)

    return apply


def stretch(img: np.ndarray, **kw) -> np.ndarray:
    return make_stretch(img, **kw)(img)


def unscreen(full: np.ndarray, starless: np.ndarray) -> np.ndarray:
    """Star layer s such that screen(starless, s) == full (both already stretched)."""
    return np.clip(1 - (1 - full) / np.maximum(1 - starless, 1e-4), 0, 1).astype(np.float32)


def curves_contrast(img: np.ndarray, amount: float, pivot: float | None = None) -> np.ndarray:
    """Gentle S-curve around `pivot` (default: the background level) applied to luminance."""
    if amount <= 0:
        return img
    L = luminance(img)
    if pivot is None:
        pivot = bg_stats(L)[0]
    # piecewise power curve: darker below pivot, brighter above, continuous at pivot
    lo = pivot * (np.clip(L, 0, pivot) / max(pivot, EPS)) ** (1 + amount)
    hi = 1 - (1 - pivot) * ((1 - np.clip(L, pivot, 1)) / max(1 - pivot, EPS)) ** (1 + amount)
    Ln = np.where(L < pivot, lo, hi)
    return np.clip(img * (Ln / np.maximum(L, EPS))[..., None], 0, 1).astype(np.float32)


def set_black(img: np.ndarray, target_bg: float) -> np.ndarray:
    """Linear rescale so the background median sits at `target_bg` (white point kept)."""
    bg = bg_stats(luminance(img))[0]
    if bg <= target_bg:
        return img
    lo = (bg - target_bg) / (1 - target_bg)
    return np.clip((img - lo) / (1 - lo), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------------------------- colour
def star_white_balance(img: np.ndarray, max_stars: int = 400) -> tuple[np.ndarray, list[float], int]:
    """Catalogue-free colour calibration: make the median star white (linear data, background neutralised).

    Stars are local maxima of a small-scale high-pass image well above the noise; their fluxes are measured
    in 5x5 apertures on the high-pass (so smooth nebulosity under them doesn't count). Returns
    (image, [kR, kG, kB], number of stars used).
    """
    img, _, _ = neutralize_background(img)
    hp = np.stack([img[..., c] - ndi.median_filter(img[..., c], size=15) for c in range(3)], -1)
    L = hp.mean(-1)
    _, sig = bg_stats(L)
    peaks = (L == ndi.maximum_filter(L, size=9)) & (L > 25 * sig) & (img.max(-1) < 0.95)
    ys, xs = np.nonzero(peaks)
    if ys.size < 20:
        return img, [1.0, 1.0, 1.0], int(ys.size)
    order = np.argsort(L[ys, xs])[::-1][:max_stars]
    ys, xs = ys[order], xs[order]
    box = np.ones((5, 5), np.float32)
    flux = np.stack([ndi.convolve(hp[..., c], box, mode="nearest")[ys, xs] for c in range(3)], -1)
    flux = flux[(flux > 0).all(-1)]
    if len(flux) < 20:
        return img, [1.0, 1.0, 1.0], len(flux)
    k = np.array([np.median(flux[:, 1] / flux[:, 0]), 1.0, np.median(flux[:, 1] / flux[:, 2])], np.float32)
    k = k / k.max()  # never push data above its current range
    bg = np.array([bg_stats(img[..., c])[0] for c in range(3)], np.float32)
    out = (img - bg) * k + bg.mean()
    # clipped star cores carry no colour information: keep them neutral instead of tinting them
    clipped = ndi.binary_dilation(img.max(-1) >= 0.95 * float(img.max()), iterations=1)
    out[clipped] = out[clipped].max(-1, keepdims=True)
    return out.astype(np.float32), [float(v) for v in k], len(flux)


def scnr_green(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Average-neutral SCNR: pull green down to the mean of red and blue where it exceeds it."""
    if amount <= 0:
        return img
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    g2 = np.minimum(g, (r + b) / 2)
    out = img.copy()
    out[..., 1] = g * (1 - amount) + g2 * amount
    return out


def signal_mask(img: np.ndarray, low: float | None = None, soft: float = 0.15, blur: float = 2.0) -> np.ndarray:
    """0 on background, ramping to 1 on signal (for applying enhancements without amplifying noise)."""
    L = ndi.gaussian_filter(luminance(img), blur)
    if low is None:
        bg, sig = bg_stats(L)
        low = bg + 2 * sig
    return np.clip((L - low) / soft, 0, 1).astype(np.float32)


def saturate(img: np.ndarray, amount: float, mask: np.ndarray | None = None, bg_amount: float = 1.0) -> np.ndarray:
    """Scale chroma about luminance. With a mask, background gets `bg_amount`, signal gets `amount`."""
    if amount == 1 and (mask is None or bg_amount == 1):
        return img
    L = luminance(img)[..., None]
    k = amount if mask is None else bg_amount + (amount - bg_amount) * mask[..., None]
    out = L + (img - L) * k
    # bring over-range pixels back without hue shift
    peak = out.max(axis=-1, keepdims=True)
    out = np.where(peak > 1, L + (out - L) * ((1 - L) / np.maximum(peak - L, EPS)), out)
    return np.clip(out, 0, 1).astype(np.float32)


def chroma_denoise(img: np.ndarray, sigma: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Blur colour (not luminance) to remove colour mottle, mostly in the background."""
    if sigma <= 0:
        return img
    L = luminance(img)[..., None]
    chroma = img - L
    blurred = np.stack([ndi.gaussian_filter(chroma[..., c], sigma) for c in range(3)], -1)
    if mask is not None:
        w = 1 - 0.7 * mask[..., None]  # keep more colour detail on the signal
        blurred = w * blurred + (1 - w) * chroma
    return np.clip(L + blurred, 0, 1).astype(np.float32)


def lum_denoise(img: np.ndarray, sigma: float, mask: np.ndarray) -> np.ndarray:
    """Light luminance smoothing on the background only (after the AI denoise, this is just polish)."""
    if sigma <= 0:
        return img
    L = luminance(img)
    Ls = ndi.gaussian_filter(L, sigma)
    w = (1 - mask) * 0.8
    Ln = L * (1 - w) + Ls * w
    return np.clip(img * (Ln / np.maximum(L, EPS))[..., None], 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------------------------- detail
def multiscale_detail(img: np.ndarray, scales: list[float], gains: list[float], mask: np.ndarray | None) -> np.ndarray:
    """Boost structure at chosen scales (difference-of-Gaussians layers added back to luminance)."""
    if not scales:
        return img
    L = luminance(img)
    out = L.copy()
    prev = L
    for s, g in zip(scales, gains):
        cur = ndi.gaussian_filter(L, s)
        layer = prev - cur
        out += (g - 1.0) * layer * (1.0 if mask is None else mask)
        prev = cur
    out = np.clip(out, 0, 1)
    return np.clip(img * (out / np.maximum(L, EPS))[..., None], 0, 1).astype(np.float32)


def local_contrast(img: np.ndarray, amount: float, tile: int = 8, clip: float = 0.01,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """CLAHE on luminance, blended in by `amount` (0..1)."""
    if amount <= 0:
        return img
    from skimage import exposure

    L = luminance(img)
    ks = (max(16, L.shape[0] // tile), max(16, L.shape[1] // tile))
    Lc = exposure.equalize_adapthist(np.clip(L, 0, 1), kernel_size=ks, clip_limit=clip).astype(np.float32)
    w = amount if mask is None else amount * mask
    Ln = L * (1 - w) + Lc * w
    return np.clip(img * (Ln / np.maximum(L, EPS))[..., None], 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------------------------- stars
def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1 - (1 - a) * (1 - b)


def tame_blue_fringe(stars: np.ndarray, limit: float) -> np.ndarray:
    """Cap the star layer's blue at `limit` x max(red, green).

    Small refractors focus blue slightly differently, so bright stars grow blue/violet discs. Real stars are
    never much bluer than B/G ~ 1.1-1.2, so the cap leaves genuine star colour alone."""
    if not limit:
        return stars
    out = stars.copy()
    cap = np.maximum(stars[..., 0], stars[..., 1]) * limit
    excess = np.maximum(stars[..., 2] - cap, 0)
    # hand the excess back as neutral light so the disc keeps its brightness (no dark ring)
    out[..., 2] -= excess * 2 / 3
    out[..., 0] += excess / 3
    out[..., 1] += excess / 3
    return np.clip(out, 0, 1)


def limit_chroma(img: np.ndarray, max_chroma: float | None) -> np.ndarray:
    """Pull colours that stray more than `max_chroma` (relative to brightness) back toward neutral.

    Used on the star layer: behind a dual-band filter the red channel only sees starlight through the narrow
    Ha band, so star halos come out strongly cyan; real star colours are subtle and survive the cap."""
    if not max_chroma:
        return img
    L = luminance(img)[..., None]
    d = img - L
    c = np.abs(d).max(-1, keepdims=True) / np.maximum(L, 1e-4)
    k = np.where(c > max_chroma, max_chroma / np.maximum(c, 1e-6), 1.0)
    return np.clip(L + d * k, 0, 1).astype(np.float32)


def desaturate_halos(stars: np.ndarray, amount: float, knee: float = 0.35) -> np.ndarray:
    """Remove colour from the faint outskirts of stars (chromatic halos, blue/violet fringes),
    keeping it in star cores. `amount` 0..1; `knee` is the star-layer brightness where colour is fully kept."""
    if amount <= 0:
        return stars
    L = luminance(stars)[..., None]
    keep = np.clip(L / knee, 0, 1) ** 0.7
    k = 1 - amount * (1 - keep)
    return np.clip(L + (stars - L) * k, 0, 1).astype(np.float32)


def shrink_stars(stars: np.ndarray, amount: float) -> np.ndarray:
    """Reduce star halos: grey erosion blended in by `amount` (0..1)."""
    if amount <= 0:
        return stars
    er = np.stack([ndi.grey_erosion(stars[..., c], size=(3, 3)) for c in range(3)], -1)
    return (stars * (1 - amount) + er * amount).astype(np.float32)


# ----------------------------------------------------------------------------------------- narrowband
def hoo(linear: np.ndarray, oiii_boost: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Split a dual-band (Ha+OIII) OSC image into Ha (red) and OIII (mean of green & blue) linear planes."""
    ha = linear[..., 0]
    oiii = (linear[..., 1] + linear[..., 2]) / 2
    return ha, oiii * oiii_boost


def mono_stretch(ch: np.ndarray, target_bg: float, shadow_sigma: float = 2.8, asinh_beta: float = 0.0) -> np.ndarray:
    bg, sig = bg_stats(ch)
    bp = max(bg - shadow_sigma * sig, 0)
    x = np.clip((ch - bp) / (1 - bp), 0, 1)
    bgn = (bg - bp) / (1 - bp)
    if asinh_beta > 0:
        x = np.arcsinh(asinh_beta * x) / np.arcsinh(asinh_beta)
        bgn = float(np.arcsinh(asinh_beta * bgn) / np.arcsinh(asinh_beta))
    return np.clip(mtf(mtf_for(bgn, target_bg), x), 0, 1).astype(np.float32)
