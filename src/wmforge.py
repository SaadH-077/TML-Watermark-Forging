"""Core utilities for the watermark-forgery attack (TML 2026 Task 4).

We are black-box attackers. For each method WM_k we have 25 watermarked CARRIER
images sharing the same hidden message, and we must transplant that watermark onto
25 clean TARGET images while keeping LPIPS low.

Design notes
------------
* LPIPS (the visual-quality half of the score) is computable locally; bit-accuracy
  (the detection half) is NOT, since we have no detector. So everything here is
  built to (a) estimate a watermark pattern robustly and (b) measure LPIPS exactly.
* Watermark estimation with NO clean originals: estimate clean_i = denoise(wm_i),
  residual_i = wm_i - clean_i, then average residuals across the 25 same-message
  carriers. Content/denoiser-error decorrelates and averages out; a consistent
  (content-independent) watermark survives. This is the "averaging / copy attack".
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "extracted"
TARGETS_DIR = DATA / "clean_targets"
SOURCES_DIR = DATA / "watermarked_sources"

# WM_k  ->  inclusive target index range (1-based, matches the PDF mapping)
WM_TO_TARGETS: Dict[str, range] = {
    f"WM_{k}": range(25 * (k - 1) + 1, 25 * k + 1) for k in range(1, 9)
}


def method_names() -> List[str]:
    return [f"WM_{k}" for k in range(1, 9)]


def carrier_paths(wm: str) -> List[Path]:
    """Sorted list of the 25 watermarked carrier images for a method."""
    paths = sorted(
        (SOURCES_DIR / wm).glob("*.png"),
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0),
    )
    return paths


def target_paths(wm: str) -> List[Path]:
    """The clean target images this method must be forged onto."""
    return [TARGETS_DIR / f"{i}.png" for i in WM_TO_TARGETS[wm]]


def all_target_paths() -> List[Path]:
    return [TARGETS_DIR / f"{i}.png" for i in range(1, 201)]


# ----------------------------------------------------------------------------
# Image IO  (work in float64/float32 [0,1], RGB, HxWx3)
# ----------------------------------------------------------------------------
def load(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_stack(paths) -> np.ndarray:
    """[N,H,W,3] float32 in [0,1].  Assumes all same size (true within a method)."""
    return np.stack([load(p) for p in paths], axis=0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def save(img: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(img)).save(path)


# ----------------------------------------------------------------------------
# Denoisers (clean-image estimators).  Input/Output float [0,1] HxWx3.
# ----------------------------------------------------------------------------
def denoise(img: np.ndarray, method: str = "nlm", **kw) -> np.ndarray:
    if method == "gauss":
        import cv2

        s = kw.get("sigma", 1.0)
        return np.clip(cv2.GaussianBlur(img, (0, 0), s), 0, 1)
    if method == "median":
        import cv2

        k = kw.get("ksize", 3)
        u = to_uint8(img)
        return cv2.medianBlur(u, k).astype(np.float32) / 255.0
    if method == "nlm":
        import cv2

        h = kw.get("h", 6)  # luminance filter strength
        hc = kw.get("hColor", 6)
        u = to_uint8(img)
        out = cv2.fastNlMeansDenoisingColored(u, None, h, hc, 7, 21)
        return out.astype(np.float32) / 255.0
    if method == "bilateral":
        import cv2

        out = cv2.bilateralFilter(img, kw.get("d", 5), kw.get("sc", 0.1), kw.get("ss", 3))
        return np.clip(out, 0, 1)
    raise ValueError(method)


def highpass(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    import cv2

    return img - cv2.GaussianBlur(img, (0, 0), sigma)


# ----------------------------------------------------------------------------
# Watermark estimation
# ----------------------------------------------------------------------------
def residuals(carriers: np.ndarray, clean: str = "nlm", **kw) -> np.ndarray:
    """Per-carrier residual = wm - denoise(wm).  [N,H,W,3]."""
    return np.stack([c - denoise(c, clean, **kw) for c in carriers], axis=0)


def estimate_watermark(carriers: np.ndarray, clean: str = "nlm", **kw) -> dict:
    """Estimate the additive watermark pattern from same-message carriers.

    Returns a dict of the mean residual ('delta') plus diagnostics that say
    whether the watermark really is a content-independent fixed pattern.
    """
    res = residuals(carriers, clean, **kw)              # [N,H,W,3]
    N = res.shape[0]
    delta = res.mean(axis=0)                            # averaged watermark estimate

    # Fixed-pattern energy ratio:  ||mean residual||^2 / mean ||residual||^2.
    # ~1/N (=0.04 for N=25) if residuals are random noise; ->1 if a fixed pattern
    # dominates every carrier.
    e_mean = float((delta ** 2).mean())
    e_ind = float((res ** 2).mean())
    energy_ratio = e_mean / (e_ind + 1e-12)

    # Mean pairwise cosine correlation between flattened residuals.
    flat = res.reshape(N, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    unit = flat / norm
    corr = unit @ unit.T
    off = corr[~np.eye(N, dtype=bool)]
    mean_pair_corr = float(off.mean())

    return {
        "delta": delta,
        "residuals": res,
        "residual_std": float(res.std()),
        "delta_std": float(delta.std()),
        "energy_ratio": energy_ratio,        # >> 1/N  => fixed additive pattern
        "random_floor": 1.0 / N,
        "mean_pair_corr": mean_pair_corr,    # > 0 => shared spatial pattern
    }


# ----------------------------------------------------------------------------
# Forging
# ----------------------------------------------------------------------------
def forge(target: np.ndarray, delta: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Add the estimated watermark to a clean target. Resizes delta if needed."""
    if delta.shape != target.shape:
        import cv2

        delta = cv2.resize(delta, (target.shape[1], target.shape[0]),
                           interpolation=cv2.INTER_CUBIC)
    return np.clip(target + alpha * delta, 0, 1)


# ----------------------------------------------------------------------------
# LPIPS (the visual-quality metric we CAN compute locally)
# ----------------------------------------------------------------------------
_LPIPS_MODEL = {}


def lpips_model(net: str = "alex"):
    if net not in _LPIPS_MODEL:
        import lpips
        import torch

        m = lpips.LPIPS(net=net, verbose=False)
        m.eval()
        _LPIPS_MODEL[net] = m
    return _LPIPS_MODEL[net]


def lpips_distance(a: np.ndarray, b: np.ndarray, net: str = "alex") -> float:
    """LPIPS between two [0,1] HxWx3 images."""
    import torch

    m = lpips_model(net)

    def t(x):
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
        return x * 2 - 1  # [0,1] -> [-1,1]

    with torch.no_grad():
        d = m(t(a), t(b))
    return float(d.item())


def quality_score(lpips_val: float) -> float:
    return float(np.exp(-8.0 * lpips_val))


# ----------------------------------------------------------------------------
# Clean bank + watermark estimators (Recipe 0 / 1)
# ----------------------------------------------------------------------------
def clean_bank(res: int, paths=None) -> np.ndarray:
    """A bank of clean images at resolution `res` for the difference-of-means
    estimator. Defaults to all 200 targets, resized to res."""
    import cv2

    if paths is None:
        paths = all_target_paths()
    out = []
    for p in paths:
        im = load(p)
        if im.shape[0] != res or im.shape[1] != res:
            im = cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA)
        out.append(im)
    return np.stack(out, 0)


def dc_remove(delta: np.ndarray) -> np.ndarray:
    """Strip per-channel spatial mean (removes flat content-difference bias)."""
    return delta - delta.reshape(-1, delta.shape[-1]).mean(0)


def estimate(carriers: np.ndarray, mode: str = "resid_median",
             bank: np.ndarray = None, denoiser: str = "nlm", dkw: dict = None,
             hp_sigma: float = None, dc: bool = False) -> np.ndarray:
    """Estimate the additive watermark pattern δ.

    modes:
      diff_means    : mean(carriers) - mean(bank)               (Yang'24 Recipe 0)
      resid_mean    : mean_i(C_i - denoise(C_i))                (Craver/Souček fallback)
      resid_median  : median_i(C_i - denoise(C_i))              (robust Recipe 1)
    Optional post: hp_sigma (high-pass) and/or dc (remove flat mean).
    """
    dkw = dkw or {}
    if mode == "diff_means":
        if bank is None:
            bank = clean_bank(carriers.shape[1])
        delta = carriers.mean(0) - bank.mean(0)
    elif mode in ("resid_mean", "resid_median"):
        res = np.stack([c - denoise(c, denoiser, **dkw) for c in carriers], 0)
        delta = res.mean(0) if mode == "resid_mean" else np.median(res, 0)
    else:
        raise ValueError(mode)
    if hp_sigma:
        delta = highpass(delta, hp_sigma)
    if dc:
        delta = dc_remove(delta)
    return delta


def coherence(carriers: np.ndarray, delta: np.ndarray, denoiser="nlm", dkw=None) -> float:
    """Label-free quality of a δ estimate: mean cosine alignment between δ and each
    carrier's own residual. Higher => δ captures the signal shared across carriers."""
    dkw = dkw or {}
    d = (delta - delta.mean()).ravel()
    d /= np.linalg.norm(d) + 1e-12
    vals = []
    for c in carriers:
        r = (c - denoise(c, denoiser, **dkw))
        r = (r - r.mean()).ravel()
        vals.append(float(d @ (r / (np.linalg.norm(r) + 1e-12))))
    return float(np.mean(vals))


# ----------------------------------------------------------------------------
# Submission building + validation
# ----------------------------------------------------------------------------
def forge_all(deltas: dict, alphas) -> dict:
    """Forge all 200 targets. `deltas[wm]` is that method's δ; `alphas` is a float
    or dict wm->float. Returns {target_index: forged_img}."""
    out = {}
    for wm in method_names():
        a = alphas[wm] if isinstance(alphas, dict) else alphas
        for i in WM_TO_TARGETS[wm]:
            t = load(TARGETS_DIR / f"{i}.png")
            out[i] = forge(t, deltas[wm], alpha=a)
    return out


def build_submission(forged: dict, out_zip) -> Path:
    """Write {index: img} as a FLAT zip of N.png (the required submission format)."""
    import io
    import zipfile

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 201):
            buf = io.BytesIO()
            Image.fromarray(to_uint8(forged[i])).save(buf, format="PNG")
            zf.writestr(f"{i}.png", buf.getvalue())
    return out_zip


# ----------------------------------------------------------------------------
# Identified schemes: native (re-)encoding via invisible-watermark
# ----------------------------------------------------------------------------
# Confirmed by decoder reconnaissance (scripts 04–06). dwtDct length is uncertain
# (all-1s message is length-robust); RivaGAN is a fixed 32-bit scheme.
IDENTIFIED = {
    "WM_1": {"lib": "iw", "scheme": "dwtDct", "L": 30},
    "WM_2": {"lib": "iw", "scheme": "rivaGan", "L": 32},
    "WM_7": {"lib": "trustmark", "variant": "Q", "L": 61},
}

_RIVAGAN_LOADED = [False]


def _ensure_rivagan():
    if not _RIVAGAN_LOADED[0]:
        from imwatermark import WatermarkEncoder

        WatermarkEncoder.loadModel()
        _RIVAGAN_LOADED[0] = True


def _to_bgr(img_rgb_float):
    return to_uint8(img_rgb_float)[:, :, ::-1].copy()


def iw_decode(img_rgb_float, scheme: str, L: int) -> np.ndarray:
    import numpy as np
    from imwatermark import WatermarkDecoder

    if scheme == "rivaGan":
        _ensure_rivagan()
    d = WatermarkDecoder("bits", L)
    bgr = np.ascontiguousarray(_to_bgr(img_rgb_float))
    return np.asarray(d.decode(bgr, scheme), dtype=int).ravel()


def recover_message(wm: str) -> np.ndarray:
    """Consensus (majority-vote) message decoded from a method's 25 carriers,
    using whichever library identified the scheme."""
    info = IDENTIFIED[wm]
    rows = []
    if info["lib"] == "iw":
        for p in carrier_paths(wm):
            try:
                b = iw_decode(load(p), info["scheme"], info["L"])
                if b.size == info["L"]:
                    rows.append(b)
            except Exception:
                pass
    elif info["lib"] == "trustmark":
        for p in carrier_paths(wm):
            try:
                b, _ = tm_decode(Image.open(p).convert("RGB"), info["variant"])
                if b.size == info["L"]:
                    rows.append(b)
            except Exception:
                pass
    return (np.stack(rows).mean(0) >= 0.5).astype(int)


# ----------------------------------------------------------------------------
# TrustMark (identified WM_7): high-quality learned watermark with an encoder
# ----------------------------------------------------------------------------
_TM_MODELS = {}


def tm_model(variant: str = "Q"):
    if variant not in _TM_MODELS:
        from trustmark import TrustMark

        _TM_MODELS[variant] = TrustMark(verbose=False, model_type=variant)
    return _TM_MODELS[variant]


def tm_decode(img_pil, variant: str = "Q"):
    out = tm_model(variant).decode(img_pil, MODE="binary")
    sec = out[0] if isinstance(out, (tuple, list)) else out
    present = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
    bits = np.array([int(c) for c in str(sec) if c in "01"])
    return bits, present


def tm_encode(img_rgb_float: np.ndarray, message: np.ndarray, variant: str = "Q") -> np.ndarray:
    cover = Image.fromarray(to_uint8(img_rgb_float))
    stego = tm_model(variant).encode(cover, "".join(map(str, message)), MODE="binary")
    stego = stego.convert("RGB").resize(cover.size)
    return np.asarray(stego, np.float32) / 255.0


def native_forge(wm: str, img_rgb_float: np.ndarray, message: np.ndarray) -> np.ndarray:
    """Re-embed the recovered message into a clean target with the genuine encoder
    of the identified scheme. Dispatches on the library."""
    info = IDENTIFIED[wm]
    if info["lib"] == "iw":
        return iw_encode(img_rgb_float, info["scheme"], message)
    if info["lib"] == "trustmark":
        return tm_encode(img_rgb_float, message, info["variant"])
    raise ValueError(info["lib"])


def iw_encode(img_rgb_float: np.ndarray, scheme: str, message: np.ndarray) -> np.ndarray:
    """Embed `message` into a clean image with the genuine library encoder.
    Returns an RGB float [0,1] image at the input resolution."""
    from imwatermark import WatermarkEncoder

    if scheme == "rivaGan":
        _ensure_rivagan()
    enc = WatermarkEncoder()
    enc.set_watermark("bits", list(map(int, message)))
    out_bgr = enc.encode(_to_bgr(img_rgb_float), scheme)
    return out_bgr[:, :, ::-1].astype(np.float32) / 255.0


def validate_zip(path) -> dict:
    """Check the submission contract: 200 flat PNGs named exactly 1..200.png."""
    import zipfile

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    expected = {f"{i}.png" for i in range(1, 201)}
    got = set(names)
    return {
        "ok": got == expected and len(names) == 200,
        "n": len(names),
        "missing": sorted(expected - got)[:10],
        "unexpected": sorted(got - expected)[:10],
        "has_subdirs": any("/" in n for n in names),
    }
