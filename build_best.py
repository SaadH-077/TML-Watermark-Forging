#!/usr/bin/env python3
"""Recreate our best leaderboard submission: public S_final = 0.8364.

Output: submissions/candidate_best.zip  — a flat zip of 200 PNGs (1.png..200.png), where WM_k
covers targets 25(k-1)+1 .. 25k. Assembled from:
  * WM_2,4,5,6,7,8 : genuine native re-encodes (forged_native_masked/<WM>/), frozen
  * WM_1           : native dwtDct re-encode (forged_native_masked/WM_1), residual-amplified s=1.5
                     (widens the QIM margin -> more robust to the server's pre-detection transforms)
  * WM_3           : color-axis (R+B-2G) chroma transplant, bilateral-denoised template, fit to LPIPS 0.023

Run:  python3 build_best.py     (from inside this folder)
Needs: data/extracted/{clean_targets,watermarked_sources/WM_3}, forged_native_masked/, and pip deps
       in requirements.txt (numpy, opencv-python, torch, torchvision, lpips, Pillow). No venvs needed.
"""
import io, zipfile, warnings, os
warnings.filterwarnings("ignore"); os.environ.setdefault("TQDM_DISABLE", "1")
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
TARG = ROOT / "data/extracted/clean_targets"
SRC  = ROOT / "data/extracted/watermarked_sources"
MASK = ROOT / "forged_native_masked"
OUT  = ROOT / "submissions"; OUT.mkdir(exist_ok=True)
WM_TO_TARGETS = {f"WM_{k}": range(25 * (k - 1) + 1, 25 * k + 1) for k in range(1, 9)}

def load(p): return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
def to_u8(x): return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)
def carrier_paths(wm):
    return sorted((SRC / wm).glob("*.png"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))

_m = None
def lpips_d(a, b):
    global _m
    if _m is None:
        import lpips
        _m = lpips.LPIPS(net="alex", verbose=False).eval()
        _m = _m.cuda() if torch.cuda.is_available() else _m
    def t(x):
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
        return x.cuda() if torch.cuda.is_available() else x
    with torch.no_grad():
        return float(_m(t(a), t(b)).item())
def Sqlt(l): return float(np.exp(-8 * l))

RES = {}

# --- WM_3: color-axis (R+B-2G) transplant, bilateral template, fit to LPIPS 0.023 ---
P = np.array([0.37, -0.834, 0.41], np.float32); P /= np.linalg.norm(P)
def axproj(img): return np.tensordot(img, P, ([2], [0]))
def bilat(c):    return cv2.bilateralFilter(c, 9, 0.1, 7)
carr = [load(p) for p in carrier_paths("WM_3")]
T3 = np.mean([cv2.GaussianBlur(axproj(c) - axproj(bilat(c)), (0, 0), 4) for c in carr], 0)
TS3 = [load(TARG / f"{i}.png") for i in WM_TO_TARGETS["WM_3"]]
def f_axis(t, a): return np.clip(t + a * T3[:, :, None] * P[None, None, :], 0, 1)
def fit_alpha(budget, lo=0.0, hi=400.0):
    for _ in range(13):
        mid = (lo + hi) / 2
        ml = np.mean([lpips_d(t, f_axis(t, mid)) for t in TS3[::5]])
        lo, hi = (mid, hi) if ml < budget else (lo, mid)
    return (lo + hi) / 2
a3 = fit_alpha(0.025)
for i, t in zip(WM_TO_TARGETS["WM_3"], TS3):
    RES[i] = f_axis(t, a3)
print(f"WM_3 color-axis transplant: alpha={a3:.2f}")

# --- WM_2,4,5,6,7,8: genuine native re-encodes, frozen ---
for wm in ["WM_2", "WM_4", "WM_5", "WM_6", "WM_7", "WM_8"]:
    for i in WM_TO_TARGETS[wm]:
        RES[i] = load(MASK / wm / f"{i}.png")

# --- WM_1: native dwtDct re-encode, residual-amplified s=1.5 ---
S_WM1 = 1.5
for i in WM_TO_TARGETS["WM_1"]:
    t = load(TARG / f"{i}.png"); f0 = load(MASK / "WM_1" / f"{i}.png")
    RES[i] = np.clip(t + S_WM1 * (f0 - t), 0, 1)
print(f"WM_1 dwtDct re-encode amplified s={S_WM1}")

# --- integrity + quality table ---
assert set(RES) == set(range(1, 201)), "coverage != 1..200"
print(f"\n{'WM':5}{'LPIPS':>8}{'Sqlt':>7}")
for wm in [f"WM_{k}" for k in range(1, 9)]:
    for i in WM_TO_TARGETS[wm]:
        assert RES[i].shape == load(TARG / f"{i}.png").shape, f"{wm}/{i} resolution mismatch"
    lp = float(np.mean([lpips_d(load(TARG / f"{i}.png"), RES[i]) for i in WM_TO_TARGETS[wm]]))
    print(f"{wm:5}{lp:>8.4f}{Sqlt(lp):>7.3f}")

# --- build + validate flat zip ---
ZIP = OUT / "candidate_best.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for i in range(1, 201):
        buf = io.BytesIO(); Image.fromarray(to_u8(RES[i])).save(buf, format="PNG")
        zf.writestr(f"{i}.png", buf.getvalue())
with zipfile.ZipFile(ZIP) as zf:
    names = set(zf.namelist())
assert names == {f"{i}.png" for i in range(1, 201)} and len(names) == 200, "SUBMISSION INVALID"
print(f"\nOK  {ZIP}  (200 flat PNGs, 1.png..200.png)  ->  submit for public S_final 0.8364")
