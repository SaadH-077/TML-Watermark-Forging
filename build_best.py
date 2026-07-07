#!/usr/bin/env python3
"""Recreate our best leaderboard submission: public S_final = 0.8553.

Output: submissions/candidate_best.zip, a flat zip of 200 PNGs (1.png..200.png), where WM_k
covers targets 25(k-1)+1 .. 25k. All eight schemes are genuine native re-encodes, frozen under
forged_native_masked/<WM>/:
  * WM_2..WM_8 : native re-encodes, loaded as-is
  * WM_3       : ArtificialGANFingerprints (StegaStamp) native re-embed (see encoders/afgf_wm3_encode.py)
  * WM_1       : native dwtDct re-encode, residual-amplified s=1.5 at build time
                 (widens the QIM margin -> more robust to the server's pre-detection transforms)

Run:  python3 build_best.py     (from inside this folder)
Needs: data/extracted/clean_targets (for the WM_1 amplification + the per-scheme LPIPS table),
       forged_native_masked/, and the pip deps in requirements.txt (numpy, torch, torchvision,
       lpips, Pillow). No venvs, no GPU, and no dataset carriers required.
"""
import io, zipfile, warnings, os
warnings.filterwarnings("ignore"); os.environ.setdefault("TQDM_DISABLE", "1")
from pathlib import Path
import numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
TARG = ROOT / "data/extracted/clean_targets"
MASK = ROOT / "forged_native_masked"
OUT  = ROOT / "submissions"; OUT.mkdir(exist_ok=True)
WM_TO_TARGETS = {f"WM_{k}": range(25 * (k - 1) + 1, 25 * k + 1) for k in range(1, 9)}

def load(p): return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
def to_u8(x): return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)

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

# --- WM_2..WM_8: genuine native re-encodes, frozen (WM_3 = ArtificialGANFingerprints re-embed) ---
for wm in ["WM_2", "WM_3", "WM_4", "WM_5", "WM_6", "WM_7", "WM_8"]:
    for i in WM_TO_TARGETS[wm]:
        RES[i] = load(MASK / wm / f"{i}.png")
print("WM_3 <- ArtificialGANFingerprints native re-embed (frozen)")

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
print(f"\nOK  {ZIP}  (200 flat PNGs, 1.png..200.png)  ->  submit for public S_final 0.8553")
