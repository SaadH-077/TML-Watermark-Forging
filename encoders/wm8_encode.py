"""WM_8 = TrustMark variant P (100-bit raw, use_ECC=False). Native re-encode.
Identified by the fresh-ID sweep (notes/decoder_id_retest_native.json): carrier_agree 0.999,
clean 0.61 -> a real crack. We only tested TrustMark Q (=WM_7) before, never P on the 512² carriers.
Decode 25 carriers -> consensus message -> TrustMark-P encode onto clean targets 176..200 (native 512)
-> verify round-trip bitacc + LPIPS -> save forged_native/WM_8/<id>.png.
"""
import os, sys, glob
import numpy as np
from PIL import Image
from trustmark import TrustMark

PROJECT = os.environ.get("A4_ROOT", "/path/to/Assignment4")  # set to your project root
DATA = os.path.join(PROJECT, "data", "extracted")
WM_DIR = os.path.join(DATA, "watermarked_sources", "WM_8")
CLEAN_DIR = os.path.join(DATA, "clean_targets")
OUT_DIR = os.path.join(PROJECT, "forged_native", "WM_8")
os.makedirs(OUT_DIR, exist_ok=True)
KNOWN = "1010001101101000000100011110001101011111111011001010000110001111111101011011011011101101100001111000"

tm = TrustMark(use_ECC=False, verbose=False, model_type="P")


def decode_bits(img_pil):
    out = tm.decode(img_pil, MODE="binary")
    # trustmark decode returns (secret, present, ...) or a string depending on version
    if isinstance(out, tuple):
        secret = out[0]
    else:
        secret = out
    if isinstance(secret, str):
        return np.array([int(c) for c in secret if c in "01"], dtype=np.int8)
    return np.array(list(secret), dtype=np.int8)


# 1. consensus from the 25 carriers (sanity vs KNOWN)
carriers = sorted(glob.glob(os.path.join(WM_DIR, "*.png")))
assert len(carriers) == 25, len(carriers)
cb = [decode_bits(Image.open(p).convert("RGB")) for p in carriers]
L = min(len(b) for b in cb)
arr = np.stack([b[:L] for b in cb], 0)
consensus = (arr.mean(0) >= 0.5).astype(np.int8)
agree = float(np.mean([(b[:L] == consensus).mean() for b in cb]))
M = "".join(map(str, consensus.tolist()))
print(f"[carriers] agree={agree:.4f}  L={L}", file=sys.stderr)
print(f"[carriers] consensus == KNOWN: {M == KNOWN}", file=sys.stderr)
msg = M if agree > 0.9 else KNOWN   # use carrier consensus if clean, else fall back to validated msg

# 2. encode into targets 176..225? -> WM_8 targets are 176..200
ids = list(range(176, 201))
import lpips, torch
import torchvision.transforms as T
loss = lpips.LPIPS(net="alex", verbose=False)
def to_t(im): return (T.ToTensor()(im).unsqueeze(0) * 2 - 1)

accs, lps = [], []
for tid in ids:
    src = os.path.join(CLEAN_DIR, f"{tid}.png")
    cover = Image.open(src).convert("RGB")
    stego = tm.encode(cover, msg, MODE="binary", WM_STRENGTH=1.0)
    stego.save(os.path.join(OUT_DIR, f"{tid}.png"))
    rec = decode_bits(stego)[:L]
    accs.append(float((rec == consensus).mean()))
    with torch.no_grad():
        lps.append(loss(to_t(cover), to_t(stego)).item())
    print(f"  {tid}.png bitacc={accs[-1]:.3f} LPIPS={lps[-1]:.4f}", file=sys.stderr)

mb, ml = float(np.mean(accs)), float(np.mean(lps))
print(f"\n=== WM_8 TrustMark-P forge ===", file=sys.stderr)
print(f"mean bitacc={mb:.4f}  mean LPIPS={ml:.4f}  Sqlt={np.exp(-8*ml):.4f}", file=sys.stderr)
print(f"saved {len(ids)} -> {OUT_DIR}", file=sys.stderr)
print(f"Sfinal~={max(0,2*(mb-0.5))*np.exp(-8*ml):.4f}", file=sys.stderr)
