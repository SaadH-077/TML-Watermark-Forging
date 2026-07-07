#!/usr/bin/env python3
"""WM_3 native forge: ArtificialGANFingerprints (StegaStamp autoencoder).

Unlike the other native encoders here, this one is SELF-CONTAINED: it bundles the StegaStamp
encoder/decoder definitions (from ningyu1991/ArtificialGANFingerprints) and auto-downloads the
public AFHQ cat2dog 256x256 checkpoint. No extra repo or venv is needed, only the deps in
../requirements.txt plus `gdown`.

WM_3 was long thought unidentifiable. It is ArtificialGANFingerprints (Yu et al., ICCV 2021),
a StegaStamp (Tancik et al., CVPR 2020) autoencoder. The public decoder reads the 25 WM_3
carriers at cross-carrier agreement 0.9996 (clean targets ~0.48 = chance), which can only happen
if the benchmark used these exact public weights, so the hidden detector IS this decoder.

Pipeline: (1) decode the 25 carriers, majority-vote the 100-bit message; (2) re-embed it into each
target with the matching encoder, forged = clip(y + a*r), r = Enc(msg,y) - y; (3) sweep the
per-image strength a, score AFTER uint8/PNG rounding, and pick the a that maximises the worst-case
score across the AlexNet and VGG LPIPS backbones (the grader's backbone is unknown).

Run:  python3 encoders/afgf_wm3_encode.py     (from the repo root)
Out:  forged_native_masked/WM_3/51.png .. 75.png  (the frozen forge build_best.py consumes)
Needs: data/extracted/{watermarked_sources/WM_3, clean_targets}  (python3 fetch_data.py), and gdown.
"""
import os, math, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("TQDM_DISABLE", "1")
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn.functional import relu, sigmoid

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data/extracted/watermarked_sources/WM_3"
TARG_DIR = ROOT / "data/extracted/clean_targets"
WDIR = ROOT / "encoders/afgf_weights"
ENC_PATH = WDIR / "AFHQ_cat2dog_256x256_encoder.pth"
DEC_PATH = WDIR / "AFHQ_cat2dog_256x256_decoder.pth"
GDRIVE_FOLDER = "1k5Ezb2Do5oBiN-Ei6P0SY6CJfIsJXMX9"
OUT_DIR = ROOT / "forged_native_masked/WM_3"
TARGET_IDS = list(range(51, 76))
RES = 256
device = torch.device("mps") if torch.backends.mps.is_available() else \
         (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


# ---- StegaStamp encoder / decoder (verbatim from ningyu1991/ArtificialGANFingerprints) ----
class StegaStampEncoder(nn.Module):
    def __init__(self, resolution=32, IMAGE_CHANNELS=1, fingerprint_size=100, return_residual=False):
        super().__init__()
        self.fingerprint_size = fingerprint_size; self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.return_residual = return_residual
        self.secret_dense = nn.Linear(fingerprint_size, 16 * 16 * IMAGE_CHANNELS)
        lr = int(math.log(resolution, 2))
        assert resolution == 2 ** lr, "resolution must be a power of 2"
        self.fingerprint_upsample = nn.Upsample(scale_factor=(2 ** (lr - 4), 2 ** (lr - 4)))
        self.conv1 = nn.Conv2d(2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1); self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1); self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1)); self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2)); self.conv6 = nn.Conv2d(256, 128, 3, 1, 1)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1)); self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2)); self.conv7 = nn.Conv2d(128, 64, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1)); self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2)); self.conv8 = nn.Conv2d(64, 32, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1)); self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(32 + 32 + 2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1); self.residual = nn.Conv2d(32, IMAGE_CHANNELS, 1)

    def forward(self, fingerprint, image):
        fp = relu(self.secret_dense(fingerprint)).view((-1, self.IMAGE_CHANNELS, 16, 16))
        inputs = torch.cat([self.fingerprint_upsample(fp), image], dim=1)
        c1 = relu(self.conv1(inputs)); c2 = relu(self.conv2(c1)); c3 = relu(self.conv3(c2))
        c4 = relu(self.conv4(c3)); c5 = relu(self.conv5(c4))
        u6 = relu(self.up6(self.pad6(self.upsample6(c5)))); c6 = relu(self.conv6(torch.cat([c4, u6], 1)))
        u7 = relu(self.up7(self.pad7(self.upsample7(c6)))); c7 = relu(self.conv7(torch.cat([c3, u7], 1)))
        u8 = relu(self.up8(self.pad8(self.upsample8(c7)))); c8 = relu(self.conv8(torch.cat([c2, u8], 1)))
        u9 = relu(self.up9(self.pad9(self.upsample9(c8)))); c9 = relu(self.conv9(torch.cat([c1, u9, inputs], 1)))
        res = self.residual(relu(self.conv10(c9)))
        return res if self.return_residual else sigmoid(res)


class StegaStampDecoder(nn.Module):
    def __init__(self, resolution=32, IMAGE_CHANNELS=1, fingerprint_size=1):
        super().__init__()
        self.resolution = resolution; self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU())
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512), nn.ReLU(),
            nn.Linear(512, fingerprint_size))

    def forward(self, image):
        x = self.decoder(image).view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return self.dense(x)


def ensure_weights():
    if ENC_PATH.exists() and DEC_PATH.exists():
        return
    WDIR.mkdir(parents=True, exist_ok=True)
    import gdown
    print("Downloading AFGF AFHQ cat2dog 256x256 weights ...")
    gdown.download_folder(f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER}",
                          output=str(WDIR), quiet=True, use_cookies=False)
    assert ENC_PATH.exists() and DEC_PATH.exists(), "weight download failed"


def load01(p): return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
def to_t(a):   return torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
def round_u8(a):
    u8 = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8); return u8, u8.astype(np.float32) / 255.0


@torch.no_grad()
def decode(dec, a):  return (dec(to_t(a)) > 0).long().cpu().numpy().reshape(-1)
def bitacc(a, b):    return float(np.mean(a == b))
def Sqlt(l):         return math.exp(-8.0 * l)
def Sdet(b):         return max(0.0, 2.0 * (b - 0.5))

_lp = {}
def lpips_dist(net, a, b):
    import lpips as _l
    if net not in _lp:
        _lp[net] = _l.LPIPS(net=net, verbose=False).eval().to(device)
    def t(x): return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device) * 2 - 1
    with torch.no_grad():
        return float(_lp[net](t(a), t(b)).item())


def main():
    ensure_weights(); OUT_DIR.mkdir(parents=True, exist_ok=True)
    dsd = torch.load(DEC_PATH, map_location="cpu"); esd = torch.load(ENC_PATH, map_location="cpu")
    FP = int(dsd["dense.2.weight"].shape[0])
    dec = StegaStampDecoder(RES, 3, FP); dec.load_state_dict(dsd); dec.eval().to(device)
    enc = StegaStampEncoder(RES, 3, FP, False); enc.load_state_dict(esd); enc.eval().to(device)

    src = sorted(SRC_DIR.glob("src_*.png"), key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    assert len(src) == 25, f"expected 25 WM_3 carriers, got {len(src)}"
    bits = np.stack([decode(dec, load01(p)) for p in src])
    msg = (bits.mean(0) >= 0.5).astype(np.int64)
    consist = float(np.mean([bitacc(bits[i], msg) for i in range(25)]))
    clean = np.stack([decode(dec, load01(TARG_DIR / f"{i}.png")) for i in TARGET_IDS])
    clean_agree = float(np.mean([bitacc(clean[i], msg) for i in range(len(TARGET_IDS))]))
    print(f"decode-consistency={consist:.4f}  clean-control={clean_agree:.4f}  (device={device})")

    msg_t = torch.from_numpy(msg.astype(np.float32)).view(1, FP).to(device)
    A_GRID = [round(x, 2) for x in np.arange(0.30, 1.801, 0.10)]
    accs = []
    for idx in TARGET_IDS:
        y = load01(TARG_DIR / f"{idx}.png")
        with torch.no_grad():
            r = (enc(msg_t, to_t(y)) - to_t(y)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        best = None
        for a in A_GRID:
            _, f01 = round_u8(np.clip(y + a * r, 0, 1))
            ba = bitacc(decode(dec, f01), msg)
            sw = Sdet(ba) * min(Sqlt(lpips_dist("alex", y, f01)), Sqlt(lpips_dist("vgg", y, f01)))
            if best is None or sw > best[0]:
                best = (sw, a, ba)
        u8, _ = round_u8(np.clip(y + best[1] * r, 0, 1))
        Image.fromarray(u8).save(OUT_DIR / f"{idx}.png")
        accs.append(best[2])
        print(f"  {idx}: a={best[1]:.2f}  bitacc={best[2]:.3f}")
    print(f"\nWM_3 forge -> {OUT_DIR}   mean bit-accuracy={np.mean(accs):.4f}")


if __name__ == "__main__":
    main()
