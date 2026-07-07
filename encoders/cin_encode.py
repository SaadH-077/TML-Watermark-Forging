import os, sys, glob
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

PROJECT = os.environ.get("A4_ROOT", "/path/to/Assignment4")  # set to your project root
REPO = os.path.join(PROJECT, "CIN_repo")
sys.path.insert(0, os.path.join(REPO, "codes"))

from utils.yml import parse_yml, dict_to_nonedict
import torch.nn as nn
from models.IM import IM
from models.FSM import FSM
from models.DEM import DEM
from models.modules.InvDownscaling import InvDownscaling
from models.NIAM import NIAM

DATA = os.path.join(PROJECT, "data", "extracted")
WM_DIR = os.path.join(DATA, "watermarked_sources", "WM_5")
CLEAN_DIR = os.path.join(DATA, "clean_targets")
CKPT = os.path.join(REPO, "pth", "cinNet_nsmNet.pth")
OUT_DIR = os.path.join(PROJECT, "forged_native", "WM_5")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")
opt = dict_to_nonedict(parse_yml(os.path.join(REPO, "codes", "options", "opt.yml")))


class CINCodec(nn.Module):
    """Encode + decode-only module (no Noise_pool / nsm_model)."""
    def __init__(self, opt, device):
        super().__init__()
        self.h, self.w = opt['network']['H'], opt['network']['W']
        self.msg_length = opt['network']['message_length']
        self.invertible_model = IM(opt).to(device)        # invertible_model.*
        self.cs_model = FSM(opt).to(device)               # (no params)
        self.fusion_model = DEM(opt).to(device)           # fusion_model.*
        self.invDown = InvDownscaling(opt).to(device)     # invDown.*
        self.decoder2 = NIAM(self.h, self.w, self.msg_length).to(device)  # decoder2.*

    def encoder(self, image, msg):
        cover_down = self.invDown(image)                       # [64]
        fusion = self.fusion_model(cover_down, msg, self.invDown)
        inv_encoded = self.invertible_model(fusion)            # forward
        cs = self.cs_model(inv_encoded, cover_down)            # rev=False
        watermarking_img = self.invDown(cs, rev=True).clamp(-1, 1)  # [128]
        return watermarking_img


model = CINCodec(opt, device).to(device)
model.eval()

# load ckpt: top key 'cinNet', params prefixed 'module.'
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
flat = {}
for topk, sub in ck.items():
    for k, v in sub.items():
        flat[k] = v
msd = model.state_dict()
loaded, skipped = 0, 0
new = {}
for k, v in msd.items():
    cand = "module." + k
    if cand in flat and flat[cand].shape == v.shape:
        new[k] = flat[cand]; loaded += 1
    elif k in flat and flat[k].shape == v.shape:
        new[k] = flat[k]; loaded += 1
    else:
        new[k] = v; skipped += 1
model.load_state_dict(new)
print(f"[ckpt] loaded={loaded} skipped(kept-init)={skipped} of {len(msd)}", file=sys.stderr)

norm = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def load_img(path):
    im = Image.open(path).convert("RGB")
    if im.size != (128, 128):
        im = im.resize((128, 128), Image.BILINEAR)
    t_ = T.ToTensor()(im)        # [0,1]
    t_ = norm(t_)                # [-1,1]
    return t_.unsqueeze(0).to(device)


@torch.no_grad()
def decode_inv(x):
    down = model.invDown(x)
    cs_rev = model.cs_model(down, rev=True)
    inv_back = model.invertible_model(cs_rev, rev=True)
    _, msg1 = model.fusion_model(inv_back, None, model.invDown, rev=True)
    return msg1.squeeze(0).cpu().numpy()


def to_bits(msg):
    return (np.array(msg) > 0.5).astype(np.int8)


# ---------- 1. Decode all 25 WM_5 carriers, get consensus M ----------
carrier_files = sorted(glob.glob(os.path.join(WM_DIR, "*.png")))
assert len(carrier_files) == 25, f"expected 25 carriers, got {len(carrier_files)}"
carr_bits = [to_bits(decode_inv(load_img(f))) for f in carrier_files]
arr = np.stack(carr_bits, 0)
consensus = (arr.mean(0) > 0.5).astype(np.int8)
agree = float(np.mean([np.mean(b == consensus) for b in carr_bits]))
M_str = "".join(str(int(b)) for b in consensus)
print(f"[decode] carrier consensus agreement={agree:.4f}", file=sys.stderr)
print(f"[decode] M = {M_str}", file=sys.stderr)
expected = "000101111111000000100011001101"
print(f"[decode] matches expected={M_str == expected}", file=sys.stderr)

# message tensor: mod_a -> raw 0/1 floats in [0,1], shape [1, 30]
msg_t = torch.tensor(consensus.astype(np.float32)).unsqueeze(0).to(device)


# ---------- 2-5. Encode M into each target 101..125, verify, save ----------
def denorm_to_uint8(x):
    # x in [-1,1] -> [0,255] uint8
    img = (x.squeeze(0).cpu().clamp(-1, 1) + 1.0) / 2.0   # [0,1]
    img = (img * 255.0).round().clamp(0, 255).byte()
    return img.permute(1, 2, 0).numpy()                   # HWC


bit_accs = []
l1_diffs = []
saved = 0
for idx in range(101, 126):
    src = os.path.join(CLEAN_DIR, f"{idx}.png")
    cover = load_img(src)
    with torch.no_grad():
        wm = model.encoder(cover, msg_t)                  # [-1,1]
        # verify: decode forged
        rec = to_bits(decode_inv(wm))
    acc = float(np.mean(rec == consensus))
    bit_accs.append(acc)

    forged_u8 = denorm_to_uint8(wm)
    clean_u8 = np.array(Image.open(src).convert("RGB").resize((128, 128), Image.BILINEAR)) \
        if Image.open(src).size != (128, 128) else np.array(Image.open(src).convert("RGB"))
    l1 = float(np.mean(np.abs(forged_u8.astype(np.int32) - clean_u8.astype(np.int32))))
    l1_diffs.append(l1)

    Image.fromarray(forged_u8, "RGB").save(os.path.join(OUT_DIR, f"{idx}.png"))
    saved += 1
    print(f"  {idx}.png  bit_acc={acc:.4f}  L1={l1:.3f}", file=sys.stderr)

mean_acc = float(np.mean(bit_accs))
mean_l1 = float(np.mean(l1_diffs))
print("\n=== SUMMARY ===", file=sys.stderr)
print(f"mean round-trip bit-accuracy = {mean_acc:.6f}", file=sys.stderr)
print(f"PNGs saved = {saved}", file=sys.stderr)
print(f"mean per-pixel L1 (0-255) = {mean_l1:.4f}", file=sys.stderr)
print(f"M = {M_str}", file=sys.stderr)
print(f"out_dir = {OUT_DIR}", file=sys.stderr)
print("RESULT_JSON=" + repr({
    "mean_bit_acc": mean_acc,
    "pngs_saved": saved,
    "mean_l1": mean_l1,
    "M": M_str,
    "M_matches_expected": M_str == expected,
    "carrier_agreement": agree,
    "out_dir": OUT_DIR,
}))
