"""WM_4 = VINE-R native forge. Encode the recovered 100-bit WM_4 message onto clean targets
76..100 (WM_4's batch, 256x256 == VINE's native encode resolution) using the genuine VINE-R encoder
(VINE_Turbo, SDXL-Turbo-adapted, one-step). Output -> forged_native/WM_4/<id>.png.
Validate afterwards by decoding with .venv_vine (CustomConvNeXt) -> bitacc.
Runs on GPU if available, else CPU (slow but works for 25 images).
"""
import os, sys, time, torch
from PIL import Image
from torchvision import transforms

# CPU-fallback shim: VINE_Turbo hardcodes .cuda()/'cuda'/empty_cache internally (skip-convs,
# __init__ device default, scheduler). On a CPU-only box, neutralize/remap them to CPU so the
# genuine encoder runs without a GPU. No-op when CUDA is present.
if not torch.cuda.is_available():
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self
    _m_to = torch.nn.Module.to
    def _mod_to(self, *a, **k):
        a = tuple('cpu' if x == 'cuda' else x for x in a)
        if k.get('device') == 'cuda': k['device'] = 'cpu'
        return _m_to(self, *a, **k)
    torch.nn.Module.to = _mod_to
    _t_to = torch.Tensor.to
    def _ten_to(self, *a, **k):
        a = tuple('cpu' if x == 'cuda' else x for x in a)
        if k.get('device') == 'cuda': k['device'] = 'cpu'
        return _t_to(self, *a, **k)
    torch.Tensor.to = _ten_to
    _mk = torch.tensor
    def _cpu_tensor(*a, **k):
        if k.get('device') == 'cuda': k['device'] = 'cpu'
        return _mk(*a, **k)
    torch.tensor = _cpu_tensor
    torch.cuda.empty_cache = lambda *a, **k: None

PROJ = os.environ.get("A4_ROOT", "/path/to/Assignment4")  # set to your project root
REPO = os.path.join(PROJ, "VINE_repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "vine", "src"))
from vine_turbo import VINE_Turbo

# WM_4 consensus message (notes/vine_oracle_VINE-R-Dec.json, carrier_agree 1.0, all 25 unanimous)
MSG = "1100100101010001111111010110110011110000011111100011100101011111000010000001000010001101100011110001"
CLEAN = os.path.join(PROJ, "data", "extracted", "clean_targets")
OUT = os.path.join(PROJ, "forged_native", "WM_4")
os.makedirs(OUT, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[vine] device={device}; loading VINE-R-Enc ...", flush=True)
enc = VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc")
enc.to(device).eval()
wm = torch.tensor([int(c) for c in MSG], dtype=torch.float).unsqueeze(0).to(device)
assert wm.shape[1] == 100, wm.shape

t256 = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

ids = list(range(76, 101))  # WM_4 -> targets 76..100
t0 = time.time()
for k, tid in enumerate(ids, 1):
    im = Image.open(os.path.join(CLEAN, f"{tid}.png")).convert("RGB")
    size = im.size  # WM_4 targets are 256x256
    r = (2.0 * t256(im) - 1.0).unsqueeze(0).to(device)           # [1,3,256,256] in [-1,1]
    full = (2.0 * transforms.ToTensor()(im) - 1.0).unsqueeze(0).to(device)
    with torch.no_grad():
        enc256 = enc(r, wm)
    resid = enc256 - r
    if size != (256, 256):
        resid = transforms.Resize((size[1], size[0]),
                                  interpolation=transforms.InterpolationMode.BICUBIC)(resid)
    out = torch.clamp((resid + full) * 0.5 + 0.5, 0.0, 1.0)
    transforms.ToPILImage()(out[0].cpu()).save(os.path.join(OUT, f"{tid}.png"))
    print(f"[vine] {k}/25 -> {tid}.png ({(time.time()-t0)/k:.1f}s/img avg)", flush=True)

print(f"[vine] DONE: 25 forged WM_4 PNGs in {OUT} ({time.time()-t0:.0f}s total)", flush=True)
