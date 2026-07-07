import os, sys, glob
import numpy as np
import torch
from PIL import Image

PROJECT = os.environ.get("A4_ROOT", "/path/to/Assignment4")  # set to your project root
REPO = os.path.join(PROJECT, "MBRS_repo")
CKPT = os.path.join(PROJECT, "mbrs_pretrained/MBRS_256_256/EC_42.pth")
WM_DIR = os.path.join(PROJECT, "data/extracted/watermarked_sources/WM_6")
TGT_DIR = os.path.join(PROJECT, "data/extracted/clean_targets")
OUT_DIR = os.path.join(PROJECT, "forged_native/WM_6")
sys.path.insert(0, REPO)

from network.Encoder_MP_Decoder import EncoderDecoder  # noqa

device = torch.device("cpu")
H = W = 256
MSG_LEN = 256
STRENGTH = 1.0


def build_model():
    # noise_layers needs at least one identity-ish layer; use Identity() via combined string
    model = EncoderDecoder(H, W, MSG_LEN, noise_layers=["Identity()"])
    sd = torch.load(CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # only noise layer (no params) should be absent; encoder/decoder must load fully
    enc_dec_missing = [k for k in missing if k.startswith("encoder.") or k.startswith("decoder.")]
    assert len(enc_dec_missing) == 0, f"enc/dec missing: {enc_dec_missing[:5]}"
    enc_dec_unexpected = [k for k in unexpected if k.startswith("encoder.") or k.startswith("decoder.")]
    assert len(enc_dec_unexpected) == 0, f"enc/dec unexpected: {enc_dec_unexpected[:5]}"
    model.eval().to(device)
    return model


def img_to_tensor(path):
    im = Image.open(path).convert("RGB").resize((W, H), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5  # [-1,1]
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_uint8(t):
    # t: (1,3,H,W) in [-1,1]
    a = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    a = (a + 1.0) / 2.0 * 255.0  # [0,255]
    a = np.clip(np.round(a), 0, 255).astype(np.uint8)
    return a


def decode_bits(model, t):
    with torch.no_grad():
        out = model.decoder(t)
    return (out > 0.5).cpu().numpy().astype(np.int32).reshape(-1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = build_model()

    # ---- 1. consensus message from 25 carriers ----
    carriers = sorted(glob.glob(os.path.join(WM_DIR, "*.png")))
    assert len(carriers) == 25, f"expected 25 carriers, got {len(carriers)}"
    cbits = np.stack([decode_bits(model, img_to_tensor(f)) for f in carriers])
    # check perfect agreement
    n_agree_perfect = all(np.array_equal(cbits[0], cbits[i]) for i in range(len(carriers)))
    consensus = (cbits.mean(0) > 0.5).astype(np.int32)
    # decisiveness
    perbit = cbits.mean(0)
    decisiveness = float(np.mean(np.abs(perbit - 0.5) * 2))
    M = consensus.copy()
    msg_str = "".join(str(int(b)) for b in M)
    print(f"[consensus] all 25 carriers identical: {n_agree_perfect}  decisiveness={decisiveness:.4f}")
    print(f"[consensus] M = {msg_str}")

    # message tensor for encoder: float in {0,1}, shape (1, MSG_LEN)
    msg_t = torch.from_numpy(M.astype(np.float32)).unsqueeze(0).to(device)

    # ---- 2-5. encode into each target, verify, save ----
    bit_accs = []
    l1_diffs = []
    saved = 0
    for idx in range(126, 151):
        tgt_path = os.path.join(TGT_DIR, f"{idx}.png")
        img = img_to_tensor(tgt_path)
        with torch.no_grad():
            enc = model.encoder(img, msg_t)
        forged = img + (enc - img) * STRENGTH
        forged = torch.clamp(forged, -1.0, 1.0)

        # verify decode
        fbits = decode_bits(model, forged)
        acc = float(np.mean(fbits == M))
        bit_accs.append(acc)

        # save uint8 png
        u8 = tensor_to_uint8(forged)
        Image.fromarray(u8, mode="RGB").save(os.path.join(OUT_DIR, f"{idx}.png"))
        saved += 1

        # L1 diff vs clean target (0-255 units), using same uint8 round-trip of clean
        clean_u8 = tensor_to_uint8(img)  # clean re-encoded the same way -> matches input bicubic-resized
        l1 = float(np.mean(np.abs(u8.astype(np.float32) - clean_u8.astype(np.float32))))
        l1_diffs.append(l1)

    mean_acc = float(np.mean(bit_accs))
    mean_l1 = float(np.mean(l1_diffs))
    print(f"\n[RESULT] PNGs saved: {saved}")
    print(f"[RESULT] mean round-trip bit-accuracy: {mean_acc:.6f}  (min={min(bit_accs):.6f}, max={max(bit_accs):.6f})")
    print(f"[RESULT] mean per-pixel L1 (0-255): {mean_l1:.4f}")
    print(f"[RESULT] output dir: {OUT_DIR}")
    print(f"[RESULT] M ({len(msg_str)} bits) = {msg_str}")


if __name__ == "__main__":
    main()
