# Native encoders (reference: how each native forge was produced)

**You do NOT need these to reproduce the best result.** The genuine native re-encodes are already in
`../forged_native_masked/`, and `../build_best.py` assembles the submission from them. These scripts document
*how* those forges were made: identify the scheme, decode the 25 carriers to recover the shared message,
re-embed it into the 25 targets with the scheme's genuine encoder, and round-trip-verify bit-accuracy = 1.0.

Each requires its own repo + pinned virtual-env (not bundled; they are the "bigger artifacts / venvs" left out)
and the full dataset (`python3 ../fetch_data.py`). Paths inside the scripts point at the original project root and
must be adjusted.

| script | scheme | needs |
|---|---|---|
| (in `../src/wmforge.py`) | WM_1 dwtDct, WM_2 RivaGAN, WM_7 TrustMark-Q | `pip install invisible-watermark trustmark` |
| `wm8_encode.py` | WM_8 TrustMark-P (`use_ECC=False`) | `pip install trustmark` |
| `cin_encode.py`  | WM_5 CIN (MM'22) | `CIN_repo` + `.venv_cin` (checkpoint `cinNet_nsmNet.pth`) |
| `mbrs_encode.py` | WM_6 MBRS (MM'21) | `MBRS_repo` + `.venv_mbrs` (checkpoint `EC_42.pth`) |
| `vine_encode.py` | WM_4 VINE-R (ICLR'25) | `VINE_repo` + `.venv_vine_enc` (SDXL-Turbo encoder, GPU) |
| `afgf_wm3_encode.py` | WM_3 ArtificialGANFingerprints / StegaStamp | `pip install gdown` (weights auto-download); **self-contained, no repo/venv** |

**WM_3** was long thought unidentifiable, but it is **ArtificialGANFingerprints** (Yu et al., ICCV 2021),
a StegaStamp autoencoder, public checkpoint `AFHQ_cat2dog_256x256`. The public decoder reads the 25 WM_3
carriers at cross-carrier agreement **0.9996** (clean targets ~0.48 = chance), which can only happen if the
benchmark used these exact public weights, so the hidden detector *is* this decoder. `afgf_wm3_encode.py` is
**self-contained** (it bundles the StegaStamp encoder/decoder and auto-downloads the weights): it decodes the
25 carriers, majority-votes the 100-bit message, and re-embeds it into each target with the matching encoder
(`forged = clip(y + a·r)`, `r = Enc(msg, y) − y`), choosing each image's strength `a` to maximise the
worst-case score across the AlexNet and VGG LPIPS backbones. Mean bit-accuracy 0.96, robust to ±1px and
Gaussian blur. It writes the 25 forges to `../forged_native_masked/WM_3/`, which `../build_best.py` consumes
directly like the other seven native forges:

```bash
python3 encoders/afgf_wm3_encode.py     # from the repo root; -> forged_native_masked/WM_3/51..75.png
```
