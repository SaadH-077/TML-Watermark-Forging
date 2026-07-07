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

WM_3 was never identified (see the report / `../HOW_WE_GOT_HERE.md`); its forge is the color-axis chroma
transplant, regenerated directly inside `../build_best.py` from the WM_3 carriers.
