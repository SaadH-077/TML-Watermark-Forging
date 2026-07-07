# TML 2026 · Task 4, Watermark Forgery, team_V

Forging image watermarks by **scheme identification and native re-encoding**. This repository is a
self-contained package that **reproduces our best leaderboard submission** and documents **how we got
there**. It ships only what is needed, no virtual-environments and no large model repositories.

**Best public score: `S_final = 0.8364`.**

---

## The task

For each of 8 unknown watermarking schemes `WM_1..WM_8` we are given 25 watermarked *carrier* images that
all carry **one** shared hidden message. We must transplant that watermark onto 25 assigned clean *target*
images (different images, true transplantation) so that a hidden detector reads the message, while
keeping each image perceptually close to its target. `WM_k` maps to targets `25(k−1)+1 .. 25k`.

The score is `S_final = S_det · S_qlt`, with `S_det = max(0, 2·(bitacc − 0.5))` and
`S_qlt = exp(−8·LPIPS)`. LPIPS is measurable locally; bit-accuracy is not (there is no public detector),
so the only detection signal is the leaderboard (1 submission / 60 min, best-kept, 30% public).

## Our approach in one line

**Identify each scheme, then re-embed the recovered message with the scheme's own genuine encoder** so the
forgery *is* a real watermark (bit-accuracy 1.0, and it generalizes). A blind additive transplant fails on
frequency-domain and content-adaptive marks. We identified 7 of 8 schemes; the eighth (WM_3) is a custom
content-keyed chroma mark with no public decoder, for which we use a color-axis chroma transplant.

### What we did for each watermark

| WM | targets | scheme (encoder) | how we forged it | LPIPS / S_final |
|----|---------|------------------|------------------|-----------------|
| WM_1 | 1–25 | **dwtDct** (`invisible-watermark`), 30-bit | Decoded the shared message from the 25 carriers and re-embedded it into each target with the genuine dwtDct encoder, then residual-amplified the forge at `s=1.5` (`clip(T + s·(F0−T))`) to widen the QIM quantization margin for robustness against the server's pre-detection transform. | 0.006 / 0.89 |
| WM_2 | 26–50 | **RivaGAN** (`invisible-watermark`), 32-bit | Decoded the 32-bit message and re-embedded it with the genuine RivaGAN encoder. Its LPIPS of 0.016 is encoder-inherent and is the quality floor among the native schemes. | 0.016 / 0.86 |
| WM_3 | 51–75 | **unidentified** custom low-frequency chroma | No public decoder matched (about 27 ruled out with positive controls). We transplant the shared chroma template: the mean of the bilateral-denoised carrier residuals projected on the color axis `R+B−2G` (unit vector `[0.37, −0.834, 0.41]`), low-pass filtered, added to each target at strength α fit so LPIPS ≈ 0.023. | 0.023 / ≈0.43 |
| WM_4 | 76–100 | **VINE-R** (ICLR 2025), 100-bit | Decoded the 100-bit message with the released VINE decoder and re-embedded it with the genuine VINE-R encoder (SDXL-Turbo based, one step; GPU, with a CPU fallback shim) at the native 256². | 0.003 / 0.94 |
| WM_5 | 101–125 | **CIN** (ACM MM 2022), 30-bit | Decoded the consensus 30-bit message and re-embedded it with the genuine CIN encoder at the native 128². | 0.001 / 0.99 |
| WM_6 | 126–150 | **MBRS** (ACM MM 2021), 256-bit | Decoded the consensus 256-bit message and re-embedded it with the genuine MBRS encoder at the native 256². | 0.002 / 0.96 |
| WM_7 | 151–175 | **TrustMark-Q**, 61-bit | Decoded the message with pip `trustmark` variant Q and re-embedded it with the genuine TrustMark-Q encoder at the native 512². | 0.001 / 0.99 |
| WM_8 | 176–200 | **TrustMark-P** (`use_ECC=False`), 100-bit raw | Same family as WM_7 but a different variant; it only matched once we tried variant P with error-correction disabled. Decoded the 100-bit raw message and re-embedded it with the genuine TrustMark-P encoder at the native 512². | 0.001 / 0.99 |

For the seven identified schemes the forge is a genuine re-encode, so it decodes at bit-accuracy 1.0 and
generalizes across the public/private split. WM_3 is the only scheme without a public decoder, and its
color-axis transplant is the single largest remaining gap in our score.

---

## Reproduce the best submission (one command, offline)

```bash
pip install -r requirements.txt
python3 build_best.py
# -> submissions/candidate_best.zip   (200 flat PNGs, 1.png..200.png)
```

Everything `build_best.py` needs is already in this folder: the WM_3 carriers under
`data/extracted/watermarked_sources/WM_3/`, the 200 clean targets under `data/extracted/clean_targets/`,
and the seven native forges under `forged_native_masked/`. It regenerates the WM_3 color-axis transplant,
assembles the 200 images, validates the flat-zip contract, and prints the per-scheme LPIPS / `S_qlt` table.

## Submit it

```bash
echo "TML_API_KEY=<your key>" > .env
python3 -m src.submit submissions/candidate_best.zip --yes
```

Endpoint `POST http://34.63.153.158/submit/22-forging-task`; 1 submission / 60 min, best-kept.
Leaderboard: <http://34.63.153.158/leaderboard_page>. Provide your own API key, none is bundled, and the
`.env` file is git-ignored.

Without `--yes`, `src/submit.py` performs a dry run that validates the zip and the key but does not submit.

---

## What's in this repository

```
build_best.py            # the reproduce script (one command, offline)
requirements.txt         # pip dependencies (numpy, opencv-python, torch, torchvision, lpips, Pillow)
fetch_data.py            # optional: re-download the full course dataset from Hugging Face
report/report.pdf        # the report (approach + results); report.tex is the LaTeX source
HOW_WE_GOT_HERE.md       # readable narrative of the strategy, what worked, and what didn't
EXPERIMENT_LOG.md        # full experiment & decision log: every submission, number, and lesson
src/wmforge.py           # core forging library + the WM_1/2/7 encoders/decoders
src/submit.py            # leaderboard submit + score reader (dry-run by default)
encoders/                # reference: how each native forge was produced (per-scheme repos + envs)
forged_native_masked/    # the 7 native forge outputs (25 PNGs each), inputs to build_best.py
data/extracted/          # clean_targets (200) + watermarked_sources/WM_3 (25 carriers)
```

## Not included (large artifacts)

To keep the package light we omit the per-scheme model repositories (`*_repo`), all virtual-environments,
the full `data/Dataset.zip`, and alternate candidate submissions. Only WM_3's carriers are shipped, since
that is the only scheme `build_best.py` re-derives at run time; run `python3 fetch_data.py` to download all
eight schemes' carriers if you want to re-derive the native forges from scratch with the `encoders/` scripts.

Start with **`HOW_WE_GOT_HERE.md`** for the narrative, `EXPERIMENT_LOG.md` for the full log, and
`report/report.pdf` for the write-up.
