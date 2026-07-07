# TML 2026 · Task 4, Watermark Forgery, team_V

Forging image watermarks by **scheme identification and native re-encoding**. This repository is a
self-contained package that **reproduces our best leaderboard submission** and documents **how we got
there**. It ships only what is needed, no virtual-environments and no large model repositories.

**Best public score: `S_final = 0.8553`** (all eight schemes forged by native re-encoding).

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
frequency-domain and content-adaptive marks. We identified **all 8 schemes**; the eighth (WM_3) took longest,
a content-adaptive learned mark that resists the averaging attack, but it is **ArtificialGANFingerprints**
(a StegaStamp autoencoder), which we identified by decode-consistency and re-embed with the matching encoder.

### What we did for each watermark

| WM | targets | scheme (encoder) | how we forged it | LPIPS / S_final |
|----|---------|------------------|------------------|-----------------|
| WM_1 | 1–25 | **dwtDct** (`invisible-watermark`), 30-bit | Decoded the shared message from the 25 carriers and re-embedded it into each target with the genuine dwtDct encoder, then residual-amplified the forge at `s=1.5` (`clip(T + s·(F0−T))`) to widen the QIM quantization margin for robustness against the server's pre-detection transform. | 0.006 / 0.89 |
| WM_2 | 26–50 | **RivaGAN** (`invisible-watermark`), 32-bit | Decoded the 32-bit message and re-embedded it with the genuine RivaGAN encoder. Its LPIPS of 0.016 is encoder-inherent and is the quality floor among the native schemes. | 0.016 / 0.86 |
| WM_3 | 51–75 | **ArtificialGANFingerprints** (StegaStamp), 100-bit | Content-adaptive learned mark that resists averaging; we treated it as unidentifiable (≈27 decoders ruled out) until a decode-consistency sweep at the native 256² matched the public AFHQ cat2dog checkpoint (carrier-agreement 0.9996, clean ≈0.48). We majority-vote the 100-bit message and re-embed with the matching encoder (`clip(y+a·r)`), picking each image's strength `a` to maximise the worst case across the AlexNet and VGG LPIPS backbones. | 0.024 / 0.67–0.78 |
| WM_4 | 76–100 | **VINE-R** (ICLR 2025), 100-bit | Decoded the 100-bit message with the released VINE decoder and re-embedded it with the genuine VINE-R encoder (SDXL-Turbo based, one step; GPU, with a CPU fallback shim) at the native 256². | 0.003 / 0.94 |
| WM_5 | 101–125 | **CIN** (ACM MM 2022), 30-bit | Decoded the consensus 30-bit message and re-embedded it with the genuine CIN encoder at the native 128². | 0.001 / 0.99 |
| WM_6 | 126–150 | **MBRS** (ACM MM 2021), 256-bit | Decoded the consensus 256-bit message and re-embedded it with the genuine MBRS encoder at the native 256². | 0.002 / 0.96 |
| WM_7 | 151–175 | **TrustMark-Q**, 61-bit | Decoded the message with pip `trustmark` variant Q and re-embedded it with the genuine TrustMark-Q encoder at the native 512². | 0.001 / 0.99 |
| WM_8 | 176–200 | **TrustMark-P** (`use_ECC=False`), 100-bit raw | Same family as WM_7 but a different variant; it only matched once we tried variant P with error-correction disabled. Decoded the 100-bit raw message and re-embedded it with the genuine TrustMark-P encoder at the native 512². | 0.001 / 0.99 |

All eight forges are genuine native re-encodes, so they decode at high bit-accuracy (1.0 for the seven with
public decoders, 0.96 mean for WM_3) and generalize across the public/private split. With WM_3 now native,
the remaining gap to the top of the board is the quality and robustness of the public encoders, not a missing
scheme.

---

## Reproduce the best submission (one command, offline)

```bash
pip install -r requirements.txt
python3 build_best.py
# -> submissions/candidate_best.zip   (200 flat PNGs, 1.png..200.png)
```

`build_best.py` needs the 200 clean targets under `data/extracted/clean_targets/` (for the WM_1
residual amplification and the per-scheme LPIPS table) and the eight native forges under
`forged_native_masked/`. All eight forges, WM_3 included, are already frozen there, so the script simply
loads them, amplifies WM_1 at `s=1.5`, assembles the 200 images, validates the flat-zip contract, and prints
the per-scheme LPIPS / `S_qlt` table. To re-derive the WM_3 forge from scratch, run
`python3 encoders/afgf_wm3_encode.py` (self-contained, auto-downloads the AFGF weights; reproduces the frozen
`forged_native_masked/WM_3/` byte-for-byte).

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
requirements.txt         # pip dependencies (numpy, torch, torchvision, lpips, Pillow, ...)
fetch_data.py            # optional: re-download the full course dataset from Hugging Face
report/report.pdf        # the report (approach + results); report.tex is the LaTeX source
HOW_WE_GOT_HERE.md       # readable narrative of the strategy, what worked, and what didn't
EXPERIMENT_LOG.md        # full experiment & decision log: every submission, number, and lesson
src/wmforge.py           # core forging library + the WM_1/2/7 encoders/decoders
src/submit.py            # leaderboard submit + score reader (dry-run by default)
encoders/                # how each native forge was produced; afgf_wm3_encode.py is self-contained
forged_native_masked/    # the 8 native forge outputs (25 PNGs each), inputs to build_best.py
data/extracted/          # clean_targets (200) + watermarked_sources/WM_3 (25 carriers)
```

## Not included (large artifacts)

To keep the package light we omit the per-scheme model repositories (`*_repo`), all virtual-environments,
the full `data/Dataset.zip`, alternate candidate submissions, and the AFGF checkpoints (auto-downloaded by
`encoders/afgf_wm3_encode.py`). All eight native forges are already frozen under `forged_native_masked/`, so
`build_best.py` runs offline with just the clean targets. Run `python3 fetch_data.py` to download all eight
schemes' carriers if you want to re-derive the native forges from scratch with the `encoders/` scripts.

Start with **`HOW_WE_GOT_HERE.md`** for the narrative, `EXPERIMENT_LOG.md` for the full log, and
`report/report.pdf` for the write-up.
