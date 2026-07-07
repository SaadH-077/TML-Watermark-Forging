# TML 2026 · Task 4 — Watermark Forgery — team_V

Forging image watermarks by **scheme identification and native re-encoding**. This repository is a
self-contained package that **reproduces our best leaderboard submission** and documents **how we got
there**. It ships only what is needed — no virtual-environments and no large model repositories.

**Best public score: `S_final = 0.8364`.**

---

## The task

For each of 8 unknown watermarking schemes `WM_1..WM_8` we are given 25 watermarked *carrier* images that
all carry **one** shared hidden message. We must transplant that watermark onto 25 assigned clean *target*
images (different images — true transplantation) so that a hidden detector reads the message, while
keeping each image perceptually close to its target. `WM_k` maps to targets `25(k−1)+1 .. 25k`.

The score is `S_final = S_det · S_qlt`, with `S_det = max(0, 2·(bitacc − 0.5))` and
`S_qlt = exp(−8·LPIPS)`. LPIPS is measurable locally; bit-accuracy is not (there is no public detector),
so the only detection signal is the leaderboard (1 submission / 60 min, best-kept, 30% public).

## Our approach in one line

**Identify each scheme, then re-embed the recovered message with the scheme's own genuine encoder** — the
forgery then *is* a real watermark (bit-accuracy 1.0, and it generalizes). A blind additive transplant
fails on frequency-domain and content-adaptive marks. We identified 7 of 8 schemes; the eighth (WM_3) is a
custom content-keyed chroma mark with no public decoder, for which we use a color-axis chroma transplant.

| schemes | forge |
|---|---|
| WM_1, WM_2, WM_4, WM_5, WM_6, WM_7, WM_8 | **identified → native re-encode** with the scheme's genuine encoder (dwtDct, RivaGAN, VINE-R, CIN, MBRS, TrustMark-Q, TrustMark-P). WM_1 is residual-amplified `s=1.5`. |
| WM_3 | **unidentified custom chroma scheme** → color-axis `R+B−2G` chroma transplant (bilateral template), fit to LPIPS ≈ 0.023. |

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
Leaderboard: <http://34.63.153.158/leaderboard_page>. Provide your own API key — none is bundled, and the
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
forged_native_masked/    # the 7 native forge outputs (25 PNGs each) — inputs to build_best.py
data/extracted/          # clean_targets (200) + watermarked_sources/WM_3 (25 carriers)
```

## Not included (large artifacts)

To keep the package light we omit the per-scheme model repositories (`*_repo`), all virtual-environments,
the full `data/Dataset.zip`, and alternate candidate submissions. Only WM_3's carriers are shipped, since
that is the only scheme `build_best.py` re-derives at run time; run `python3 fetch_data.py` to download all
eight schemes' carriers if you want to re-derive the native forges from scratch with the `encoders/` scripts.

Start with **`HOW_WE_GOT_HERE.md`** for the narrative, `EXPERIMENT_LOG.md` for the full log, and
`report/report.pdf` for the write-up.
