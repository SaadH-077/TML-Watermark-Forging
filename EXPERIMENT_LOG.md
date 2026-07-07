# Experiment & decision log — team_V, TML 2026 Task 4 (Watermark Forgery)

This is our full working log: every experiment we ran, every submission we made, the number it
produced, and the decision it led to. It is the detailed companion to `HOW_WE_GOT_HERE.md` (the
readable narrative) and `report/report.pdf` (the write-up). It is organised by theme so the reasoning
is easy to follow; the chronological submission table near the end records the exact climb.

**Final best public score: `S_final = 0.8364`.**

---

## 1. The strategy in one screen

**Governing fact:** `S_final = S_det · S_qlt`, with `S_det = max(0, 2·(bitacc − 0.5))` and
`S_qlt = exp(−8·LPIPS)`. `S_det` is a **hard gate**: bit-accuracy 0.5 makes `S_det = 0`, which zeroes
the whole image regardless of quality. `S_qlt` is only a multiplier, and — crucially — **we can compute
LPIPS locally, but we cannot compute bit-accuracy** (we hold no detector). The only detection signal is
the leaderboard: 1 submission per 60 minutes, best-kept, 30% public / 70% private.

Two consequences shaped everything:
1. **Land the bits first, claw back LPIPS second.** The LPIPS budget turned out to be large (adding an
   averaged watermark at α up to ~3–4 costs almost nothing in `S_qlt`), so detection strength — not
   quality — is the binding constraint. Bias watermark strength high.
2. **Identify the scheme, then re-encode with its own genuine encoder.** A blind additive transplant of
   the averaged carrier residual fails on frequency-domain and content-adaptive marks. But if we can
   identify which public watermarking scheme produced the carriers, we decode the shared message and
   re-embed it into the target with that scheme's real encoder — the forgery then *is* a genuine
   watermark: bit-accuracy 1.0 and it generalizes across the public/private split.

**The oracle test (identification without a detector).** The 25 carriers of a scheme all share one
hidden message. So for any candidate public decoder we decode all 25 carriers: if they agree on a single
message (**carrier-agreement ≈ 1.0**) while clean images decode randomly (**clean-agreement ≈ 0.5**), the
scheme is identified. We guarded every "hit" two ways:
- a **validated positive control** — a genuine encode→decode round-trip proving the decoder actually
  reads real marks of that scheme;
- a **known-negative control** — an already-identified scheme (e.g. WM_1) must not falsely match.

This rigor mattered: many apparent hits were artifacts — a decoder returning a *constant* output
regardless of input, or content-correlation bias that also elevates clean images. Without the controls
we would have declared several false identifications.

---

## 2. Dataset & submission facts

**Dataset (`data/extracted/`).**
- 200 clean targets `{1..200}.png`; watermarked carriers under `watermarked_sources/WM_k/`, 25 per scheme.
- Carriers are **different images** from the targets (mean |target − carrier| ≈ 107/255) — this is true
  transplantation, not re-stamping the same picture.
- The 25 carriers within a scheme share one hidden message (this is the signal that averaging exploits).
- **Resolution is locked per scheme and matches the target batch:** WM_1,2,3,4,6 = 256²; WM_5 = 128²;
  WM_7,8 = 512². (Targets: 125 at 256², 25 at 128², 50 at 512².)
- All images are lossless PNG, RGB, uint8.

**Scheme → target mapping:** `WM_k` covers targets `25(k−1)+1 .. 25k`.

**Submission contract.**
- `POST http://34.63.153.158/submit/22-forging-task`, header `X-API-Key`, multipart `file=<zip>`.
- The zip must be **flat**: exactly 200 PNGs named `1.png .. 200.png`, no subfolders, no extras.
- 1 submission / 60 min (a malformed/failed submission has a 2 min cooldown). The server keeps your best
  result. Leaderboard page: `http://34.63.153.158/leaderboard_page` (the score is embedded in a JS map,
  which `src/submit.py::fetch_score` reads). The API key lives in a local `.env` (`TML_API_KEY=...`) that
  is never committed.

---

## 3. Watermark characterization (`scripts/01_explore.py`, `scripts/02_signal_probe.py`)

Coherent-signal fraction of the NL-means residual per scheme (random floor for N=25 carriers = 1/25 = 0.04):

| WM | res  | δσ (px) | signal_frac | FFT coh_peak | grid(8×8) | reading                              |
|----|------|---------|-------------|--------------|-----------|--------------------------------------|
| WM_1 | 256² | 1.24 | 1.6%  | 18.6 | 4.7  | low-amplitude **frequency** mark      |
| WM_2 | 256² | 1.20 | 1.4%  | 15.5 | 2.3  | low-amplitude frequency mark          |
| WM_3 | 256² | 1.51 | 6.3%  | 18.3 | 3.5  | moderate spatial                      |
| WM_4 | 256² | 1.30 | 5.7%  | 7.8  | 2.5  | moderate spatial                      |
| **WM_5** | 128² | 2.30 | **19.4%** | 10.5 | 3.0 | **strong spatial fixed pattern**  |
| WM_6 | 256² | 1.56 | 8.6%  | 41.0 | 5.2  | spatial + peaky frequency             |
| **WM_7** | 512² | 0.96 | 2.1% | **54.7** | **15.1** | **strong DCT / block-frequency mark** |
| WM_8 | 512² | 0.89 | 2.8%  | 20.6 | 11.0 | DCT / block-frequency mark            |

None look like pure content-adaptive noise, so an averaging-family estimate lands *some* signal on all 8.
For the frequency-domain marks (WM_1/2/7/8), a frequency-aware δ (keeping coherent FFT peaks) beats raw
pixel averaging — but the decisive lever turned out to be identification, not estimator polish.

**LPIPS budget (local AlexNet), adding an averaged δ to a representative target:**

| rep | α=1 | α=2 | α=3 | α=4 |
|-----|-----|-----|-----|-----|
| WM_1 (256²) | 0.000/1.00 | 0.002/0.98 | 0.005/0.96 | 0.008/0.94 |
| WM_5 (128²) | 0.000/1.00 | 0.000/1.00 | 0.001/1.00 | 0.001/0.99 |
| WM_7 (512²) | 0.001/0.99 | 0.004/0.97 | 0.012/0.91 | 0.024/0.82 |

(LPIPS / `S_qlt`.) Quality is nearly free → push α high; detection is the bottleneck.

---

## 4. Identification results — 7 of 8 schemes cracked

Each identified scheme was decoded on its 25 carriers to recover the shared message, then re-encoded
into the assigned targets with the scheme's own genuine encoder, and round-trip-verified (decode the
forgery → bit-accuracy 1.0).

| WM | scheme | library / repo | forge | LPIPS / S_final |
|----|--------|----------------|-------|-----------------|
| WM_1 | **dwtDct** (invisible-watermark), length 30 | `imwatermark` | native re-encode (later residual-amplified s=1.5) | 0.006 / 0.89 |
| WM_2 | **RivaGAN** (invisible-watermark), 32-bit | `imwatermark` | native re-encode | 0.016 / 0.86 |
| WM_4 | **VINE-R** (ICLR'25) | `VINE_repo`, SDXL-Turbo encoder (GPU) | native re-encode | 0.003 / 0.94 |
| WM_5 | **CIN** (MM'22), 30-bit | `CIN_repo` (`cinNet_nsmNet.pth`) | native re-encode | 0.001 / 0.99 |
| WM_6 | **MBRS** (MM'21), 256-bit | `MBRS_repo` (`EC_42.pth`) | native re-encode | 0.002 / 0.96 |
| WM_7 | **TrustMark-Q**, 61-bit | pip `trustmark` | native re-encode | 0.001 / 0.99 |
| WM_8 | **TrustMark-P** (`use_ECC=False`), 100-bit raw | pip `trustmark` | native re-encode | 0.001 / 0.99 |
| WM_3 | **unidentified** (custom chroma) | — | color-axis chroma transplant (fallback) | 0.023 / ≈0.43 |

Recovered messages and per-scheme carrier-agreement values are recorded in `notes/*.json`. Notes on the
identifications:
- **WM_1 dwtDct** decodes to an all-ones message (length-robust); this matched across every length we tried.
- **WM_7 vs WM_8** was the key subtlety: both are TrustMark and both are 512². We initially only tried
  variant **Q** (WM_7). WM_8 only cracked once we tried variant **P** with `use_ECC=False` (the 100-bit
  raw variant) — carrier-agreement 0.999, clean 0.61.
- **WM_5 CIN** and **WM_6 MBRS** were found with a parallel deep-decoder sweep and re-encoded on CPU.
- **WM_4 VINE-R** decodes with the released VINE decoder (carrier-agreement 1.0); the encoder is
  SDXL-Turbo-based and runs best on GPU (we added a CPU-fallback shim in `encoders/vine_encode.py`).

**Source benchmark.** All 7 identified schemes belong to the Shilin-LU/VINE (ICLR'25) "W-Bench" baseline
set (MBRS, CIN, PIMoG, RivaGAN, SepMark, TrustMark, DWTDCT, DWTDCTSVD, SSL, StegaStamp, EditGuard, VINE-R,
VINE-B). This benchmark was the map that let us enumerate and rule out candidates systematically.

---

## 5. WM_3 — the one scheme we could not identify

WM_3 was the entire remaining gap to the top of the board. We attacked it three ways — identification,
better transplant estimators, and a learned forger — and hit a wall on all three.

### 5.1 Identification: ~27 public decoders ruled out (with validated positive controls)

We ran the oracle test with a genuine positive control for each candidate. Under the bar "a real
same-scheme decode gives WM_3 carrier-agreement ≥ 0.95 with ≥ 90 unanimous bits, while content noise
gives 0.5–0.72 and ≤ 17 unanimous," every candidate failed:

- **Classical / spread-spectrum:** dwtDct, dwtDctSvd (120 configs, 63 positive-control-valid), blind
  watermark, 972+ spread-spectrum PN-key configs (PCG64/MT19937/Gaussian × seeds × lengths 16–256;
  DCT/Hadamard/grid bases; segment-CDMA). WM_3 stayed at chance under every valid transform. Its coherent
  template is a **single smooth low-frequency mode** (cos-vs-mean 0.909), not a multi-bit codeword.
- **Learned CNN enc-dec:** HiDDeN, MBRS, CIN, all VINE W-Bench baselines (VINE-R, VINE-B, SSL, PIMoG,
  StegaStamp, EditGuard), WAM, SepMark, ARWGAN, SteganoGAN. Several produced *elevated* carrier-agreement
  (VINE 0.64, RoSteALS 0.81, InvisMark 0.96, LaWa 0.73, CRMark 0.79) — but every one of these was a
  **content-correlation artifact**: clean images decoded almost identically, the ECC/BCH message test
  returned 0/25 valid codewords, and the positive control proved the decoder reads *real* marks of its
  own scheme at ≥ 0.99. So the elevation was the decoder's prior, not a recovered WM_3 message.
- **VAE-latent family:** RoSteALS, genuine 2-part Stable Signature (KL-f8 AE + fine-tuned decoder +
  48-bit whitened extractor), LaWa (ECCV'24) — all with positive controls at bit-accuracy 1.0, all NO_MATCH
  on WM_3. This closed the gap left by an earlier Stable-Signature rule-out that had used only the SSL
  48-bit extractor with no positive control.
- **Chroma-channel input:** because WM_3's mark lives in chroma, we also fed the R+B−2G / U / V channels
  into the learned decoders. Some classical decoders reported carrier-agreement 1.000 on chroma — but that
  was pure degeneracy: those decoders return the **identical constant bits on WM_3 carriers, on clean
  targets, and on pure random noise** (100% of bit positions identical across all inputs). A decoder whose
  output is invariant to its input has identified nothing. No public chroma-native learned decoder exists.

**Verdict:** WM_3 is an instructor-custom, content-keyed, low-frequency green–magenta chroma watermark
with no public decoder. Identification is exhausted on every axis.

### 5.2 Transplant ceiling — measured, not assumed (Craver diagnostic)

Because we *can* decode the analogous learned schemes (WM_4/VINE, WM_6/MBRS), we calibrated the transplant
method on them. A norm-matched transplant of the averaged template reaches bit-accuracy **0.882** on
WM_4/VINE — essentially at the ceiling (the optimal adversarial perturbation gets 0.999). So
content-adaptivity per se does **not** defeat transplanting.

Yet WM_3 has a near-identical template (top-1 SVD energy 0.088 vs WM_4's 0.085, cos-vs-mean 0.909 vs
0.917) and its real detector accepts the same-quality transplant at only **~0.72 bit-accuracy**
(`S_final ≈ 0.38`). That gap — same template quality, much lower acceptance — is positive evidence that
WM_3's detector demands a **per-image content-keyed** component that a copied template cannot supply.

### 5.3 Estimator studies — robust averaging helps analogs but not WM_3

We ran a full estimator study on the WM_8 (TrustMark-P) local oracle, the hardest *decodable*
content-adaptive analog of WM_3:
- **Winner: the per-pixel 20%-trimmed mean** of the NL-means residuals (`trim20`). At matched LPIPS it
  beat the plain arithmetic mean (WM_8 bit-accuracy 0.741 → 0.779; +0.054 `S_final`) and **generalized**:
  on WM_4/VINE it lifted bit-accuracy 0.882 → 0.906, and it was neutral on the already-saturated WM_5/WM_6.
- **Everything "fancier" failed:** the Kutter/Craver Wiener copy-attack → chance (the Wiener residual is
  content edges, not the mark); frequency/domain-matched masking → hurts (over-attenuates the band);
  spectral whitening / content-adaptive re-embedding → chance; the median → *worse* than the mean (it
  over-trims at N=25). So "averaging is naive" only in that the *arithmetic* mean isn't robust — the fix
  is a robust mean, not a fancier transform.
- **But `trim20` did not transfer to WM_3** (submissions v18/v19 were flat). An estimator that lifts the
  decodable analogs but not WM_3 is, again, positive evidence that WM_3 is content-keyed.

The Souček preference-model (WMForger) denoiser was worse than a bilateral filter on all five transplant
schemes in our 25-carrier averaging setting: it works at a fixed 768² and resizes back, smearing the
native-resolution fixed pattern (catastrophic for WM_5 at 128² and WM_8 at 512²), and it ships no encoder.

### 5.4 Learned forger (WMCopier) — below transplant parity at 25 carriers

WMCopier trains a small per-scheme diffusion UNet on the 25 carriers to memorize the shared watermark,
then injects it into targets via shallow DDIM inversion + score refinement. We built the full pipeline and
validated it end-to-end:
- **On WM_5 (CIN, content-independent), it works:** the jitter-only augmentation was essential (geometric
  augmentation destroys the frame-aligned watermark; the first run trained on watermark-free noise and gave
  chance). After the fix, forge bit-accuracy 0.996. Tuning the injection (the "beta" knob controls the
  LPIPS/bits trade-off) reached `S_final ≈ 0.77` on WM_5.
- **On the content-adaptive 256² marks it stays below the transplant.** Decoding a WMCopier WM_4 forge with
  the genuine VINE decoder gave bit-accuracy 0.765 (old injection) rising to 0.867 after a re-tuned
  injection sweep — but the LPIPS cost of the stronger injection tanks the product: best WM_4
  `S_final ≈ 0.498`, far below the transplant's 0.89.
- **On WM_3, two leaderboard probes (v20 old injection, and a re-tuned config) were both flat** — WMCopier
  did not beat the chroma transplant. The data wall is real: the paper's lowest tested regime used ~1000
  images; we have 25.

### 5.5 What we actually shipped for WM_3

Forensics localize WM_3's watermark to a low-frequency chroma mode (86% low-band, green–magenta
color-opponent, luma at the noise floor) that nonetheless retains a coherent additive chroma template
(SVD top component sign-consistent, ~9% energy) — more transplantable than a template-less mark **if
placed in-band**. Our progression of WM_3 forges:
1. plain `resid_mean` transplant → `S_final ≈ 0.33`;
2. chroma-domain transplant (low-pass, luma-suppressed, in-band) → ~0.38 (submission v13, a real gain);
3. **color-axis `R+B−2G` transplant with a bilateral-denoised template, swept to higher strength** →
   ~0.43, our final WM_3 forge. This is what `build_best.py` regenerates directly from the WM_3 carriers
   (α fit to an LPIPS budget of ~0.023). The strength sweep (v11, v16) and estimator sweep (v18, v19) all
   plateaued — WM_3's bit-accuracy does not rise with strength, confirming the content-keyed ceiling.

---

## 6. Native robustness audit — locating the residual gap to the top

The leaderboard grades bit-accuracy *after* an unknown pre-detection transform, so clean-PNG bit-accuracy
**overestimates** the server. We built a local **robustness oracle** (decode a forge → apply
JPEG 90/75/60, resize .5/.75, gaussian noise, blur → re-decode → measure surviving bits vs the clean-forge
decode) and audited all seven native forges with their genuine decoders:

| WM | scheme | robust bit-accuracy @ shipped strength | verdict |
|----|--------|----------------------------------------|---------|
| **WM_1** | **dwtDct** | **0.600** (jpeg75 0.30, blur 0.54; fine on resize/noise) | **fragile — the outlier** |
| WM_2 | RivaGAN | 0.979 | robust, quality-limited (amplifying only hurts) |
| WM_4 | VINE | 0.993 | robust |
| WM_5 | CIN | ~0.99 (inferred) | robust |
| WM_6 | MBRS | 0.982 | robust (JPEG-trained) |
| WM_7 | TrustMark-Q | 0.983 | robust |
| WM_8 | TrustMark-P | 0.989 | robust |

Six of seven natives are already robust and near-optimal. Only **WM_1 (classical dwtDct)** is fragile.
dwtDct is a QIM (quantization) watermark; we cannot change the scale the server decodes at, but residual
amplification `F_s = clip(T + s·(F0 − T))` widens the quantization margin. The oracle confirmed it: s=1.0
robust 0.60 → s=1.5 robust 0.71, clean-decode stays 0.985.

**Honest negative result.** We probed WM_1 at s=1.5 on the leaderboard (submission 4709): 0.836409 vs the
prior 0.836243 — a **+0.0002 wash**. The quality cost (`S_qlt` 0.955 → 0.893) exactly cancelled the
detection gain, and dwtDct is JPEG-fragile at *every* strength (jpeg75 stays ~0.32 even at s=1.8). The
natives sit near their achievable ceiling with these public encoders; the remaining gap to the ~0.90 top
score is essentially the one scheme we could not crack, WM_3, not a fixable native.

---

## 7. Score decomposition — where the gap actually is

Decomposing our best total `0.8362 = (7·native + WM_3)/8` with the WM_3 axis-transplant at ≈0.43 puts our
seven natives at an average of ≈0.894, and our per-native `S_qlt` averages 0.967. So if our native `S_det`
were ~1.0 we would already be near 0.90. The top score of 0.9022 fits the decomposition
`(7×0.984 + 0.33)/8 = 0.902` — i.e. **~0.97 on the natives with WM_3 also uncracked** — at least as well as
"WM_3 cracked." In other words the leaders' edge is most plausibly slightly stronger natives on the
server's robust detector plus the same uncracked WM_3, not a WM_3 breakthrough. We could not close that
last sliver without either WM_3's identity or a marginally more robust set of public encoders.

---

## 8. Chronological submission table

Team = **team_V**. Task id `22-forging-task`. Every gain came from a per-scheme method that generalizes
across the 30% public / 70% private split, not from tuning to the public sample.

| sub id | date | candidate | public S_final | what changed / learned |
|--------|------|-----------|----------------|------------------------|
| 3394 | 2026-06-21 | `candidate_recon` — WM_1,2,7 native + WM_3–8 transplant α=3 | **0.6613** | first submission; transplant lands bits on the unidentified schemes |
| 3441 | 2026-06-22 | `candidate_v2` — per-scheme α tuned to LPIPS 0.02 | 0.6617 | +0.0004: raw α-tuning is essentially flat |
| 3453 | 2026-06-22 | `candidate_v3` — bilateral denoiser for cleaner δ | ≤0.6617 | cleaner estimate didn't help → blind-transplant route exhausted |
| 3461 | 2026-06-22 | `candidate_v4` — **5 native** (WM_1,2,5,6,7) + transplant WM_3,4,8 | **0.7188** | +0.057: CIN + MBRS native re-encode delivered |
| 3602 | 2026-06-24 | `candidate_v5` — WMCopier on WM_3,4,8 | 0.7188 | flat: WMCopier did not beat transplant on the hard schemes |
| 3608 | 2026-06-24 | `candidate_v6` — WMCopier WM_3,4 + transplant WM_8 | 0.7188 | flat: confirms WMCopier ≤ transplant here |
| 3612 | 2026-06-24 | `candidate_v8` — **7 native** + WM_3 `resid_mean` transplant | **0.8211** | +0.049: WM_4 VINE-R and WM_8 TrustMark-P cracked |
| 4122 | 2026-07-01 | `candidate_v9` — WM_3 `diff_means`+hp+texmask | ≤0.8211 | `diff_means` is the wrong estimator for a hard scheme |
| 4126 | 2026-07-01 | `candidate_v11` — WM_3 `resid_mean`+texmask+strength | ≤0.8211 | even the analog-optimal recipe didn't beat plain `resid_mean` |
| 4130 | 2026-07-01 | `candidate_v12` — native-opt (strength trims) + chroma WM_3 | ≤0.8211 | **key lesson:** cutting strength to lower LPIPS hurts the robust server detector |
| 4134 | 2026-07-01 | `candidate_v13` — v8 natives + **WM_3 chroma transplant** | **0.8268** | +0.0057: in-band chroma forensics paid off (WM_3 0.33 → ~0.38) |
| 4140 | 2026-07-01 | `candidate_v15` — perceptual-masked natives + chroma WM_3 | **0.8274** | +0.0007: robustness-verified masking transferred marginally |
| 4149 | 2026-07-01 | `candidate_v16` — chroma WM_3 at s=1.4 | 0.8274 | flat: WM_3 strength lever exhausted |
| —    | 2026-07-01 | `candidate_v18` — WM_3 `trim20`+low-freq-boost | 0.8274 | flat: boost hurt WM_3 (as it hurt the 256² analogs) |
| 4170 | 2026-07-01 | `candidate_v19` — WM_3 `trim20`-only | 0.8274 | flat: `trim20` win on analogs did not transfer → WM_3 content-keyed |
| 4175 | 2026-07-02 | `candidate_v20` — WMCopier-WM_3 (old injection) | 0.8274 | flat |
| 4223 | 2026-07-02 | WMCopier-WM_3 (re-tuned injection) | 0.8274 | flat: learned-forger path near-dead at 25 carriers |
| 4227 | 2026-07-02 | `candidate_v21` | 0.8274 | flat |
| 4244 | 2026-07-02 | `candidate_v22` — WM_3 estimator ensemble | 0.8274 | flat |
| 4643 | 2026-07-05 | **color-axis `R+B−2G` chroma transplant** (bilateral template, higher strength) | **0.8362** | +0.0088: the biggest WM_3 gain — a better estimator + more in-band chroma energy |
| 4709 | 2026-07-05 | `candidate_wm1strong` — WM_1 dwtDct residual-amplified s=1.5 | **0.836409** | +0.0002 wash: dwtDct is JPEG-fragile at every strength (robustness hypothesis falsified) |

**The climb:** `0.6613` (3 native) → `0.7188` (+CIN, +MBRS) → `0.8211` (+VINE, +TrustMark-P) →
`0.8268` (chroma WM_3) → `0.8274` (masked natives) → `0.8362` (color-axis WM_3) → **`0.8364`** (final).

---

## 9. What did *not* work (so we don't repeat it)

- **α / strength tuning of the blind transplant** — flat after the first submission; the limiter is
  estimate quality and, ultimately, scheme identity, not strength.
- **Reducing watermark strength to lower LPIPS** — a real trap (submission v12). It lowers LPIPS locally
  but reduces robustness against the server's pre-detection transform, costing more `S_det` than the
  `S_qlt` it buys. Keep full genuine-strength forgeries.
- **"Fancy" transplant estimators** — Wiener copy-attack, frequency-matched masking, spectral whitening,
  content-adaptive re-embedding all performed at or below chance on the hard analog. Only the robust
  trimmed mean beat the plain mean, and even that didn't transfer to WM_3.
- **Per-image learned encoder** (train content→watermark map from 25 carriers) — chance bit-accuracy;
  with no clean sources the watermark sits below the noise floor and 25 examples can't teach the map.
  Averaging is the right denoiser for the shared signal; a per-image learner has no equivalent.
- **WMCopier on the hard schemes** — validated on WM_5 but below transplant parity on the content-adaptive
  256² marks; the 25-carrier data wall is real and oracle-confirmed.
- **Amplifying the fragile native (WM_1)** — a wash; dwtDct is JPEG-fragile at every strength.

---

## 10. Reproducing the best submission

Everything needed to rebuild the best `S_final = 0.8364` submission offline is in this repository:

```bash
pip install -r requirements.txt
python3 build_best.py        # -> submissions/candidate_best.zip  (200 flat PNGs, 1.png..200.png)
```

`build_best.py` regenerates the WM_3 color-axis transplant from the WM_3 carriers, assembles the seven
frozen native forges (WM_1 residual-amplified s=1.5), validates the flat-zip contract, and prints the
per-scheme LPIPS / `S_qlt` table. To submit:

```bash
echo "TML_API_KEY=<your key>" > .env
python3 -m src.submit submissions/candidate_best.zip --yes
```

The native re-encoders themselves (CIN, MBRS, VINE, TrustMark) are documented in `encoders/` for
reference; each needs its own repo + pinned environment and is **not** required to rebuild the best zip —
their outputs are already frozen in `forged_native_masked/`.
