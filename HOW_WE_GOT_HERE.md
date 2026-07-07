# How we got to 0.8364 — the journey

A narrative companion to the report (`report/report.pdf`). The blow-by-blow log with every submission and
number is in `EXPERIMENT_LOG.md`; this file is the readable version.

**Best public score: `S_final = 0.8364`.**

## The task and the one fact that shaped everything

For each of 8 unknown watermarking schemes `WM_1..WM_8` we get 25 watermarked *carrier* images that all
carry **one** shared hidden message. We must transplant that watermark onto 25 assigned clean *target* images
(different images — true transplantation) so a hidden detector reads the message, while keeping each image
perceptually close. `WM_k` → targets `25(k-1)+1 .. 25k`.

Score: `S_final = S_det · S_qlt`, with `S_det = max(0, 2·(bitacc − 0.5))` and `S_qlt = exp(−8·LPIPS)`.
The decisive asymmetry: **LPIPS is measurable locally; bit-accuracy is NOT** (we have no detector). The only
detection signal is the leaderboard (1 submission/hour, best-kept, 30% public). `S_det` is a hard gate — 0.5
bit-accuracy zeroes the image. The LPIPS budget turned out to be large. So the whole game is **landing the bits**.

## Turning point: identify the scheme, don't blindly transplant

The literature's averaging attack (estimate the watermark as the mean carrier residual, add it to the target)
**fails** on frequency-domain and content-adaptive marks. On the two schemes where we could also decode, native
re-encoding scored `S_final` 0.89 / 0.86 versus **~0** for additive transplant. The lesson: if you **identify**
the scheme and re-embed the recovered message with its *own genuine encoder*, the forgery *is* the real
watermark — perfect and generalizing. Identification became the core strategy.

## The oracle test (identification without a detector)

The 25 carriers share one message, so for any candidate public decoder we decode all 25: if they agree on a
single message (**carrier-agreement ≈ 1.0**) while clean images decode randomly (**clean-agreement ≈ 0.5**),
the scheme is identified. Two guards made this trustworthy:
- a **validated positive control** — a genuine encode→decode round-trip proving the decoder actually reads marks;
- a **known-negative control** — a scheme we've already identified must not falsely match.

This rigor mattered. Many apparent "hits" were artifacts: a decoder returning a *constant* output regardless of
input, or content-correlation bias that also elevates clean images. One WM_3 "breakthrough" we investigated —
running classical decoders on WM_3's chroma channels — reported carrier-agreement 1.000, but the same decoder
returns the **identical constant bits on pure random noise**. That is a degenerate reading, not a recovered
message. It was a false positive; WM_3 stayed unidentified.

## Seven of eight, cracked

Public decoders plus a systematic sweep (one candidate scheme per isolated environment) identified 7/8,
each re-encoded natively at round-trip bit-accuracy 1.0 and tiny LPIPS:

| WM | scheme | forge |
|----|--------|-------|
| WM_1 | dwtDct (invisible-watermark) | native re-encode |
| WM_2 | RivaGAN (invisible-watermark) | native re-encode |
| WM_4 | VINE-R (ICLR'25) | native re-encode (GPU encoder) |
| WM_5 | CIN (MM'22) | native re-encode |
| WM_6 | MBRS (MM'21) | native re-encode |
| WM_7 | TrustMark-Q | native re-encode |
| WM_8 | TrustMark-P (`use_ECC=False`) | native re-encode |
| WM_3 | **unidentified custom chroma** | color-axis transplant (fallback) |

## WM_3 — the scheme we could not forge to native quality

We ruled out **~27 public decoders** for WM_3 under the positive-control bar (all VINE W-Bench baselines,
RoSteALS, InvisMark, CRMark, Stable-Signature, LaWa, SepMark, PIMoG, EditGuard, and classical / spread-spectrum
families). Forensics localize WM_3 to a **low-frequency green–magenta chroma** mark that is **content-keyed**
(per-image). We proved the transplant ceiling three independent ways:
1. **Strength / estimator sweeps plateau.** A 20%-trimmed-mean robust estimator beats the plain mean on *decodable*
   analog schemes but does **not** transfer to WM_3 — positive evidence of content-keying.
2. **Craver-style diagnostic:** a norm-matched template transplant reaches ~0.88 bit-accuracy on the analogous
   VINE mark, but WM_3's detector accepts the same-quality template at only ~0.72.
3. **Learned-manifold forger (WMCopier)** trained on the 25 carriers stayed *below* transplant parity — the
   25-carrier data wall (the paper used ~1000–5000).

Our best WM_3 forge is a **color-axis `R+B−2G` chroma transplant** with a bilateral-denoised template, capped
around `S_final ≈ 0.43`. A higher-strength version of this transplant is what took us from ~0.827 to **0.836**.

## Native robustness (final analysis) — and an honest negative result

The leaderboard grades bit-accuracy *after* an unknown pre-detection transform, so clean-PNG bit-accuracy
**overestimates** the server. We built a local **robustness oracle** (decode a forge → apply JPEG/resize/noise/
blur → re-decode → measure surviving bits) and audited all seven natives. Six are already robust; only **WM_1**
(classical dwtDct) is fragile (surviving bit-accuracy 0.60, collapses under JPEG). But a leaderboard probe that
strengthened WM_1's quantization margin was a **wash** (+0.0002): the quality cost cancelled the detection gain,
and dwtDct is JPEG-fragile at *every* strength. Conclusion: the natives sit near their achievable ceiling with
these public encoders — the remaining gap to the ~0.90 leaders is essentially the one scheme we couldn't crack, WM_3.

## The climb

`0.661` (3 schemes) → **`0.719`** (adding CIN + MBRS, +0.057) → **`0.821`** (adding VINE + TrustMark-P) →
`0.827` (chroma-domain WM_3) → **`0.836`** (color-axis WM_3 + native robustness). The jumps were driven
by *identification*, not leaderboard tuning — every gain is a per-scheme method that generalizes across the
30% public / 70% private split.

## Real-world takeaway

Watermark forgery under near-black-box conditions is practical and cheap: for any *deployed public* scheme, an
attacker with a handful of same-message watermarked images can identify the scheme and re-encode with its own
public encoder — a perfect, generalizing forgery. The one scheme we could **not** forge, WM_3, is exactly the
one that is content-keyed with a non-public detector. That is the defensive lesson: robust provenance needs
secret, content-bound, keyed watermarks whose detectors reject transplanted marks — not merely imperceptible ones.
