# How we got to 0.8553, the journey

A narrative companion to the report (`report/report.pdf`). The blow-by-blow log with every submission and
number is in `EXPERIMENT_LOG.md`; this file is the readable version.

**Best public score: `S_final = 0.8553`** (all eight schemes forged by native re-encoding).

## The task and the one fact that shaped everything

For each of 8 unknown watermarking schemes `WM_1..WM_8` we get 25 watermarked *carrier* images that all
carry **one** shared hidden message. We must transplant that watermark onto 25 assigned clean *target* images
(different images, true transplantation) so a hidden detector reads the message, while keeping each image
perceptually close. `WM_k` → targets `25(k-1)+1 .. 25k`.

Score: `S_final = S_det · S_qlt`, with `S_det = max(0, 2·(bitacc − 0.5))` and `S_qlt = exp(−8·LPIPS)`.
The decisive asymmetry: **LPIPS is measurable locally; bit-accuracy is NOT** (we have no detector). The only
detection signal is the leaderboard (1 submission/hour, best-kept, 30% public). `S_det` is a hard gate, 0.5
bit-accuracy zeroes the image. The LPIPS budget turned out to be large. So the whole game is **landing the bits**.

## Turning point: identify the scheme, don't blindly transplant

The literature's averaging attack (estimate the watermark as the mean carrier residual, add it to the target)
**fails** on frequency-domain and content-adaptive marks. On the two schemes where we could also decode, native
re-encoding scored `S_final` 0.89 / 0.86 versus **~0** for additive transplant. The lesson: if you **identify**
the scheme and re-embed the recovered message with its *own genuine encoder*, the forgery *is* the real
watermark, perfect and generalizing. Identification became the core strategy.

## The oracle test (identification without a detector)

The 25 carriers share one message, so for any candidate public decoder we decode all 25: if they agree on a
single message (**carrier-agreement ≈ 1.0**) while clean images decode randomly (**clean-agreement ≈ 0.5**),
the scheme is identified. Two guards made this trustworthy:
- a **validated positive control**: a genuine encode→decode round-trip proving the decoder actually reads marks;
- a **known-negative control**: a scheme we've already identified must not falsely match.

This rigor mattered. Many apparent "hits" were artifacts: a decoder returning a *constant* output regardless of
input, or content-correlation bias that also elevates clean images. One WM_3 "breakthrough" we investigated, 
running classical decoders on WM_3's chroma channels, reported carrier-agreement 1.000, but the same decoder
returns the **identical constant bits on pure random noise**. That is a degenerate reading, not a recovered
message, so we rejected it; WM_3 stayed unidentified until the ArtificialGANFingerprints match below.

## Eight of eight, cracked

Public decoders plus a systematic sweep (one candidate scheme per isolated environment) identified all 8,
each re-encoded natively at round-trip bit-accuracy 1.0 (0.96 mean for WM_3) and tiny LPIPS:

| WM | scheme | forge |
|----|--------|-------|
| WM_1 | dwtDct (invisible-watermark) | native re-encode |
| WM_2 | RivaGAN (invisible-watermark) | native re-encode |
| WM_3 | **ArtificialGANFingerprints** (StegaStamp) | native re-embed |
| WM_4 | VINE-R (ICLR'25) | native re-encode (GPU encoder) |
| WM_5 | CIN (MM'22) | native re-encode |
| WM_6 | MBRS (MM'21) | native re-encode |
| WM_7 | TrustMark-Q | native re-encode |
| WM_8 | TrustMark-P (`use_ECC=False`) | native re-encode |

WM_3 was the last to fall, and for a long time we thought it never would; that story is next.

## WM_3, the scheme we almost gave up on

WM_3 resists the averaging attack because it is **content-adaptive**: the shared residual sits at the
clean-image noise floor (cross-carrier correlation ≈ 0.001), so there is no fixed pattern to average out, and a
blind transplant caps around `S_final ≈ 0.43`. For a long stretch we could not identify it and treated it as a
structural ceiling. We ruled out **~27 public decoders** under the positive-control bar (all VINE W-Bench
baselines, RoSteALS, InvisMark, CRMark, Stable-Signature, LaWa, SepMark, PIMoG, EditGuard, and classical /
spread-spectrum families), built a color-axis `R+B−2G` chroma transplant that inched WM_3 from ~0.33 to ~0.43
(and the score from ~0.827 to **0.836**), and even convinced ourselves the ceiling was real with a Craver-style
diagnostic and a learned diffusion forger (WMCopier) that stayed below transplant parity at 25 carriers.

**That verdict was wrong.** The scheme *was* identifiable; we had simply never tried the right decoder at the
right resolution. The break came from taking the oracle test seriously one more time, at the native 256²:
because all 25 carriers share one message, the correct decoder must return the *same* bits on all of them. Most
candidates gave agreement near chance, but the **ArtificialGANFingerprints** (Yu et al., ICCV 2021) **AFHQ
cat2dog 256²** checkpoint, a **StegaStamp** autoencoder, read the 25 carriers at agreement **0.9996** while
clean images decoded at **0.48**. Such near-perfect agreement is only possible if the benchmark used these exact
public weights, so the hidden detector *is* this decoder. (Our earlier StegaStamp tests used only its original
400² checkpoint and were excluded on resolution; the reissued 256² fingerprinting weights were the missing
piece.) With the model in hand, WM_3 became like every other scheme: majority-vote the 100-bit message from the
carriers, re-embed it into each target with the matching encoder (`forged = clip(y + a·r)`), and it decodes at
mean bit-accuracy **0.96**. LPIPS depends on the grader's backbone (AlexNet vs VGG), which we do not know, so we
set each image's strength `a` to maximise the **worst case** across both rather than the AlexNet optimum, which
we found hides high-frequency texture that VGG penalises. This lifted WM_3 from ~0.43 to ≈0.67–0.78 and the
score from 0.836 to **0.8553**, our final result. The self-contained forge is in
`encoders/afgf_wm3_encode.py`.

## Native robustness (final analysis), and an honest negative result

The leaderboard grades bit-accuracy *after* an unknown pre-detection transform, so clean-PNG bit-accuracy
**overestimates** the server. We built a local **robustness oracle** (decode a forge → apply JPEG/resize/noise/
blur → re-decode → measure surviving bits) and audited all seven natives. Six are already robust; only **WM_1**
(classical dwtDct) is fragile (surviving bit-accuracy 0.60, collapses under JPEG). But a leaderboard probe that
strengthened WM_1's quantization margin was a **wash** (+0.0002): the quality cost cancelled the detection gain,
and dwtDct is JPEG-fragile at *every* strength. Conclusion: the natives sit near their achievable ceiling with
these public encoders. With WM_3 now identified too, the remaining gap to the top of the board is the quality
and robustness of the public encoders, not a missing scheme.

## The climb

`0.661` (3 schemes) → **`0.719`** (adding CIN + MBRS, +0.057) → **`0.821`** (adding VINE + TrustMark-P) →
`0.827` (chroma-domain WM_3) → `0.836` (color-axis WM_3 + native robustness) → **`0.8553`** (WM_3 identified as
ArtificialGANFingerprints and re-embedded natively, +0.019). The jumps were driven by *identification*, not
leaderboard tuning, every gain is a per-scheme method that generalizes across the 30% public / 70% private split.

## Real-world takeaway

Watermark forgery under near-black-box conditions is practical and cheap: for any *deployed public* scheme, an
attacker with a handful of same-message watermarked images can identify the scheme and re-encode with its own
public encoder, a perfect, generalizing forgery. WM_3 is the sharpest lesson. As a content-adaptive learned
mark it defeats the averaging forgery that breaks the additive schemes, and we briefly mistook that resistance
for security. But content-adaptivity is **not** protection once the weights are public: the per-image embedding
that hides the mark is exactly what its own encoder reproduces on any target, so an attacker need only identify
the model and re-embed. Nor is obscurity protection, WM_1 and WM_5 fall to plain averaging and WM_3 fell the
moment we matched it to a public checkpoint. Robust provenance has to rest on a secret key or a secret detector,
not on imperceptibility, content-adaptivity, or an undisclosed scheme built on public encoders.
