# Wan-Animate-2 — find the length-dependent divergence

## Premise: the model CAN do this

- ComfyUI ships **this exact checkpoint** (`wan_animate_2_int8_convrot`) with a template built around
  **81-frame** clips at **`pose_strength = 1.0`**, and the community runs it there.
- Our port **dances correctly at short T** (720x1200, 21 frames, verified driving-live:
  dancing `91c7ac51` vs static `cd1b7ad6`).
- Therefore the model is capable and **our implementation diverges somewhere that scales with sequence
  length.**

⚠️ **`PoseStrength = 1.2` is a symptom patch, not a fix.** It stays as a user knob, but if Tier 1 finds a
real divergence the 1.2 recommendation is **retracted** — it will over-drive a corrected splice.

⚠️ **The contradiction that proves a bug exists:** we argued the driving band dilutes as 1/(T+1)
architecturally, so ComfyUI must dilute too. But ComfyUI needs no amplification at 81 frames. Both cannot
be true. Exactly one of:
- **(a)** the reference's attention structure differs (gen self-attention extent, splice extent, or norm),
- **(b)** the driving band's positions differ, so trained weights can attend it preferentially and ours cannot,
- **(c)** the weights handle dilution fine and something numeric in our splice is off.

A layer-by-layer diff discriminates all three. **Do not guess between them.**

## TIER 0 RESULT (done) — all four clean; the premise INVERTS

**(a) and (b) are FALSIFIED by primary source.** The reference's generation self-attention attends the
WHOLE clip (`is_base_attention` has no query-side term, `wan_animate_2_model.py:583`), so **the reference
dilutes as 1/(T+1) exactly as we do** — that argument cannot distinguish us from it. Driving RoPE positions
agree exactly at T=4 AND T=21 (driving t: 1…20, w: 40…79; `refer_offset_w: -1` is the only sentinel and
resolves to the gen grid width, as ours does). Temporal alignment (gen j ↔ driving j−1) and all
normalization/scaling match.

⚠️ **THE REFRAME — this is the finding.** Upstream `CLIP_LEN = 81` → `lat_t = 22`. Our **failing** case
(77 frames, T=21) is therefore the **TRAINED regime**; our **working** case (21 frames, T=7) is
**off-distribution**, under a third of trained length. So this is NOT "dilution hurts as T grows". It is
**a defect that a SHORT sequence masks and that appears at the length the model was trained for.**

⇒ Tier 1 must hunt something benign at 6 driving frames / 3375 tokens-per-frame and destructive at
20 driving frames / 960 tokens-per-frame — **not an attention-share problem.**

⇒ **The `PoseStrength` doc comment blaming 1/(T+1) dilution is now WRONG and must be revised.** The knob
may still be useful, but its stated rationale is falsified.

**Real bugs found (neither scales with T, neither explains ours-vs-ComfyUI):**
1. **`Animate2LogScale` is never populated** — `WanConfigDetector` sets `IsAnimate2` but not the log scale,
   so **distill builds run `log_scale = 0` instead of `-1.3`**. Does not affect our base checkpoint, and
   ComfyUI omits `score_mod` entirely, but it is a genuine upstream divergence. FIX IT.
2. `reference_image_strength` unimplemented (cosmetic at default 1.0).

**Surviving hypothesis: (c), an engine primitive in the splice path** — invisible to desk reading, exactly
the class the LTX decoder layer-diff caught. Ranked suspects: `SliceRows`/`ScatterSeqHeadMajor` at the
shapes only the long case reaches (s = 20160, buffer 21120, 840 slice+scatter pairs per forward at T=21 vs
240 at T=7); `WanRopeInterleaved` over refSeq = 19200; and the driving cache's device residency under
pressure (pinned activations re-read by every block of every step beside a resident 14B DiT).

⚠️ **"ComfyUI needs no amplification at 81 frames" is SECOND-HAND and load-bearing for this whole
investigation. Tier 1's same-checkpoint/same-inputs protocol tests it for free — do not skip that read.**

## Tier 0 — desk checks, no GPU, hours (COMPLETE, see result above)

Diff `WanVideoBlock.Animate2FrameLocalAttention` against `comfy_model_animate2.py`'s gen-frame loop
(byte-identical copy at `scratchpad/comfy_model_animate2.py`, verified vs upstream master):

1. **What does the GENERATION stream's self-attention actually attend** — full sequence, or windowed?
   Ours attends all `s = T·hw` gen tokens. If the reference is frame-local on the gen side too, the driving
   share stays constant in T and the whole dilution disappears. **This is the single highest-value check.**
2. **Driving-band RoPE.** The reference derives `shift_x` **per call** from the current grid. Verify our
   `RefRopeOffsets => (t:1, h:0, w:genGridW)` yields the *same positions* the reference does **at T=21**,
   not merely at T=4.
3. **Temporal offset `t=1` at large T.** An off-by-one aligning driving frame j with gen frame j+1 is
   nearly invisible at T=4 and destructive at T=21.
4. **K normalization / scale on the driving band vs the gen band** — must match exactly.

## Tier 1 — the layer-by-layer diff (main event)

Reuse the existing dump/parity machinery (`docs/Checklists/PARITY_VERIFICATION.md` conventions); do not
invent new infrastructure.

⚠️ **The local ComfyUI clone at `Models/bench-comfy` is `14b05228` — PRE-Animate-2.** Either fetch and
check out `a464ac33`+ (PR #15362, 2026-08-07) or instrument the upstream repo at `scratchpad/wanimate2/`.

**Protocol:**
- Same checkpoint, reference image, driving clip, fixed seed/noise.
- **T=21 (the failing regime). Parity at T=4 proves nothing — we already work there.**
- **Compare the PREPASS FIRST** (`EncodeDriving`): deterministic given inputs, isolates half the pipeline
  before fighting sampler-state alignment.
- Then per block, for a mid frame (j≈10): driving K/V after projection+RoPE → spliced attention output →
  block output. **First divergent tensor is the bug.**

This is the discipline that found 3 LTX decoder bugs pure code-reading missed
([[ltx25-diffusion-decoder-gpu]]).

## Tier 2 — cheaper probes, run in parallel

- **Attention-mass probe at T=21 with REAL weights.** We measured 20.0% at T=4 on synthetic. If the
  reference's trained weights give the driving band well above 1/(T+1) share and ours don't, that localizes
  to Q·K — i.e. RoPE/positions — with no Python run at all.
- **Single-block parity:** real weights, block 0 only, same driving latent both sides, diff K/V. Catches
  projection/norm divergence far cheaper than e2e.
- **Bisection by substitution:** if the prepass matches but e2e diverges, swap our sampler trajectory for
  the reference's dumped sigmas to rule the scheduler in or out.

## Possibly the same root cause — check before treating separately

**cfg > 1 renders hazy** (40 steps/cfg 3.0 measured worse than 20/cfg 1.0), and the block-9 uncond skip has
never been verified on GPU. If the splice is wrong, this may resolve with it. Do not chase independently
until Tier 1 lands.

## Explicitly OUT OF SCOPE

The `continue_motion` chunk washout. Separate, already-specified fix: colour-match each decoded chunk to
the **original reference image** before taking the anchor frame (see `wan-animate2-reference-contract`).
