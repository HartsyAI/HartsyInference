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

## VERDICT — there is no length-dependent defect. Closed.

**77 frames (latent T=21) at 384x640 renders sharp and follows the driver, once the checkpoint is sampled the
way its build is meant to be sampled.** `wan_animate_2_int8_convrot` is the **base** build: upstream and the
ComfyUI template both run the base at **40 steps / guidance 3.0**, and ComfyUI only reaches 6 steps by loading
`lightx2v_I2V_14B_480p_cfg_step_distill` at 1.0. Our "best known settings" copied that template with its
load-bearing LoRA removed, which is a raw base model at 6 steps with no guidance. A short clip hides that
(the reference latent frame carries proportionally more of a 7-frame result); a long one cannot.

| 384x640, `drive_half.mp4`, seed 424242, single chunk | result |
|---|---|
| 21f, 6 steps, cfg 1 | sharp |
| 21f, 40 steps, cfg 3.0 | sharp |
| 33f / 45f / 77f, 6 steps, cfg 1 | mush, worse toward the end of the clip |
| 77f, 20 steps, cfg 1 | still mush |
| **77f, 40 steps, cfg 3.0** | **sharp through frame 76** |

**A/B at the corrected settings** (77f, 40 steps, cfg 3.0, seed 424242): dancing `drive_half.mp4` vs an
81-frame still built from its frame 0. Different files (`b46620c3` vs `9c4ad0fd`), mean |A−B| 46/255, and
mean frame-to-frame motion **19.12 vs 1.55** — the static driver holds the pose, the dancing one dances, both
sharp. The driving stream reaches the output at the trained length.

**The distillation build is the better answer for few-step work.** `wan_animate_2_distill_int8_convrot`
(Comfy-Org ships it beside the base one) at **77f / 6 steps / cfg 1 / 384x640** renders sharp through frame 76
in 2.5 minutes, against 14 for the base build's 40 steps — and its A/B is cleaner still: mean |A−B| 36/255,
frame-to-frame motion **18.64 dancing vs 0.37 static**. That run is also the first time `log_scale = -1.3`
has ever executed (the fix below routes it), and the biased-attention path holds up. 480x800 / 61f behaves the
same.

### Retracted

- **"Best known settings: 6 steps, cfg 1"** — that is the *distillation* configuration. For this base
  checkpoint use **40 steps at cfg 3.0**, or switch to `wan_animate_2_distill_int8_convrot`.
- **"cfg > 1 renders hazy"** — falsified. cfg 3.0 produced the sharpest output at both 21 and 77 frames, with
  the block-9 unconditional skip active the whole time. That retires the skip as a suspect too.
- **`PoseStrength = 1.2`** — retracted. 1.0 reproduces the reference and the A/B above was measured at 1.0.
- **The Tier 0 note that our checkpoint might be a distillation build** — it is not; `log_scale = 0` is
  correct for it. Comfy-Org ships the distillation weights as a separate file.

### Real gaps this uncovered (neither is the mush)

1. **`Animate2LogScale` was never populated** — FIXED, `WanAnimate2Transformer.ResolveLogScale`.
2. **A LoRA cannot be merged into an int8-convrot checkpoint**: `LoraApplier` throws
   `Unsupported dtype conversion: I8 → F32`, and `WanLoraMapper` additionally **drops every `diff`/`diff_b`
   full-weight key with a warning** — lightx2v is full of bias and norm diffs. Supporting it needs a
   dequantize → add → requantize path per tensor plus full-weight-diff handling. Not done.

## TIER 1 RESULT — the symptom was mis-stated, and every ranked suspect is dead

**The symptom is not "ignores the driving video".** At 77 frames the subject moves plenty; the output is
**hazy, smeared and mesh-textured**, worst in the later frames of the clip. "Doesn't follow the driver" was a
misread of an unreadable image.

**Length IS the variable, now with every confound controlled.** All rows: 384x640, `drive_half.mp4`,
seed 424242, cfg 1, `dpm++2m`, single chunk.

| frames | latent T | steps | result |
|---|---|---|---|
| 21 | 7  | 6  | **sharp** — crisp face, clean detail |
| 21 | 7  | 20 | **sharp** — no mesh at all |
| 33 | 9  | 6  | frame 0 ok, degraded by frame 20, mush by frame 32 |
| 45 | 12 | 6  | mush at frame 20 |
| 77 | 21 | 6  | mush throughout |
| 77 | 21 | 20 | still mush — **more steps do not rescue it** |

The verified-good 720x1200/21f run has s = 23625, refSeq = 20250, buffer = 27000; the failing 384x640/77f run
has s = 20160, refSeq = 19200, buffer = 21120. **The working case is LARGER on every magnitude the plan's
suspects #1 and #2 name.** Frame index alone is not it either: frame 20 is sharp at T=7 and mush at T=12.

**Falsified in Tier 1 (do not re-chase):**
1. **Steps.** 6 steps renders beautifully at T=7; 20 steps does not save T=21.
2. **Resolution.** 384x640 is fine at T=7.
3. **`SliceRows` / `ScatterSeqHeadMajor` / `WanRopeInterleaved` (suspects #1, #2).** `WanAnimate2LongSequenceParityTests`
   runs a whole spliced forward CPU-vs-CUDA on the REAL token grid (40x24, hw = 960) at genFrames 7/11/16/21;
   relL2 is 4.7–5.0e-4 at every length, flat in T. Caveat: 1 layer, 4 heads, synthetic F32 weights — it
   exonerates the splice primitives at long shapes, not the int8 GEMM path or memory management.
4. **The streamed VAE encode.** Streamed vs whole-clip is bit-exact at 13, 41 and 77 frames on the real
   `[false,true,true]` temporal-downsample layout.
5. **A per-frame-index VAE decode decay.** Output frame 20 is sharp at T=7 and mush at T=12.

### The settings we have been running are not a supported configuration

**Upstream (`infer/wan_animate_2*.yaml`, `wan_animate_2_demo.py`):** base = 40 steps / guidance 3.0 /
`log_scale 0.0`; distillation = 10 steps / guidance 1.0 / `log_scale -1.3`. Both sample at shift 5.0.

**ComfyUI's shipped template for THIS checkpoint** (`comfyui_workflow_templates_json/video_wan_animate2.json`)
runs `wan_animate_2_int8_convrot` at 6 steps, `lcm`, `simple`, cfg 1, `ModelSamplingSD3` shift 5 — **with
`LoraLoaderModelOnly(lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16, 1.0)`**. The few-step behaviour is
the LoRA's. The checkpoint itself is the **base** build, so `log_scale = 0` is correct for it.

⇒ **"Best known settings: 6 steps, cfg 1" reproduced that template with its load-bearing LoRA removed.** A raw
base Wan-14B at 6 steps with no guidance is expected to render hazy; a short clip masks it because the
reference latent frame carries proportionally more of the result. That alone predicts the whole T curve
without any engine defect, so it has to be excluded before any layer-diff.

⚠️ **We cannot apply that LoRA at all**: `LoraApplier` throws `Unsupported dtype conversion: I8 → F32` on the
int8-convrot checkpoint, and the Wan LoRA reader additionally **skips every `diff`/`diff_b` full-weight key
with a warning** — lightx2v is full of bias and norm diffs. Two real gaps, and together they are why the
supported configuration has never been run here.

⚠️ **Every failing datapoint so far was collected with two non-reference switches ON in the service unit:**
`HARTSY_ANIMATE2_BF16_DRIVING_CACHE=1` and `HARTSY_ANIMATE2_POSE_STRENGTH_X100=120`. The BF16 driving cache has
no correctness coverage anywhere (the parity test above runs the F32 default).

**Still open, in order:** the BF16-cache / pose-strength knobs; driving-cache residency under block streaming
(suspect #3, untouched); and the ComfyUI e2e at the exact failing config, which is the only thing that can
still tell "our port has a bug" from "this checkpoint cannot do 77 frames at a quarter of its trained area".

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
