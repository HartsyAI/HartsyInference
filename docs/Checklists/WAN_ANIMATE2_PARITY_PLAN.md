# Wan-Animate-2 — length-dependent divergence investigation

## VERDICT — CLOSED. There is no length-dependent defect.

`wan_animate_2_int8_convrot` is a **base** build; it needs base sampling (40 steps, cfg 3.0, `log_scale 0.0`).
The settings previously used here (6 steps, cfg 1) reproduced ComfyUI's shipped template with its
load-bearing distillation LoRA removed — a raw base Wan-14B at 6 steps with no guidance renders hazy, and a
short clip masks it because the reference latent frame carries proportionally more of the result. Sampled
correctly, 77 frames (latent T=21) at 384×640 renders sharp through the whole clip and follows the driver
(A/B: dancing vs static-pose driver, mean frame-to-frame motion 19.12 vs 1.55, both sharp).

The distillation build (`wan_animate_2_distill_int8_convrot`, Comfy-Org ships it alongside the base one) is
the better answer for few-step work: 77f/6 steps/cfg 1 renders sharp in 2.5 min vs 14 min for the base
build's 40 steps.

**Retracted along the way:** `PoseStrength = 1.2` (1.0 reproduces the reference); "cfg > 1 renders hazy"
(cfg 3.0 was the sharpest arm measured). **Fixed along the way:** `Animate2LogScale` was never populated —
distill builds were silently running `log_scale = 0` instead of `-1.3` (`WanAnimate2Transformer.ResolveLogScale`);
a build/settings-mismatch warning was added so this class of error surfaces immediately next time.

Full diagnostic trail (four tiers of falsified hypotheses — attention-share dilution, splice primitives,
RoPE positions, VAE streaming, all exonerated by real measurements before the settings mismatch was found):
`git log 4b1a38e1..675297e6` on `WanAnimate2*` source paths.

## The distillation build's biased attention no longer OOMs at 480x800 / 61 frames

Routing `log_scale = -1.3` handed `ScaledDotProductAttention` a `[Sq, Skv]` additive mask. That mask shape was
never the problem — `CudnnMaskCompatible` accepts it and the fused engine takes it as a bias score-modifier.
The problem was the failure classification underneath it: a **VRAM shortfall inside the fused path was
classified as a STRUCTURAL cuDNN failure**, which permanently disabled D=128 for the process, after which every
masked call fell through to the materialized `[heads, Sq, Skv]` score matrix — `40 x 1500 x 25500 x 4 B =
5836 MB` at 480x800 / 61f, which is exactly the allocation that was reported failing. Demoting on an OOM to a
path that needs an order of magnitude MORE memory guarantees the run dies.

Three changes, all in the engine (not the recipe):

1. **An `OutOfVramException` from the fused path is transient**, never a structural kill. Unknown exception
   types stay permanent — that conservatism was deliberate and is unchanged.
2. **A bias that depends only on the key is stored as one `[1, Skv]` row and broadcast.** `score_mod` here has
   no query-side condition, so the `[Sq, Skv]` form was `Sq` identical copies (162 MB per call at this
   geometry, resident via auto-promotion). cuDNN takes the query axis as a dim-1 broadcast and stays on the
   fused engine; the CPU backend and the materialized GEMM path broadcast it too.
3. **The memory-bounded query-tiled path now accepts a key-only bias** (a rank-1 `ones (x) bias` accumulate per
   tile, so no `[Sq, Skv]` duplicate is built anywhere). A masked call whose score matrix does not fit is no
   longer forced into the full materialization, which is what makes the no-OOM property hold even if a
   transient cuDNN failure does land mid-generation.

**Measured 2026-08-22, RTX 4090 (nvidia-smi index 1), distill checkpoint, 6 steps / cfg 1 / seed 424650,
`ref_dany_portrait.png` + `drive_half.mp4`:** 480x800 / 61 frames completes in 142 s at **22093 MiB peak** (169 s before the VAE-encode host-round-trip fix `07439663`), no
OOM, `[cuDNN SDPA] fused flash-attention engaged (D=128)` and no `permanently disabled` line in the journal.
Frames 2 / 30 / 58 at full resolution are sharp — fabric weave and background props legible, identity held,
and the subject genuinely turns through the clip rather than holding a smeared portrait.

Headroom is real but not large: the DiT streams all 40 blocks (`resident prefix 0`) and the driving cache
alone is 9.16 GiB at this geometry, up from 2.20 GiB at 384x640 / 21f. It scales with driving tokens, so it is
the next wall for anything longer or larger, not the attention mask.

**`DitShardGpuId` remains a no-op for this recipe** (verified by reading `WanAnimate2Recipe`, which consumes
only `context.Backend`), but it is no longer a SILENT one: the recipe already calls
`PlacementSupport.WarnIfIgnored`, so a configured shard device logs "DiT sharding is configured but not wired
for this model" at load. Pooling the 3060's 12 GB was not needed to fit this geometry.

## Open items

- **A LoRA cannot be merged into an int8-convrot checkpoint.** `LoraApplier` throws
  `Unsupported dtype conversion: I8 → F32`, and `WanLoraMapper` additionally drops every `diff`/`diff_b`
  full-weight key with a warning (lightx2v-style LoRAs are full of bias/norm diffs). Needs a
  dequantize→add→requantize path per tensor plus full-weight-diff handling. Not done — this is why the
  distillation LoRA has never actually been run through this checkpoint here.
- **`continue_motion` chunk washout** (separate from the above). Specified fix not yet applied: colour-match
  each decoded chunk to the original reference image before taking the anchor frame (see
  `wan-animate2-reference-contract`).

Neither of these has a home yet in `MODEL_STATUS_VIDEO.md` (which currently has no Wan-Animate-2 entry at
all) — add one there rather than letting this file be the only record.

## Driving-cache dtype is now automatic (`9db28bf5`)

The BF16 driving cache is no longer a machine-local env drop-in. `HARTSY_ANIMATE2_BF16_DRIVING_CACHE`
accepts `auto` (default when unset) / `on` / `off`; auto honours the global low-VRAM policy, then measures
free VRAM (trim-first) and keeps F32 only when it fits beside the weight plan. Resolved once per
generation in the recipe and relayed per-request. A pre-flight check refuses infeasible geometry before
any VAE encode, naming the largest frame count that fits.
