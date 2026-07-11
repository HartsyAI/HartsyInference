# Parameter Parity — Fix Plan

## IMPLEMENTATION STATUS (updated 2026-06-28)

**Shipped + build-verified (Diffusion / Video / Interactive all compile clean):**
- **Phase 1.1** HunyuanImage `V21`/`V21Distilled` → 3584/28, `GuidanceEmbed` un-inverted (full=false, distilled=true), added `UseMeanflow` + `SamplingShift` (5.0/4.0), pipeline now uses config shift (was hardcoded 3.0).
- **Phase 1.2** AuraFlow `PosEmbedMaxSize` 1024→9216 + new `V02` preset (4096); doc comments corrected.
- **Phase 1.3** Qwen-Image `ContextDim` 4096→3584, `PooledProjectionDim` 2048→768.
- **Phase 1.4** F-Lite `V1_7B` → 3072/12/28 (was placeholder 2560/10/32).
- **Phase 1.5** SD3.5 `Medium35` dual-attn [0..12] + `PosEmbedMaxSize`=384; `Large35` dual-attn → null (none).
- **Phase 1.6** Flux Fill: detection broadened (`XEmbedInputDim > 64`), Fill (≥384) explicitly diagnosed (masked-latent+mask path is a documented TODO), `FluxConfig.Flux1Fill` preset added, `FluxToolsConfig` channel units corrected to packed.
- **Phase 1.7** LTX-Video `V097` 13B preset (head_dim 128 / 48 layers / cross-attn 4096) + `VaeTimestepConditioned`/`DecodeTimestep`/`DecodeNoiseScale` config fields.
- **Phase 1.8** HunyuanVideo plain `T2V` preset (16-ch, embedded guidance) + `EmbeddedGuidanceScale` field.
- **Phase 2 smalls:** Flux2 `_hiddenLayers` now defaults by encoder (Mistral→[10,20,30], Qwen→[9,18,27]) + log fix; ERNIE scheduler shift default 1.0→4.0; Kandinsky `AttentionType` field added.

**SECOND WAVE shipped + build-verified (full solution incl. tests compiles clean) — user approved the
breaking change and validation-pending numerics:**
- **Phase 0.1/0.2 nullable defaults** — `TextToImageRequest.Steps/CfgScale/Width/Height` are now `int?`/`float?`
  (null = model default). New `Requests/GenerationDefaults.cs` holds per-model reference defaults; every image +
  video pipeline (28 files) resolves via `?? modelDefault` and the dead video `<=0`-fallback branches were
  removed so each model's config defaults (Wan/LTX/Chroma/FLite/Lens/Lance `NumInferenceSteps`/`DefaultSteps`/
  etc.) are now LIVE. ⚠️ This is a breaking change to a type the external SwarmUI extension consumes — bump it.
- **Phase 0.3** `Requests/VideoGenerationRequest.cs` added (NumFrames/Fps/FlowShift). Type is available; wiring
  it into video pipeline signatures (replacing the loose method args) is a follow-up.
- **Phase 0.4 Flux true_cfg** — `FluxPipeline.GenerateFromTokens` gained negative-token params + `trueCfgScale`;
  when >1 with a negative prompt it runs a real uncond pass each step (layered on embedded guidance). The only
  in-repo Flux caller is the external extension; ModelManager has a wiring note.
- **Phase 2 HIGH numerics (validation-pending, marked `// VALIDATION-PENDING` in code):**
  - Wan → `FlowUniPCMultistepScheduler` across all Wan pipelines (V2V left on Euler w/ a noted reason).
  - HiDream real top-2-of-4 MoE routing (+ shared expert) replacing the single-expert fallback.
  - OmniGen2 text/image dual-guidance defaults + `cfg_range` gating (image triple-pass stubbed for when image
    conditioning lands).
  - Lumina2 `cfg_normalization` (on by default) + `cfg_trunc_ratio` + dynamic shift.
  - LTX 0.9.1+/13B timestep-conditioned VAE decode (decode_timestep 0.05 / noise 0.025).
  - Krea2/Chroma cond-anchored CFG (`CfgHelper.ApplyCfgCondAnchored`) + Chroma `InitialNoise` honoring.

**STILL deliberately NOT done (need real weights):**
- **HunyuanVideo embedded-guidance FORWARD path** — config field + plain T2V preset are in (Phase 1.8), but the
  `guidance_in` MLP in the DiT forward, plus a plain HunyuanVideo T2V pipeline (Phase 3.1), are unbuilt.
- **Wan Animate/S2V variant presets + transformer wiring** (Phase 3.2), **SD3.5 SLG** (3.3), **Flux Fill
  masked-latent+mask conditioning** (now explicitly diagnosed, not silently wrong), **STG / ByT5 glyph /
  reference-image editing / batch>1** (Phase 4) — all remain.
- **Lance timestep shift** — sources CONFLICT (README 3.5 both / paper Table 2 4.0); left unchanged pending weight verification.
- **Validation harness (Phase 5)** — EVERY numerical change above is `// VALIDATION-PENDING`; none has been
  diffed against real weights. The preset round-trip test + per-model parity diffs are the gating next step.

---


Remediation plan for every gap in [PARAM_PARITY_AUDIT.md](PARAM_PARITY_AUDIT.md). Ordered by leverage:
shared-infrastructure fixes that unblock many models at once, then cheap load-breaker preset corrections,
then the real numerical-correctness work, then variant/preset additions and feature gaps.

**Effort key:** S = <1h / one-liner-ish · M = a few hours / one component · L = a day+ / new subsystem.
**Risk key:** numbers that change *output* must be validated against real weights or a Python reference
before being marked done (CLAUDE.md rule "validate against references"). Each item says what proves it.

**Two things to keep in mind while reading:**
- Several Tier-1 "WRONG preset" items are *mitigated at runtime* by a `FromWeights`/`AutoDetect` path that
  reads dims from the checkpoint. Fixing the hardcoded preset is still correct (it's the fallback and the
  documented truth), but those are lower-urgency than items with no autodetect.
- Some configs are placeholders for **unreleased** models (Krea2 is actually verified; Boogu/Lens partial;
  MG3 mostly unknown). For those, the "fix" is often *confirm against weights when available*, not *change a number now*.

---

## Phase 0 — Shared infrastructure (do first; unblocks the systemic Tier-3 problems across ALL models)

These four changes fix the single most pervasive class of findings (wrong defaults, dead-code fallbacks,
missing user knobs) for every model simultaneously, and they're prerequisites that make the per-model
phases small.

### 0.1 — Make generation params "unset-aware" so per-model defaults can apply · M · low risk
**Problem:** `Requests/TextToImageRequest.cs` uses concrete defaults (`Steps=20`, `CfgScale=7.5`,
`Width/Height=512`). A pipeline can't distinguish "user wants 20" from "user left the default", so it can
never substitute the model-correct value. This is the root of ~25 "WRONG default" findings.
**Fix:** change the four fields to nullable (`int? Steps`, `float? CfgScale`, `int? Width`, `int? Height`;
keep the old concrete defaults only as the final fallback). Each pipeline resolves
`request.Steps ?? modelDefaults.Steps`.
**Then:** add a `GenerationDefaults` record (Steps, Cfg, Width, Height, + flow/shift where relevant) and a
static table of per-model values (the audit's reference defaults are the source — e.g. SDXL 50/5.0/1024,
SD3.5 28/7.0/1024, Flux dev 28/3.5/1024, schnell 4, Qwen 50/4.0-true-cfg/1024, Z-Image-Turbo 8/1.0, etc.).
Wire it through `PipelineFactory` so each constructed pipeline carries its defaults.
**Proves done:** unit test that a request with null fields resolves to the model's reference defaults; no
output-number change for callers who set values explicitly (SwarmUI already does).

### 0.2 — Kill the `<=0`-fallback dead code in video pipelines · S · low risk
**Problem:** Wan/LTX/LTX-2 pipelines fall back to `_config.NumInferenceSteps`/`GuidanceScale` only when the
request value is `<=0`, which never happens, so their correct defaults are dead.
**Fix:** once 0.1 lands, replace the `<=0` checks with the `?? modelDefault` resolution. Remove the dead branches.
**Proves done:** code review; the resolution test from 0.1 covers video too.

### 0.3 — Add a `VideoGenerationRequest` carrying video-only knobs · M · low risk
**Problem:** `num_frames`, `fps`, `flow_shift` have no home in any request type; they're passed as loose
method args, and video models reuse the image-shaped request.
**Fix:** `VideoGenerationRequest : TextToImageRequest` with `NumFrames`, `Fps`, optional `FlowShift?`,
optional first/last-frame conditioning handles, and video-correct default resolution. Update video pipeline
signatures to take it. Carry `EmbeddedGuidanceScale?` for HunyuanVideo (see 2.7).
**Proves done:** video pipelines compile against the new type; defaults match the audit (Wan 50/5.0/480x832/81f,
LTX 50/3.0/161f, LTX-2 40/4.0/121f, HunyuanVideo 50/embedded-6.0/129f, Kandinsky 50/5.0/121f).

### 0.4 — Wire negative-prompt / `true_cfg_scale` where the field is silently ignored · M · MEDIUM risk
**Problem:** `NegativePrompt` exists on the request but Flux and Flux2 never encode or use it; Qwen-Image's
true-cfg path is correct but defaults wrong. Diffusers exposes `true_cfg_scale` (real uncond pass) on these.
**Fix:** add `float? TrueCfgScale` to the request. In `FluxPipeline`/`Flux2Pipeline` (Klein), when
`TrueCfgScale > 1` and a negative prompt is present, run the real two-pass CFG (encode negative, uncond
forward, combine). Leave guidance-distilled paths (Flux dev, Flux2 dev) on embedded guidance.
**Proves done:** Python parity — a Flux generation with `true_cfg_scale=4` + negative prompt matches diffusers
`FluxPipeline` within tolerance. (This is the one Phase-0 item that changes output; validate it.)

---

## Phase 1 — Tier-1 load-breaker config fixes (cheap, high value; mostly preset edits)

All of these are wrong *architecture* numbers. Do them as a batch; each is a few lines. Validation for each =
load the real checkpoint and confirm weights bind with no shape mismatch (+ a forward pass producing finite output).

| # | Model | File | Change | Effort | Autodetect mitigates? |
|---|---|---|---|---|---|
| 1.1 | **HunyuanImage 2.1** | `HunyuanImageConfig.cs` | `V21` & `V21Distilled`: `HiddenSize 3072→3584`, `NumHeads 24→28`. **Swap `GuidanceEmbed`** (full=false, distilled=true). Add `UseMeanflow` flag (distilled+refiner=true). Fix backwards doc comment. | M | No — must fix |
| 1.2 | **AuraFlow** | `AuraFlowConfig.cs` | `PosEmbedMaxSize 1024→9216` (V03). Add `V02` preset (=V03 but 4096). Fix the "32×32" doc comments. | S | No — must fix |
| 1.3 | **Qwen-Image** | `QwenImageConfig.cs` | `ContextDim 4096→3584`; `PooledProjectionDim 2048→768`; rename `InChannels`(16) to reflect it's out-channels, or document the packed-64 vs 16 distinction; ensure RoPE axes [16,56,56] are applied (verify `QwenImageRope`). | S–M | Partial (encoder dim must match) |
| 1.4 | **F-Lite-7B** | `FLiteConfig.cs` | `V1_7B`: `HiddenSize 2560→3072`, `NumHeads 10→12`, `Depth 32→28`. | S | No — must fix |
| 1.5 | **SD3.5** | `Sd3Config.cs` | `Medium35.DualAttentionLayers` → `[0..12]` (add layer 12); `Large35.DualAttentionLayers` → `[]` (empty). `Medium35.PosEmbedMaxSize=384` (Large stays 192). | S | Yes (AutoDetect reads attn2.* + weight) — still fix presets |
| 1.6 | **Flux Fill** | `FluxConfig.cs`/`FluxToolsConfig.cs`/`FluxPipeline.cs` | Add Fill preset `InChannels=384`; change Tools detection from `XEmbedInputDim==128` to `>64`; build conditioning per-variant (Canny/Depth +64, Fill +320). Fix `AdditionalInChannels` units (1→64, 17→320) or delete the dead config. | M | No — detection gate is the bug |
| 1.7 | **LTX-Video 13B** | `LtxVideoConfig.cs` | Add `V097` preset: `HeadDim 64→128`, `NumLayers 28→48`, `CrossAttentionDim 2048→4096` (0.9.8 shares it). | S | No |
| 1.8 | **HunyuanVideo plain T2V** | `HunyuanVideoConfig.cs` | Add a plain preset (`NumDoubleBlocks=20, NumSingleBlocks=40, InChannels=16`) distinct from GameCraft. (Pipeline work is 3.x.) | S (config) | No |

---

## Phase 2 — Tier-2 numerical correctness (real code; each MUST be reference-validated)

These change output. Treat each as: implement → diff against Python/diffusers on real weights → mark done only
when within tolerance. Roughly priority-ordered.

### 2.1 — Wan: UniPCMultistepScheduler · L · HIGH
Reference TI2V-5B ships `UniPCMultistepScheduler` (order 2, bh2, predict_x0, use_flow_sigmas, exponential
shift); we use Euler everywhere. Implement a `FlowUniPCMultistepScheduler` variant matching these settings
(we already have a `FlowUniPCMultistepScheduler` for Matrix-Game — reuse/parameterize it). Route all Wan
pipelines through it. **Proves done:** latent-trajectory diff vs diffusers `WanPipeline` at 50 steps.

### 2.2 — OmniGen2: dual guidance · M–L · HIGH
Add `text_guidance_scale` (4.0) + `image_guidance_scale` (1.0) and the triple-pass CFG (uncond / text-only /
text+image) combine. Add `cfg_range` (start,end) gating. For pure-t2i this still means using text_guidance,
not the generic CfgScale. **Proves done:** diffusers `OmniGen2Pipeline` parity on a t2i prompt.

### 2.3 — HiDream: real MoE FFN routing · L · HIGH
Implement top-2-of-4 gated expert routing in `HiDreamBlock` (currently single-expert fallback per the
`// TODO`). Also pick the correct scheduler per variant (UniPC flow for Full, LCM shift-6 for Dev).
**Proves done:** per-layer diff vs diffusers `HiDreamImagePipeline`; confirm expert gate weights load.

### 2.4 — Lumina-2: cfg_normalization + dynamic shift · M · HIGH
`cfg_normalization` defaults **True** upstream (renormalize guided velocity to conditional norm) and is
unimplemented — implement it on by default. Replace static `SchedulerShift=6.0` with dynamic shift
(base 0.5 / max 1.15, mu from image_seq_len). Add `cfg_trunc_ratio` gating (no-op at 1.0). Also apply the
same `cfg_normalization`/trunc to Z-Image's Base/CFG path (Z-Image defaults False, lower urgency).
**Proves done:** diffusers `Lumina2Pipeline` parity at default settings.

### 2.5 — Krea2 & Chroma: guidance convention · S–M · MEDIUM
Krea2/Chroma reference velocity = `cond + s*(cond-uncond)`; our `CfgHelper.ApplyCfg` = `uncond + s*(...)`.
Either add a `cond-anchored` CFG mode and use it for these models, or map `CfgScale→scale+1` for them.
Separately, make Chroma/Radiance/Zeta pipelines fall back to their `DefaultCfgScale`/`DefaultSteps` (Phase 0.1
makes this trivial) and have `ChromaPipeline` honor `request.InitialNoise`. **Proves done:** Krea2 (already a
verified model) re-diff at the corrected scale; Chroma diff vs diffusers `ChromaPipeline`.

### 2.6 — HunyuanImage: sampling shift + embedded-guidance constant · M · HIGH (pairs with 1.1)
Source the sampling `shift` from config (5 full / 4 distilled) instead of the literal `3.0f`; use Hunyuan's
custom sigma scheduler (`get_timesteps_sigmas`) not plain FlowMatchEuler. Feed the distilled embedded-guidance
**constant** (~6016) not `cfgScale`. **Proves done:** Hunyuan reference parity once 1.1's dims are fixed.

### 2.7 — HunyuanVideo: embedded guidance · M · HIGH (pairs with 1.8 + 3.x)
Add a `guidance_in` MLP path feeding `embedded_guidance_scale=6.0` into the modulation vector (the signature
distilled-CFG mechanism). Carry `EmbeddedGuidanceScale` on the video request (0.3). **Proves done:** plain
HunyuanVideo reference parity.

### 2.8 — LTX-Video: timestep-conditioned VAE decode · M · HIGH
Pass `decode_timestep=0.05` + `decode_noise_scale=0.025` into the VAE decode for 0.9.1+/13B (block already
supports timestep conditioning; build the decoder with `timestepCond:true` for those presets and thread the
values). **Proves done:** decoded-frame diff vs diffusers `LTXPipeline` on a 0.9.7 checkpoint.

### 2.9 — Anima/Cosmos-Predict2: RoPE extrapolation axes · S–M · MEDIUM (verify intent first)
`RopeScale=(2.0,1.0,1.0)` puts 2.0 on temporal; Cosmos is 4.0 on H/W. Confirm what the **Anima** checkpoint
(not upstream Cosmos) actually uses before changing — Anima is a flow-match retrain and may differ. Also decide
the CFG form (`CfgHelper.ApplyCfg` vs the unused in-file `CosmosCfg`) and delete the dead EDM helpers.
**Proves done:** Anima checkpoint forward diff (this is research-gated, not a blind change).

### 2.10 — Lance: timestep-shift values · S · MEDIUM (verify, may be data-only)
`VideoTimestepShift=4.0` likely should be **3.5** (README) and `ImageTimestepShift=3.5` likely **4.0**
(paper Table 2) — they look swapped. Confirm against the upstream inference config, then correct.
**Proves done:** Lance reference parity once weights are downloadable.

### 2.11 — ERNIE-Image: scheduler shift default · S · LOW
Default `schedulerShift` to **4.0** (from `scheduler_config.json`) instead of 1.0; confirm the `[-2]`
hidden-state tap. **Proves done:** ERNIE reference parity.

### 2.12 — SDXL: micro-conditioning + guidance_rescale + steps_offset · M · MEDIUM
Expose `original_size`/`target_size`/`crops_coords_top_left` (+ negative variants) on an SDXL request instead
of hardcoding `(H,W)`/`(0,0)`. Add `guidance_rescale` (Lin et al.) to the shared CFG helper (benefits SD/SDXL/
others). Verify the scheduler applies `steps_offset=1` for SD1.5/SDXL `Leading` spacing (off-by-one if not).
**Proves done:** SDXL crop-conditioning visibly changes composition as expected; guidance_rescale matches diffusers.

---

## Phase 3 — Variant & preset additions (new presets/pipelines, no new math)

| # | Item | Work | Effort |
|---|---|---|---|
| 3.1 | **HunyuanVideo plain T2V pipeline** | New `HunyuanVideoPipeline` (num_frames 129, embedded guidance 6.0, 720×1280, Llava-Llama-3 + CLIP text path) using the plain preset from 1.8 + guidance from 2.7. | L |
| 3.2 | **Wan Animate / S2V presets + transformer wiring** | Add `WanVideoConfig` presets carrying the Animate (motion/face encoder dims, image_dim 1280, added_kv_proj_dim **5120**) and S2V (audio_dim 1024, audio_inject_layers, motion_token_num, adain_mode) fields; confirm `WanAnimateTransformer`/`WanS2VTransformer` read them (currently hardcoded/absent). | L |
| 3.3 | **SD3.5 Skip-Layer Guidance** | Add `skip_guidance_layers`/`skip_layer_guidance_scale`(2.8)/`_start`(0.01)/`_stop`(0.2) to a SD3 request + the SLG forward path. StabilityAI-recommended for Medium. | M |
| 3.4 | **SD3.5 Large-Turbo, HiDream Fast, ERNIE/Boogu/Lens/HunyuanImage distilled** presets | Add the few-step distilled presets (steps/cfg only differ; most backbones identical). | S each |
| 3.5 | **AuraFlow v0.2**, **F-Lite-7B** | Covered by 1.2/1.4 (presets). | — |
| 3.6 | **Kandinsky-5: `attention_type` field** | Add `AttentionType` (regular/nabla); key the NABLA warning off it not off frame count; needed for 10s video checkpoints. | S |
| 3.7 | **GameCraft / MG2 default presets** | Encode GameCraft standard/distilled (steps 50/8, cfg 2.0/1.0); verify MG2 `LocalAttnSize` window + TempleRun 4-step list (currently assumed). | S–M |

---

## Phase 4 — Feature gaps (larger; schedule after correctness, several are legitimately deferrable)

- **LTX-Video/LTX-2 STG** (spatiotemporal skip guidance) + `skip_block_list` + per-step guidance schedules +
  `guidance_rescale`/`modality_scale`. Biggest 13B/LTX-2 quality lever. · L
- **HunyuanImage ByT5 glyph branch** (`glyph_byT5_v2`, ByT5Mapper) — text-rendering prompts wrong without it. · L
- **Reference-image conditioning** for Flux.2 (edit / Klein-KV), OmniGen2 (input_images + image_guidance), Qwen-Image-Edit. · L each
- **Flux.2 dev: caption upsampling** (Mistral VLM, temp 0.15) + fix `text_encoder_out_layers` to `[10,20,30]` for the Mistral path (Qwen path keeps `[9,18,27]`). · M (the out-layers fix is S and should ride in Phase 1/2)
- **HunyuanImage refiner**, **SDXL dedicated 9-ch inpaint UNet**, **APG for F-Lite**, **IP-Adapter** (Chroma/SD/SDXL), **Ideogram4 json_prompt** (structured layout). · M–L each, all deferrable.
- **Engine-wide knobs**: `num_images_per_prompt` (batch>1), custom `sigmas`/`timesteps` override, per-encoder
  `prompt_2`/`prompt_3`. Batch>1 is a cross-cutting refactor — schedule deliberately. · L

---

## Phase 5 — Verification & guard-rails (runs alongside every phase)

1. **Per-model parity harness** — for each model touched, run its `*DebugDump` against the Python
   `dump_*_full_forward.py` / `diff_*_layers.py` flow (the existing pattern in PHASE_4) on real weights. No
   output-changing fix is "done" until first-divergent-layer `avg_err < 1e-3`.
2. **Preset round-trip test** — for every config, assert the hardcoded preset equals what `FromWeights`/
   `AutoDetect` infers from the real checkpoint's keys (catches the Tier-1 class permanently).
3. **Defaults resolution test** — assert each pipeline resolves null request fields to the documented
   reference defaults (locks in Phase 0).
4. **Update [PARAM_PARITY_AUDIT.md](PARAM_PARITY_AUDIT.md) + [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md)**
   status per model as items close.
5. **Confirm-when-available list** — track the UNKNOWN/placeholder items (Flux.2 gated config, Lens mu schedule,
   Boogu epsilons, MG3 everything, Lance backbone, Krea2 template) so they're verified the moment weights land,
   not silently trusted.

---

## Suggested execution order (dependency-aware)

1. **Phase 0** (0.1→0.2→0.3→0.4) — unblocks everything, low risk, immediate broad improvement.
2. **Phase 1** (all eight) as one batch — cheap, fixes the load-breakers; gate each on a real-weight load test.
3. **Phase 2** by priority: 2.1 Wan-UniPC, 2.2 OmniGen2, 2.3 HiDream-MoE, 2.4 Lumina2, 2.8 LTX-VAE,
   2.6+2.7 Hunyuan(image+video), then the smaller 2.5/2.9/2.10/2.11/2.12. Each independently validated.
4. **Phase 3** presets/pipelines as the relevant weights become downloadable.
5. **Phase 4** features by demand.
6. **Phase 5** continuously.

**Reality check on "fix all":** Phases 0–1 are days of work and remove every load-breaker. Phase 2 is the bulk
of the real engineering and is *gated on having the real checkpoints downloaded* to validate — that's the true
critical path, not the code. Phase 4 is genuinely optional surface area. Recommend committing to 0+1+the HIGH
items of 2 as the "fix all the things that are actually wrong" milestone, and treating 3/4 as opt-in breadth.
