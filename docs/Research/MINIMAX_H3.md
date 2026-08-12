# MiniMax-H3 — research notes

> **Status 2026-08-08: SHIPPED AND VERIFIED ON REAL WEIGHTS.** H3 generates video plus its jointly
> generated stereo soundtrack end to end on the fp8 build, including fl2va keyframes, ref2va references,
> LoRA merge, and DiT sharding across two GPUs. Everything below the "Release status" heading is
> **historical provenance** — it records what was known before the weights dropped and how the port was
> derived from Kijai's ComfyUI PR **#15224**. Where this doc and the code disagree, the code is right.
>
> For current state, read these instead: `docs/Checklists/MODEL_STATUS_VIDEO.md` (what works),
> `docs/Checklists/PARITY_VERIFICATION.md` (real-weight parity), and the recipe/pipeline sources
> themselves. Still-live findings kept here: the NVFP4-AWQ text-encoder conventions, and the
> SageAttention F16-V measurement in "Bring-up notes".

## What H3 is

MiniMax-H3 (internally "Hailuo 03") is an omni-modal generative model announced **2026-07-31**. It
takes text, images, video, and audio as input and generates video with a **jointly generated native
stereo soundtrack** — the audio is not a separate TTS/foley pass. Output is up to **15 seconds at 2K,
24 fps**.

MiniMax state they "plan to open up the model weights in the coming days, subject to applicable laws
and regulations," and that a full H3 Technical Report is coming. Neither has shipped.

## Release status — verified, not assumed

Checked 2026-08-01:

| Source | Result |
|---|---|
| `modelscope.cn/api/v1/models/MiniMax/MiniMax-H3` file tree | `获取模型失败` — repo does not exist / not public |
| `huggingface.co/api/models?search=MiniMax-H3` | `[]` |
| `huggingface.co/api/models?author=MiniMaxAI` | newest is `MiniMax-M3` (2026-07-23); no H3 |
| H3 Technical Report | not published |
| `config.json` / any state-dict key list | does not exist publicly |

**ComfyUI's H3 support is API-only and contains zero architectural information.** The nodes in
`comfy_api_nodes/nodes_minimax.py` (`MinimaxHailuo03TextToVideoNode`,
`MinimaxHailuo03FirstLastFrameNode`, `MinimaxHailuo03ReferenceNode`) POST to
`/proxy/minimax/v2/video_generation` and poll `/proxy/minimax/v2/query/video_generation/{task_id}` —
they are a thin cloud client. ComfyUI's own announcement says "Native open-weight support is coming
soon." **SUPERSEDED 2026-08-02** — that prediction was right about *where* (`comfy/ldm/minimax/`) and it
has since arrived as PR #15224. The paragraph above describes only the merged-to-main API nodes; there is
now a full native implementation to port from.

## Capability contract (KNOWN — this part is not speculation)

Extracted from ComfyUI's node schemas and validators, which encode MiniMax's documented API limits.
This is the user-facing surface the engine must eventually satisfy, and it is safe to design against
now.

**Output**

| Field | Value |
|---|---|
| Resolution | `2K` only (no other option exposed) |
| Aspect ratio | `adaptive`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`, `21:9` — T2V only; frame/reference modes inherit AR from the inputs |
| Duration | integer seconds, **5–15** per the Comfy node (`min=5,max=15`). Some press reports say 4–15; trust the node until MiniMax's own docs are readable |
| Frame rate | 24 fps |
| Audio | native stereo on every clip; sample rate not documented |

**Inputs — three modes**

1. **Text-to-video** — prompt + resolution + ratio + duration + seed.
2. **First/last frame** — `first_frame` (required) + `last_frame` (optional), each tagged with a
   `role` field in the request `content` array. Output AR follows the images; no `ratio` input.
3. **Reference-to-video** — up to **9 reference images**, **3 reference videos**, **3 reference
   audio clips**. Referenced positionally from the prompt text as `"Image 1"`, `"Video 1"`,
   `"Audio 1"`, … At least one image or video is required; **audio references cannot be used
   alone**.

**Validation limits**

- Images: min 256×256, aspect ratio within **0.4–2.5**, ≤30 MB.
- Reference videos: **23.976–60 fps**, ≥2 s each, ≤15 s total across all clips, ≤50 MB.
- Reference audio: ≥2 s each, ≤15 s total, ≤15 MB.
- Mixed reference inputs capped at **12 files** total; prompt ≤7,000 characters.
- Seed: uint32. MiniMax explicitly document it as **not** guaranteeing identical results — same seed
  gives "similar, but not guaranteed identical" output. Do not build a bit-exactness gate on it.

## Architecture — the four named components (marketing-level only)

MiniMax name four pieces. Each is a label with a claimed benefit and **no disclosed structure** —
treat every one as an open research question, not a design input.

- **H3-VAE** — replaces the conventional video tokenizer. Claimed **4× gain in effective sequence
  length** from a higher compression ratio, and this is credited as the enabling technology for
  native 2K. Compression ratio, latent channel count, and whether video and audio share one latent
  space are all undisclosed.
- **H3-Omni Transformer** — one pre-training framework spanning T2I, T2V, I2V, and audio. Claimed to
  "separate understanding and generation workloads" for better hardware utilization, worth ~30%
  end-to-end training throughput. Whether that separation is a two-tower design, an MoE, or a
  routing scheme is undisclosed.
- **Contextual Omni Representation** — language used as the bridge between modalities. ~100K tokens
  of source material distilled to an average of ~4K tokens at inference.
- **In-Context Regeneration** — 2K output is produced by the base model **regenerating** a
  low-resolution result, explicitly *not* a bolt-on super-resolution module. Implies a second
  conditioned pass through the same backbone rather than a separate upscaler stage.

Unverified: a widely-reshared tweet claims H3 "can run on RTX 3060." That is tweet-level, contradicts
nothing but is supported by nothing. Do not size the implementation around it.

## Pre-drop intel (2026-08-02) — the text encoder is ANSWERED

**`DeepBeepMeep/MiniMax-H3` on HuggingFace is staging H3 components** (last modified 2026-08-02 14:51).
DeepBeepMeep is the Wan2GP/WanGP author, who repacks models for low-VRAM use — the README says these are
"the MiniMax H3 image models used with WanGP". Files present so far:

```
Qwen3-VL-32B-Instruct/Qwen3-VL-32B-Instruct-layer50_bf16.safetensors
Qwen3-VL-32B-Instruct/Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors
Qwen3-VL-32B-Instruct/{config,tokenizer,tokenizer_config,preprocessor_config,chat_template,vocab}.json
```

So **H3's conditioning is Qwen3-VL-32B-Instruct, tapped at layer 50** — one of the day-one unknowns below is
now answered. From its `config.json`: `Qwen3VLForConditionalGeneration`, text hidden 5120, 64 layers,
64 heads / 8 KV heads (GQA), vocab 151936, rope_theta 5e6, max_position 262144; vision hidden 1152.
The layer-50-of-64 tap is the same middle-layer-hidden-state idiom Hunyuan/Qwen-Image use.

**This is a big head start: the engine already has Qwen3-VL** — `Qwen3VlMultimodalEncoder`,
`Qwen3VlEncoder`, `Qwen3VlVisionEncoder`, `Qwen3VlImageProcessor`, `Qwen3VlVisionConfig`. The conditioning
tower should be reuse, not a port.

Only the text encoder is staged; the DiT and VAEs are not up yet, which is why a watcher polls for them.

**ComfyUI still has NO native support.** Verified: `Add minimax h3 support (#15167)` (2026-07-31) touches
only `comfy_api_nodes/apis/minimax.py` and `comfy_api_nodes/nodes_minimax.py` — the cloud client, no
`comfy/ldm` changes; and `comfy/ldm/` contains no h3/hailuo/minimax directory. Today's
`[Partner Nodes] feat(Minimax): add 768P resolution for H3 model (#15227)` is also API-only, but it does
tell us **H3 now exposes 768P as well as 2K**. Note the repo moved to **`Comfy-Org/ComfyUI`**
(`comfyanonymous/ComfyUI` 301-redirects; the API needs `-L` or the new slug).

## THE ARCHITECTURE IS PUBLIC — Kijai's ComfyUI PR #15224 (2026-08-02)

**`feat: Support MiniMax-H3 (CORE-375)`**, open (not merged), +2599/−6 across 16 files, branch
`kijai:minimax_h3`, head `e2ab36d933356bc8cd6ecb39c655fe8be75af4e5`. This is the **native open-weight
implementation** — it supersedes everything below that says "undisclosed". Re-fetch with:
`https://raw.githubusercontent.com/kijai/ComfyUI/<sha>/comfy/ldm/minimax/model.py` (also `vae.py`,
`audio_vae.py`, `comfy/text_encoders/minimax.py`, `comfy_extras/nodes_minimax_h3.py`).

### DiT (`comfy/ldm/minimax/model.py`, 646 lines)
Single-stream **packed-token** transformer denoising video and audio jointly. Defaults:

| | |
|---|---|
| hidden_size | 5376 |
| num_layers | 50 |
| token_refiner_num_layers | 2 |
| num_attention_heads / head_dim | 56 / 128 |
| ffn_hidden_size | 14336 |
| latents_dim (video) | 24, patch (1,2,2) |
| audio_latents_dim | 32 |
| text_dim | 5120 (Qwen3-VL hidden) |
| rope_inv_freq_len | 16 |
| norm/qk_norm/final eps | 1e-5 |
| sigma_shift video / audio | **12.0 / 3.0** |
| dtypes | bf16 + fp32; `memory_usage_factor` 0.114 |

Packed sequence: `[text | cond rows | audio | video]` for t2va/fl2va, `[text | reference blocks | audio |
video]` for ref2va (`PackedLayout` class). Constants: `FRAME_PER_TOKEN=(1,4,4,4,4)`, `FRAME_RESCALE=5/3`,
`VISUAL_COND_TIMESTEP=0.999`, `AUDIO_COND_TIMESTEP=1.0`. Structure: `TimeEmbedder`, `Attention` (qkv_proj +
q_norm/k_norm), `MLP` (fc1 gated — `//2`), `AdalnProj`, `RefinerBlock`/`TokenRefiner`, `DiTBlock`,
`FinalLayer` (separate `video_out` / `audio_out`).

**The audio timestep trick** — worth reading even outside H3, given our LTX-2 audio history: the sampler
supplies the *video* sigma; per-token timesteps are `t = 1 − sigma`; the audio stream runs its own shifted
schedule derived in closed form, and **the returned audio velocity is scaled by `d(sigma_a)/d(sigma_v)`**
(`time_shift_sigma` / `time_shift_slope`) so a stock sampler integrating the flat AV pack still solves each
stream's true ODE. `sampling_settings = {"shift": 12.0}`.

**Two checkpoint variants**: the original time-embedder, and a pruned (~40% smaller) variant that replaces
the time embedder + full-width adaln weights with a precomputed `adaln_t_table` curve basis
(`adaln_curve_grid`). Detection keys off `adaln_t_table` being present.

### VAEs
- **Video** (`vae.py`, 694 lines): 3D causal CNN encoder + **ViT3D decoder**, internal spatial tiling and
  temporal chunking. 24 latent channels, **16× spatial / 4× temporal** downscale, `scale_factor` 1.0.
- **Audio** (`audio_vae.py`, 442 lines): **DAC-lineage encoder + BigVGAN decoder**, stereo at **32 kHz,
  800 samples per latent frame** (→ the 40 Hz latent rate). Our LTX-2 BigVGAN work is adjacent prior art.
- `latent_formats.MiniMaxH3AV` packs both streams at `latent_channels = 32` (max of video 24 / audio 32).

### Detection (`comfy/model_detection.py`)
Signature is `video_patch_proj.weight` **and** `audio_patch_proj.weight`; everything else is derived from
shapes (`blocks.N.attn.qkv_proj`, `blocks.0.attn.q_norm`, `blocks.0.mlp.fc1`, `condition_proj`,
`final_layer.video_out` `//4` for patch 1×2×2, `rope.inv_freq`). Prefixes: `vae.`, `text_encoders.`.

### Text encoder (`comfy/text_encoders/minimax.py`)
Qwen3-VL-32B **truncated to 50 layers**, consumed as the **unnormalized last hidden state**,
**non-chat-templated**, with `<Picture i>` / `<Video k>` / `<Audio j>` labels and **2 fps timestamped video
blocks**. State-dict prefix `text_encoders.qwen3vl_32b.transformer.`.

### Nodes
`EmptyMiniMaxH3LatentAV`, `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`, `MiniMaxH3SigmaShift`
(exposes `minimax_h3_sigma_shift_video` / `_audio`). Reuses LTXV's AV latent concat/separate nodes.

### Non-model changes in the PR
ModelOpt AWQ-style `pre_quant_scale` in quant ops, and a fused activation+quantize path for INT8 linears.

## Device-op plan (audited 2026-08-02) — write the DiT device-first

Every H3 stage mapped to an existing `IBackend` op, so the port composes rather than inventing kernels.
**`IBackend` gives almost every op a host default body**, so a backend that lacks an override does not fail —
it silently runs on the CPU through `DataPointer`, draining the stream. Absence of an override is therefore
a *performance and correctness* trap, not a compile error. Coverage below is what the audit found.

### Per-block hot path (runs x50 — must never touch the host)

| H3 stage | op | CUDA | Vulkan |
|---|---|---|---|
| `norm1`/`norm2`, refiner norms | `RmsNorm` | ✅ | ✅ |
| adaln scale/shift (`_mod_scale_shift`) | `AffineBroadcastLastDim` (`out = in*scale + shift`; precompute `1+scale` once per step on the small adaln tensor) | ✅ | ✅ |
| gated residual (`_mod_gate` addcmul) | `GatedResidualLastDim` (`out = residual + gate*value`) — exact match | ✅ | ✅ |
| qkv / out / fc1 / fc2 | `Linear` | ✅ | ✅ |
| attention | `ScaledDotProductAttention` / `FlashAttention` | ✅ | ✅ |
| per-head q/k RMS + **partial split-half RoPE** | `ApplyRopeSingle(x, cos, sin, rotaryDim)` | ✅ | ❌ host |
| SwiGLU (`fc1` gated pair → `fc2`) | `GluActivate` | ✅ | ❌ host |

**RoPE is the one that had to be gotten right.** H3 rotates **96 of 128** head dims with NEOX split-half
pairing `(i, i+48)`, leaving dims 96–127 untouched (`rope_freqs` builds `[S,48*3=144?]` no — `cat(t,h,w)`
= `[S,48]` then `cat(half,half)` = `[S,96]`; `rope_rotation_table` reads `angles[:, :48]`, and the attention
computes `rot_dim = 48*2 = 96`). Two nearby ops are WRONG for this:
- `Ltx2SplitRope` hardcodes `r = headDim / 2` — full rotary only, would rotate 64/64 instead of 48/48.
- the interleaved family (`WanRopeInterleaved`, `ApplyRopeInterleaved`) pairs `(2i, 2i+1)`, not split-half.

`ApplyRopeSingle` is the correct one: NEOX `(i, i+rotaryDim/2)` pairing, partial-rotary aware, cos/sin kept
at `headDim` stride with only the first `rotaryDim` entries read, `x` shaped `[B,L,heads,headDim]`. The CUDA
override forwards `rotaryDim` to the kernel and clears the stale in-place callbacks (pitfall #17). Getting
this wrong is the coherent-but-wrong failure class (the GLM-4 partial-rotary and Qwen3.5 split-half bugs).

### Once per forward
`Permute0213` (patchify/unpatchify video, pack/unpack audio) ✅ both; `SliceRows` ✅ both;
`RowGather` ✅ CUDA / ❌ Vulkan (only needed when conditioning rows interleave with target rows).

### Legitimately host-side — not "host glue"
The time embedder's sin/cos and the adaln curve-table lerp run over the **unique timesteps only**, of which
there are at most four per step (video, audio, and optionally the visual/audio conditioning pins). That is a
handful of scalars per sampling step, not per token or per block, and it is fp32 in the reference too.
Keep it host-side and don't let a reviewer mistake it for the SeedVR2-class host-math problem.

### Vulkan status — honest
The per-block path is fully device-resident **on CUDA**. On Vulkan, **RoPE and SwiGLU fall back to host**,
which would mean ~100 stream drains per forward (50 blocks x 2). This is **pre-existing and not H3-specific**:
`ApplyRopeSingle`, `Ltx2SplitRope`, `ApplyRopeInterleaved` and `WanRopeInterleavedPerHead` have **no Vulkan
override at all**, so every video DiT in the repo (Wan, LTX-2, …) has the same gap. Closing it means SPIR-V
shaders for a rope family + `GluActivate`, which is its own task — tracked in MODEL_STATUS_VIDEO, not
something to discover during H3 bring-up. Until then, H3 on Vulkan will run but slowly, and that is a
known-and-stated limitation rather than a surprise.

## Comfy's 66% memory reduction — decoded (2026-08-03)

Comfy get 123.6 GB (full precision) down to 42.5 GB via three levers. Two are cheap for us; one is real work.

### 1. Pruned modulation → lookup table — **ALREADY IMPLEMENTED HERE**
The adaln modulation weights are ~40% of parameters. The pruned release deletes the time embedder and the
full-width adaln input, replacing them with `adaln_t_table F32 [1025, 8]` — a 1025-point curve sampled over
t, projected by a **[96768, 8]** adaln linear instead of [96768, 2688]. That is a 336x reduction on the
adaln input side.

`MiniMaxH3Config.Detect` already reads it (`adaln_curve_grid=1025`, `time_embed_dim=8`), and
`MiniMaxH3Transformer` already implements both the curve lerp (`BuildTimeEmbedding`) and the SiLU-skip the
curve form requires (`Adaln`). Verified against the real pruned header. **No work needed.**

### 2. int8 convrot — the missing piece, now fully specified
Each quantized linear ships three tensors (4 linears x 50 blocks = 200 groups):

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `X.weight` | **I8** | [out, in] | quantized weights |
| `X.weight_scale` | **F32** | **[out, 1]** | per-output-channel scale |
| `X.comfy_quant` | U8 | [72] | JSON descriptor |

The descriptor decodes to plain JSON:
`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`

So "convrot" is a **block-diagonal rotation over groups of 256 input channels** — the QuaRot/SpinQuant
family. Weights are stored rotated (`W' = W R`); at inference you rotate the *activation* by the same R
(`x W^T = (x R) W'^T`), which spreads outliers so int8 stays accurate. R is orthogonal, so it cancels.

Only `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2` are quantized — norms stay BF16, patch
projections and `adaln_t_table` stay F32, adaln linears are F16.

**Two implementation phases:**
- **Phase 1 (correctness, cheap):** at load, dequantize `W = W_i8 * scale[o]` and fold the rotation out
  (`W_orig = W' R^T`, R orthogonal). Gives a runnable pruned checkpoint with **no** memory win — weights
  land in F32/BF16. Good for proving the pruned path end to end.
- **Phase 2 (the actual win):** keep int8 resident and apply the grouped Hadamard to activations at
  runtime, then an int8 GEMM with per-output-channel dequant. This is where the 66% comes from. Reuse the
  existing W8A8/SmoothQuant int8 GEMM work rather than starting fresh; the new kernel needed is the
  size-256 grouped Hadamard on activations.

### 3. Dynamic VRAM offloading
Comfy's third lever. Our equivalent already exists: `BlockStreamingController` / `IStreamingBlock`, built
for LTX-2.3's 19 GB fp8 DiT in a ~1.2 GB resident window and reused by Wan. H3's 50 blocks need wiring to
it — plumbing, not new machinery.

**Footprint math for the smallest variant:** DiT pruned+int8 20.97 GB + text encoder nvfp4_awq 15.69 GB +
video VAE fp16 5.21 GB + audio VAE 0.61 GB = **42.5 GB**, matching Comfy's number. The nvfp4 text encoder
additionally needs NVFP4 support, which is roadmap-only here — the int8 text encoder (27.1 GB) is the
nearer target once convrot lands.

## NVFP4-AWQ text encoder — decoded conventions (2026-08-03)

The `qwen3vl_32b_minimax_h3_nvfp4_awq` release (15.69 GB vs 51.5 GB bf16) is loadable with the existing
`Nvfp4Codec`. Descriptor read from the real file: **`{"format": "nvfp4", "full_precision_matrix_mult": true}`**
— dequantize, then a normal full-precision GEMM. No nvfp4 tensor-core kernel required.

Layout per quantized linear: `X.weight` U8 `[out, in/2]` (nibble-packed FP4), `X.weight_scale` F8_E4M3
`[out, in/16]` (**block size 16**, matching `Nvfp4Codec`), `X.weight_scale_2` F32 scalar global scale.

**Three conventions that shape alone cannot discriminate** — established numerically against the bf16
release of the same encoder, so treat these as settled, not guesses:

1. **Block scales are swizzled** (`BlockScaleSwizzle`). Reading them row-major, or applying `W·s` instead
   of the dequant direction, lands far off. Check: `amax(W/s per 16-element block) / (6·scale)` has median
   **1.0019**, range [0.943, 1.058] — exactly E4M3's ±6% rounding band.
2. **`pre_quant_scale` DIVIDES the stored weight**, so the activation must be MULTIPLIED:
   `x·Wᵀ = (x⊙s)·(W/s)ᵀ`. It exists only on `down_proj` and `o_proj` (50 each).
3. **AWQ scale migration**: for `q/k/v/gate/up` the scale was folded into the PRECEDING RMSNorm, which is
   why they carry no explicit `pre_quant_scale`. `input_layernorm` ratio (nvfp4 vs bf16) median **0.49099**
   matches `q_proj`'s amax ratio **0.4886**. **Consequence: use the nvfp4 file's own norm weights.** Mixing
   in norms from the bf16 file double-applies the migration and silently corrupts conditioning.

Also: **`model.embed_tokens` is I8, not nvfp4** — `[151936, 5120]` I8 with F32 `weight_scale [151936, 1]`,
`amax/scale = 127.000` exactly, so `w = i8 · scale[row]`. Dequantize only the referenced rows.

Cost note: transient per-linear dequant (the `GptOssMoeFfn` discipline, ~500 MB peak) means ~24 GB of
dequant work per encode — correct and memory-bounded, but slow on CPU. A per-layer dequant cache is the
obvious lever if prompt-encode latency matters.

## Unknowns — the day-one checklist

When weights land, these are the questions to answer in order. Every one of them currently blocks
writing a single line of forward pass.

- [x] **Parameter count / layout** — DiT hidden 5376, 50 layers, 56x128 heads, ffn 14336 (PR #15224).
- [x] **H3-VAE** — video: 24 latent ch, 16x spatial / 4x temporal, 3D causal CNN encoder + ViT3D decoder.
- [x] **Audio latent path** — SEPARATE audio VAE: DAC-lineage encoder + BigVGAN decoder, stereo 32 kHz,
      32 latent ch at 40 Hz (800 samples/latent frame).
- [x] **Text encoder** — Qwen3-VL-32B truncated to 50 layers, unnormalized last hidden state,
      non-chat-templated with `<Picture i>`/`<Video k>`/`<Audio j>` labels + 2 fps timestamped video blocks.
- [x] **DiT block structure / RoPE** — single-stream packed tokens, qkv_proj + q/k RMS-norm, gated MLP,
      adaln modulation, `rope.inv_freq` length 16.
- [x] **Scheduler / guidance** — flow-match, sigma_shift video 12.0 / audio 3.0; audio schedule derived
      from the video sigma in closed form with the velocity scaled by d(sigma_a)/d(sigma_v).
- [ ] **In-Context Regeneration mechanics** — how the low-res pass feeds back for 2K. Not obviously in the
      PR; may be a pipeline-level two-pass rather than a model feature.
- [ ] **License** — MiniMax say "subject to applicable laws and regulations," which is not a license
      name. Confirm before shipping catalog assets.

## Engine state (HISTORICAL — this section described the pre-weights seam)

Everything this section used to say is obsolete: `Construct()` no longer throws `UnsupportedModelException`,
the catalog entry is no longer `Structural`, and `VideoDefaults` are no longer placeholders. `MiniMaxH3Recipe`
is still registered in `VideoRecipeRegistry` under `minimax-h3`, `minimax-hailuo-03`, `hailuo-03`, `hailuo03`,
and now builds a real pipeline.

The one impedance mismatch noted here did survive and is worth keeping: **H3's cloud API takes duration in
integer seconds (5–15) while `VideoRequest` carries a frame count.** The engine works in frames, snapping
onto the `17k + 5` grid (`MiniMaxH3Geometry.AlignFrameCount`), with 124 frames the native default.

For what actually works today, see `docs/Checklists/MODEL_STATUS_VIDEO.md`.

## Bring-up notes to save the next session time

**Naming and hosting.** Record every alias — the sniffer must not be keyed to one guess:

- Comfy calls the nodes `MinimaxHailuo03*`; **H3 == "Hailuo 03"** internally. The architecture string
  in the checkpoint could plausibly be `hailuo`, `Hailuo03`, `hailuo-03`, or `MiniMaxH3`.
- **Org slugs differ per host**: ModelScope is `MiniMax/MiniMax-H3`; HuggingFace uses `MiniMaxAI/*`
  (so expect `MiniMaxAI/MiniMax-H3`). Check both hosts — MiniMax has historically published to
  ModelScope first.

**The audio return path is already built (2026-08-01) — H3 does not have to solve it.** This was the
one blocking prerequisite: `IVideoRecipePipeline.Generate` used to return frames only, so LTX-2.3's
generated soundtrack was logged-and-dropped. It now returns `VideoGenerationResult` (frames plus an
optional `AudioBuffer`), `VideoAudioResolver` picks and trims the track, and the SwarmUI extension's
ffmpeg mux is reconnected. An H3 pipeline that produces a stereo waveform just attaches it to its
result and the mux happens for free.

**SageAttention F16-V overflow — MEASURED CLEAR, 2026-08-08.** H3's SDPA calls take the default INT8
SageAttention path (F32 in, no mask, `d=128`, `Skv >= 2048` — every real geometry qualifies). That path
quantizes Q/K but materializes V as an F16 transpose, so any `|V| > 65504` becomes INF and the softmax
smears it across every query row with no error raised — the failure that bit Lens at its block 45. H3's
documented ~2.7e6 residual made this worth checking directly.

It is not a problem, and the reason is structural: `norm1` precedes the qkv projection, so **V is a
projection of a normalized tensor, never of the raw residual stream**. Measured with `HARTSY_H3_VPROBE=1`
over a full 30-step generation (141f@512x288, 1500 block-probes, zero non-finite):

| | max\|V\| | % of F16 max |
|---|---|---|
| block 0 | 81 | 0.12% |
| block 48 (hottest) | ~1200 | 1.83% |
| peak over all blocks/steps | **1201** | **1.83% (55x margin)** |

It grows with depth but oscillates in a band across steps rather than compounding the way Lens did — so
**probe every block, not block 0**: block 0 reads 15x lower than the real peak and would falsely reassure.
No V-damping actuator is needed; the probe stays env-gated off (it is a host-side synchronizing scan).

**Precedents worth reading first.** LTX-2.3 (`LtxVideo2Pipeline`, `Ltx2Result`) is the closest
existing architecture: dual-stream video+audio DiT with separate video and audio VAEs and a BigVGAN
vocoder. Whatever H3 turns out to be, that pipeline is the nearest template and its
`Ltx2Result`-shaped return is the natural model for the widened contract.

## Region-targeted reference conditioning (Tier 3.8) — DONE, real-weight verified 2026-08-11

The SwarmUI extension backlog's own item 3.8 asked whether `<segment:>`-style masked refinement
(Tier 3.2, `Engine/Features/SegmentRefinement.cs`) should extend to H3's reference-image conditioning
— "condition on only the face region of this reference image" instead of the whole asset. It doesn't
extend as a code path (3.2 is a post-hoc pass on the DECODED CANVAS after generation; a reference
image is an INPUT the model conditions on before generation starts — different point in the pipeline
entirely), but the underlying idea (auto-crop a reference to a text-matched region before it's
encoded) is real, and shipped as `<refcrop:N,query[,threshold]>`.

**Say the honest baseline first**: this feature is a convenience, not a capability unlock. A user can
already crop the reference image themselves before uploading it — H3's own reference encode
(`EncodeReferenceImage` below) has no canvas requirement to work around. The value is "point at what
you mean in text" instead of "open an image editor," nothing more.

**Mechanical question, resolved — no aspect-ratio padding needed.** Read
`MiniMaxH3RecipePipeline.EncodeReferenceImage` (`.cs:585-606`) directly rather than assuming: it scales
a reference DOWN ONLY to fit the generation's pixel budget and keeps the reference's OWN aspect ratio
— no squash, no letterbox, no forced canvas match ("a reference is not the canvas, so it keeps its
shape rather than being stretched," per the method's own doc comment). A CLIPSeg-cropped face/object
region has whatever aspect ratio its bounding box happens to be, and this encode path already handles
arbitrary aspect ratios by design. So a cropped reference needed ZERO new handling here — it's just a
smaller/differently-shaped `ImageData` fed into the exact same call, confirmed by shipping it. This
also means the crop doesn't touch `MiniMaxH3RefBlock` packing or the timestep-row conditioning at
all: `LatentH`/`LatentW` are already computed FROM the encoded image's own dimensions post-encode.

**Mask→bbox reuse, cleaner than expected once actually checked.** `ClipSegSegmenter.Segment(backend,
modelDirectory, image, query, threshold)` is already generic — takes any standalone `ImageData`,
reused directly. `FeatureImaging.MaskBounds(mask, width, height, grow, threshold)` — the ACTUAL
bbox/oversize math `InpaintOnlyMasked.Prepare` calls internally — turned out to already be a
standalone static utility, not tied to `ImageRequest` at all; no new helper was needed, just
orchestration (`InpaintOnlyMasked.Prepare` itself, which IS request-shaped, was the thing not directly
reusable — the design pass's "~20-line new helper" estimate was wrong in the safe direction).

**The syntax decision, resolved (user explicitly asked to match SwarmUI's own conventions and build
it): `<refcrop:N,query>` / `<refcrop:N,query,threshold>`.** Researched SwarmUI core directly before
inventing anything (`PromptRegion.cs`, `T2IPromptHandling.cs`, `docs/Features/Prompt Syntax.md`) —
confirmed: (1) `<segment:query,creativity,threshold>`'s own grammar (right-to-left, last comma-field =
threshold if it parses as a float) is the established comma-arg convention this codebase already
mirrors; (2) H3's own `<Picture N>` presentation is 1-based (`MiniMaxH3TextEncoding`'s own doc:
"1-based per-type ordinals") — `<refcrop:>`'s index matches that numbering directly, so "reference 1"
means the same thing in both places; (3) an unregistered `<...>` tag prefix passes through BOTH of
SwarmUI core's parsing layers byte-for-byte unchanged (confirmed by reading `PromptRegion`'s and
`T2IPromptHandling`'s own fallback code) — exactly how `<Picture N>` itself already survives to the
engine today — so `<refcrop:>` needed no `PromptRegion.RegisterCustomPrefix` registration; it's parsed
entirely engine-side, in `ReferenceCropResolver.cs`, mirroring `PromptRegionParser`'s own
split-and-scan style rather than a regex. One correction to this doc's earlier framing: `<Picture N>`
is not actually something SwarmUI's text encoder "resolves" from user-typed inline text — it's a
label the ENGINE ITSELF generates ahead of the prompt (`MiniMaxH3TextEncoding.cs`, `AddText($"<Picture
{++images}>: ")`), one per attached reference, in list order; a user typing `<Picture 1>` in their own
prose is just ordinary text the LLM correlates back to the vision block via its own language
understanding, not a tag SwarmUI or the engine parses.

**Implementation**: `Engine/Features/ReferenceCropResolver.cs` (new) — `HasCropTags`/`Apply`, same
public-surface shape as `SegmentRefinement`. Wired into `MiniMaxH3RecipePipeline.Generate` as the
FIRST thing that runs, before `request.Prompt` is read anywhere — learned directly from the Tier 3.2
base-prompt tag-leak bug (an un-stripped tag reaching the text encoder steers the whole generation,
not just the crop); `Apply` always strips the tag even when nothing ends up cropped. Scope matches
Tier 3.2's own precedent: reference IMAGES only, CLIPSeg free-text only (no YOLO), an empty match
warns and falls back to the whole uncropped reference. Real-weight verified in two parts (the design
pass's own prescribed verification shape): (1) a synthetic red-square-on-gray reference, cropped and
the crop PNG actually looked at — CLIPSeg correctly isolated the square with the expected oversize
margin; (2) a same-seed A/B through the real ref2va checkpoint (256x256, 5 frames, 6 steps) — cropped
vs. whole reference produced a measurably different (mean abs diff 10.91) AND visually different
frame 0, both coherent on-prompt "calm ocean at dusk" scenes, confirming the crop reaches conditioning
rather than being silently dropped. `TestPaths.MiniMaxH3.DitRef2VaFp8` added (was missing — only the
fl2va checkpoint had a constant). 7 pure-logic tests (`ReferenceCropResolverTests.cs`) lock in the
grammar's malformed/edge-case tolerance (missing comma, 0-index rejected since 1-based, non-numeric
index, empty query, unterminated tag) without touching CLIPSeg.

**Naming trap, still true, worth repeating for whoever picks this up**:
`MiniMaxH3SegmentKind` (`Diffusion/Models/Denoisers/MiniMaxH3SegmentKind.cs`) is an unrelated packed-
sequence bookkeeping enum (`Text, Cond, RefImage, RefAudio, Audio, Video` — which token-range a block
occupies), not connected to `<segment:>` prompt syntax at all. A grep for "Segment" near MiniMaxH3
will hit this and can mislead.

## Chunked-attention activation peak (Tier 3.9) — design pass 2026-08-12

The roadmap's "CPU-offloaded activations" item resolves here rather than as a paging feature; the
general reasoning is in [`MEMORY_SCHEDULING_SERVING.md`](MEMORY_SCHEDULING_SERVING.md) §9. H3 is the
engine's one *measured* activation OOM, so it is the case that decides the mechanism.

`AttentionChunked` pass 1 holds q + k + v full-sequence simultaneously (3x `seq·inner·4`). q cannot be
scattered away like k/v were — the `qChunks` list is already exactly one full-size buffer, not the
redundant second copy that the `ScatterSeqHeadMajor` fix removed for k/v. So the only ways down are
paging q to host or not holding it. Paging costs ~2 GB of round trip **per block per step** across 50
blocks through a pageable, synchronous path — the wrong mechanism (§9).

**The fix: split the fused projection across the two passes.** Pass 1 projects k+v only (2/3 of the
GEMM), pass 2 re-projects q per chunk (1/3). Identical total FLOPs; pass-1 peak 3x → 2x. Enabled by the
packing being `[q(inner) | k(inner) | v(inner)]` per token, so both weight slices are contiguous row
ranges of `qkv_proj.weight` (see `QkvSplitNormHeadMajor`'s indexing). The fused split/norm/head-major op
is generalized to emit a subset rather than gaining two near-duplicate siblings. Separately, both
`AttentionChunked` pass 2 and `MlpChunked` stop accumulating output chunks for a final `Concat` and
scatter each chunk into the destination instead (`ScatterRowsGeneric` — the byte-offset inverse of
`SliceRowsGeneric`, a plain D2D copy with no kernel).

**Not bit-identical.** q's GEMM narrows from `[c,hidden]x[hidden,inner·3]` to `[hidden,inner]`, so
cuBLASLt selects a different algorithm — the same divergence class as chunked-vs-unchunked, which is
why the sub-`chunkRows` dispatch deliberately keeps the legacy `Attention` path untouched. The scatter
half *is* bit-identical. Gate on a real generation, not byte-equality.

**Floor math.** `EstimateFloorBytes` goes from `residual + 3·fullSeqInner + chunkScratch + fudge` to
`residual + max(2·fullSeqInner + chunkScratch, 2·fullSeqInner + seq·hidden·bodyBytes) + fudge` — for
this config, slope `107520·seq` → `78848·seq` bytes, a **+36% sequence-length ceiling**. It must change
in the same commit or the pre-flight keeps refusing exactly the geometries the fix enables.

**Recorded before-state (4090, 2026-08-12):** resident DiT 19,988 MB, 22,634 MB free → 2,646 MB for
activations. `56f@768x768` (seq 10,490) needed 2,771 MB — **refused, 125 MB short**. Predicted post-fix
floor 2,484 MB (~161 MB margin). Note the refusal threshold moves with whatever else holds VRAM
(an idle RustDesk process held 424 MB during this measurement), so re-measure rather than reusing
these numbers; `MiniMaxH3ActivationEstimateTests`' class-comment floors are themselves stale, fitting
an older `100352·seq` slope that the current code no longer computes.

## Sources

- [MiniMax H3 announcement blog](https://www.minimax.io/blog/minimax-h3)
- [ComfyUI `comfy_api_nodes/nodes_minimax.py`](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_api_nodes/nodes_minimax.py) — API node schemas, the source of the capability contract
- [ComfyUI announcement (partner nodes, "native open-weight support coming soon")](https://x.com/ComfyUI/status/2083071877891682784)
- [MarkTechPost writeup](https://www.marktechpost.com/2026/08/01/minimax-releases-minimax-h3-an-omni-modal-video-model-that-generates-15-second-2k-clips-with-native-stereo-audio/) — reference-input limits, file size caps
- [Comfy docs — Hailuo MiniMax in ComfyUI](https://comfy.org/p/supported-models/hailuo-minimax/)
