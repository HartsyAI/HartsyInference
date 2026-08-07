# Ideogram 4 Architecture — Research Notes

> **Status:** Complete (read-only from upstream code; no checkpoint inspected on disk yet) | **Last Updated:** 2026-06-07 | **Needed Before:** `Ideogram4Transformer`, `Ideogram4Pipeline`, `Ideogram4CheckpointConverter`, and the structured-prompt builder ([STRUCTURED_PROMPT_BUILDER.md](STRUCTURED_PROMPT_BUILDER.md))
>
> **Sources of truth:**
> - GitHub: [ideogram-oss/ideogram4](https://github.com/ideogram-oss/ideogram4) — `src/ideogram4/modeling_ideogram4.py`, `pipeline_ideogram4.py`, `scheduler.py`, `sampler_configs.py`, `latent_norm.py`, `constants.py`, `autoencoder.py`, `magic_prompt.py`, `quantized_loading.py`
> - GitHub docs: `docs/model_architecture.md`, `docs/pipeline.md`, `docs/inference.md`, `docs/prompting.md`
> - HuggingFace (official, gated): `ideogram-ai/ideogram-4-nf4`, `ideogram-ai/ideogram-4-fp8`
> - HuggingFace (ComfyUI, ungated): [`Comfy-Org/Ideogram-4`](https://huggingface.co/Comfy-Org/Ideogram-4)
> - Text encoder config: [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) `config.json` (`text_config`)
> - ComfyUI reference (treat as **example, not gospel** — several known mistakes; see § Differences Between Implementations): [docs.comfy.org Ideogram v4](https://docs.comfy.org/tutorials/image/ideogram/ideogram-v4)
>
> **License:** "Ideogram 4 Non-Commercial" (gated on HuggingFace). The DiT weights and inference code are non-commercial. The ComfyUI repackage (`Comfy-Org/Ideogram-4`) mirrors the same weights. **This is a non-commercial license — flag for the package/legal boundary; HartsyInference itself stays MIT/permissive, the model weights carry their own terms (same handling as the existing GameCraft license-acceptance gate).**

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Ideogram 4 is a **9.3B-parameter single-stream Diffusion Transformer** (DiT) text-to-image foundation model, open-weight release of the `ideogram-oss` org. Unlike the dual-stream MMDiT designs already in this codebase (Flux, SD3.5, Lens), Ideogram 4 concatenates text tokens and image-latent tokens into **one unified sequence** and runs them through **34 identical single-stream blocks** — there are no separate text/image branches. Conditioning comes from **Qwen3-VL-8B-Instruct**: the pipeline runs the prompt through the VLM's 36-layer language model and **concatenates the hidden states of 13 layers** `(0, 3, 6, …, 33, 35)` channel-wise (4096 × 13 = **53248**), then RMSNorms and projects that down to the model width 4608. The other headline trick is **3D MRoPE** (Qwen-VL-style multimodal rotary embedding with sections `(24, 20, 20)`, θ=5e6) that puts text and image tokens in a **shared positional space**, plus a `segment_ids` block-diagonal attention mask and an `image_indicator` embedding to keep the two modalities from cross-contaminating.

The model is trained on **structured JSON captions** (scene summary + style block + per-object descriptions with bounding boxes and hex color palettes), which is why community users report prompting is hard — you get the most out of it by hand-authoring JSON or running an LLM "magic prompt" expander. That motivates the companion [STRUCTURED_PROMPT_BUILDER.md](STRUCTURED_PROMPT_BUILDER.md) design: a model-agnostic structured-prompt data model with a per-model serializer dialect (Ideogram-4 JSON first, regional-attention prompting for other models later).

VAE is the **Flux.2 semantic VAE** (`flux2-vae.safetensors`, 32-channel, 8× spatial) — **already implemented** in this codebase for Flux.2/Lens — but Ideogram does **NOT** use the Flux.2 BN running-stats un-normalize. Instead it applies its own **fixed 128-value per-channel `LATENT_SHIFT`/`LATENT_SCALE`** constants at the pipeline boundary (`z = z * scale + shift` before decode). Sampling is **flow-matching Euler** with a **logit-normal timestep schedule** whose mean auto-adjusts for resolution, and **asymmetric CFG** (the unconditional branch is the image-only token sequence with zeroed text features — shorter and cheaper than a symmetric pass) with a **two-stage guidance schedule** (gw≈7 for the bulk, gw≈3 for the last few "polish" steps).

For HartsyInference, the genuinely new pieces are: (1) **single-stream unified-sequence DiT** with **scale-only AdaLN** (no shift) + tanh-gated residuals + sandwich norms; (2) **3D MRoPE with non-equal interleaved sections** and a large image-position offset; (3) **multi-layer Qwen3-VL hidden-state capture** (13 layers — the `LlamaStyleEncoder` family already supports Qwen, needs the multi-layer tap that Lens also needs); (4) the **fixed-constant latent normalization** (replaces the BN un-normalize used by Flux.2/Lens); (5) **asymmetric CFG** with a per-step guidance schedule; (6) the **structured-JSON prompt path**. The VAE, flow-match Euler scheduler core, RMSNorm, SwiGLU, and Qwen tokenizer are all reusable.

## Key Numbers / Constants

| Constant | Value | Source |
|---|---|---|
| Parameters (DiT) | 9.3 B | README |
| Layers | 34 | `modeling_ideogram4.py` |
| Hidden (inner) dim | 4608 | model_architecture.md |
| Attention heads | 18 | " |
| Head dim | 256 | 4608/18 |
| MLP hidden (SwiGLU) | 12288 | " |
| AdaLN dim | 512 | `adaln_proj` |
| In/out channels (transformer) | 128 / 128 | `input_proj` / `final_layer.linear` |
| VAE latent channels | 32 | Flux.2 VAE |
| Patch (pipeline) | 2×2 | 32 × 4 = 128 |
| VAE spatial downsample | 8× | autoencoder |
| RoPE θ (base) | 5_000_000 | `Ideogram4MRoPE` |
| MRoPE sections | (24, 20, 20) | " |
| Text encoder | Qwen3-VL-8B-Instruct (lang tower) | pipeline |
| TE hidden | 4096 | Qwen config |
| TE layers (total / used) | 36 / 13 | Qwen config / `QWEN3_VL_ACTIVATION_LAYERS` |
| TE tapped layers | (0,3,6,9,12,15,18,21,24,27,30,33,35) | `constants.py` |
| Text feature concat dim | 53248 | 4096 × 13 |
| Max text tokens | 2048 | pipeline |
| Per-block modulation outputs | 4 × 4608 = 18432 | `adaln_modulation` |
| Final modulation | 4608 (scale only) | `final_layer.adaln_modulation` |
| Latent norm channels | 128 | `latent_norm.py` |
| LATENT_SHIFT range | ≈ −0.35 .. +0.38 | `latent_norm.py` (copy verbatim) |
| LATENT_SCALE range | ≈ +1.53 .. +1.94 | `latent_norm.py` (copy verbatim) |
| Scheduler logsnr_max / min | 18.0 / −15.0 | `scheduler.py` |
| Schedule known resolution | (512, 512) | `scheduler.py` |
| Default preset | V4_QUALITY_48 (48 steps) | `inference.md` |
| Guidance main / polish | 7.0 / 3.0 | `sampler_configs.py` |
| SEQUENCE_PADDING_INDICATOR | −1 | `constants.py` |
| OUTPUT_IMAGE_INDICATOR | 2 | `constants.py` |
| LLM_TOKEN_INDICATOR | 3 | `constants.py` |
| IMAGE_POSITION_OFFSET | 65536 | `constants.py` |
| Resolution range | 256–2048, ×16 | README |
| Aspect ratio max | 6:1 | README |

## Data Layouts / Formats

### Tensor shapes through the pipeline (1024×1024 example)

```
prompt → chat template → tokenize → input_ids                 [B, L_text ≤ 2048]
↓ Qwen3-VL-8B language tower, tap 13 layers
13 × [B, L_text, 4096]  → concat                              [B, L_text, 53248]
↓ (inside DiT) llm_cond_norm + llm_cond_proj
text features                                                 [B, L_text, 4608]

noise (pure Gaussian at t=1)                                  [B, grid_h·grid_w, 128]   grid = 64×64
↓ input_proj
image tokens                                                 [B, 4096, 4608]
+ embed_image_indicator                                       (image tag added)

unified sequence  [text tokens | image tokens]               [B, L_text + 4096, 4608]
+ 34× single-stream blocks (3D MRoPE, block-diag mask)
↓ final_layer (slice image tokens only)
velocity                                                      [B, 4096, 128]

z₀ (after Euler loop)                                         [B, 4096, 128]
↓ z·latent_scale + latent_shift  → reshape/unpatchify
                                                             [B, 32, 128, 128]   (H/8 × W/8)
↓ Flux.2 VAE decode
RGB                                                          [B, 3, 1024, 1024]  [-1,1]
```

### ComfyUI checkpoint layout (`Comfy-Org/Ideogram-4`, total ~46.8 GB)

```
diffusion_models/ideogram4_fp8_scaled.safetensors              ~13.8 GB  FP8 scaled (per-tensor scale_weight)
diffusion_models/ideogram4_unconditional_*.safetensors         (verify: same weights as conditional? — see Open Q)
diffusion_models/ideogram4_*_nvfp4_mixed.safetensors           NVFP4 (reuse Nvfp4Codec)
text_encoders/qwen3vl_8b_fp8_scaled.safetensors                ~8 GB    Qwen3-VL-8B language tower
text_encoders/qwen3vl_8b_nvfp4.safetensors                     NVFP4 variant
text_encoders/gemma4_e4b_it_fp8_scaled.safetensors             ~2 GB    ⚠ MAGIC-PROMPT LLM, NOT a conditioning encoder
vae/flux2-vae.safetensors                                      ~336 MB  Flux.2 VAE (reused)
```

## Reference Implementations

- **`ideogram-oss/ideogram4 — modeling_ideogram4.py`** — `Ideogram4Transformer`, `Ideogram4TransformerBlock`, `Ideogram4Attention`, `Ideogram4MLP`, `Ideogram4MRoPE`, `Ideogram4EmbedScalar`, `Ideogram4FinalLayer`. **Primary reference for the C# transformer.**
- **`pipeline_ideogram4.py`** — `Ideogram4Pipeline`: Qwen tap-and-concat, asymmetric CFG, Euler loop, latent norm, unpatchify. **Primary reference for the C# pipeline.**
- **`scheduler.py`** — `LogitNormalSchedule`, `get_schedule_for_resolution`, `make_step_intervals`. **Primary reference for the new scheduler helper.**
- **`sampler_configs.py`** — the three presets + guidance schedules.
- **`latent_norm.py`** — the 128 `LATENT_SHIFT` / `LATENT_SCALE` constants (copy verbatim).
- **`constants.py`** — indicator/offset constants + the 13 tap layers.
- **`autoencoder.py`** + `flux2-vae.safetensors` — Flux.2 VAE; already implemented (`Flux2CheckpointConverter`, `VaeDecoder`).
- **`magic_prompt.py` + `magic_prompt_system_prompts/`** — the LLM JSON-expander system prompts. Relevant to [STRUCTURED_PROMPT_BUILDER.md](STRUCTURED_PROMPT_BUILDER.md), not the inference core.
- **`Qwen/Qwen3-VL-8B-Instruct` `config.json`** — language-tower dims. The `LlamaStyleEncoder` family already supports Qwen3.
- **diffusers `AutoencoderKLFlux2`** — already in tree via Flux.2.

## Differences Between Implementations

The reference is single-source (`ideogram-oss/ideogram4`). The divergences worth pinning are vs **ComfyUI** (which the user flagged as having mistakes) and vs **other DiTs in this codebase**:

1. **Gemma-4 is NOT a conditioning text encoder.** ComfyUI bundles `gemma4_e4b_it_fp8_scaled.safetensors` (~2 GB) as a **local LLM to generate the structured-JSON prompt** (its "magic prompt"). The actual model conditioning comes from **Qwen3-VL only**. Treating Gemma-4 as a second text encoder would be wrong. The official repo uses an API/Claude/Ideogram-hosted LLM for the same job. **In HartsyInference, the JSON builder is a separate utility (the prompt builder), with an optional pluggable LLM expander — not a model component.**
2. **The "unconditional" checkpoint is (almost certainly) not a separate network.** Official CFG runs one transformer twice. See § Open Questions — verify by hashing.
3. **Asymmetric CFG, not symmetric.** Unlike Lens/SD3.5 (duplicate one tensor batch-of-2, identical shapes), Ideogram's negative pass is a *shorter, image-only* sequence with zeroed text. Two forwards of different length.
4. **Scale-only AdaLN (no shift), tanh gates, sandwich norms.** Every other DiT here has shift terms and a single pre-norm. Do not copy `FluxDoubleStreamBlock` / `JointBlock` modulation wholesale.
5. **Logit-normal schedule, not the SD3 time-shift.** The existing `FlowMatchEulerDiscreteScheduler` is the wrong reuse target; write a small `LogitNormalSchedule`.
6. **Fixed-constant latent norm, not BN.** Flux.2/Lens decode applies VAE `bn.running_mean/var`; Ideogram applies its own 128 constants. Confirm the Flux.2 VAE's internal BN is bypassed (see Open Q).
7. **3D MRoPE with non-equal sections `(24,20,20)` and a 65536 image offset** — a third RoPE flavor distinct from Flux pair-rotation and Lens/Qwen complex-polar. The interleave slicing is bespoke.
8. **head_dim = 256** is large; the existing SDPA kernels must handle 256-wide heads (verify the `sdpa_f32.ptx` shared-memory tiling at 256, or fall back to the tiled path).

## Implementation Notes (recommendations for HartsyInference)

### What can be reused

- **`VaeDecoder` (Flux.2 preset) + `Flux2CheckpointConverter`** — the VAE is identical; only swap BN un-norm for the constant latent-norm.
- **`LlamaStyleEncoder` (Qwen3 preset)** — already supports Qwen3. Needs the **multi-layer hidden-state tap** (13 layers) — the **same net-new capability Lens needs** (Lens taps GPT-OSS layers [5,11,17,23]). Build one shared `EncodeTapLayers(int[] layerIndices)` capability and both models use it. **Do NOT duplicate.** (See [MICROSOFT_LENS_ARCHITECTURE.md](MICROSOFT_LENS_ARCHITECTURE.md) § multi-layer capture and the AGENTS.md reuse rule.)
- **Fused-QKV split helper** — existing (Flux/SD3 converters split `qkv`).
- **SwiGLU `w1/w2/w3`** — existing `SwiGluFfn` (bias=False). Note Ideogram's `w2(silu(w1)·w3)` matches.
- **RMSNorm (learned scale)** — existing primitive.
- **Euler step** (`z += v·Δ`) — trivial; the scheduler warp is the only new math.
- **FP8 `fp8_scaled` scale-companion folding** — existing (`Tensor.Fp8ScaleFactor`).
- **`Nvfp4Codec`** — added for Lens; reuse for the nvfp4 variants.
- **`DeterministicRng`** — seeded Gaussian noise.
- **`Qwen3Tokenizer` / `Microsoft.ML.Tokenizers.BpeTokenizer`** — Qwen3 chat template already in tree.

### What's net-new

| Component | Effort | Why |
|---|---|---|
| **`Ideogram4Config.cs`** | Low | One preset; all dims above. |
| **`Ideogram4Transformer.cs` + `Ideogram4Block.cs`** | Medium (~4-6 days) | Single-stream unified-sequence; scale-only tanh-gated AdaLN + sandwich norms; new block, not a Flux clone. |
| **`Ideogram4Mrope.cs`** | Medium (~2-3 days) | 3D sectioned MRoPE `(24,20,20)`, θ=5e6, 65536 offset, bespoke interleave. Validate vs Python dump. |
| **Multi-layer Qwen tap** | Medium (~2-3 days) | Shared with Lens. Add `EncodeTapLayers` to the encoder family. |
| **`LogitNormalSchedule.cs`** | Low (~1 day) | `ndtri`/`erfinv` + `expit` + resolution-mean adjust + clamp. New scheduler. |
| **`Ideogram4Pipeline.cs`** | Medium (~3 days) | Asymmetric CFG (two-length forwards), two-stage guidance, constant latent-norm, Flux.2 decode. |
| **`Ideogram4CheckpointConverter.cs`** | Low (~1-2 days) | Diffusers-naming passthrough + fused-QKV split + fp8_scaled / nvfp4 folding; route VAE to Flux.2 converter. |
| **Latent-norm constants** | Low | Copy 128 `LATENT_SHIFT`/`LATENT_SCALE` floats verbatim. |
| **`Ideogram4DebugDump.cs` + `dump_ideogram4_full_forward.py` + `diff_ideogram4_layers.py`** | Medium (~2-3 days) | Layer-by-layer diff harness (SD3.5/Lens template). MRoPE + asymmetric CFG are the likely first-run bug hotspots. |
| **Structured-prompt builder** | — | Separate workstream, see [STRUCTURED_PROMPT_BUILDER.md](STRUCTURED_PROMPT_BUILDER.md). |

### VRAM budget (12 GB target, RTX 3060)

- DiT fp8 (9.3 B) ≈ **9.5 GB** → tight. fp8_scaled is the realistic path; activations at 1024² (≤2048 text + 4096 image tokens × 4608 dim) are non-trivial.
- Qwen3-VL-8B language tower fp8 ≈ **8 GB** (encode once, then `FreeWeights` before the DiT).
- Flux.2 VAE ≈ **~5 GB** (load after `FreeWeights(transformer)`).
- **Eviction plan (same as Lens/SD3.5/Qwen-Image):** encode → free Qwen → run DiT → free DiT → VAE decode. 12 GB is feasible at fp8 with discipline; 8 GB likely needs nvfp4 + tiled VAE decode.

### Suggested implementation order

1. **Multi-layer Qwen tap** (shared with Lens) — unblocks both.
2. **`Ideogram4Config` + `Ideogram4Mrope` + `Ideogram4Block` + `Ideogram4Transformer`** — port block-by-block; validate MRoPE against a dump early.
3. **`LogitNormalSchedule`** + **latent-norm constants**.
4. **`Ideogram4CheckpointConverter`** (diffusers passthrough + fp8_scaled/nvfp4).
5. **`Ideogram4Pipeline`** (asymmetric CFG, two-stage guidance).
6. **Validation harness** — expect 1-3 first-run bug iterations (MRoPE interleave, asymmetric-CFG seq lengths, latent-norm vs BN are the suspects).
7. **SwarmUI extension wiring** — register the new arch loader + Qwen side-model in the SwarmUI-HartsyInference extension (see Phase 4 checklist + the [[swarmui_extension]] memory).

### What NOT to do

- **Don't treat Gemma-4 as a text encoder.** It's the optional JSON-prompt LLM. Conditioning = Qwen3-VL only.
- **Don't reuse `FlowMatchEulerDiscreteScheduler`** — Ideogram's schedule is logit-normal, not SD3 time-shift.
- **Don't reuse the Flux.2/Lens BN latent un-normalize** — Ideogram uses fixed constants.
- **Don't assume symmetric batch-of-2 CFG** — the negative pass is a shorter image-only sequence.
- **Don't add a shift term to the AdaLN modulation** — Ideogram modulation is scale-only.
- **Don't implement two transformers for the "unconditional" file** until you've confirmed the weights actually differ.
- **Don't pull in bitsandbytes/NF4** — fp8 / fp8_scaled / nvfp4 cover the weights with our existing codecs.
</content>
</invoke>
