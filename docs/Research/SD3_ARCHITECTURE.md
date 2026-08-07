# SD3 Architecture — Research Notes

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Stable Diffusion 3 (SD3) uses a **Multi-Modal Diffusion Transformer (MMDiT)** architecture that jointly processes image and text tokens through transformer blocks with shared attention. It replaces the U-Net used in SD1.5/SDXL with a pure transformer operating on latent patches. Three text encoders (CLIP-L/14, OpenCLIP bigG/14, T5-v1.1-XXL) provide conditioning. The model uses **rectified flow matching** instead of DDPM noise schedules, with a logit-normal timestep distribution during training. QK-norm via RMSNorm stabilizes attention at scale. The architecture is parameterized by a single depth value `d` from which hidden size and head count are derived (`hidden_size = 64 * d`, `num_heads = d`).

SD3 shares the flow-matching paradigm with Flux but differs significantly in block structure: SD3 uses symmetric "joint blocks" where both modalities have independent weights but share a single concatenated attention operation, whereas Flux uses double-stream blocks (separate attention per modality with cross-attention) followed by single-stream blocks (concatenated sequence with shared weights).

---

## Key Numbers/Constants

### SD3 Medium (2B parameters)
| Parameter | Value |
|-----------|-------|
| Depth (num_layers) | 24 |
| Hidden size | 1536 (= 64 * 24) |
| Attention heads | 24 (= depth) |
| Head dimension | 64 |
| MLP hidden dim | 6144 (= 4 * 1536, with SwiGLU) |
| Patch size | 2 |
| Input/output channels | 16 (VAE latent channels) |
| Sample size (latent) | 128 (for 1024px images) |
| Pos embed max size | 192 |
| Joint attention dim | 4096 (context embedder input) |
| Caption projection dim | 1536 (= hidden_size) |
| Pooled projection dim | 2048 (= 768 + 1280) |
| ADM in channels | 2048 |
| Context sequence length | 154 (= 77 CLIP + 77 T5) |
| Latent scale factor | 1.5305 |
| Latent shift factor | 0.0609 |
| Flow matching shift | 3.0 |
| QK-norm | RMSNorm (learnable scale), eps=1e-6 |
| MLP activation | SwiGLU (SiLU gating) |
| AdaLN activation | SiLU |

### SD3.5 Large (8B parameters)
| Parameter | Value |
|-----------|-------|
| Depth (num_layers) | 38 |
| Hidden size | 2432 (= 64 * 38) |
| Attention heads | 38 |
| Head dimension | 64 |
| Architecture variant | MMDiT-X (dual attention) |
| Dual attention layers | First ~13 layers |
| QK-norm | RMSNorm (confirmed) |

### SD3.5 Medium (2.5B parameters)
| Parameter | Value |
|-----------|-------|
| Architecture variant | MMDiT-X (dual attention) |
| Dual attention layers | First ~12 layers |
| QK-norm | RMSNorm |

### Text Encoder Dimensions
| Encoder | Hidden | Layers | Heads | Vocab | Context Len |
|---------|--------|--------|-------|-------|-------------|
| CLIP-L/14 | 768 | 12 | 12 | 49408 | 77 |
| CLIP-G/14 | 1280 | 32 | 20 | 49408 | 77 |
| T5-v1.1-XXL | 4096 | 24 | 64 | 32128 | 77-256 |

### VAE (AutoEncoder)
| Parameter | Value |
|-----------|-------|
| Latent channels | 16 |
| Downsampling factor | 8 |
| Base channels | 128 |
| Channel multipliers | (1, 2, 4, 4) |
| Resolution blocks per level | 2 |

### Scaling Study Model Sizes (from paper)
| Depth | Params (approx) |
|-------|-----------------|
| 15 | 450M |
| ~24 | ~2B |
| 38 | 8B |

---

## Data Layouts/Formats

### Input Latent
- Shape: `[B, 16, H/8, W/8]` (e.g., `[B, 16, 128, 128]` for 1024x1024)
- After patch embedding: `[B, (H/8/2)*(W/8/2), hidden_size]` = `[B, 4096, 1536]` for 1024x1024

### Context (Text Conditioning)
- Combined context: `[B, 154, 4096]` (77 CLIP tokens + 77 T5 tokens, each 4096-dim after padding)
- After context_embedder projection: `[B, 154, 1536]`

### Pooled Projection
- Shape: `[B, 2048]` (concatenated CLIP-L + CLIP-G pooled outputs)

### Timestep
- Scalar per batch element, range [0, 1] in flow matching (or [0, 1000] in discrete steps)

### Joint Attention Sequence
- Concatenated: `[B, num_image_tokens + num_text_tokens, hidden_size]`
- For 1024x1024: `[B, 4096 + 154, 1536]` = `[B, 4250, 1536]`

---

## Reference Implementations

| Source | URL | Notes |
|--------|-----|-------|
| Stability AI SD3 reference | [github.com/Stability-AI/sd3-ref](https://github.com/Stability-AI/sd3-ref) | Original MMDiT implementation (`mmdit.py`, `sd3_impls.py`, `other_impls.py`) |
| Stability AI SD3.5 | [github.com/Stability-AI/sd3.5](https://github.com/Stability-AI/sd3.5) | MMDiT-X variant (`mmditx.py`) with dual attention |
| HuggingFace Diffusers | [SD3Transformer2DModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_sd3.py) | Production implementation with JointTransformerBlock |
| HuggingFace Diffusers (attention) | [attention.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py) | JointTransformerBlock class |
| FlowMatchEulerDiscreteScheduler | [scheduling_flow_match_euler_discrete.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py) | Flow matching Euler scheduler |
| SD3 Paper | [arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206) | "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" |
| SD3 Paper PDF | [Stability AI S3](https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf) | Direct PDF link |
| Lucidrains MMDiT | [github.com/lucidrains/mmdit](https://github.com/lucidrains/mmdit) | Minimal single-layer MMDiT implementation |
| HuggingFace SD3 blog | [huggingface.co/blog/sd3](https://huggingface.co/blog/sd3) | Diffusers integration overview |
| Sayak Paul attention analysis | [sayak.dev/posts/attn-diffusion.html](https://sayak.dev/posts/attn-diffusion.html) | Comparison of attention flavors across diffusion models |
| DeepWiki SD3 text encoders | [deepwiki.com/Stability-AI/sd3-ref/4.4-text-encoders](https://deepwiki.com/Stability-AI/sd3-ref/4.4-text-encoders) | Text encoder combination details |
| SD3 HuggingFace model card | [huggingface.co/stabilityai/stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) | Model card and weights |

---

## Differences Between Implementations

### SD3 MMDiT vs Flux DiT

| Aspect | SD3 (MMDiT) | Flux (DiT) |
|--------|-------------|------------|
| **Block type** | All JointBlocks (symmetric dual-stream) | Double-stream blocks (19) + single-stream blocks (38) |
| **Attention** | Concatenate Q/K/V from both modalities, single attention op | Double-stream: separate self-attn + cross-attn; Single-stream: concat into one sequence |
| **Context handling** | Context contributes to every block's attention, final block is context_pre_only | Context fully participates in double-stream blocks, then merged for single-stream |
| **Positional encoding** | 2D sinusoidal (additive) | RoPE (Rotary Position Embeddings) for both 2D image and text |
| **QK-norm** | RMSNorm (optional in SD3, standard in SD3.5) | RMSNorm (always enabled) |
| **MLP type** | SwiGLU | GELU (approximate=tanh) in some blocks |
| **Text encoders** | CLIP-L + CLIP-G + T5-XXL | CLIP-L + T5-XXL (no CLIP-G) |
| **Pooled conditioning** | CLIP-L + CLIP-G pooled (2048-dim) | CLIP-L pooled (768-dim) |
| **Flow matching shift** | shift=3.0 (static) | Dynamic shifting based on image resolution |
| **Guidance** | Standard CFG (two forward passes) | Guidance-distilled (single pass with guidance embedding) for Schnell |
| **Depth formula** | hidden_size = 64 * depth, heads = depth | Fixed: hidden_size=3072, heads=24 (for Flux-dev/schnell) |
| **Total params** | ~2B (Medium) / ~8B (Large) | ~12B |

### SD3 (Stability AI ref) vs Diffusers Implementation

| Aspect | Stability AI Reference | HuggingFace Diffusers |
|--------|----------------------|----------------------|
| **Class name** | `MMDiT` | `SD3Transformer2DModel` |
| **Block class** | `JointBlock` (wraps `DismantledBlock`) | `JointTransformerBlock` |
| **QK-norm param** | `qk_norm="rms"` or `"ln"` | `qk_norm="rms_norm"` or `"layer_norm"` |
| **Config format** | Derived from weight shapes at load time | Explicit `config.json` |
| **Depth derivation** | `depth = x_embedder.proj.weight.shape[0] // 64` | Explicit `num_layers` parameter |
| **MLP** | `SwiGLUFeedForward` class | Standard `FeedForward` with gelu-approximate |
| **Modulation** | `modulate()` function + manual shift/scale | `AdaLayerNormZero` module |

### SD3 vs SD3.5 (MMDiT vs MMDiT-X)

| Aspect | SD3 (MMDiT) | SD3.5 (MMDiT-X) |
|--------|-------------|------------------|
| **Dual attention** | No | Yes (first ~12-13 layers) |
| **QK-norm** | Optional/absent in released checkpoint | Always enabled (RMSNorm) |
| **AdaLN** | Standard AdaLN-Zero (6 params: shift, scale, gate for norm1 and MLP) | Extended AdaLN with extra modulation params for dual attention |
| **Depth** | 24 (Medium, 2B) | 24 (Medium, 2.5B) / 38 (Large, 8B) |
| **Extra params** | N/A | Second self-attention module per dual-attention layer |

---

## Implementation Notes

### For HartsyInference

1. **Code reuse with Flux**: The joint attention mechanism is fundamentally different from Flux's double-stream/single-stream split. However, the following can be shared:
   - Flow matching scheduler (same `FlowMatchEulerDiscreteScheduler`, different shift values)
   - VAE decoder (same 16-channel architecture)
   - T5-XXL text encoder
   - CLIP-L text encoder
   - Basic transformer infrastructure (attention, norms, MLPs)
   - Patch embedding / unpatchify operations

2. **SD3-specific components** that need dedicated implementation:
   - `JointBlock` with symmetric dual-stream attention (different from Flux's asymmetric blocks)
   - Text encoder combination logic (three encoders -> context + pooled)
   - AdaLN-Zero modulation with the specific parameter count
   - Context embedder (Linear projection from 4096 -> hidden_size)
   - 2D sinusoidal positional embeddings (vs RoPE in Flux)
   - QK-norm as optional per-model feature

3. **Memory considerations**:
   - T5-XXL alone is ~10GB in fp16; can be dropped for lighter inference
   - CLIP-L: ~0.5GB, CLIP-G: ~1.5GB
   - SD3 Medium transformer: ~4GB in fp16
   - Total with all encoders: ~16GB in fp16
   - Quantization (fp8 for T5, fp8/int8 for transformer) reduces this significantly

4. **Weight loading**: The Stability AI reference derives all architecture parameters from weight tensor shapes at load time. Key formula: `depth = x_embedder.proj.weight.shape[0] // 64`. This means HartsyInference can auto-detect model configuration from safetensors metadata without requiring a separate config file.

5. **Latent scaling**: After VAE decode, apply: `latent_for_decode = (x / 1.5305) + 0.0609`. This is different from SD1.5/SDXL which use a simple scale factor.

6. **Scheduler defaults**: Use `shift=3.0` for SD3 Medium, `num_inference_steps=28` is a common default. The scheduler computes sigmas as: `sigma = 3.0 * t / (1 + 2.0 * t)` where t is linearly spaced.
