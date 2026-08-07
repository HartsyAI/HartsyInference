# Microsoft Lens Architecture — Research Notes

> **Status:** Complete (read-only from upstream code; no checkpoint inspected on disk yet) | **Last Updated:** 2026-05-27 | **Needed Before:** `LensTransformer`, `LensGptOssEncoder`, `LensPipeline` implementation
>
> **Sources of truth:**
> - GitHub: [microsoft/Lens](https://github.com/microsoft/Lens) — `lens/transformer.py` (LensTransformer2DModel, ~700 lines), `lens/pipeline.py` (LensPipeline, ~580 lines), `lens/text_encoder.py` (LensGptOssEncoder, ~130 lines), `lens/resolution.py`, `lens/reasoner.py`
> - HuggingFace: [microsoft/Lens](https://huggingface.co/microsoft/Lens) (RL-tuned, 30.7 GB), [microsoft/Lens-Turbo](https://huggingface.co/microsoft/Lens-Turbo) (4-step distilled), `microsoft/Lens-Base` (supervised baseline)
> - `transformer/config.json` and `text_encoder/config.json` extracted verbatim below
>
> **License:** MIT (for both DiT weights and the inference code). The GPT-OSS text encoder it depends on is licensed separately as Apache-2.0 by OpenAI under `openai/gpt-oss-20b` upstream — Microsoft re-publishes a Lens-trimmed copy alongside Lens.

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Lens is a **3.8B-parameter dual-stream MMDiT** image generator from Microsoft Research. Released 2026-05-25. Architecturally similar to Flux's double-stream block, but at one-fifth the parameter count: 48 layers × hidden=1536 × 24 heads × 64 head-dim, with **3-axis complex-polar RoPE** (frame=8, h=28, w=28; total 64 = head_dim) and a **SwiGLU MLP** (hidden=4096). The headline trick is the text encoder: it does **not** use T5, CLIP, or any diffusion-tuned LLM — instead it concatenates four layer hidden states (layers 5, 11, 17, 23) from a frozen **GPT-OSS** MoE causal LM, normalizes each layer with its own RMSNorm, and projects the concat (4×2880 = 11520) down to 1536. This is what Microsoft means by "massive text encoder training on GPT image outputs" — quality scales because the conditioning signal is much richer than a single T5/CLIP pool, while inference stays cheap because the DiT itself is small.

VAE is the same **Flux.2 semantic VAE** (`AutoencoderKLFlux2`) already implemented for Flux.2 Klein in this codebase: 16× spatial downsample × 4× channel patchify = 32-channel latent → 128-channel transformer input. Scheduler is the standard `FlowMatchEulerDiscreteScheduler` with an **empirical mu** computed per-resolution. CFG is a dual-pass batch-of-2 with **norm-rescaling** (Microsoft's twist — combined prediction is rescaled to match the conditional branch's L2 norm per token). Default sampling is **20 steps, CFG 5.0** for the RL-tuned variant; **4 steps, CFG 1.0** for Turbo; **50 steps, CFG 5.0** for Base.

For HartsyInference the genuinely new piece is **GPT-OSS as a text encoder** — Mixture-of-Experts (32 local experts, 4 active per token, MXFP4-native), GQA (64 query heads : 8 KV heads), alternating sliding/full attention (window 128) — none of which the existing `LlamaStyleEncoder` supports. The transformer block itself is close enough to Flux's double-stream that ~70% of the block code can be reused (modulation, fused QKV, joint attention, AdaLN-Continuous final). RoPE is its own thing (complex-polar with scale_rope=True centered around zero), but mathematically identical to Qwen-Image's 3-axis RoPE just with different per-axis dims.

## Key Numbers / Constants

| Constant | Value | Source |
|---|---|---|
| Parameters (DiT) | 3.8 B | README |
| Layers | 48 | `num_layers` |
| Hidden (inner) | 1536 | `inner_dim` |
| Attention heads | 24 | `num_attention_heads` |
| Head dim | 64 | `attention_head_dim` |
| MLP hidden (SwiGLU) | 4096 | `int(1536 / 3 * 8)` |
| Patch size | 2 | `patch_size` |
| In channels (transformer) | 128 | `in_channels` (= 4× of 32 after pipeline patchify) |
| Out channels (transformer) | 32 | `out_channels` |
| Text encoder hidden | 2880 | `enc_hidden_dim`; matches `openai/gpt-oss-20b` |
| Text encoder layers used | [5, 11, 17, 23] | `selected_layer_index`; 0-indexed |
| Text input projection dim | 4 × 2880 = 11520 | concat width |
| Modulation outputs per stream | 6 × 1536 = 9216 | `Linear(SiLU(temb))` width |
| RoPE θ | 10000 | `LensEmbedRope.__init__` |
| RoPE per-axis dims | (8, 28, 28) | `axes_dims_rope` |
| RoPE scale_rope | True | `LensTransformer2DModel.__init__` (hardcoded, not a config field) |
| Max position-grid pre-compute | 4096 | `pos_index = arange(4096)` |
| Scheduler | FlowMatchEulerDiscreteScheduler | `model_index.json` |
| Empirical mu — low-seq slope (a1) | 8.73809524e-5 | `compute_empirical_mu` |
| Empirical mu — low-seq intercept (b1) | 1.89833333 | " |
| Empirical mu — high-seq slope (a2) | 0.00016927 | " |
| Empirical mu — high-seq intercept (b2) | 0.45666666 | " |
| Sigmas | `linspace(1.0, 1.0/N, N)` | pipeline |
| Default CFG (Lens / Lens-Base) | 5.0 | README, pipeline default 4.0 (overridden in CLI) |
| Default steps (Lens / Lens-Base) | 20 / 50 | README |
| Default steps (Lens-Turbo) | 4 | README |
| Default CFG (Lens-Turbo) | 1.0 | README |
| Tokenizer pad token | EOS (`tokenizer.eos_token`) | pipeline |
| Padding side | right | pipeline |
| Max sequence length (text) | 512 | pipeline default |
| `txt_offset` (system-prompt strip) | 97 | `DEFAULT_TXT_OFFSET` |
| VAE downsample × patchify | 16 × (2×2) = 16× spatial, 4× channels | `vae_scale_factor=16`, pipeline patchify |
| Default sample size | 1024 | pipeline (used when no base_resolution / aspect_ratio) |
| Total repo size | 30.7 GB | HF page |

### GPT-OSS encoder

| Constant | Value |
|---|---|
| Hidden size | 2880 |
| Layers (model has) | 24 |
| Layers (Lens uses) | up to 23 (last selected layer) |
| Heads (Q) | 64 |
| Heads (KV) | 8 (GQA 8:1) |
| Intermediate (per expert) | 2880 |
| Vocab | 201,088 |
| Activation | SiLU (SwiGLU FFN per expert) |
| Local experts | 32 |
| Active experts per token | 4 |
| Sliding window | 128 tokens |
| Attention pattern | alternating sliding/full per layer |
| Native dtype | MXFP4 (4-bit packed with 32-element block scales) |

## Data Layouts / Formats

### Latent tensor shapes through the pipeline

```
RGB image                          [B, 3, H, W]                   uint8 [0, 255]
↓ VAE encode (decode is symmetric — Lens is t2i so we only decode)
VAE latent                         [B, 32, H/16, W/16]            F32/BF16
↓ pipeline 2×2 patchify (channel)
Packed latent (transformer input)  [B, (H/16)·(W/16), 128]        BF16
                                   = (B, S_img, in_channels)

img_in projection                  [B, S_img, 1536]
+ 48× blocks (joint with txt)      [B, S_img, 1536]   stays put; encoder grows/shrinks alongside
proj_out                           [B, S_img, 128]
↓ pipeline rearrange + unpatchify + BN un-normalize
                                   [B, 32, H/16, W/16]            BF16
↓ VAE decode
RGB image                          [B, 3, H, W]                   F32 [-1, 1]
```

### Text feature shapes

```
prompt → chat-templated text → tokenize → input_ids [B, S_padded]
↓ GPT-OSS forward, capture layers [5,11,17,23]
List[4] of [B, S_padded, 2880]
↓ strip first 97 tokens (system + chat template wrapper)
List[4] of [B, S_txt, 2880]      where S_txt = S_padded - 97
↓ per-layer RMSNorm + channel-concat + Linear
[B, S_txt, 1536]   (now joins the image stream in each block)
```

### Safetensors file layout (microsoft/Lens)

```
transformer/diffusion_pytorch_model.safetensors      ~7.6 GB BF16   (all LensTransformer2DModel keys)
text_encoder/model.safetensors                       ~12-13 GB MXFP4 packed (LensGptOssEncoder weights — full GPT-OSS layers)
text_encoder/model.safetensors.index.json            (multi-shard pointer if sharded)
vae/diffusion_pytorch_model.safetensors              ~5-10 GB FP16   (AutoencoderKLFlux2; same file as Flux.2 ships)
tokenizer/tokenizer.json + tokenizer_config.json     GPT-OSS BPE
scheduler/scheduler_config.json                      FlowMatchEulerDiscreteScheduler config
model_index.json                                     pipeline manifest (415 B)
```

### ComfyUI distribution (`Comfy-Org/Lens`) — actual checkpoints in use

The diffusers `microsoft/Lens` repo (above) is the reference, but the **checkpoints we actually load** are the ComfyUI-repackaged ones at [`huggingface.co/Comfy-Org/Lens`](https://huggingface.co/Comfy-Org/Lens) (MIT, ungated). These differ from diffusers in three load-bearing ways, all handled by `LensCheckpointConverter.ConvertComfy*` + the `Mxfp8Codec` / `Nvfp4Codec`:

| File | Size | Format | Notes |
|---|---|---|---|
| `diffusion_models/lens_bf16.safetensors` / `lens_turbo_bf16` | 8.2 GB | plain **BF16** | diffusers-native key names, **no `transformer.` prefix** (`transformer_blocks.{i}.attn.img_qkv`, `img_mlp.w1/w2/w3`, `norm_out.linear`, `proj_out`, `time_text_embed.timestep_embedder`). Loads through the existing converter passthrough + fused-QKV split. |
| `diffusion_models/lens_mxfp8.safetensors` / `lens_turbo_mxfp8` | 5.5 GB | **MXFP8** (`mxfp8_block32`) | per-Linear: `{name}.weight` F8E4M3 `[out,in]` + `{name}.weight_scale` U8 (E8M0, group 32 along in, **swizzled**) + `{name}.comfy_quant` JSON blob. Dequant `w = decode_e4m3(weight)·2^(scale-127)` → BF16. No transpose. |
| `text_encoders/gpt_oss_20b_nvfp4.safetensors` | 13.2 GB | **NVFP4** (`nvfp4`) | GPT-OSS-20B, **no `model.` prefix** (`layers.{i}.…`, `embed_tokens.weight`, `norm.weight`, plus an embedded `tokenizer_json` blob). MoE experts only are quantized: `experts.{gate_up,down}_proj.weight` U8 (FP4 E2M1, **high nibble = even elem**) + `.weight_scale` F8E4M3 (group 16, **swizzled**) + `.weight_scale_2` F32 per-expert global + `.comfy_quant`. Attn/router/embed/norm stay BF16; biases use `gate_up_proj.bias` (renamed to HF `gate_up_proj_bias`). Dequant `w = e2m1(nibble)·global·decode_e4m3(block_scale)`, then transpose `[E,out,in]→[E,in,out]` to the runtime layout. |
| `vae/flux2-vae.safetensors` | 336 MB | FP16/BF16 | the Flux.2 semantic VAE, reused as-is. |

**Swizzled block scales.** Both MXFP8 and NVFP4 store their per-block scale tensors in NVIDIA's cuBLAS "blocked" layout (ComfyUI's `comfy.float.to_blocked`): the logical `[out, in/group]` scale matrix is zero-padded to `[128·ceil(out/128), 4·ceil((in/group)/4)]` and permuted. `BlockScaleSwizzle.SwizzledIndex(row, blockCol, paddedCols)` inverts the permutation (verified by an exact swizzle round-trip against `to_blocked`). This is why NVFP4 `down_proj.weight_scale` shows the padded `[32, 2944, 180]` shape (out 2880 → 2944).

**MXFP4 vs MXFP8/NVFP4.** The earlier `Mxfp4Codec` (E8M0 group-32 FP4, no global) matches the **diffusers** `microsoft/Lens` text encoder. The ComfyUI repo uses MXFP8 (DiT) + NVFP4 (TE) instead. All three codecs coexist.

**Memory caveat (≤12 GB target).** Dequant-at-load of the NVFP4 20B encoder to F32 needs ~76 GB of host RAM (experts dominate) — this OOM-killed a 62 GB host on 2026-07-16, so `ConvertComfyTextEncoder` now passes the NVFP4 expert banks through PACKED (mmap-backed) and `GptOssMoeFfn` dequantizes one expert at a time transiently during the forward (`Nvfp4Codec.DequantExpertSlice` into ~100 MB slices; tokens are bucketed by routed expert so each bank is dequantized once per layer forward and evaluated as two `backend.Linear` GEMMs). **Backend concurrency contract:** on CPU, experts run in parallel with per-worker reusable slices; on CUDA they run strictly sequentially with a fresh slice tensor per expert + `Sync`/`FreeWeights` before disposal — the CUDA backend's stream, reference-keyed caches, lazy D2H sync, and upload auto-promotion are not safe against concurrent calls or reused-and-mutated tensors (the parallel shape corrupted the native heap in the first SwarmUI deploy: `malloc(): unsorted double linked list corrupted`). Measured on the real checkpoint (`LensTeMemoryBoundedLoadTests`, also runnable with `LENS_TE_BACKEND=cuda`): load 9.4-9.6 s at ~7 GB peak RSS; load + two back-to-back 112-token encodes peak at **15.2-15.7 GB** VmHWM. Encode: ~61 s on a 3060 (GPU steady-state <400 MB VRAM — slices are freed per expert) vs ~5 min on CPU (single-threaded per-expert GEMMs at DOP 8). A threaded CPU GEMM or a persistent-device MoE path is the perf follow-up. The DiT (BF16/MXFP8→BF16) fits with the existing eviction discipline.

## Reference Implementations

- **microsoft/Lens — `lens/transformer.py`** ([github.com/microsoft/Lens/blob/main/lens/transformer.py](https://github.com/microsoft/Lens/blob/main/lens/transformer.py)) — `LensTransformer2DModel`, `LensTransformerBlock`, `LensJointAttention`, `LensEmbedRope`, `apply_rotary_emb_lens`. ~700 lines. **Primary reference for the C# transformer port.**
- **microsoft/Lens — `lens/pipeline.py`** ([github.com/microsoft/Lens/blob/main/lens/pipeline.py](https://github.com/microsoft/Lens/blob/main/lens/pipeline.py)) — `LensPipeline`, `compute_empirical_mu`. ~580 lines. **Primary reference for the C# pipeline port.**
- **microsoft/Lens — `lens/text_encoder.py`** ([github.com/microsoft/Lens/blob/main/lens/text_encoder.py](https://github.com/microsoft/Lens/blob/main/lens/text_encoder.py)) — `LensGptOssEncoder`. ~130 lines. **Subclass of `GptOssForCausalLM` from transformers.**
- **microsoft/Lens — `lens/resolution.py`** ([github.com/microsoft/Lens/blob/main/lens/resolution.py](https://github.com/microsoft/Lens/blob/main/lens/resolution.py)) — 18 fixed resolution buckets.
- **diffusers `AutoencoderKLFlux2`** — same VAE as Flux.2 Klein/Dev. Already implemented in this codebase via `Flux2CheckpointConverter` + `VaeDecoder`.
- **transformers `GptOssForCausalLM`** ([github.com/huggingface/transformers — models/gpt_oss/](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt_oss)) — the base GPT-OSS implementation. **Reference for MoE routing semantics, MXFP4 unpack, sliding/full attention mask construction.**
- **diffusers `FlowMatchEulerDiscreteScheduler`** — existing scheduler (used by Flux, SD3.5, Z-Image, F-Lite). The `set_timesteps(sigmas, mu)` form is what Lens calls.
- **diffusers `AdaLayerNormContinuous`** — `Linear(hidden → 2*hidden) → [shift, scale]`; identical to the version used by SD3.5/Qwen-Image final layers in this codebase.

## Differences Between Implementations

The reference is single-source (microsoft/Lens), but there are a few places where the upstream code diverges from common idioms in this codebase:

1. **Block return order is `(encoder_hidden_states, hidden_states)`** — text first, image second. Flux's `FluxDoubleStreamBlock.forward` returns `(hidden_states, encoder_hidden_states)` (image first). **Don't blindly copy the unpacking pattern.**
2. **`_modulate` chunks the half-mod into `(shift, scale, gate)` along dim=-1.** The chunked tensor has shape `[B, 3·1536]`; chunking yields three `[B, 1536]` tensors, then `.unsqueeze(1)` broadcasts across the sequence dim. Flux uses similar shape mechanics but a different overall chunk order; the C# port should follow Lens's `(shift1, scale1, gate1, shift2, scale2, gate2)` exactly.
3. **RoPE is complex-polar (Qwen-Image style), not pair-rotation (Flux style).** The C# port should re-use Qwen-Image's `RopeApplyComplex` kernel rather than Flux's `RopeApplyPair`.
4. **CFG batch ordering is `[positive, negative]`** in the doubled tensor (concat along dim=0). After the forward, `chunk(2)` gives `(cond, uncond) = (positive_pred, negative_pred)`. **Note that this is opposite to some other pipelines in the codebase that batch as `[uncond, cond]`** — pay attention to this in the C# port.
5. **`scale_rope=True` is hardcoded** in `LensTransformer2DModel.__init__` but **not in the JSON config**. The C# `LensConfig` should hardcode it the same way (don't expose it as a knob unless we ever need scale_rope=False).
6. **`rope_cache` is a runtime cache.** Upstream stores it as a plain dict on the module so the first forward per (h, w) computes freqs and subsequent forwards hit the cache. Important for the C# port to do the same — recomputing 4096-entry tables per step is wasteful.
7. **`pos_freqs` / `neg_freqs` are NOT registered as buffers** — register_buffer strips imaginary parts on safetensors save/load. They live as ordinary tensor attributes that get re-built in `__init__`. The C# port has no such constraint (we use raw `Tensor` with our own dtype handling), so we can construct them once at model creation and treat them as constants.

## Implementation Notes (recommendations for HartsyInference)

### What can be reused

- **`FlowMatchEulerDiscreteScheduler`** — used by Flux, SD3.5, Z-Image, F-Lite. Just plumb `mu` through.
- **`VaeDecoder` (Flux.2 preset)** — already loaded by `Flux2Pipeline.cs`. Lens uses the same decode path with BN un-normalization at the pipeline boundary.
- **`Flux2CheckpointConverter`** — handles the Flux.2 VAE weights file. Already in tree.
- **`AdaLayerNormContinuous` pattern** — same `Linear(1536 → 3072) → [shift, scale]` as SD3.5 and Qwen-Image final layers.
- **`RMSNorm` (with and without learned scale)** — existing primitive. Lens uses learned-scale RMSNorm everywhere (`eps=1e-5` for QK norms, `eps=1e-6` for stream norms).
- **`SwiGLU` MLP (`w1, w2, w3` naming)** — Flux double-stream block uses this same pattern. Bias=False on all three.
- **HiDream `NumRoutedExperts` config plumbing** — start here for the MoE FFN side of the encoder.
- **Qwen-Image complex-polar RoPE** — reuse `RopeApplyComplex` if it exists; otherwise port from `QwenImageRope.cs`.
- **`Microsoft.ML.Tokenizers.BpeTokenizer`** — already powers `Qwen3Tokenizer` and `ClipTokenizer`. Add a new `GptOssTokenizer` class with the appropriate special-token set and chat template.

### What's net-new

| Component | Effort | Why |
|---|---|---|
| **MoE FFN with real top-k routing** | High (~1-2 weeks) | Existing HiDream uses single-expert fallback. Need a proper grouped-by-expert dispatch primitive on CPU and CUDA. Shared with future text-LLM MoE work, so worth doing right. |
| **MXFP4 dequant-at-load** | Medium (~3-4 days) | New dtype handler in `Tensor.LoadAs`. Standard MXFP4 unpack is well documented; we already do FP8 scale-companion folding. |
| **Sliding-window + full-attention alternating mask** | Low (~1 day) | New flag in `LlamaStyleEncoderConfig`: `LayerTypes : string[]` (per-layer "sliding" or "full"). Mask builder branches per layer. |
| **`LensTransformer.cs` + `LensTransformerBlock.cs` + `LensRope.cs`** | Medium (~3-5 days) | Mostly a Flux-double-block clone with the differences flagged above. |
| **`LensGptOssEncoder.cs`** | Medium (~3-5 days) | Subclass `LlamaStyleEncoder` (or write a new `MoeLlamaStyleEncoder`) that captures multi-layer hidden states and exits early after the last selected layer. Reuse GPT-OSS-specific layer math. |
| **`LensPipeline.cs`** | Low (~2 days) | Standard pipeline once the pieces are wired. Norm-rescaled CFG is one extra op. |
| **`LensCheckpointConverter.cs`** | Low (~1 day) | Probably diffusers-naming passthrough for the transformer; MXFP4 unpack for the encoder; Flux.2 VAE converter for the VAE. |
| **`dump_lens_full_forward.py` + `diff_lens_layers.py`** | Medium (~2-3 days) | Standard layer-by-layer diff harness following the SD3.5 / Z-Image template. |

### VRAM budget on a 12 GB card (RTX 3060 / 4070 / etc.)

- DiT (FP16, 3.8 B) ≈ **7.6 GB** → at FP8 cast-on-load ≈ **3.8 GB**
- GPT-OSS encoder (MXFP4 native) ≈ **12 GB** packed; **24 GB** dequant'd to FP16
- Flux.2 VAE (FP16) ≈ **~5 GB**
- Activations at 1024×1024 (4096 tokens × 1536 dim × batch-of-2 for CFG, 48 layers worth of carryover): ~2-3 GB peak

**Path on 12 GB:** can't fit dequant'd encoder + DiT + VAE simultaneously. Plan:
1. Encode prompt with the encoder loaded; capture 4-layer hidden states (small — `4 × S_txt × 2880` is well under 1 GB).
2. `backend.FreeWeights(textEncoder)` before loading the DiT.
3. Run the denoise loop on the DiT alone (~4-8 GB peak depending on FP8/FP16).
4. `backend.FreeWeights(transformer)` before VAE decode (mirrors PHASE_3_DEVIATIONS #18, #33).
5. VAE decode.

This is the same eviction-discipline pattern Flux/SD3.5/Qwen-Image use today. **Native MXFP4 encoder GEMM would let the encoder stay resident**, but isn't needed for correctness — only throughput on repeated generations.

### Suggested implementation order

1. **GPT-OSS infrastructure** — MXFP4 dequant, MoE FFN routing, alternating attention masks, tokenizer. Land these as additions to the existing `LlamaStyleEncoder` family. This unblocks Lens AND any future MoE-LM text encoder.
2. **`LensConfig` + `LensTransformer` scaffold** — port the transformer block by block from `lens/transformer.py`. Reuse Qwen-Image RoPE, Flux2 VAE.
3. **`LensCheckpointConverter`** — most likely a diffusers-naming passthrough with FP8/MXFP4 scale-companion folding.
4. **`LensPipeline`** — straightforward once the encoder is up.
5. **Validation harness** — first-run debug; expect 1-3 pipeline-level bug iterations per the SD3.5 / Z-Image / Flux first-run pattern.

### What NOT to do

- **Don't try to call into `HartsyInference.LLM` for the GPT-OSS encoder.** Lens needs the multi-layer hidden states with mid-network capture, which is not a public API of the LLM package. Re-implement the relevant forward as a single self-contained `LensGptOssEncoder` class.
- **Don't fold the norm-rescale into the scheduler step.** Keep it explicit in the pipeline so the layer-diff harness can validate it against the upstream code exactly.
- **Don't expose `selected_layer_index` as a runtime knob.** It's part of how the DiT was trained; changing it means re-training. Hardcode in `LensConfig`.
- **Don't reuse `T5TextEncoder.EncodeAtLayer` semantics.** That helper re-applies the final layer norm; Lens does NOT — each captured hidden state is the raw output of the selected decoder layer (no extra normalization), and the per-layer `txt_norm` in the DiT does the only normalization that matters.
