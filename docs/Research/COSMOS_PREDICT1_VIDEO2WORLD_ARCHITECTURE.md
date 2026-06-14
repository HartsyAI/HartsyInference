# Cosmos-Predict1 Video2World — Research Notes

> Status: Complete (code + configs + tokenizer captured; HF tensor key dump still required) | Last Updated: 2026-05-24 | Needed Before: HartsyInference.Video (Phase 9, AR video continuation pipeline). Discrete tokenizer (Cosmos DV) + AR transformer infra reused by HartsyInference.Interactive (Phase 10) for action-conditioned world models.
> License: NVIDIA Open Model License (commercial OK — see § License)
> Source of truth: [nvidia-cosmos/cosmos-predict1 GitHub](https://github.com/nvidia-cosmos/cosmos-predict1), [arXiv 2501.03575 "Cosmos World Foundation Model Platform for Physical AI"](https://arxiv.org/abs/2501.03575), [HF nvidia/Cosmos-Predict1-5B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict1-5B-Video2World), [HF nvidia/Cosmos-Predict1-13B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict1-13B-Video2World), [HF nvidia/Cosmos-Tokenizer-DV8x16x16](https://huggingface.co/nvidia/Cosmos-Tokenizer-DV8x16x16)
> Related: [`LANCE_ARCHITECTURE.md`](LANCE_ARCHITECTURE.md) (joint image+video unified pipeline lineage), [`TEXT_ENCODERS.md`](TEXT_ENCODERS.md) (T5-11B is used here), [`VAE_ARCHITECTURE.md`](VAE_ARCHITECTURE.md) (continuous side of Cosmos Tokenizer family for diffusion decoder), and the forthcoming `WORLD_MODELS_AR_ACTION.md` (Phase 10, action-conditioned world models built on the same DV-token primitives)

## Summary

Cosmos-Predict1 Video2World (V2W) is NVIDIA's discrete-token **autoregressive video continuation** family — Llama3-style transformers that predict the next discrete video token until a clip is complete. Two variants ship: **5B** (4B base + cross-attn) and **13B** (12B base + cross-attn). Both consume text + image (1 frame) or text + video (9 frames) and emit a 24-frame or 32-frame clip at **1024×640 @ 25 fps**. Internally the model is **strictly next-token**: it autoregresses over the discrete tokens produced by the **Cosmos-Tokenize1-DV8x16x16-720p** tokenizer (8× temporal × 16× spatial × 16× spatial compression, FSQ codebook of **64,000** entries). T5 prompt embeddings are injected through cross-attention layers inserted **every transformer block** (`insert_cross_attn_every_k_layers=1`, `context_dim=1024`). After AR sampling, a **separate 7B latent diffusion decoder** ("Cosmos-Predict1-7B-Decoder-DV8x16x16ToCV8x8x8-720p") upsamples DV tokens into a CV8x8x8 continuous latent and decodes RGB — the AR model itself does not own pixel reconstruction.

**Framing for HartsyInference.** Cosmos-Predict1 V2W does **not** take action inputs (no joystick, no keyboard, no robot pose). It is not an "interactive world model" in the Matrix-Game / Oasis / GameCraft / DriveDreamer sense. It *is* a state-of-the-art autoregressive video predictor, and the same Python repo contains an `action_dim`/`action_embedding_mode` post-training path (`create_video2world_model` ll. 476–479) that wires an MLP action embedding into the same context stream, so the same backbone is the on-ramp for the Phase 10 action-conditioned models. The reusable Phase-10 infrastructure is:

1. **Cosmos-Tokenize1-DV8x16x16** — discrete video tokenizer (JIT encoder + JIT decoder, FSQ levels [8,8,8,5,5,5], 64,000-entry vocab, 8×16×16 compression). This is the single most reusable artifact — *any* Cosmos-lineage AR world model (Matrix-Game-2 included) consumes/produces tokens in this exact space.
2. **CosmosAR backbone** — Llama3-shaped decoder with **3D RoPE** over (T,H,W), QK-norm, GQA (32 Q heads / 8 KV heads), SwiGLU FFN, RMSNorm. Same body for 4B/12B base and 5B/13B V2W.
3. **Cross-attn-every-layer adapter** — light, frozen-base finetune that turns the base AR LM into a conditioned video generator. The same hook is what an action-conditioned world model will reuse, just swapping T5 embeddings for action embeddings (or concatenating both).
4. **3D RoPE** — per-axis split of `head_dim` into temporal/H/W ranges.

For HartsyInference this means: **build Cosmos V2W in `HartsyInference.Video` (Phase 9)** as `CosmosV2WPipeline`, and design the DV tokenizer + AR backbone + 3D RoPE as standalone reusable types under `HartsyInference.Video/Models/Cosmos/` so Phase 10 (`HartsyInference.Interactive`) can compose them with an `ActionEmbedder`.

Predict1 has been **superseded by Cosmos-Predict2 / Cosmos-Predict2.5** in the diffusion family — Predict2.5-2B (Oct 6 2025) is **diffusion, not AR** and replaces the diffusion-side Predict1 pipelines. The autoregressive V2W path remains Predict1's domain; there is no AR Predict2.5 as of 2026-05-24. So Predict1 5B/13B V2W are still the SOTA AR-token world models from NVIDIA.

## License

**NVIDIA Open Model License (OML), June 2024** — applied to all Cosmos-Predict1 weights and the Cosmos-Tokenizer weights. The cosmos-predict1 source **code** in GitHub is Apache 2.0 (see `SPDX-License-Identifier: Apache-2.0` at the top of every `.py`).

### What's permitted (verbatim or paraphrased from OML June 2024 + HF model card summaries)

- **Commercial use: yes.** "Models are commercially usable."
- **Derivative models: yes.** "You are free to create and distribute Derivative Models. Derivative Model means all (a) modifications to the Model, (b) works based on the Model, and (c) any other derivative works of the Model."
- **Output ownership: yours.** "NVIDIA does not claim ownership to any outputs generated using the Models or Derivative Models." → HartsyInference users keep ownership of every video Cosmos generates.
- **Relicensing of derivatives: allowed.** "You may add your own copyright statement to your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of your modifications, or for any such Derivative Models as a whole, provided your use, reproduction, and distribution of the Model otherwise complies with the conditions stated in this Agreement."
- **Redistribution of the base model: must include OML.** "if you distribute the Model, you must give any other recipients of the Model a copy of this Agreement and include the following attribution notice within a 'Notice' text file with such copies: 'Licensed by NVIDIA Corporation under the NVIDIA Open Model License'."

### Hard restrictions (load-bearing for HartsyInference)

- **Cannot bypass safety guardrails.** Every Cosmos model card states the license **automatically terminates** if you "circumvent" NVIDIA's safety guardrails. NVIDIA's reference code ships a "guardrail" pipeline (text classifier + face-blur + content filter) that runs before/after generation. The OML text frames this as a model-specific use restriction.
  - **HartsyInference implication:** we can ship Cosmos V2W *without* implementing NVIDIA's guardrail pipeline (it's not architecturally required), but our distribution must not include code that detects or removes guardrail enforcement when present. Practical stance: do not port the guardrail; document its absence; do not actively suppress it. This is the same posture we took for Anima.
- **Trade compliance.** OML requires you to comply with U.S. export controls when redistributing. Standard.
- **No "hate, harm, harass" content as marketed output.** The "intended use" sections in the model cards (which the OML incorporates by reference for Cosmos-Predict1) say outputs must not be used to harm individuals. Not enforceable in code; ship as a user-facing warning.

### Attribution notice text (required in any redistribution bundle)

```
Licensed by NVIDIA Corporation under the NVIDIA Open Model License
```

To be placed in a `NOTICE` file alongside any Cosmos weights or derivatives that HartsyInference ships. The Apache-2.0 code in the GitHub repo additionally requires retaining the per-file `SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.` notices if we port any code (we won't — pure C# reimplementation — so this only applies to weights).

### Permissibility summary

| Activity | OML status |
|---|---|
| Use Cosmos-Predict1 weights in a closed-source commercial SaaS | ✅ |
| Ship Cosmos-Predict1 weights in HartsyInference's distribution | ✅ (include NOTICE file) |
| Generate videos and sell them | ✅ (you own them) |
| Fine-tune V2W on game footage, redistribute as `MatrixGameLike-7B` | ✅ Derivative Model — relicense at your option |
| Strip the guardrail pipeline from NVIDIA's Python and ship that | ❌ License termination |
| Re-implement Cosmos-Tokenize1-DV in pure C# | ✅ (architecture is not patented; weights still under OML) |

## Detailed Findings

### 1. Naming: Predict1 vs Cosmos-1.0

`Cosmos-Predict1-5B-Video2World` and `Cosmos-1.0-Autoregressive-5B-Video2World` are **the same weights**, with the same SHA. NVIDIA renamed the suite in early 2025 when they split the world-model platform into `Cosmos-Predict` (forecasting / video gen), `Cosmos-Transfer` (translation), and `Cosmos-Reason` (reasoning) families. HF still hosts both names; the `nvidia/Cosmos-Predict1-*` variants are the supported entry point.

### 2. Family / variants

| Variant | Params | Cross-attn? | Base backbone | `.pt` file size | Conditioning |
|---|---|---|---|---|---|
| Cosmos-Predict1-4B | 4 B | no | "4b" (16L × 4096) | ~7.5 GB BF16 | continue-only AR (no text) |
| Cosmos-Predict1-12B | 12 B | no | "12b" (40L × 5120) | ~22 GB BF16 | continue-only AR (no text) |
| **Cosmos-Predict1-5B-Video2World** | **5 B** | **yes** | "4b" + CA every layer | **9.17 GB BF16 `model.pt`** | text + image(1f) or text + video(9f) |
| **Cosmos-Predict1-13B-Video2World** | **13 B** | **yes** | "12b" + CA every layer | **26.6 GB BF16 `model.pt`** | text + image(1f) or text + video(9f) |
| Cosmos-Predict1-7B-Decoder-DV8x16x16ToCV8x8x8-720p | 7 B (diffusion) | — | DiT | n/a (post-AR refiner) | DV tokens → CV latent |
| Cosmos-Tokenize1-DV8x16x16-720p | ~30 M | — | causal conv + Haar wavelet + FSQ | JIT (`encoder.jit`, `decoder.jit`) | RGB ↔ discrete tokens |
| Cosmos-Tokenize1-CV8x8x8-720p | ~50 M | — | causal conv + Haar wavelet + AE | JIT | CV continuous latent ↔ RGB |

**The "5B/13B" parameter count is the base 4B/12B + the per-layer cross-attention adapters and their norms.** No new self-attn or FFN is added; the cross-attn is `q_proj` (from query stream) + `k_proj`+`v_proj` (from T5 context) + `o_proj`, sized by the same `dim`/`n_heads`, plus a `cross_attention_norm` RMSNorm — see `transformer.py:74-79`.

### 3. AR transformer backbone (exact constants from `cosmos_predict1/autoregressive/configs/base/model_config.py`)

**Per-family shared base config:**

```python
BASE_CONFIG = {
    "n_kv_heads": 8,
    "norm_type": "rmsnorm",
    "norm_eps": 1e-5,
    "ffn_hidden_size": 14336,
}
```

**Per-size architecture:**

```python
COSMOS_ARCHITECTURES = {
    "1b":  {"n_layers": 16, "dim": 2048, "n_heads": 32},                   # not released for V2W
    "4b":  {"n_layers": 16, "dim": 4096, "n_heads": 32},                   # base for 5B V2W
    "12b": {"n_layers": 40, "dim": 5120, "n_heads": 32, "head_dim": 128},  # base for 13B V2W
}
```

**4B / 5B-V2W resolved:**

| Field | Value |
|---|---|
| `n_layers` | **16** |
| `dim` (hidden) | **4096** |
| `n_heads` | **32** |
| `head_dim` | 4096 / 32 = **128** (default — fits the explicit 12B value) |
| `n_kv_heads` | **8** (GQA factor **4**) |
| `ffn_hidden_size` (SwiGLU intermediate) | **14336** |
| `norm_type` | **RMSNorm** |
| `norm_eps` | **1e-5** |
| `vocab_size` | **64000** (= prod(FSQ levels), no text vocab) |

**12B / 13B-V2W resolved:**

| Field | Value |
|---|---|
| `n_layers` | **40** |
| `dim` | **5120** |
| `n_heads` | **32** |
| `head_dim` | **128** (explicit) |
| `n_kv_heads` | **8** (GQA factor **4**) |
| `ffn_hidden_size` | **14336** |
| `norm_type` | **RMSNorm** |
| `norm_eps` | **1e-5** |
| `vocab_size` | **64000** |

**Activation: SwiGLU** — confirmed in `cosmos_predict1/autoregressive/modules/mlp.py:77`:
```python
output = self.w2(F.silu(self.w1(x)) * self.w3(x))
```
i.e. `down(silu(gate(x)) * up(x))`, classic Llama-style. The `ffn_hidden_size = 14336` matches Llama3-8B exactly — NVIDIA literally reused Llama3 dimensions for the 4B body, just shaving layers.

**QK normalization is ON for V2W** — `create_video2world_model_config(..., use_qk_normalization: bool = True, ...)`. Per-head RMSNorm on Q and K before RoPE. Confirmed in `modules/attention.py:109-115`. The base 4B/12B don't enable it by default; the 5B/13B V2W finetunes do.

### 4. Cross-attention adapter (the +1B / +1B of "5B"/"13B")

From `cosmos_predict1/autoregressive/networks/transformer.py:71-79`:

```python
self.has_cross_attention = False
self.cross_attention, self.cross_attention_norm = None, None
if args["insert_cross_attn"] and layer_id % args["insert_cross_attn_every_k_layers"] == 0:
    self.has_cross_attention = True
    cross_attention_args = attention_args.copy()
    cross_attention_args.update({"context_dim": args["context_dim"], "fuse_qkv": False, "attn_type": "cross"})
    self.cross_attention = Attention(**cross_attention_args)
    self.cross_attention_norm = create_norm(args["norm_type"], dim=args["dim"], eps=args["norm_eps"])
```

**V2W defaults (`world_generation_pipeline.py:100-109`):**

```python
insert_cross_attn = True
insert_cross_attn_every_k_layers = 1   # EVERY layer
context_dim = 1024                     # T5-11B encoder hidden
training_type = "text_to_video"
apply_abs_pos_emb = True
```

So both 5B and 13B have **a cross-attention block in every single transformer layer** (16 for 5B, 40 for 13B). Forward order is:

```
h = x + self_attn(norm(x))                  # GQA self-attn, 3D-RoPE on Q/K
if has_cross_attention:
    h = h + cross_attn(cross_attn_norm(h),
                       context=t5_embed,    # (B, 512, 1024)
                       context_mask=t5_mask)
h = h + ffn(ffn_norm(h))                    # SwiGLU
```

Cross-attn is **full attention** (no causal mask — `modules/attention.py:206` "For cross-attention, it's always full-attn without causal mask") and does **not** use RoPE (`attn_type == "cross"` skips the RoPE call at `attention.py:181`).

### 5. Positional encoding: **3D RoPE**

From `cosmos_predict1/autoregressive/modules/embedding.py:231-247`. The head_dim is split into three contiguous frequency ranges over (T, H, W):

- `dim_temporal_range`, `dim_h_range`, `dim_w_range` partition the head_dim.
- `temporal_inv_freq = 1.0 / (rope_theta ** dim_temporal_range)`
- `spatial_inv_freq = 1.0 / (rope_theta ** dim_spatial_range)` (shared between H and W)
- Per-token position is the (t, h, w) latent coordinate.

`rope_theta` default in `BASE_CONFIG` is the Llama3-style **500,000** (inherited via `LLAMA3_ARCHITECTURES`; verify Open Q § 4 — `BASE_CONFIG` doesn't set it explicitly so the model dataclass default applies. The Cosmos paper uses 500,000 in their RoPE setup for AR; confirm by reading `cosmos_predict1/autoregressive/configs/base/model.py:ModelConfig` defaults locally).

**3D RoPE shape** (latent grid axes):

- Video latent shape = `[T_latent, H_latent, W_latent]`.
- For the default V2W setup at 1024×640 with `compression_ratio=[8,16,16]` and `pixel_chunk_duration=33`, `num_video_frames=33`:
  - `latent_chunk_duration = (33-1)/8 + 1 = 5`
  - `latent_height = 640/16 = 40`
  - `latent_width = 1024/16 = 64`
  - `video_latent_shape = [5, 40, 64]` → `num_token_video_latent = 5*40*64 = 12,800`
  - **`max_seq_len = 12,800`** (no `+3` for special tokens because `add_special_tokens=False` for 3D-RoPE V2W).

### 6. Absolute positional embedding (additional to RoPE)

V2W sets `apply_abs_pos_emb=True` (`world_generation_pipeline.py:107`). From `transformer.py:206-209`, a learned/frozen `PositionEmbedding3D`-style table is added to embeddings before the layer stack:

```python
if self.params["apply_abs_pos_emb"]:
    self.pos_emb_config = self._create_abs_pos_emb_config()
    self.pos_emb, self.abs_pos_emb = self._initialize_abs_pos_emb()
```

So the V2W path uses **both** 3D RoPE in attention *and* an additive 3D abs-pos at input. The base 4B/12B (no V2W finetune) do not have `apply_abs_pos_emb`; this is a V2W-only addition.

### 7. Discrete video tokenizer: **Cosmos-Tokenize1-DV8x16x16-720p**

Tokenizer module: `cosmos_predict1/autoregressive/tokenizer/discrete_video.py`. Quantizer: `cosmos_predict1/autoregressive/tokenizer/quantizers.py:FSQuantizer`.

**Hard constants (default constructor args):**

```python
levels = [8, 8, 8, 5, 5, 5]           # FSQ per-dim levels (6 dims)
compression_ratio = [8, 16, 16]       # (T, H, W) downsample
```

→ **Codebook size = 8 × 8 × 8 × 5 × 5 × 5 = 64,000** (== AR vocab size). Computed as `self._levels.prod().item()` in `FSQuantizer.__init__`. Each token is a single integer in `[0, 64000)`.

**Encoder architecture** (from HF model card + Cosmos paper):

- Input: `(B, 3, T, H, W)` RGB in bfloat16, normalized to `[-1, 1]`.
- Front-end: **2-level Haar wavelet transform** (4× downsample in space + temporal channel split; produces 12-channel pre-conv input).
- Body: causal 3D convolutions interleaved with downsampling stages → total 8× temporal, 16× spatial.
- Head: project to 6-dim continuous code → FSQ quantize (per-dim round to one of {8,8,8,5,5,5} levels) → single integer index via the per-dim mixed-radix basis.

**Decoder architecture:** mirror of encoder — inverse Haar wavelet at the end. Encoder and decoder are shipped as **separate `encoder.jit` and `decoder.jit` TorchScript blobs** plus a combined `autoencoder.jit`. Cosmos-Predict1 V2W loads only `encoder.jit` (to tokenize the 9-frame video prefix during inference) and only `decoder.jit` (only used for non-diffusion-decoder fallback — the main pixel path goes through the 7B diffusion decoder instead).

**Output latent grid** for 9 input frames at 1024×640:

- `T = (9-1)/8 + 1 = 2` latent frames
- `H = 640/16 = 40`
- `W = 1024/16 = 64`
- → **5,120 tokens** for the 9-frame conditioning chunk.

For the full 33-pixel-frame output (4× chunks of 9 = wait, math: `num_video_frames=33, pixel_chunk_duration=33`):

- `T = 5, H = 40, W = 64` → **12,800 tokens** total. The model autoregresses ~7,680 new tokens after the first 5,120 conditioning tokens (or first `H*W = 2,560` if using `num_input_frames=1` for image input).

**Constraint:** `pixel_chunk_duration % compression_ratio[0] == 1` must hold (asserted in `model_config.py:357-359`). That's why valid prefix lengths are **1, 9** (`_SUPPORTED_CONTEXT_LEN = [1, 9]` in `utils/inference.py:33`) — both satisfy `n*8 + 1`.

**Pretrained tokenizer constants for HartsyInference porting:**

- Levels: `[8, 8, 8, 5, 5, 5]` (FSQ basis = cumprod = `[1, 8, 64, 512, 2560, 12800]`)
- Codebook: 64,000
- Compression: 8× T, 16× H, 16× W
- Encoder/decoder live in TorchScript JIT; **no `.safetensors` from NVIDIA** — porting to C# requires either (a) JIT introspection + re-emission, or (b) a community pure-pytorch re-implementation. See [TokenBench](https://github.com/NVlabs/TokenBench) for reference shapes.

### 8. Text encoder: T5-11B (not T5-XXL)

From `cosmos_predict1/auxiliary/t5_text_encoder.py:30`:

```python
def __init__(self, model_name: str = "google-t5/t5-11b", ...):
```

T5-11B has hidden size **1024** (matches `context_dim=1024`), 24 encoder layers, 64 attention heads, and FFN 65,536. Token limit per `encode_prompts(..., max_length: int = 512)`. Standard SentencePiece tokenizer (`T5TokenizerFast`). Embeddings are extracted from `last_hidden_state`, with positions past the actual prompt length **zero-masked** (line 99-101 of `t5_text_encoder.py`).

Note: T5-11B (1024 dim) is *not* the T5-XXL (4096 dim, often called "T5xxl" in SD3/Flux). Cosmos chose T5-11B specifically so the cross-attn `k`/`v` projection can stay narrow (1024→4096 for 4B or 1024→5120 for 12B). This is unusual — most modern diffusion/video models use T5-XXL. If HartsyInference already has T5-XXL plumbing, we will need a separate **T5-11B** loader.

T5 weights are loaded from a local cache at `checkpoints/google-t5/t5-11b/` (HF cache layout), not from the Cosmos HF repo itself.

### 9. Diffusion decoder (two-stage pipeline)

After AR sampling produces 12,800 DV tokens, the reference inference path runs them through a **separate diffusion-based decoder** to upsample DV8x16x16 → CV8x8x8 → pixels:

- Model: **Cosmos-Predict1-7B-Decoder-DV8x16x16ToCV8x8x8-720p** (`model.pt` ≈ 14 GB BF16).
- Config name: `DD_FT_7Bv1_003_002_tokenizer888_spatch2_discrete_cond_on_token`.
- Operates on **continuous** Cosmos-Tokenize1-CV8x8x8-720p latents (16 channels, 8× temporal, 8× spatial).
- Sampling defaults (`DiffusionDecoderSamplingConfig` in `inference_config.py:52-77`):
  - `guidance = 1.8`
  - `sigma_min = 0.02`
  - `sigma = 8` (initial noise)
  - `num_steps = 15`
  - `overlap = 2` (frame overlap between video chunks)
  - `continuous_tokenizer_channel = 16`
  - `continuous_tokenizer_spatial_compression_ratio = 8`
  - `dd_train_num_video_frames = 57`
  - `fps = 24`

A "generic prompt" T5 embedding (`"high quality, 4k, high definition, smooth video"`) ships in `aux_vars.pt` and is used when the user prompt is unavailable. Without the diffusion decoder, the AR-only path can fall back to `decoder.jit` of the DV tokenizer, but quality is **noticeably worse** — the README notes this is the "fast/lower-quality" mode.

**For HartsyInference:** the diffusion decoder is large (7B params, 14 GB). Phase 9 should implement Cosmos V2W in two stages, with `--disable_diffusion_decoder` equivalent as the always-on default, and the diffusion decoder as an opt-in quality upgrade. This is also the boundary where pure-AR cleanly separates from diffusion — Phase 10 world models will not need the diffusion decoder if they target playable latency.

### 10. Inference pipeline (end-to-end)

`cosmos_predict1/autoregressive/inference/video2world.py` + `world_generation_pipeline.py`.

**Default sampling config (`SamplingConfig`):**

```python
temperature = 0.6        # not the 1.0 the README shows; 0.6 is the dataclass default
top_k       = None
top_p       = 0.9        # nucleus
compile_prefill  = False
compile_sampling = True
logprobs    = False
echo        = False
```

The README/example scripts pass `--temperature 1.0 --top_p 0.8`, so users typically override the dataclass defaults. **The model is sensitive to temperature** — at `0.6` it's deterministic and motion-stable; at `1.0` it's diverse but more failure-prone.

**Stop tokens:** `stop_tokens = self.model.tokenizer.stop_tokens` (V2W has `add_special_tokens=False` so the stop tokens for V2W are implicit — generation halts when `num_token_video_latent` tokens have been emitted, not on an EOS).

**Number of tokens to generate:**

```python
num_gen_tokens = int(np.prod([T - latent_context_t_size, H, W]))
```

For 9-frame video prefix: `T=5, H=40, W=64`, latent_context_t = 1 (the 9 input frames are one latent chunk of size 1+... actually `latent_chunk_duration = (9-1)/8 + 1 = 2` but only the first is the conditioning chunk; the AR generates from token 5120 onward → ~7,680 new tokens).

**Hard-coded resolution / frame counts:**

- `video_height = 640`, `video_width = 1024`
- `pixel_chunk_duration = 33`, `num_video_frames = 33`
- Output fps: **25 fps** (saved via `imageio.mimsave(..., fps=25)`), but the *diffusion decoder* trains/produces 24 fps. Slight mismatch — NVIDIA appears to consciously stretch the output by ~4% on save. This is a Predict1 quirk.

**Output:** 33-frame MP4. From an image input, all 32 generated frames + the 1 input frame = 33. From a 9-frame video input, 24 generated frames + the 9 conditioning frames = 33.

### 11. Generation loop (eager, no KV-cache caveats)

The reference uses Megatron's tensor-parallel infrastructure and TorchScript-compiled sampling. Internally, autoregressive generation is the standard "prefill once, then per-token loop with KV-cache". Speed numbers from the HF model cards:

- **5B V2W, H100 BF16, no offload:** 66.2 GB VRAM, **~73 s** end-to-end for one 33-frame video.
- **5B V2W, H100, full offload:** 21.1 GB VRAM (slower).
- **13B V2W, H100, no offload:** >80 GB VRAM, **~150 s**.
- **13B V2W, H100, full offload:** 30.9 GB VRAM.

Failure rate (NVIDIA's own measurement, 100 prompts):
- 5B: 7% (image input), 2% (video input)
- 13B: 3% (image input), 0% (video input)

### 12. Cosmos-Predict2.5-2B (context — NOT AR)

Released **October 6, 2025**, model card explicitly says "developed based on Cosmos-Predict2-2B". It is **diffusion (DiT)**, not autoregressive: "interleaved self-attention, cross-attention, and feedforward layers with adaptive layer normalization for time embedding". Parameters: **2,059,174,912 (2.06 B)**. Output: 720p / 1280×704, 5 s clips at 16 fps. VRAM 32.5 GB. License: NVIDIA Open Model License (some Predict2 variants additionally tagged Apache-2.0 in the model card UI — verify per-variant).

**Important:** Predict2.5 does **not** supersede Predict1 V2W for the AR-token approach. NVIDIA has not released an AR variant of Predict2/2.5 as of 2026-05-24. So if you want AR-token video continuation from NVIDIA, **Predict1 5B/13B V2W are it**.

## Key Numbers / Constants

| Constant | Value | Where it's used |
|---|---|---|
| **AR 4B/5B backbone** | | |
| `n_layers` | 16 | Decoder depth |
| `dim` (hidden) | 4096 | Token feature dim |
| `n_heads` | 32 | head_dim = 128 |
| `n_kv_heads` | 8 | GQA factor 4 |
| `ffn_hidden_size` | 14336 | SwiGLU inner dim |
| **AR 12B/13B backbone** | | |
| `n_layers` | 40 | |
| `dim` | 5120 | |
| `n_heads` | 32 | head_dim = 128 (explicit) |
| `n_kv_heads` | 8 | GQA factor 4 |
| `ffn_hidden_size` | 14336 | |
| **Shared AR** | | |
| `norm_type` | RMSNorm | |
| `norm_eps` | 1e-5 | |
| `vocab_size` | 64000 | = DV codebook size, no text vocab |
| `use_qk_normalization` | true | V2W variants only |
| `rope_dim` | "3D" | 3D RoPE over (T, H, W) |
| `rope_theta` | 500,000 (likely; verify locally — Open Q § 4) | |
| `apply_abs_pos_emb` | true | V2W only |
| `context_dim` (cross-attn KV) | 1024 | == T5-11B encoder hidden |
| `insert_cross_attn` | true | V2W only |
| `insert_cross_attn_every_k_layers` | 1 | Every layer has CA |
| `add_special_tokens` | false | V2W (3D RoPE doesn't need BOV) |
| **Discrete tokenizer (DV8x16x16)** | | |
| FSQ levels | [8, 8, 8, 5, 5, 5] | 6-dim FSQ |
| FSQ basis (cumprod) | [1, 8, 64, 512, 2560, 12800] | mixed-radix index |
| Codebook size | 64,000 | = vocab_size |
| Compression (T × H × W) | 8 × 16 × 16 = 2048× | |
| Pre-conv wavelet | 2-level Haar | 4× spatial reduction before conv |
| Tokenizer ckpt | `Cosmos-Tokenize1-DV8x16x16-720p/ema.jit` | TorchScript |
| **Video shapes** | | |
| Pixel resolution | 1024 × 640 | hard-coded |
| Pixel chunk duration | 33 frames | |
| Latent chunk duration | (33-1)/8 + 1 = 5 | |
| Latent H | 640 / 16 = 40 | |
| Latent W | 1024 / 16 = 64 | |
| Latent shape | [5, 40, 64] | |
| Total video tokens | 5 × 40 × 64 = 12,800 | per 33-frame clip |
| Supported input prefix | 1 or 9 frames | `_SUPPORTED_CONTEXT_LEN` |
| Output fps | 25 (save) / 24 (DD train) | |
| **T5 text encoder** | | |
| Model name | `google-t5/t5-11b` | not T5-XXL |
| Hidden size | 1024 | == context_dim |
| Max tokens | 512 | from `encode_prompts(max_length=512)` |
| **Sampling defaults** | | |
| `temperature` | 0.6 (config) / 1.0 (script) | |
| `top_p` | 0.9 (config) / 0.8 (script) | nucleus |
| `top_k` | None | disabled by default |
| **Diffusion decoder (optional)** | | |
| Params | 7 B | DiT |
| Source tokenizer | DV8x16x16 | discrete in |
| Target tokenizer | CV8x8x8 | continuous out |
| `guidance` | 1.8 | |
| `num_steps` | 15 | |
| `sigma_min` | 0.02 | |
| `sigma` (start) | 8 | |
| `overlap` | 2 frames | chunk overlap |
| CV channels | 16 | |
| CV spatial compression | 8× | |
| **Storage** | | |
| 5B `model.pt` | 9.17 GB | BF16 pickle |
| 13B `model.pt` | 26.6 GB | BF16 pickle |
| 7B DD `model.pt` | ~14 GB | BF16 pickle |
| DV tokenizer JIT pair | ~150 MB combined | TorchScript |
| T5-11B (separate dl) | ~45 GB FP32 / ~22 GB BF16 | from HF `google-t5/t5-11b` |

## Data Layouts / Formats

### HF repo: `nvidia/Cosmos-Predict1-5B-Video2World/`

```
Cosmos-Predict1-5B-Video2World/
├── .gitattributes        2.79 kB
├── README.md            24.1 kB
├── config.json            480 B   (model metadata; NOT a transformers AutoConfig)
└── model.pt             9.17 GB   (PyTorch pickle, BF16)
```

### HF repo: `nvidia/Cosmos-Predict1-13B-Video2World/`

```
Cosmos-Predict1-13B-Video2World/
├── .gitattributes        2.79 kB
├── README.md            23.1 kB
├── config.json            621 B
└── model.pt             26.6 GB
```

**No safetensors.** Weights ship as **`model.pt`** PyTorch pickles. The pickle root is `collections.OrderedDict` containing `torch.BFloat16Storage` tensors via `torch._utils._rebuild_tensor_v2`. **For HartsyInference this is a concrete blocker**: we don't have a PyTorch pickle loader and don't want one. The conversion approach is:

1. **One-off Python script** (off-ship) that loads `model.pt` and re-emits as `.safetensors`. HartsyInference's existing safetensors loader then consumes it.
2. Document this in `samples/ConvertCosmosCheckpoint/` so the user runs it once per model.

The `config.json` is a Cosmos-private dataclass dump, not a `transformers.PretrainedConfig`. It's tiny (480-621 bytes); we'll read it as a JSON metadata header (or just hard-code the architecture in `CosmosCheckpointConverter` keyed by file size / model name).

### HF repo: `nvidia/Cosmos-Tokenize1-DV8x16x16-720p/` (separate gated download)

```
Cosmos-Tokenize1-DV8x16x16-720p/
├── encoder.jit            ~80 MB    TorchScript encoder
├── decoder.jit            ~80 MB    TorchScript decoder
├── ema.jit                ~160 MB   combined (encoder + decoder, EMA weights)
├── config.json            ~1 kB
└── README.md             ~10 kB
```

The V2W pipeline loads `ema.jit` (combined): `tokenizer_ckpt_path = ".../Cosmos-Tokenize1-DV8x16x16-720p/ema.jit"`.

### Expected tensor key prefixes inside `model.pt` (5B)

Based on the Python module hierarchy in `Transformer` (`networks/transformer.py`) and `TransformerBlock`:

```
tok_embeddings.weight                                  (vocab=64000, dim=4096)
layers.{0..15}.attention.wq.weight                     (4096 × 4096)        # or wqkv fused if fuse_qkv
layers.{0..15}.attention.wk.weight                     (n_kv_heads*head_dim × dim) = (1024 × 4096)
layers.{0..15}.attention.wv.weight                     (1024 × 4096)
layers.{0..15}.attention.wo.weight                     (4096 × 4096)
layers.{0..15}.attention.q_norm.weight                 (head_dim,) = (128,)  # QK-norm
layers.{0..15}.attention.k_norm.weight                 (128,)
layers.{0..15}.attention_norm.weight                   (4096,) RMSNorm
layers.{0..15}.cross_attention.wq.weight               (4096 × 4096)        # V2W only
layers.{0..15}.cross_attention.wk.weight               (1024 × 1024)        # context_dim=1024 in, context_dim out? verify
layers.{0..15}.cross_attention.wv.weight               (1024 × 1024)
layers.{0..15}.cross_attention.wo.weight               (4096 × 4096)
layers.{0..15}.cross_attention_norm.weight             (4096,)
layers.{0..15}.feed_forward.w1.weight                  (14336 × 4096)       # gate
layers.{0..15}.feed_forward.w2.weight                  (4096 × 14336)       # down
layers.{0..15}.feed_forward.w3.weight                  (14336 × 4096)       # up
layers.{0..15}.ffn_norm.weight                         (4096,)
norm.weight                                            (4096,)
output.weight                                          (64000 × 4096)       # not tied
pos_emb.weight                                         (some 3D table)      # apply_abs_pos_emb=True
```

For 13B replace `n_layers=16 → 40`, `dim=4096 → 5120`. All cross-attention K/V projection shapes need verification once we have a real `model.pt` to inspect — this is Open Q § 1.

### DV tokenizer index encoding

Each integer index in `[0, 64000)` is decomposed back to the 6-dim FSQ code via mixed-radix:

```
basis = [1, 8, 64, 512, 2560, 12800]
code_d = (index // basis[d]) % levels[d]  for d in 0..5
half_width_d = levels[d] // 2
zhat_d = (code_d - half_width_d) / half_width_d        # back to [-1, 1]
```

Then `project_out(zhat)` (a `Linear(6 → dim)` inside the encoder/decoder) gives the continuous code. For HartsyInference we only need the integer codebook for AR sampling; the FSQ project_out is internal to the tokenizer decoder.

## Algorithm Steps

### Video2World inference (text + video, 5B)

```
1.  Load T5-11B → encode prompt (max_length=512) → t5_embed [1, 512, 1024], t5_mask [1, 512]
2.  Load DV tokenizer (ema.jit). Read 9 input frames at 1024x640, normalize to [-1, 1].
3.  tokens_in = DV_encoder(input_video)                  # shape [1, 2, 40, 64], flatten → [1, 5120]
4.  Build empty token buffer of length max_seq_len = 12,800.
    Fill positions 0..5119 with tokens_in. Positions 5120..12799 will be generated.
5.  Prefill the AR transformer:
        h = embed(tokens_in) + abs_pos_emb[0..5119]
        for layer in 16:
            h = h + self_attn_with_3DRoPE_GQA(rmsnorm(h))      # KV-cache populated
            h = h + cross_attn(cross_norm(h), context=t5_embed, mask=t5_mask)  # no causal
            h = h + swiglu_ffn(ffn_norm(h))
        # final hidden state of position 5119 used for first logit
6.  For step in 5120..12799:
        x = embed(tokens[step-1]) + abs_pos_emb[step-1]
        run 1-token attention with cached K/V (causal append).
        logits = output_proj(rmsnorm(h_step))                  # [64000]
        logits = logits / temperature
        if top_p < 1.0: nucleus-prune; else if top_k: top-k prune
        sampled = multinomial(softmax(logits))
        tokens[step] = sampled
7.  Reshape tokens → [B=1, 5, 40, 64].
8.  IF diffusion decoder:
        cv_latents = DD_diffusion(tokens, t5_embed, num_steps=15, guidance=1.8, sigma=8→sigma_min=0.02)
        pixels = CV_decoder(cv_latents)                        # 33 frames, 1024x640
    ELSE:
        pixels = DV_decoder(tokens)                            # lower quality, no DD needed
9.  Save MP4 at 25 fps.
```

### Image2World (text + 1 image)

Same as above but `num_input_frames=1`, `tokens_in.shape = [1, 1, 40, 64] = 2,560 tokens`, AR generates 10,240 tokens.

## Reference Implementations

**Primary:** [github.com/nvidia-cosmos/cosmos-predict1](https://github.com/nvidia-cosmos/cosmos-predict1) (Apache-2.0 code, OML weights)

Source-of-truth files (always cite these in implementation PRs):

- [`cosmos_predict1/autoregressive/inference/video2world.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/autoregressive/inference/video2world.py) — V2W entry script. CLI args, T5 offload toggle.
- [`cosmos_predict1/autoregressive/inference/world_generation_pipeline.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/autoregressive/inference/world_generation_pipeline.py) — `ARBaseGenerationPipeline`, `ARVideo2WorldGenerationPipeline`. Contains `detect_model_size_from_ckpt_path` and `create_inference_config` — these are the two functions that resolve "5B" → "4B base + cross-attn".
- [`cosmos_predict1/autoregressive/configs/base/model_config.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/autoregressive/configs/base/model_config.py) — `COSMOS_ARCHITECTURES`, `BASE_CONFIG`, `create_video2world_model_config`, `create_video2world_model`. **All architecture constants live here.**
- [`cosmos_predict1/autoregressive/configs/inference/inference_config.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/autoregressive/configs/inference/inference_config.py) — `SamplingConfig` (defaults), `DiffusionDecoderSamplingConfig`.
- [`cosmos_predict1/autoregressive/networks/transformer.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/autoregressive/networks/transformer.py) — `TransformerBlock`, `Transformer`. Cross-attn insertion logic at lines 71-79.
- [`cosmos_predict1/autoregressive/modules/attention.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/autoregressive/modules/attention.py) — `Attention` (GQA + QK-norm + self/cross/full modes).
- [`cosmos_predict1/autoregressive/modules/embedding.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/autoregressive/modules/embedding.py) — `RotaryPositionEmbedding` (1D/2D/3D variants).
- [`cosmos_predict1/autoregressive/modules/mlp.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/autoregressive/modules/mlp.py) — SwiGLU `MLP`.
- [`cosmos_predict1/autoregressive/tokenizer/discrete_video.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/autoregressive/tokenizer/discrete_video.py) — `BaseDiscreteVideoFSQTokenizer`, `DiscreteVideoFSQJITTokenizer`. Default `levels`, `compression_ratio`.
- [`cosmos_predict1/autoregressive/tokenizer/quantizers.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/autoregressive/tokenizer/quantizers.py) — `FSQuantizer`. The mixed-radix codebook math.
- [`cosmos_predict1/auxiliary/t5_text_encoder.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/auxiliary/t5_text_encoder.py) — `CosmosT5TextEncoder` wrapping `google-t5/t5-11b` with `max_length=512`.
- [`cosmos_predict1/utils/base_world_generation_pipeline.py`](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/utils/base_world_generation_pipeline.py) — base class with `_load_text_encoder_model`.

**Paper:** [arXiv 2501.03575 — Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). 80+ pages, multiple versions through July 2025. The tokenizer + AR sections are the technical content for V2W; the rest of the paper covers training data curation and downstream post-training examples.

**Tokenizer paper companion repo:** [github.com/NVIDIA/Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) (read-only, archived Feb 10 2025; continues at the main Cosmos repo). [TokenBench](https://github.com/NVlabs/TokenBench) for evaluation harness.

**Docs:** [docs.nvidia.com/cosmos/latest/predict1/autoregressive/](https://docs.nvidia.com/cosmos/latest/predict1/autoregressive/) — quickstart, post-training, reference.

**HuggingFace collection:** [nvidia/cosmos-predict1](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7) (12 items: AR, diffusion, tokenizers).

**Diffusers integration:** **None as of 2026-05-24.** `diffusers` upstream has a `CosmosTextToWorldPipeline` and `CosmosVideoToWorldPipeline` for **Predict2** (diffusion), but no AR Cosmos pipeline. The AR Predict1 path lives only in NVIDIA's reference repo and a NeMo integration. There is also a [vLLM tracking issue #11968](https://github.com/vllm-project/vllm/issues/11968) ("[New Model]: Cosmos-1.0-Autoregressive") — still open, no merged code. **HartsyInference will be the first non-NVIDIA pure inference path for the AR Predict1 V2W family.**

**Community ports:** none of substance. ComfyUI has nodes for Cosmos *Predict2* diffusion but not AR Predict1.

## Differences Between Implementations (5B vs 13B vs Predict2.5)

| Aspect | Cosmos-Predict1-5B-V2W | Cosmos-Predict1-13B-V2W | Cosmos-Predict2.5-2B |
|---|---|---|---|
| Architecture family | AR transformer (Llama3-shape) | AR transformer (Llama3-shape) | **Diffusion (DiT)** |
| Params | 5 B (4B base + CA) | 13 B (12B base + CA) | 2.06 B |
| Layers | 16 | 40 | (DiT — different shape) |
| Hidden | 4096 | 5120 | n/a in this scope |
| Heads / KV-heads | 32 / 8 (GQA-4) | 32 / 8 (GQA-4) | n/a |
| FFN | 14336 (SwiGLU) | 14336 (SwiGLU) | n/a |
| QK-norm | yes | yes | n/a |
| RoPE | 3D, theta≈500k (verify) | 3D, theta≈500k (verify) | n/a |
| Cross-attn every layer | yes | yes | yes (DiT cross-attn) |
| Text encoder | T5-11B (`google-t5/t5-11b`) | T5-11B | T5-XXL? (verify) |
| Tokenizer | DV8x16x16 (discrete) | DV8x16x16 (discrete) | CV8x8x8 (continuous) |
| Vocab | 64,000 | 64,000 | n/a (continuous latent) |
| Pixel res | 1024 × 640 | 1024 × 640 | 1280 × 704 (720p) |
| Frame count | 33 (out) | 33 (out) | 80 (5s @ 16fps) |
| FPS | 25 (save) / 24 (DD) | 25 / 24 | 16 |
| VRAM (no offload) | 66.2 GB | >80 GB | 32.5 GB |
| Inference @ H100 | ~73 s | ~150 s | ~228 s |
| Failure rate (image / video in) | 7% / 2% | 3% / 0% | n/a (different eval) |
| File format | `.pt` pickle | `.pt` pickle | `.safetensors` (Predict2 series) |

**Only diff between 5B and 13B at the architecture level:** the base body. The cross-attn adapter shape is the same (per-layer CA with `context_dim=1024`); 13B just has more layers + wider hidden.

**Predict2.5 vs Predict1** is a totally different model (diffusion vs AR). They are not drop-in replacements. Picking between them:
- **AR-token world model lineage / Phase 10 hand-off** → Predict1 V2W.
- **Pure diffusion video generation** → Predict2.5 (separate research doc TBD).

## Open Questions

1. **Tensor key dump.** `model.pt` is a 9.17 GB / 26.6 GB pickle; HF web viewer can't load it. The expected keys in § Data Layouts are inferred from the Python module hierarchy and need verification on a real `model.pt`. The implementer should run:
   ```python
   import torch
   sd = torch.load("model.pt", map_location="cpu", weights_only=True)
   for k, v in sd.items(): print(f"{k:80s}  {tuple(v.shape)}  {v.dtype}")
   ```
   and store the result alongside the C# `CosmosCheckpointConverter`.

2. **`fuse_qkv` toggle.** `create_inference_config` sets `inference_config.model_config.fuse_qkv = False` (line 146). The base config defaults to `False` too. Confirm cross-attn never fuses (`cross_attention_args.update({"fuse_qkv": False})` at transformer.py:77 is explicit). HartsyInference can implement unfused QKV first; fused is an optimization.

3. **Cross-attn K/V dim ambiguity.** The `Attention(attn_type="cross")` constructor takes `context_dim=1024` and `dim=4096` (or 5120). Need to verify whether the cross-attn `wk`/`wv` projection is `(context_dim → n_kv_heads*head_dim)` = `(1024 → 1024)` for 4B-base or `(1024 → 1024)` for 12B-base (both have `n_kv_heads*head_dim = 8*128 = 1024`, so it lines up — but verify against the actual tensor shapes in `model.pt`).

4. **Exact `rope_theta` value.** `BASE_CONFIG` doesn't set it. The default in the `ModelConfig` dataclass or the `RotaryPositionEmbedding` constructor (in `modules/embedding.py:155: rope_theta: Optional[float] = 10000.0`) is the fallback. The Cosmos paper uses Llama3 conventions which suggests 500,000, but the V2W config uses 3D RoPE over a small latent (T=5, H=40, W=64) so a much smaller theta (10,000) would also work. Verify locally by printing `model.rope.rope_theta` after instantiation.

5. **`apply_abs_pos_emb` shape and weight name.** Need the actual tensor shape of the abs-pos table. From `transformer.py:206-209` and `_create_abs_pos_emb_config`, it's a `PositionEmbedding3D` over `[T_latent, H_latent, W_latent] = [5, 40, 64]` at `dim=4096`. Whether it's a learned `nn.Embedding` (frozen at training end) or a sin-cos table is unverified — the existence of `pos_emb.weight` in the safetensors dump will tell us.

6. **Stop tokens.** `stop_tokens = self.model.tokenizer.stop_tokens`. For V2W with `add_special_tokens=False`, this might be an empty list / a sentinel `[64000]` outside the vocab / nothing. Generation halts at `num_token_video_latent` count regardless, so this is mostly cosmetic, but verify before omitting from the C# pipeline.

7. **Diffusion decoder DiT architecture.** The 7B "DD_FT_7Bv1_003_002" decoder is a separate DiT model. We deferred its architecture to a follow-up research doc (it's effectively a video DiT conditioned on DV tokens and T5 — substantial work). Phase 9 first pass should ship V2W with the **DV-decoder-only** path (worse quality but ~7 B less VRAM) and add the DD in a follow-up PR.

8. **Cosmos Tokenize1 vs Cosmos-0.1-Tokenizer-DV8x16x16.** The HF model card I read was for `Cosmos-0.1-Tokenizer-DV8x16x16` (the original public release). The pipeline actually loads `Cosmos-Tokenize1-DV8x16x16-720p` (the "720p" specialization, presumably retrained for the Predict1 resolutions). Architecture is the same family; weights differ. Verify which HF repo NVIDIA has gated for download (likely under `nvidia/Cosmos-Tokenizer` collection).

9. **Action conditioning hook.** `create_video2world_model` exposes `use_action_condition: bool = False`, `action_dim: int = 8`, `action_embedding_mode: Optional[str] = "mlp"`, `concat_action_to_context: bool = False`. The wgp inference path does **not** wire actions. This is the post-training hook NVIDIA used for their DROID / robotics demos. For Phase 10 HartsyInference world models, we'd add an `ActionEmbedder` that produces a (B, 1, context_dim)-or-similar embedding and concatenates onto the T5 context. Verify the exact wiring in `cosmos_predict1/autoregressive/training/model.py` before designing the abstract layer.

10. **License for the Cosmos-Tokenizer JIT files specifically.** They're tagged OML (same as the AR models), but JIT TorchScript blobs are a different distribution shape. HartsyInference must either include them as-is (with NOTICE) or recreate the encoder/decoder architecture in C# from scratch and ship a separate weights-conversion step.

11. **`tokenizer_offset`.** `VideoTokenizerConfig(tokenizer_offset=0)` because the AR vocab is only video tokens — no text vocab share. This means the embedding table is `nn.Embedding(64000, dim)` exactly. No offset arithmetic is needed in HartsyInference's token-id handling.

12. **fps mismatch (25 save vs 24 train).** Documented above. Replicate NVIDIA's behavior (save at 25) for output-bit-identical results, or expose `--output_fps` and let the user pick. First pass: save at 24 fps (the actual training frame rate) and document the deviation.

## Implementation Notes

### How this maps to HartsyInference packages

**`HartsyInference.Video`** (Phase 9 — primary home) adds:

- `Models/Cosmos/CosmosArConfig.cs` — backbone config record (5B / 13B variants).
- `Models/Cosmos/CosmosArTransformer.cs` — the Llama3-shape decoder with optional per-layer cross-attn, 3D RoPE, QK-norm, GQA, SwiGLU, RMSNorm. **Designed as a reusable backbone, not bound to V2W specifically** — Phase 10 instantiates the same class with `useActionEmbeddings: true`.
- `Models/Cosmos/CosmosArBlock.cs` — `TransformerBlock` (self-attn → optional cross-attn → SwiGLU FFN).
- `Models/Cosmos/Cosmos3DRoPE.cs` — 3D RoPE with per-axis frequency partition.
- `Models/Cosmos/CosmosAbsPos3D.cs` — additive abs-pos table over (T, H, W).
- `Models/Cosmos/Tokenizer/CosmosDvTokenizer.cs` + `CosmosDvEncoder.cs` + `CosmosDvDecoder.cs` — pure-C# port of the 2-level Haar + causal-conv-3D + FSQ encoder/decoder. Reusable for Phase 10.
- `Models/Cosmos/Tokenizer/Fsq.cs` — 6-dim FSQ (`levels`, mixed-radix basis, `index↔code`).
- `Pipelines/CosmosV2WPipeline.cs` — orchestrates T5 → DV-encode (if video prefix) → AR sample → DV-decode (or DD-decode).

**`HartsyInference.Diffusion`** (deferred follow-up):

- `Pipelines/CosmosDDPipeline.cs` — 7B latent DiT decoder. Only needed for quality mode; defer to a second PR.

**`HartsyInference.ModelHandler`** adds:

- `CheckpointConverters/CosmosArCheckpointConverter.cs` — load NVIDIA's `model.pt`-converted safetensors (after the one-off PyTorch script). Maps tensor keys to `CosmosArTransformer` parameters; demuxes the cross-attn sibling weights into a separate dict if present.
- `CheckpointConverters/CosmosDvTokenizerConverter.cs` — load DV encoder/decoder weights (after a separate conversion script that extracts state from the JIT).
- A `tools/ConvertCosmosCheckpoint/` sample Python helper (one-off, off-ship) that re-emits `model.pt` → `.safetensors` and `encoder.jit` / `decoder.jit` → `.safetensors`.

**`HartsyInference.TextEncoders`** adds (or extends if not yet present):

- `T5_11B.cs` — T5-11B encoder (24 layers, hidden 1024, 64 heads, FFN 65536, SentencePiece tokenizer). **Distinct from T5-XXL.** If the codebase already has a generic T5 encoder, this is a config selection only.

**`HartsyInference.Interactive`** (Phase 10 — *future*, this doc is the foundation):

- `Models/CosmosLike/ActionEmbedder.cs` — MLP/matrix embed of action vector → context tokens.
- `Pipelines/CosmosActionPipeline.cs` — composes `CosmosArTransformer` + `CosmosDvTokenizer` + `ActionEmbedder`. **Reuses ~95% of Phase 9 code; only the conditioning stream differs.**
- This is also where Matrix-Game-2, Oasis, GameCraft port loaders go; if those models share the DV-token vocabulary (TBD per-model), they share `CosmosDvTokenizer` too.

### Net-new backend / kernel work required

1. **3D Causal Conv** (for DV tokenizer encoder/decoder). Already covered by the Lance work (`IBackend.Conv3D`); reuse. The DV tokenizer's causal conv is slightly different from Wan's CausalConv3d (no frame-cache plumbing because we always tokenize the full 9-frame prefix at once and decode the full 33-frame output at once), so a non-streaming Conv3D suffices.
2. **Haar wavelet 3D transform** — 2-level forward + 2-level inverse. Tiny custom kernel (4× spatial reduction by averaging/differencing 2×2 blocks per channel). Implement on CPU and Vulkan; CUDA can wait.
3. **FSQ quantizer** — pure scalar ops on a 6-dim code per spatiotemporal latent. No new kernel; element-wise on tensor.
4. **3D RoPE** — per-axis frequency split. Already implemented for 2D RoPE in `RoPE.cs`; extend to a 3-axis split. Tiny addition.
5. **QK-norm** — per-head RMSNorm on Q and K before attention. Existing `RmsNorm` kernel, just applied at a different point in the attention path.
6. **GQA-4** (32 Q / 8 KV). Existing GEMM paths handle. Same factor as Qwen3-4B.
7. **Cross-attention without causal mask** — already supported (cross-attn primitives are standard). Just ensure the attention helper accepts a separate `context_mask` distinct from the causal mask.
8. **AR sampling loop with KV-cache** — **net-new for HartsyInference's video path**. Whisper has it for audio; we need a generic `IKvCache` that supports growing per-step appends and works across CPU / CUDA / Vulkan. Design this as a reusable primitive in `HartsyInference.Core/Inference/KvCache.cs`. Lance's `DenoiseKvCache` (planned for diffusion) is a different shape (fixed-prefix + recompute-noisy-slot) and should be kept separate.
9. **Top-p (nucleus) sampler** — sort logits, cumulative softmax, mask & resample. ~30 lines of C#. Top-k is even simpler. Both should live in `HartsyInference.Core/Sampling/`.
10. **T5-11B encoder forward** — if not already in the codebase, this is the standard encoder-only T5: rel-pos bias attention, GeGLU FFN, LayerNorm (T5 LayerNorm, no mean), SentencePiece tokenizer. ~1-2 weeks of work if from scratch, ~1 day if HartsyInference already has T5-XXL and just needs a smaller config.

### VRAM and viability per target GPU

| GPU | VRAM | Cosmos-Predict1-5B-V2W | Cosmos-Predict1-13B-V2W |
|---|---|---|---|
| RTX 3060 12 GB | 12 GB | **Tight.** Backbone BF16 ≈ 9.2 GB; +KV cache 12,800 × 16 layers × 1024 (kv) × 2 (k+v) × 2 B ≈ 800 MB; + T5-11B 22 GB → **infeasible at full FP16/BF16**. Need FP8/INT8 cast of backbone (~4.6 GB) AND T5 offload to CPU. AR loop possible at ~10 GB total VRAM if T5 lives off-GPU. Diffusion decoder won't fit. |  Not viable. 26.6 GB just for backbone. |
| RTX 4090 24 GB | 24 GB | **Comfortable** at BF16 with T5 offload. AR backbone 9.2 GB + KV 0.8 GB + activations ~3 GB + DV tokenizer 0.3 GB ≈ 13 GB. Diffusion decoder (14 GB) needs swap-in/swap-out. | **Tight.** Backbone 26.6 GB alone won't fit at BF16. Need FP8 cast (~13.3 GB) + T5 offload. Feasible only with offload. |
| RTX 5090 32 GB | 32 GB | Comfortable, all three stages co-resident if FP8. | Feasible at BF16 with T5 offload; or FP8 backbone + diffusion decoder co-resident. |
| H100 80 GB | 80 GB | NVIDIA's reference target. All stages co-resident at BF16 (~66 GB). | All stages co-resident at BF16 (~80+ GB). NVIDIA documents this as the un-offloaded path. |
| A100 40 GB | 40 GB | Requires partial offload. NVIDIA's recommended offload set: guardrails + T5 → 41.3 GB total. | Requires aggressive offload. |

**Recommended HartsyInference quality presets** (matching `QualityProfileApplier` conventions):

- 5B V2W `Maximum`: BF16 backbone, FP16 T5, FP16 DV tokenizer, BF16 diffusion decoder (24 GB+ card).
- 5B V2W `High`: BF16 backbone, FP16 T5 with offload, no diffusion decoder (12 GB card achievable).
- 5B V2W `Performance`: FP8 backbone, FP8 T5 with offload, no diffusion decoder, top_p=0.9 + temperature 0.6 (lower variance = fewer regenerations).
- 13B V2W `Maximum`: needs 32 GB+ at BF16; gate behind VRAM probe.
- 13B V2W `High`: FP8 backbone + offload (24 GB feasible).

### Ordering / dependencies for the build

1. **Cosmos-Tokenize1-DV8x16x16 port first.** It's standalone (encode + decode RGB ↔ tokens), testable against reference frame-by-frame with a small Python diff harness, and **is the single most reusable artifact** for Phase 10. Get this working before any AR code. Validation: encode a 9-frame test clip with both NVIDIA's JIT and our C# port, compare token IDs (must match exactly — FSQ is deterministic).
2. **T5-11B encoder.** If not already present, add it. Validate against `transformers.T5EncoderModel.from_pretrained("google-t5/t5-11b")` on a few prompts (max cos-sim should be ≥ 0.999 for fp32).
3. **CosmosArTransformer (5B, no diffusion decoder).** Build the body, validate layer-by-layer against a PyTorch dump using the standard HartsyInference debug-dump pattern (see Anima / Z-Image). The cross-attn is the only novel piece vs other Llama3-shape decoders we already have.
4. **AR sampling loop with KV-cache.** Use 5B at 1024×640 with temperature=0.6, top_p=0.9 for first validation. Match output tokens against NVIDIA's reference at seed=42 (deterministic comparison).
5. **DV-decoder-only pixel reconstruction.** Acceptable for v1 — quality is lower than DD-decoded but works. Document the quality trade-off.
6. **13B variant.** Same pipeline class with different config. Should be a config swap once 5B works.
7. **(Deferred) 7B diffusion decoder.** Separate Phase 9 follow-up.
8. **(Phase 10 follow-up) Action-conditioned variant.** Add `ActionEmbedder`, swap the T5 context for `concat([t5_embed, action_embed])` or just `action_embed`, retrain (or load NVIDIA's robotics post-trained checkpoints when they release).

### Test-skipping discipline

Following project convention (every `*GenerationTests` skips cleanly when env vars or VRAM are missing):

- `CosmosV2WPipelineTests` should require:
  - `COSMOS_PREDICT1_5B_V2W_PATH` (folder containing converted `model.safetensors` + `config.json`)
  - `COSMOS_DV_TOKENIZER_PATH` (folder containing converted `encoder.safetensors` + `decoder.safetensors`)
  - `T5_11B_PATH` (HF cache layout)
  - VRAM probe ≥ 12 GB free for FP8 / ≥ 18 GB for BF16
- `CosmosDvTokenizerTests` should require only `COSMOS_DV_TOKENIZER_PATH` (no AR). Cheap to run — should always be in PR CI when path is set.
- Diffusion-decoder tests gated separately (`COSMOS_PREDICT1_7B_DD_PATH`).

### Reuse opportunities (priority order)

- **Cosmos-Tokenize1-DV8x16x16** — reusable by **any** future Cosmos-lineage AR world model (Matrix-Game-style, robotic-action, driving). Treat as a first-class component, not a private utility of the V2W pipeline.
- **CosmosArTransformer** — reusable by Phase 10 action-conditioned models. Design `insertCrossAttention` and `contextDim` as constructor args, not hard-coded.
- **Cosmos3DRoPE** — reusable by any future video-AR or video-diffusion model that uses 3D positional encoding (Wan, LTX, OpenSora-style).
- **FSQ quantizer (6-dim)** — small but generic; could be reused by other FSQ-tokenized models (some MAGVIT variants).
- **T5-11B encoder** — fewer reuse opportunities since most newer models use T5-XXL or LLM-based encoders (Qwen, Llama). Build it lean.
- **KV-cache primitive** — reusable across **every** future AR model in HartsyInference. This is the most important infrastructure investment.

### What does *not* belong in this pipeline

- **No guardrails port.** NVIDIA ships a Llama-Guard-style text classifier + face-blur + post-gen content filter. HartsyInference doesn't replicate them — see § License. Document in `CosmosV2WPipeline.cs` XML doc that the user is responsible for content review.
- **No NeMo/Megatron parallelism.** Pure single-GPU C#; the reference's TP/CP machinery is N/A.
- **No `transformer_engine` backend hook.** N/A.
- **No FSDP / training paths.** Inference only — `TrainingModelConfig` and `act_ckpt_enabled` fields are ignored.

---

**Summary for the impatient implementer:** Cosmos-Predict1 V2W is a Llama3-shape (16L/4096 or 40L/5120) decoder with **3D RoPE**, GQA-4, SwiGLU, RMSNorm, QK-norm, plus a cross-attention block in **every** layer fed by T5-11B (`context_dim=1024`) plus an additive 3D abs-pos table. It autoregresses 64,000-vocab tokens produced by the **Cosmos-Tokenize1-DV8x16x16** tokenizer (2-level Haar wavelet → causal conv 3D → FSQ levels [8,8,8,5,5,5]). The output 12,800 tokens reshape to a [5, 40, 64] latent that the DV decoder (or a separate 7B diffusion decoder) renders as 33 frames @ 1024×640. License is permissively commercial under NVIDIA OML; only the guardrails-bypass clause matters for our distribution. Build the tokenizer first — it's the single most reusable artifact for the Phase 10 world-model lineage.
