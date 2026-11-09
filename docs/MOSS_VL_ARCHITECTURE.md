# MOSS-VL-Realtime

OpenMOSS's open-weights vision-language model for realtime video understanding (arXiv 2608.15045). Realtime means the
model reads frames as they arrive, emits text incrementally, and can stay silent with `<|silence|>` when nothing needs a
reply. It is **not** the full-duplex speech system MOSS-Live (API-only, no weights). ASR, TTS and memory come from the
demo layer, not the model.

## Sources (snapshot 2026-11-08)

- Weights: `OpenMOSS-Team/MOSS-VL-Realtime-SGLANG` (sha `3d6407cc`, not gated, Apache-2.0). Five BF16 shards,
  22,672,742,416 bytes in total, 895 tensors. Quantized builds (all Apache-2.0, not gated; checked 2026-11-08):
  - `OpenMOSS-Team/MOSS-VL-Realtime-NF4`: bitsandbytes NF4, three shards, 11,841,998,136 bytes (about 11.8 GB). Layers 0–3 and
    44–47, cross-attention, vision and `lm_head` stay BF16.
  - `OpenMOSS-Team/MOSS-VL-Realtime-FP8`: compressed-tensors FP8, four shards, 15,729,921,752 bytes (about 15.7 GB). Quantizes
    the self-attention and MLP linears of the 36 self-attention layers (per-channel FP8 weights, dynamic per-token FP8
    activations). Cross-attention, vision, merger and `lm_head` stay BF16.
  - `SkyContact/MOSS-VL-Realtime-GGUF`: third-party GGUF `Moss-VL-Realtime-q4k.gguf`, 8,423,234,016 bytes (about 8.4 GB). Its
    tags name the NF4 repo as base. The first 8 MiB of the header show text blocks with cross-attention, the vision blocks,
    the four-stream merger (`mm.ln0`–`ln3`), `mm.separator` and `general.architecture = moss_vl`. Provenance and per-tensor
    quant types are not checked yet.
- Code: `modeling_moss_vl.py` (4,773 lines, transformers 5.12.1 style), `configuration_moss_vl.py`, `processing_moss_vl.py`,
  `video_processing_moss_vl.py`, `inference.md`. Downloaded to `Models/LLM/moss-vl-realtime/` (gitignored).
- Hashes and file sizes come from the Hub's LFS listing. The shard sha256 values match the listing.

## Verified facts (from `config.json`, the index and the shard headers)

| item | value |
|---|---|
| text decoder | 48 layers, hidden 4096, 32 heads, 8 KV heads, head_dim 128, FFN 12288, vocab 151936 |
| cross-attention layers | `[2, 6, 10, …, 46]`, 12 of them; the other 36 are self-attention |
| RoPE | theta 5,000,000; interleaved M-RoPE, sections `[24, 20, 20]` |
| norms and bias | RMSNorm eps 1e-6; per-head Q/K RMSNorm on every layer; no attention bias |
| embeddings | `tie_word_embeddings: false`; `lm_head.weight` separate |
| ViT | 27 blocks, hidden 1152, FFN 4304, 16 heads, patch 16, merge 2, temporal patch 1, 2304 learned positions |
| deepstack | taps at blocks 8, 16, 24; merger input = 4 × 4608 = 18432 (last block plus three taps) |
| merger out | 4096 (the decoder width); cross-attention reads these features |
| special tokens | image 151655, video 151656, vision start 151652, vision end 151653, separator is a learned parameter |
| dtype | all 895 tensors BF16 |
| checkpoint layout | HF transformers names (`model.language_model.*`, `model.visual.*`) |

Checkpoint keys match the config exactly: 552 text-decoder keys (396 self-attention, 156 cross-attention), 324 ViT keys,
12 merger keys, 2 patch-embed keys, 1 position key, 4 embedding/head keys, plus `model.separator_token`.

Cross-attention layers store their two gates as **layer-level** names (`layers.N.cross_attn_attn_gate`, not `.weight`).
These are the same names `MllamaCrossAttentionLayer.LoadWeights` already reads.

## Architecture (from `modeling_moss_vl.py`)

- **Vision.** Qwen3-VL-style ViT: conv patch embed, learned absolute positions interpolated to the grid, 2-D RoPE,
  LayerNorm blocks with fused `attn.qkv`, GELU-tanh MLP. The merger LayerNorms each of the 4 streams separately,
  concatenates them, then runs a 2-layer MLP to 4096.
- **Text self-attention.** GQA with per-head Q/K RMSNorm, then interleaved M-RoPE. Causal.
- **Cross-attention layer.** `h = x + tanh(attn_gate) · o(attn(q_norm(q_proj(rms(x))) rope, k_norm(k_proj(v)) rope, v_proj(v)))`,
  `out = h + tanh(mlp_gate) · mlp(rms(h))`. Vision keys and values are not causal. Unlike Mllama, the **query is
  RoPE-rotated** too (`query_position_embeddings`), and the keys carry the vision RoPE once, when they are first
  computed. The cache holds rotated keys.
- **Realtime.** A session queues frames and text. Each cycle drains the queues, encodes new frames, appends their
  features to the cross-attention cache, then decodes. `<|silence|>` as the sampled token ends the cycle and waits for
  new input. A separator embedding is inserted between segments. Realtime M-RoPE is computed per segment
  (`_compute_realtime_mrope_for_segment`).

## Reuse map

| need | existing owner | fit |
|---|---|---|
| text decoder stack, GQA, RMSNorm, SwiGLU, KV cache | `LLM/Transformer/GenericTransformer` with `TransformerConfig` | direct. Config reader is `Models/MossVl/MossVlConfig` (new). |
| gated cross-attention layer | `LLM/Multimodal/MllamaCrossAttentionLayer` | same weights and gates; **gap:** no query/key RoPE (it already has Q/K RMSNorm) |
| cross-attention plumbing (`crossStates`, stage handoff) | `GenericTransformer.ForwardEmbeds*` | direct |
| interleaved M-RoPE tables | `Core/Rope` (`MRopePositions`, `MRopePairAxes`, `MRopeTable`) | direct for Qwen-VL; check it matches `[24,20,20]` interleaved |
| Qwen3-VL ViT | `Diffusion/Models/TextEncoders/Qwen3VlVisionEncoder` (safetensors), `LLM/Multimodal/Qwen3VlEncoder` (GGUF) | **gap:** neither has the deepstack merger (input 4× wide). Both are missing it per the code comments |
| merger | none | new, but small |
| tokenizer | `ModelAssets/Tokenizers` (`tokenizer.json`) | direct |
| HF loader dispatch | `Engine/Services/HfTextDirectoryLoader` (`RequireSupported`) | add MOSS-VL to the Text path only after a text-only path exists; multimodal goes through the vision path |

## Gaps and open items

1. Cross-attention query/key RoPE and deepstack merger are not in the engine. Both are needed for the real model.
2. Realtime session loop is not implemented. Offline prefill and decode come first.
3. Weight format is a decision, not a fallback. BF16 (22.7 GB) and the F32 widening (about 42 GB) do not fit the RTX 3060's
   12 GB, and F32 does not fit 40 GB of RAM. The loader must accept every published format and keep weights in their stored
   precision, dequantizing per matmul or per layer, not widening the whole model at load. Target for 12 GB VRAM: the GGUF
   Q4_K (8.4 GB). The official NF4 (11.8 GB) and FP8 (15.7 GB) are further formats; FP8 also exceeds 12 GB.
   Existing owners: `HfQuantFlavorDetector` and `HfQuantCheckpoint` (compressed-tensors, bitsandbytes), `Nf4Codec`,
   `Fp8BlockF32Codec`, and `GgufLoader` / `GgufModelLoader` with `GgufCodecRegistry`. None is wired to the MOSS-VL text
   decoder yet. The NF4 companion layout and the GGUF tensor types are unverified.
4. Python reference: Python 3.12 is installed and on PATH. Reference dumps still need `torch` and `transformers==5.12.1`.
   Loading the BF16 shards needs roughly their 22.7 GB in RAM (estimate); this machine has 40 GB.
5. The GGUF-side deepstack note in `Qwen3VlEncoder` says a future mmproj with deepstack tensors would need it. This model
   has them, so that path is a useful reference but not a drop-in.

## Status

- Config reader: `src/HartsyInference.LLM/Models/MossVl/MossVlConfig.cs`. Unit tests on the real config.
- Layout check: `tests/HartsyInference.LLM.Tests/MossVlCheckpointTests.cs` (`Category=Integration`, `HARTSY_MOSS_VL_INDEX`).
  Passes on the real index: 895 keys, exact match with the config-derived set.
- Vision tower: `src/HartsyInference.Diffusion/Models/TextEncoders/MossVlVisionTower.cs`, which reuses
  `Qwen3VlVisionEncoder` and adds the `UnifiedDeepstackMerger` mode (per-stream LayerNorm, concatenate, one MLP).
  Parity against `tests/python-reference/dump_moss_vl_vision.py` on the real weights, grid 1×16×24:
  max|diff| 7.5e-4 on max|ref| 279, relative 2.7e-6 (`MossVlVisionParityTests`, `HARTSY_MOSS_VL_INDEX` +
  `HARTSY_MOSS_VL_REF`). The reference is a numpy transcription of `modeling_moss_vl.py`, not the official code executed.
- Cross-attention RoPE: `MllamaCrossAttentionLayer.Forward` takes an optional `MultimodalRope` (text cos/sin for the queries,
  vision cos/sin for the keys, and a per-row mask). `GenericTransformer.ForwardEmbeds` takes the same tables; null keeps the mllama path.
- Text model: `src/HartsyInference.LLM/Models/MossVl/MossVlTextDecoder.cs` (`GenericTransformer` with `lm_head`, and the separator) and
  `MossVlPositions.cs` (single-image positions per `compute_position_ids` and `compute_vision_position_ids`, interleaved M-RoPE).
  Parity: `tests/python-reference/dump_moss_vl_text.py` against `MossVlTextParityTests` (layers 0–2, real weights). The cross
  layer's delta matches at relative 5e-7. The checkpoint's gates are about 1e-2, so the full hidden state alone would not catch a
  cross-attention error; the test checks the delta.
- Logits and decode: `MossVlDecodeSession` (`src/HartsyInference.LLM/Models/MossVl/MossVlGenerator.cs`) runs prefill and KV-cached
  steps, with the decode token at the prompt's next position and the same vision features read by the cross layers. Parity
  against the numpy recompute: relative ~1e-6, same greedy token. Reuses `GenericTransformer.ProjectLogits`, `SamplerChain` and
  `VlmDecodeOps`, and `HfTokenizerJson` for `tokenizer.json`.
- Not built: multi-image cross-attention masks (refused), image preprocessing and vision-tower wiring (Engine composition),
  realtime loop, CUDA kernels. The full F32 text model is about 42 GB and does not fit this machine's RAM.
- GGUF text source: `MossVlTextDecoder.OpenGguf` with `MossVlKeyMapper` (`moss_vl`). Q4_K and Q6_K projections stay packed
  (`keepQuantized`). CPU parity on layers 0–2 against the BF16 reference: relative 5.0e-3, cross-layer delta 6.5e-3. Vulkan
  parity matches the CPU result. CUDA needs cuBLAS 13, which is not installed here, so its test cannot run yet.
- Test generation (`MossVlGenerateTests`, Vulkan, all 48 layers, packed quant): the reply was "A static, full-frame image of a
  television test pattern with multicolored vertical bars and a grainy texture." The image features come from the numpy vision
  dump, not a fresh run of the vision tower. The reply has not been checked against that image beyond this one run.
