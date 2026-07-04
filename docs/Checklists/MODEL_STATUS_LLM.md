# LLM + Text Encoders + VLMs — status

Concise status for native text generation, text/vision encoders, and vision-language models. Goal:
support every architecture Ollama / llama.cpp runs via the config-driven `GenericTransformer` spine, so a
new model is *a preset + key-mapper + (rarely) one engine knob*, never a new transformer class. Full
phased plan + per-model bring-up notes live in [LLM_MODEL_COVERAGE.md](LLM_MODEL_COVERAGE.md) and
[PHASE_12_LANGUAGE.md](PHASE_12_LANGUAGE.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

Two bars apply: **Runnable@3060** (fits 12 GB at Q4/Q8, verified coherent e2e on this box) and
**build-defer** (arch + key-map + slice tests land, e2e marked pending-hardware for >12 GB models).

> **Decode throughput (2026-07-04):** benchmarked vs a CUDA `llama-bench` baseline and optimized from
> **20-54× slower → 1.94-2.88× off llama.cpp** (Llama-3.2-1B under 2×). Fused quantized GEMV decode
> kernels (Q4_K/Q6_K/Q8_0) + quantized lm_head + split-K flash-decode attention + vectorized loads.
> Full record: [LLM_THROUGHPUT_BENCHMARK.md](LLM_THROUGHPUT_BENCHMARK.md) + [LLM_DECODE_PERF_GRIND.md](LLM_DECODE_PERF_GRIND.md).
> CUDA-graph decode (last lever for small models) foundation verified, full build deferred.

## LLM — verified end-to-end (✅, Runnable@3060)

| Family | Verified | Notes |
|---|---|---|
| **Llama 1/2/3.x** | Llama-3.2-1B Q8 | Interleaved (NORM) RoPE; Llama3 `rope_freqs` divisor fix. |
| **Mistral / TinyLlama / SmolLM / Yi** | Mistral-7B-v0.3, SmolLM2-1.7B, TinyLlama-1.1B, Yi-1.5-6B | All run via the llama-family path unchanged. |
| **Qwen2 / Qwen2.5** | 0.5B Q4_K_M + Q8 | — |
| **Qwen3** | 0.6B (incl. `<think>`) | — |
| **Gemma 2 / Gemma 3** (text) | Gemma-3-1B, Gemma-2-2B Q4_K_M | GeGLU + √d embed scale + sandwich norm + dual-RoPE; attn + final logit soft-cap. SPM-from-GGUF tokenizer. |
| **Phi-3 / Phi-3.5-mini / Phi-4-mini** | all three | Fused QKV split, LongRope, partial rotary, non-pow2 head-dim fix, gpt-4o tokenizer. |
| **StableLM-2** | 1.6B | Partial rotary + QKV bias. |
| **Granite-3** | 3.1-2B | Embedding/attention/residual/logit scalar multipliers. |
| **Cohere Command-R** (cohere2) | Command-R7B | LayerNorm + parallel residual + interleaved RoPE + NoPE global layers + logit-scale. |
| **OLMoE** | 1B-7B-0924 | MoE wired from GGUF; whole-vector Q/K norm. |
| **Granite-MoE** | granite-3.0-1b-a400m | Scalars + MoE combined. |

Qwen-MoE shared-expert path is unit-test-verified against an HF reference (no 14 GB GGUF needed).

## LLM — build-defer (🔧, wired but >12 GB)

| Model | Notes |
|---|---|
| **Mixtral 8x7B** (47B) | `llama` arch + experts, interleaved RoPE, renorm; config + mapper + stacked-expert split wired. |
| **Qwen3-MoE 30B-A3B / 235B** | `qwen3moe`, per-head Q/K norm, no shared expert; wired. |
| **DeepSeek-V2-Lite** | MLA + DeepSeek-MoE built + `MlaTests` pass; loads but OOMs the 3060 at preload. |
| **DeepSeek-V3 671B / Kimi-K2 1T** | MLA + MoE + **V3 node-limited routing (sigmoid + e_score bias + group top-k + routed_scaling) + q-LoRA query** all built & **slice-verified** (`MoeTests` group-routing vs HF `noaux_tc`, `MlaTests` q-LoRA block vs host ref). e2e >12 GB. |
| **GPT-OSS 20B / 120B** | Per-head **attention sinks** built (CPU+CUDA, PTX recompiled) & **slice-verified** (`FlashAttentionTests.Flash_Sink_*`); `gpt-oss` arch/mapper/config wired. MoE + o200k tokenizer reused. e2e 20B+ deferred. |
| **Llama-3.2-Vision-11B (mllama)** | ✅ **VERIFIED e2e on the 3060** (leafspark Q4_K_M + mmproj-F16, low-VRAM): red circle→"red", blue square→"a blue square with a white outline…". The only splice-free VLM — vision feeds gated cross-attention layers `[3,8,13,18,23,28,33,38]`. `MllamaVisionEncoder` (560px ViT, class embd, pre/post-tile + dual-gated position embeds, 32 local + 8 gated-global, intermediate-concat `[3,7,15,23,30]`→7680→`mm.0`→4096) **reference-validated cos=1.000000 on every stage** (`dump_mllama_vision_ref.py`). Key finding: Ollama's converter **pre-tanh's all gates** (and `1−tanh` for `position_embd.gate`, splitting HF's single gate into two) so the forward multiplies gates directly; MLP names clip-swapped; q/k permute is a no-op (no vision RoPE). `MllamaCrossAttentionLayer` slice-verified (`MllamaCrossAttentionTests`); `MllamaGenerator` (no token splice, `crossStates` threaded through `ForwardEmbeds` every step). Covers the whole mllama family. |

## VLMs (vision-language) — verified end-to-end (✅)

| Model | Notes |
|---|---|
| **Gemma-3-4B-vision** | SigLIP + avg-pool/RMSNorm/Linear projector. Tower corr 1.0 vs reference; e2e coherent. 2 bugs fixed (swapped SigLIP MLP names; relabel-not-transpose). |
| **SmolVLM2-2.2B** | SigLIP + idefics3 pixel-shuffle projector. Tower corr 1.0; e2e correct. |
| **LLaVA-1.5-7B** | CLIP ViT (CLS token, pre-LN, quick-GELU, penultimate layer) + MLP projector. Tower corr 1.0; e2e "a red circle … a Japanese flag". |
| **Qwen2.5-VL-3B** | Own ViT — Conv3D patch embed, 2D-RoPE, window attention (full at 7/15/23/31), SwiGLU, 2×2 merger. All stages corr 1.0; e2e correct. |
| **Qwen2.5-VL-7B** | Same `Qwen25VlEncoder` + qwen2 text as the 3B. **Verified e2e on the 3060** (unsloth Q4_K_M + mmproj-F16, low-VRAM): blue→"Blue.", red→"Red." in ~5s. Bring-up fixes: metadata-based Qwen mmproj detection (`clip.projector_type`, not filename) + a CUDA int-overflow in the cast byte-size math that OOM'd the 152k-vocab Q6_K lm_head (`count * SizeInBytes` widened to 64-bit — affected any large-vocab quantized head). |

Shared `SiglipVlmEncoder` (SigLIP + CLIP towers, 3 projectors auto-detected) + dedicated `Qwen25VlEncoder` +
`MllamaVisionEncoder`, behind `IVlmImageEncoder` / the mllama cross-attention path. Production wiring done: real
`SamplerChain`, reusable `VlmImagePreprocessor`, real PNG decode path. All four small VLM families + the two
larger ones (Qwen2.5-VL-7B, Llama-3.2-Vision-11B) are now verified e2e on the 3060.

## Embeddings — verified end-to-end (✅)

| Model | Notes |
|---|---|
| **bge-small-en-v1.5** | `BertEmbeddingModel` (BERT encoder), CLS pooling. **cosine = 1.000000** vs HF transformers. |
| **all-MiniLM-L6-v2** | `BertEmbeddingModel`, mean pooling. **cosine = 1.000000** vs HF. |
| **nomic-embed-text-v1.5** | `BertEmbeddingModel` config-driven path: rotary position + fused QKV + SwiGLU + no biases. **cosine = 1.000000** vs HF. |
| **Qwen3-Embedding-0.6B** | `DecoderEmbeddingModel` (reuses the qwen3 decoder + last-token pooling). **cosine = 1.000000** vs HF. Covers gte-Qwen2 / e5-mistral. |
| **bge-reranker-v2-m3** (reranker) | `BertEmbeddingModel.Score` — xlm-roberta encoder + cross-encoder head (cls→tanh→out_proj). Logit matches HF (relevant 4.63, irrelevant −11.04). |

Bidirectional post-norm BERT + CLS/mean pooling + L2-normalize; `bert`/`nomic-bert` registered as passthrough archs.
E2E via GGUF-vocab `BertWordPieceTokenizer`: cos(cat,kitten) 0.91 > cos(cat,car) 0.78. Quant decode verified
(Q8_0/Q5_K/Q4_K/Q3_K all >0.99 vs F32; codecs for all K-quants + legacy + IQ4_NL).

## Non-transformer architectures (Phase 7 — the new frontier)

| Family | Status | Notes |
|---|---|---|
| **Mamba-1 (SSM)** | ✅ verified | `MambaModel` — selective state-space scan + causal Conv1d, no attention. mamba-130m next-token logits **cosine = 1.000000** + argmax match vs HF. GGUF `ssm_a` is pre-baked `−exp(A_log)`. Mamba-2 / Falcon-Mamba reuse the path. |
| **RWKV-6** | ✅ verified | `RwkvModel` — WKV6 recurrence + data-dependent token-shift LoRA + GroupNorm. C# runs at **cosine 1.0** (argmax 281) vs the validated Python ref (= official `rwkv` package). No-copy `Reshape` relabel views to fit the 1.6B model in host RAM. RWKV-7 = near-variant. |
| **Hybrids** (Jamba/Zamba2/Granite-4) | ⬜ planned | Mamba + attention + MoE interleave (7c). |
| **Encoder-decoder** (T5/FLAN-T5) | ✅ verified | `T5Model` — full seq2seq (rel-pos bias, no 1/√d scaling, cross-attn, GeGLU). flan-t5-small encoder + decoder **cosine = 1.0** vs HF; e2e "Das Haus ist schön." BART is a near-variant. |

## Text / vision encoders

| Encoder | Status | Notes |
|---|---|---|
| **GenericTransformer** (Qwen2/Qwen3/Llama-3) | ✅ | Parity tests; backs the text encoders + native LLM. Llama-3 RoPE NTK-by-parts validated. |
| **T5 / UMT5 / Pile-T5 (AuraFlow) / BERT / SigLIP / Qwen3-VL vision tower** | 🔬 | Diff tests present (`T5EncoderDiff`, `BertModel`, `Siglip`, `Qwen3VlVisionTower`); Pile-T5 == UMT5 (per-layer rel-attn-bias). |
| **Native LLM package** (Phase 12) | 🚧 | One config-driven transformer backing both LLM + text encoders; CUDA-first quant; GPU-resident decode is the gating blocker. |

## Remaining for FULL LLM support — see [LLM_MODEL_COVERAGE.md § Completion plan](LLM_MODEL_COVERAGE.md) (Phases 6–9)

Transformer-family LLMs (dense · MoE · MLA · VLM · embeddings) are effectively complete for common/SOTA models.
What's left, by phase:

- **Phase 6 — cheap reuse wins (verifiable@3060):** decoder-based embeddings (gte-Qwen2 / e5-mistral, last-token/mean
  pooling on `GenericTransformer`); more BERT encoders (nomic-bert rotary, jina ALiBi) + rerankers; exotic IQ-quant codecs.
- **Phase 7 — NEW architecture families (the real frontier, new core code):** Mamba/Mamba-2 (SSM selective scan),
  RWKV v6/v7 (WKV recurrence), hybrids (Jamba/Zamba2/Granite-4), encoder-decoder seq2seq (T5/BART). These are the only
  families the engine fundamentally cannot run yet; all have small variants that fit the 3060.
- **Phase 8 — build-defer code gaps (the new primitives are now built + slice-verified; full e2e still needs >12 GB):**
  DeepSeek-V3/Kimi sigmoid+group-routing+q-LoRA ✅ slice-verified; GPT-OSS attention sinks ✅ slice-verified;
  Llama-3.2-Vision gated cross-attention layer ✅ slice-verified (vision encoder + e2e still deferred); remaining e2e of
  Mixtral/Qwen3-MoE/Qwen2.5-VL-7B/DeepSeek-V2-Lite needs a bigger GPU.
- **Phase 9 — serving/quality (optional):** batch>1 / paged KV, speculative decode, long-context stress, CPU GGUF
  parity tests + the pre-existing `RoundTrip_SimpleF32` test fix, tool-calling/grammar-constrained decode.
