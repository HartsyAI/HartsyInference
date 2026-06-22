# Phase 12 — Language (Native LLM Inference)

> **Goal:** First-class LLM text generation via one generic config-driven transformer that also backs the
> existing text encoders. Replaces the planned dotLLM dependency (GPLv3, clean-room our own).
> **Package:** HartsyInference.Language (Core, ModelHandler, Tokenizers, Cpu/Cuda/Vulkan).
> **Full design:** [`docs/Design/LLM_LANGUAGE_PACKAGE.md`](../Design/LLM_LANGUAGE_PACKAGE.md).
> **Initial coverage:** Llama-3.x, Qwen2.5/Qwen3, Mistral (dense). Quant: CUDA first.

---

## 0. Research

- [ ] `GENERIC_TRANSFORMER.md` — config matrix; decoder/encoder/enc-dec unification
- [ ] `LLM_DECODE_LOOP.md` — GPU-resident prefill/decode, fused per-layer dispatch, scratch plan
- [ ] `GGUF_QUANTIZED_MATMUL.md` — Q4_K_M/Q6_K/Q8_0 dequant + quantized GEMV/GEMM, R4 repacking
- [ ] `LLM_ATTENTION.md` — flash-attn (causal/GQA/cache), sliding-window, sinks, paged KV
- [ ] `ROPE_SCALING.md` — linear / NTK / YaRN / llama3 / 2D-3D mRoPE
- [ ] `LLM_SAMPLING.md` — sampler chain + penalties + (future) constrained decode
- [ ] `CHAT_TEMPLATES.md` — per-family templating + tool/function-call formatting
- [ ] `MOE_INFERENCE.md` — router + expert dispatch *(deferred to M5)*
- [x] Reuse: DOTLLM_ARCHITECTURE, FLASH_ATTENTION, GGUF_FORMAT, T5_ARCHITECTURE, QWEN3_TTS_ARCHITECTURE

## M0 — Decode performance spike (GATING BLOCKER)

- [ ] Prototype GPU-resident, fused-per-layer decode on existing `Qwen3Model` (CUDA, 3060)
- [ ] Measure tokens/sec vs per-op auto-transfer baseline; confirm viability
- [ ] Decide overlap with diffusion GPU-residency work (Ideogram4 pattern)
- [ ] **Go/no-go gate for M1+**

## M1 — Generic transformer + F16/F32 decode

- [ ] `TransformerConfig` + presets (Qwen2/Qwen3 first)
- [ ] `GenericTransformer` + `DecoderLayer` + `Attention` + `Mlp` + `RotaryEmbedding` + `TransformerForwardState`
- [ ] Port/refactor `Audio/Models/LanguageModels/{Qwen2,Qwen3}` into the core (Audio consumes presets)
- [ ] Promote `StreamingKvCache` → `KvCache` (TensorRef hot path, `Rollback`)
- [ ] `SamplerChain` (promote `NucleusSampler`; add penalties + greedy argmax)
- [ ] `IChatTemplate` registry (extract from tokenizer classes)
- [ ] `TextGenerationPipeline` + `LanguagePipelineBase` + request/result types
- [ ] Validate text-out vs HF on a small model (greedy, fixed seed)

## M2 — Quantized inference (CUDA first)

- [ ] `DequantMatMul` Q8_0 → Q4_K_M → Q6_K (build on `native/cuda/kernels/dequant_q8`/`dequant_q4k`)
- [ ] Quantized GEMV (decode) + GEMM (prefill); R4 weight repacking at load
- [ ] GGUF → `TransformerConfig` + converter path through ModelHandler
- [ ] Validate logits vs llama.cpp (clean-room, reference only)
- [ ] Vulkan + CPU dequant-matmul fallbacks

## M3 — Architecture coverage

- [ ] Llama-3.x preset + `LlamaCheckpointConverter`
- [ ] Mistral preset (sliding window) + converter
- [ ] Qwen2.5 preset + `QwenCheckpointConverter`
- [ ] Flash-attention (causal/GQA/cache) CUDA + Vulkan; CPU keeps tiled SDPA
- [ ] RoPE scaling kernels (linear/NTK/YaRN/llama3)

## M4 — Encoder unification (kill duplication)

- [ ] Re-target Diffusion `T5TextEncoder` / `ClipTextEncoder` / `LlamaStyleEncoder` onto generic core (encoder mode)
- [ ] Re-target Audio Qwen3-TTS talker / YuE / Higgs onto generic core
- [ ] Delete duplicated transformer/encoder code; re-run affected diffusion + audio tests

## M5 — MoE + serving

- [ ] `MoeLayer` (router + expert dispatch); Mixtral / Qwen-MoE presets
- [ ] `PagedKvCache` + continuous batching
- [ ] Server `/v1/chat/completions` (+ streaming SSE), OpenAI-compatible

## M6 — VLM (optional)

- [ ] Reuse Vision (CLIP/DINOv2/SigLIP) + generic transformer; Qwen-VL lineage

## Testing & Review

- [ ] SIMD-vs-scalar + CPU-vs-CUDA parity on new kernels
- [ ] Known-value logits vs HF/llama.cpp per family
- [ ] Chat-template golden tests vs HF `apply_chat_template`
- [ ] E2E greedy-decode determinism per preset (small in CI; large gated)
- [ ] Code review; deviations documented; merge
