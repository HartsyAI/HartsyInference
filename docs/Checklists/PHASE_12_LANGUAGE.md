# Phase 12 — Language (Native LLM Inference)

> **Goal:** First-class LLM text generation via one generic config-driven transformer that also backs the
> existing text encoders. Replaces the planned dotLLM dependency (GPLv3, clean-room our own).
> **Package:** HartsyInference.LLM (Core, ModelHandler, Tokenizers, Cpu/Cuda/Vulkan). Not pulled in by
> the `HartsyInference` meta package; consumers add it explicitly (`dotnet add package HartsyInference.LLM`).
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

## M0 — Decode performance spike (GATING BLOCKER) — ✅ GO (2026-06-22)

- [x] GPU-resident decode built: `Qwen2ResidentDecoder` (100% `IBackend` ops) + `DeviceKvCache`
      (Concat-grown) + new `IBackend.RepeatKvHeads` (`lm_f32.ptx`) + `ApplyRopeSingle` (GQA RoPE)
- [x] Reused the Ideogram4 residency machinery (activation cache + stream pools + `GetD2hSyncCount`)
- [x] CPU parity tests green (`Qwen2ResidentDecoderTests`): resident == `Qwen2Model` within 2e-3, argmax agrees
- [x] Harness `samples/HartsyInference.TextGen.Cli` (3-way matrix, D2H instrumentation)
- [x] **Measured on RTX 3060 (Qwen2.5-0.5B-Instruct, greedy):**
      - resident: **72.6 tok/s decode, 77 ms prefill, 0 transformer D2H syncs**, coherent text, stops at EOS
      - cuda-ref (non-resident `Qwen2Model`): 0.59 tok/s, 2170 ms prefill, 2208 D2H syncs
      - cpu-ref: 0.64 tok/s
      - **~123× faster decode than the non-resident path; all three emit identical token ids (correct)**
      - VibeVoice-1.5B run also confirmed residency (34.9 tok/s, 0 syncs, identical ids vs ref)
- [x] **GO** — residency works, decode is compute-bound, far past the 10× gate. Proceed to M1.
- Note: residency hinges on no `DataPointer` reads of activations + preloaded weights (the existing
  `Qwen2Model` re-transposes/re-allocates weights per call via `WhisperOps.ProjectLinear`, defeating the
  weight cache — that is the gap M1 closes by moving the generic transformer onto this resident pattern).

## M1 — Generic transformer + F32 decode (new `HartsyInference.LLM` package) — ✅ DONE (2026-06-22)

- [x] New package `HartsyInference.LLM` (deps Core/ModelHandler/Tokenizers) + `HartsyInference.LLM.Tests`, both in slnx
- [x] `TransformerConfig` (record) + presets `Qwen2_5_0_5B`/`Qwen2_5_1_5B`/`Qwen3_0_6B` — matrix axes: AttentionBias, QkNorm, decoupled HeadDim, TieWordEmbeddings
- [x] `GenericTransformer` (resident forward + inner `Layer`): generalizes the M0 decoder with per-head q/k RMSNorm + decoupled head_dim + optional QKV bias; 100% `IBackend` ops
- [x] `KvCache` (device-resident, `Concat`-grown) in the LLM package (M0 `DeviceKvCache` moved here)
- [x] `SamplerChain` + `ISamplerStep` (Temperature/RepetitionPenalty/TopK/TopP/MinP/Greedy) + `SamplingOptions` (clean-room)
- [x] `IChatTemplate` + `ChatMlTemplate` + `ChatTemplateRegistry` (ChatML; emits token ids; golden-matches `Qwen2Tokenizer.EncodeChat`)
- [x] `TextGenerationPipeline` + `GenerationRequest`/`GenerationResult`; `Qwen2Tokenizer.Decode` added
- [x] Deleted M0 spike artifacts (`Qwen2ResidentDecoder`, Audio `DeviceKvCache`, `Qwen2ResidentDecoderTests`); repointed `TextGen.Cli` to the LLM pipeline
- [x] Tests green (9/9 on net8.0 + net10.0): Qwen2 parity vs `Qwen2Model` (~2e-3), Qwen3 structural, sampler, template golden, GQA layout. Full solution builds clean.
- [x] **Validated on RTX 3060 (greedy):** Qwen2.5-0.5B-Instruct — coherent, identical token ids to M0; **Qwen3-0.6B** — coherent text, proving the q/k-norm + decoupled-head-dim + **split-half RoPE** path. Both GPU-resident (only the per-token logits read crosses H2D/D2H).
- Note: repo `Qwen3Model` is the **TTS** backbone (interleaved RoPE), so it is NOT a parity oracle for HF Qwen3 **text** (split-half) — Qwen3 text correctness is proved by the e2e run, not CPU parity.
- Deferred to later milestones: `Rblback`/fixed-buffer KV (M2), migrating audio models onto the core + encoder unification (M4), `LanguagePipelineBase`. Cosmetic: `Qwen2Tokenizer.Decode` leaks GPT-2 byte-level space marker `Ġ` (tokens correct; byte-level decode is a Tokenizers-package fix).

## M2 — Quantized (GGUF) inference (CUDA first) — ✅ DONE (2026-06-22)

Most of the GGUF/quant stack already existed (loader, key-remap, all quant DTypes, CPU codecs, GPU dequant
kernels Q8_0/Q4_K/Q5_K/Q6_K, quantized `Linear`). M2 added the wiring + a low-VRAM path:
- [x] **Wave A — GGUF LLMs run.** `GgufConfigFactory.FromGguf` (metadata→`TransformerConfig`; infers
      bias/qk-norm/tie + vocab); `GenericTransformer.LoadWeights` keeps proj weights quantized (only embed +
      norms forced to F32); `GgufLanguageModel.Load` (owns the GGUF mmap; relabels GGUF `[in,out]`→`[out,in]`;
      F32 fallback for non-GPU quant types e.g. Q5_0 in a Q4_K_M mix); harness `arch=gguf`.
      Validated on RTX 3060: Qwen2.5-0.5B **Q8_0 and Q4_K_M** → coherent text, EOS, resident (~52 tok/s).
- [x] **Wave B — low-VRAM path.** `IBackend.QuantizedMatMul` (CUDA override = `LinearImpl` with
      `cacheWeightCast:false` → quant weight stays resident-compressed, F16 dequant is transient per call;
      default throws). `TransformerConfig.LowVramQuant` opt-in routes quant projections to it (default = fast
      F16-cached `Linear`). Parity: `CudaQuantizedMatMulTests` (Q8_0/Q4_K/Q6_K == `Linear`) + e2e byte-identical
      to the fast path. Tradeoff measured: fast 54 tok/s vs low-VRAM 16 tok/s (per-token re-dequant).
- [x] GGUF→`TransformerConfig` (factory, no separate converter needed — `LlamaKeyMapper` already maps Qwen).
- [x] Full solution builds; LLM (9) + CUDA (53) tests green.
- **Deviation from plan:** Wave B is "transient dequant + cuBLAS, no F16 cache" — it DOES materialize F16
      transiently per op (not a true in-kernel fused GEMV). It delivers the resident low-VRAM footprint
      (weights stay quantized) but is slower (per-token re-dequant) and the F16 transient is not zero. A true
      fused in-kernel dequant-GEMV (faster decode + zero transient) is the documented follow-up.
- Deferred: true fused GEMV kernels; quantized embed/lm_head GPU gather (embed dequants to F32 on load, which
      dominates small-model VRAM); Vulkan quant path; logits-vs-llama.cpp numeric check (parity vs our F16
      path used instead).

## M3 — Architecture coverage

- [ ] Llama-3.x preset + `LlamaCheckpointConverter`
- [ ] Mistral preset (sliding window) + converter
- [ ] Qwen2.5 preset + `QwenCheckpointConverter`
- [ ] Flash-attention (causal/GQA/cache) CUDA + Vulkan; CPU keeps tiled SDPA
- [ ] RoPE scaling kernels (linear/NTK/YaRN/llama3)

## M4 — Audio decoder unification (kill duplication) — ✅ DONE (2026-06-22)

- [x] `IKvCache` abstraction (LLM); device `KvCache` + host `StreamingKvCache` both implement it (`AppendStep`/`KeyPrefix`/`ValuePrefix`); `GenericTransformer` is cache-agnostic
- [x] `GenericTransformer` grew: `LoadWeightsHeadless`, `ForwardEmbeds(applyFinalNorm/startLayer/endLayer)`, `RopeStyle` (SplitHalf | Interleaved) + `IBackend.ApplyRopeInterleaved` (CPU default)
- [x] Audio→LLM package edge (no cycles)
- [x] Audio `Qwen2Model` + `Qwen3Model` reimplemented as **thin wrappers** over `GenericTransformer` (Qwen2 = split-half/bias; Qwen3 = interleaved/qk-norm/decoupled-headDim); public API unchanged so the ~10 consumers (VibeVoice, CSM, Kyutai, Spark, YuE, Fish-Speech, PocketTTS, Chatterbox, CosyVoice, Qwen3-TTS talker+MTP) are untouched
- [x] Deleted duplicate internals: `Qwen2DecoderLayer/Attention/Mlp/Qwen2RotaryEmbedding` + Qwen3 `Layer`/`HeadRmsNorm`. `GenericTransformer` is now the single Qwen-family decoder implementation
- [x] **Zero regression:** full Audio suite 357 pass / 5 gated-skip, LLM 9 pass, full solution builds; e2e Qwen2.5-0.5B + Qwen3-0.6B emit byte-identical token ids vs pre-migration
- Note: dual-embedding/dual-head/MTP stay consumer-side (they wrap the headless decoder). Qwen3 3D mRoPE remains unimplemented (was never implemented — wrapper preserves the prior 1D interleaved behavior).
- Deferred (owner decision): T5/CLIP/LlamaStyleEncoder stay standalone (structurally different / already config-driven). Resident GPU interleaved-rope kernel is a follow-up (CPU default keeps Qwen3 audio correct, matching prior behavior).

## M5 — MoE + serving

- [ ] `MoeLayer` (router + expert dispatch); Mixtral / Qwen-MoE presets
- [ ] `PagedKvCache` + continuous batching
- [x] Server `/v1/chat/completions` (+ streaming SSE), OpenAI-compatible — shipped in `HartsyInference.API`
      (`CompatEndpoints.cs`), a thin wrapper over `IInferenceEngine.Text`. Native `/v1/native/text`(+`/stream`)
      also exists for the fuller request/result contract chat's OpenAI schema can't express. Text generation
      remains consumable in-process via `HartsyInference.LLM` too, and end users are still pointed at the
      SwarmUI backend extension as the recommended surface.

## M6 — VLM (optional)

- [ ] Reuse Vision (CLIP/DINOv2/SigLIP) + generic transformer; Qwen-VL lineage

## Testing & Review

- [ ] SIMD-vs-scalar + CPU-vs-CUDA parity on new kernels
- [ ] Known-value logits vs HF/llama.cpp per family
- [ ] Chat-template golden tests vs HF `apply_chat_template`
- [ ] E2E greedy-decode determinism per preset (small in CI; large gated)
- [ ] Code review; deviations documented; merge
