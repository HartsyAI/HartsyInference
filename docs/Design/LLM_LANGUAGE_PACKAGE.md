# Implementation Plan: HartsyInference.Language (Native LLM Inference)

> Back to [Core Design](CORE_DESIGN.md). Decision record + file-by-file architecture for adding
> first-class LLM text generation to HartsyInference, replacing the planned dotLLM dependency.
> Status: **design / not yet implemented**. Authored 2026-06-21.

## 0. Decision Record

- **Build our own; do not depend on dotLLM.** dotLLM is GPLv3 (linking it relicenses the engine + the
  SwarmUI extension). We already own ~80% of the pieces, and dotLLM's "IBackend = memory only, call static
  kernels directly" model contradicts HartsyInference's deliberate "IBackend = op-dispatch" choice. We mine
  dotLLM's *documented patterns* clean-room (patterns aren't copyrightable); we copy no code.
- **One generic config-driven transformer**, not one class per family. Modern decoder LLMs are ~90%
  identical; variation is a config + feature-flag matrix. The same core runs **decoder LLMs (causal +
  KV-cache)** and **text encoders (bidirectional, no cache)** and **encoder-decoder (T5)** — this is how
  text encoders "reuse the same backend logic" with zero duplication.
- **Initial coverage:** Llama-3.x, Qwen2.5/Qwen3, Mistral (dense). Gemma2/Phi/MoE/VLM are later milestones.
- **Quantized matmul: CUDA first** (3060 target), then Vulkan, then CPU. `native/cuda/kernels/` already has
  `dequant_q8` / `dequant_q4k` stubs to build on.
- **Refactor in:** the existing `HartsyInference.Audio/Models/LanguageModels/{Qwen2,Qwen3}` decoders and
  `StreamingKvCache` move into the shared core; Audio (Qwen3-TTS, YuE, Higgs, Bark, Moshi) and Diffusion
  (`LlamaStyleEncoder`, `T5TextEncoder`, `ClipTextEncoder`) consume it via presets.

## Prerequisites

- [x] Research: `DOTLLM_ARCHITECTURE.md`, `FLASH_ATTENTION.md`, `GGUF_FORMAT.md`, `T5_ARCHITECTURE.md`,
      `QWEN3_TTS_ARCHITECTURE.md` (Qwen3 backbone), existing tokenizers.
- [ ] Research to write (see §6): `GENERIC_TRANSFORMER.md`, `LLM_DECODE_LOOP.md`,
      `GGUF_QUANTIZED_MATMUL.md`, `LLM_ATTENTION.md`, `ROPE_SCALING.md`, `LLM_SAMPLING.md`,
      `CHAT_TEMPLATES.md` (+ `MOE_INFERENCE.md` deferred).
- [ ] **Blocker (must de-risk before Milestone 1):** GPU-resident, fused-per-layer decode path on CUDA.
      The current backend auto-transfers activations H2D/D2H per op (~33× slow for diffusion; fatal for
      hundreds of sequential single-token steps). See §5 Risk 1.

---

## 1. Package & Dependency Graph

New package **`HartsyInference.Language`**.

```
Core
 ├─ ModelHandler ─┐
 ├─ Tokenizers ───┤
 ├─ Cpu / Cuda / Vulkan
 └─────────────── HartsyInference.Language   (NEW)
                       │  (consumed by)
        Audio ── Diffusion ── Server (/v1/chat/completions)
```

- **Depends on:** Core, ModelHandler, Tokenizers, Cpu, Cuda, Vulkan. **Not** Diffusion/Audio/Vision/Video.
- **Consumed by:** Server (chat endpoint); Audio + Diffusion re-target their transformer/encoder code onto
  it over Milestones 1 & 4.
- `tests/HartsyInference.Language.Tests/` and `samples/TextGeneration/` added alongside.
- Minimum install (NVIDIA chat): `HartsyInference.Language` + `Cuda` + `ModelHandler` + `Tokenizers`.

---

## 2. API Surface

### 2.1 The generic transformer (the spine)

```csharp
// Transformer/TransformerConfig.cs  — record, required/init props, static presets
public sealed record TransformerConfig
{
    public required int HiddenSize { get; init; }
    public required int NumLayers { get; init; }
    public required int NumHeads { get; init; }
    public required int NumKvHeads { get; init; }      // == NumHeads (MHA), 1 (MQA), else GQA
    public required int HeadDim { get; init; }          // decoupled from HiddenSize/NumHeads (Qwen3/Gemma)
    public required int IntermediateSize { get; init; }
    public required int VocabSize { get; init; }

    public NormKind Norm { get; init; } = NormKind.RmsNorm;
    public NormPlacement NormPlacement { get; init; } = NormPlacement.PreNorm; // + Gemma sandwich
    public bool QkNorm { get; init; }                   // Qwen3 per-head q/k RMSNorm pre-RoPE
    public bool AttentionBias { get; init; }            // Qwen2 QKV bias; Llama/Mistral false
    public MlpKind Mlp { get; init; } = MlpKind.SwiGlu; // SwiGLU / GeGLU
    public RopeConfig Rope { get; init; } = RopeConfig.Default; // theta, scaling: none/linear/ntk/yarn/llama3
    public int? SlidingWindow { get; init; }            // Mistral / Gemma2 local attention
    public float? AttnLogitSoftcap { get; init; }       // Gemma2
    public float? FinalLogitSoftcap { get; init; }      // Gemma2
    public float? EmbeddingScale { get; init; }         // Gemma sqrt(hidden)
    public bool TieWordEmbeddings { get; init; }
    public float RmsNormEps { get; init; } = 1e-6f;

    // Directionality — unifies LLMs and text encoders on one core:
    public Directionality Direction { get; init; } = Directionality.CausalDecoder; // | BidirectionalEncoder | EncoderDecoder
    public MoeConfig? Moe { get; init; }                // null = dense; later milestone

    public static TransformerConfig Llama3_8B { get; }
    public static TransformerConfig Qwen2_5_7B { get; }
    public static TransformerConfig Qwen3_8B { get; }
    public static TransformerConfig Mistral7B { get; }
    public static TransformerConfig T5Encoder(/* dims */);   // Direction=EncoderDecoder, relative bias
}
```

```csharp
// Transformer/GenericTransformer.cs
public sealed class GenericTransformer : IDisposable
{
    public GenericTransformer(TransformerConfig cfg, IBackend backend);
    public void LoadWeights(IReadOnlyDictionary<string, TensorView> weights); // post-converter, internal names
    public void PreloadToDevice();                       // backend.PreloadWeights over EnumerateWeights()

    // Prefill: full sequence, fills cache. Decode: single token (T=1), appends to cache.
    public Tensor Forward(ReadOnlySpan<int> tokenIds, KvCache? cache, int positionOffset);
    public Tensor ForwardEmbeds(TensorView embeds, KvCache? cache, int positionOffset); // for fused embed tables
    public Tensor LmHead(TensorView hiddenLastToken);    // → logits [B, vocab]; respects tied embeddings

    public IEnumerable<Tensor> EnumerateWeights();
}
```

### 2.2 Generation

```csharp
// Generation/TextGenerationPipeline.cs   (mirrors DiffusionPipelineBase conventions)
public sealed class TextGenerationPipeline : LanguagePipelineBase
{
    public TextGenerationPipeline(GenericTransformer model, ITokenizer tok,
                                  IChatTemplate template, IBackend backend);

    // Returns full result; streams tokens via callback (NOT IAsyncEnumerable — matches diffusion convention)
    public GenerationResult Generate(GenerationRequest req, Action<TokenChunk>? onToken = null);
}

// Generation/GenerationRequest.cs  — three-tier options
public sealed record GenerationRequest
{
    public required string Prompt { get; init; }          // or Messages for chat
    public IReadOnlyList<ChatMessage>? Messages { get; init; }
    public int MaxTokens { get; init; } = 512;
    public SamplingOptions Sampling { get; init; } = SamplingOptions.Default; // temp/topk/topp/minp/penalties/seed
    public IReadOnlyList<int>? StopTokenIds { get; init; }
    public IReadOnlyList<string>? StopStrings { get; init; }
}
```

### 2.3 Sampling, KV cache, chat templates

```csharp
// Sampling/ISamplerStep.cs  — promote NucleusSampler into a composable chain (shared with Audio)
public interface ISamplerStep { void Apply(Span<float> logits, GenerationState state); }
//   Temperature, RepetitionPenalty, PresencePenalty, FrequencyPenalty, TopK, TopP, MinP, GreedyArgmax
// Sampling/SamplerChain.cs  — built from SamplingOptions or composed explicitly

// KvCache/KvCache.cs        — promoted from Audio/Streaming/StreamingKvCache (pre-alloc, alloc-free append)
//   GetKeysRef/GetValuesRef return TensorRef (hot path). Rollback(length) for speculative/stop-trim.
// KvCache/PagedKvCache.cs   — block-allocated variant for batching (Milestone 5)

// ChatTemplates/IChatTemplate.cs + registry — extract templates currently hardcoded in tokenizer classes
//   ChatMlTemplate (Qwen), Llama3Template, MistralTemplate, GemmaTemplate; tool/function-call formatting
```

---

## 3. Data Flow (decode)

```
messages ─IChatTemplate─▶ prompt string ─ITokenizer─▶ ids[T]
  prefill:  GenericTransformer.Forward(ids, cache, pos=0)         → hidden[B,T,H], cache filled (len=T)
            LmHead(hidden[:, -1])                                  → logits[B,vocab]
  loop:     SamplerChain.Apply(logits) → nextId
            stop? (eos / StopTokenIds / StopStrings)               → break
            decode: Forward([nextId], cache, pos=len) (T=1)        → hidden[B,1,H]; cache len++
            LmHead(...) → logits ; emit onToken(nextId)
  detokenize accumulated ids → text
```

Shapes per attention layer (decode): `q[B,nHeads,1,headDim]`, `k/v` appended into cache
`[B,nKvHeads,maxSeq,headDim]`; GQA repeats KV across head groups; RoPE applied at `pos`.

**Critical:** the entire per-token loop must keep activations + KV cache GPU-resident; only `nextId` (and
optionally a small logits slice) crosses the PCIe boundary per step. See §5 Risk 1.

---

## 4. File Breakdown (`src/HartsyInference.Language/`)

### Transformer/
- **`TransformerConfig.cs`** — config record + presets (§2.1). *dotLLM pattern:* architecture-as-data.
- **`GenericTransformer.cs`** — block stack, embed lookup, final norm, LM head; pre-allocates
  `TransformerForwardState`. Programs against `IBackend` only. *Reuses:* `Qwen3Model`/`Qwen2Model` logic.
- **`TransformerForwardState.cs`** — all scratch buffers allocated at load (residual, normed, q/k/v, attn
  out, mlp gate/up). *dotLLM pattern:* zero per-step allocation.
- **`DecoderLayer.cs`** — pre/sandwich-norm → attention → residual → norm → MLP → residual; reads config
  flags (QkNorm, AttentionBias, softcap). *Reuses:* `Qwen3 Layer`.
- **`Attention.cs`** — GQA/MQA, optional q/k-norm, RoPE apply, sliding-window + sink masking, calls
  `backend.ScaledDotProductAttention` / flash path; KV cache append. *Reuses:* `Qwen2Attention`.
- **`Mlp.cs`** — SwiGLU / GeGLU via `backend.Silu`/`GeGlu` + matmuls. *Reuses:* `Qwen2Mlp`.
- **`RotaryEmbedding.cs`** — split-half + interleaved layouts; scaling (linear/NTK/YaRN/llama3); 2D/3D
  mRoPE hook. *Reuses:* `Qwen2RotaryEmbedding` + `Qwen3 RotaryEmbedding`; depends on `backend.ApplyRope`.
- **`MoeLayer.cs`** *(Milestone 5)* — router + expert dispatch (gather/scatter).

### Generation/
- **`LanguagePipelineBase.cs`** — holds `IBackend`, idempotent `Dispose`, `ThrowIfDisposed()`. Mirrors
  `DiffusionPipelineBase`. Pipelines do **not** own the model/tokenizer (passed in).
- **`TextGenerationPipeline.cs`** — prefill→decode→sample→stop loop; streaming callback.
- **`GenerationRequest.cs` / `GenerationResult.cs` / `TokenChunk.cs` / `ChatMessage.cs`**.
- **`StoppingCriteria.cs`** — eos / stop-id / stop-string (incremental detok) detection.

### Sampling/
- **`ISamplerStep.cs`, `SamplerChain.cs`, `SamplingOptions.cs`**, plus one file per step. *Reuses:*
  `NucleusSampler` (Audio) top-k/top-p/min-p math; adds penalties + greedy argmax.

### KvCache/
- **`KvCache.cs`** (from `StreamingKvCache`), **`PagedKvCache.cs`** (Milestone 5).

### ChatTemplates/
- **`IChatTemplate.cs`** + `ChatMlTemplate` / `Llama3Template` / `MistralTemplate` / `GemmaTemplate` +
  `ChatTemplateRegistry.cs`. Extracts logic currently inside `Qwen3Tokenizer.EncodeChat` etc.

### Configs/  (presets live on TransformerConfig; this folder holds per-family HF config.json parsing)
- **`HfConfigReader.cs`** — maps HF `config.json` → `TransformerConfig` (auto-detect family).

### CheckpointConverters/
- **`ILlmCheckpointConverter.cs`** + `LlamaCheckpointConverter` / `QwenCheckpointConverter` /
  `MistralCheckpointConverter`. HF/GGUF weight-name → internal names. *Reuses:* ModelHandler
  `GgufKeyMapperRegistry` pattern.

### Pipelines/  (thin — mostly preset + converter wiring)
- **`LlmModelLoader.cs`** — detect format (safetensors/GGUF) + family → build `GenericTransformer`.

### Backend additions (in Cpu / Cuda / Vulkan, surfaced on `IBackend`)
- **`DequantMatMul` (Q4_K_M / Q6_K / Q8_0)** — CUDA first (`native/cuda/kernels/dequant_q4k`, `dequant_q8`
  already stubbed) → quantized GEMV for decode, GEMM for prefill. *dotLLM pattern:* quantize activations,
  not dequantize weights, for single-token decode; R4 weight repacking at load.
- **GPU-resident fused decode dispatch** — see Risk 1.
- **`FlashAttentionCausal`** (CUDA/Vulkan; CPU keeps tiled SDPA), **sliding-window + sink mask**,
  **`Argmax`/on-GPU TopK**, **`LogitSoftcap`**, RoPE-scaling kernels.

---

## 5. Edge Cases & Risks

1. **🔴 Decode performance (gating blocker).** Per-op H2D/D2H auto-transfer is acceptable for ~30 big
   diffusion steps but catastrophic for thousands of tiny token steps. **GPU-resident activations + fused
   per-layer dispatch is a prerequisite, not an optimization.** Likely the largest single work item; partly
   overlaps the diffusion GPU-residency work already proven on Ideogram4 (see project memory). De-risk via
   Milestone 0 before committing to the spec's perf claims.
2. **Quantized matmul correctness** — Q4_K/Q6_K block layout is fiddly; validate known-values vs llama.cpp
   output (reference only, clean-room).
3. **Tokenizer / chat-template parity** — must match HF exactly or output silently degrades; covered by the
   template registry + golden tests.
4. **RoPE scaling math** (YaRN/NTK/llama3) — subtle; long-context degradation if wrong. Validate vs reference.
5. **KV-cache VRAM** at long context (esp. MHA) — capacity planning + `OutOfVramException` guarding.
6. **MoE** routing (gather/scatter, expert dispatch) is a distinct execution path — its own milestone.
7. **Stop-string detection** across token boundaries needs incremental detokenization.

---

## 6. Research To Write

`GENERIC_TRANSFORMER.md` (the config-matrix spine; how decoder/encoder/enc-dec map on), `LLM_DECODE_LOOP.md`,
`GGUF_QUANTIZED_MATMUL.md`, `LLM_ATTENTION.md`, `ROPE_SCALING.md`, `LLM_SAMPLING.md`, `CHAT_TEMPLATES.md`,
and (deferred) `MOE_INFERENCE.md`.

---

## 7. Milestones (see `docs/Checklists/PHASE_12_LANGUAGE.md`)

- **M0 — Decode perf spike** (de-risk Risk 1 on existing Qwen3 on 3060). Gates everything.
- **M1 — Generic transformer + F16/F32 decode.** Extract core; port Qwen2/Qwen3 as first presets; build
  pipeline, sampler chain, template registry. Validate text-out vs HF on a small model.
- **M2 — Quantized inference (CUDA).** GGUF dequant-matmul; validate vs llama.cpp. Makes 7B–70B usable.
- **M3 — Coverage:** Llama-3.x, Mistral (sliding window), Qwen2.5 presets + converters; flash-attn; RoPE scaling.
- **M4 — Encoder unification.** Re-target `T5TextEncoder`/`ClipTextEncoder`/`LlamaStyleEncoder` + audio
  Qwen3-TTS/YuE/Higgs onto the generic core; delete duplicated transformer code.
- **M5 — MoE + paged KV + Server `/v1/chat/completions` streaming.**
- **M6 *(optional)* — VLM** (reuse Vision towers + generic transformer; Qwen-VL lineage).

## 8. Testing Strategy

- SIMD-vs-scalar parity on every new CPU kernel; CPU-vs-CUDA parity on dequant-matmul + flash-attn.
- Known-value tests vs HF/llama.cpp reference logits (greedy, fixed seed) per family.
- Chat-template golden tests vs HF `apply_chat_template`.
- End-to-end greedy-decode determinism test per preset (small model in CI; large gated to bare terminal).
