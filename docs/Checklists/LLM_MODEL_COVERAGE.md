# LLM Model Coverage Plan

Goal: support text generation end-to-end for every architecture Ollama/llama.cpp can run, or
at minimum every SOTA model, via the config-driven `GenericTransformer` spine. New models should
be *a preset + key-mapping + (rarely) one engine knob*, never a new transformer class.

## Operating policy

- **Reuse over rebuild.** The spine, tokenizers, vision encoders, and RoPE machinery already cover
  most of the surface. Each phase lists what is *reused* vs what is genuinely *new engine work*.
- **Two distinct bars:**
  - **Runnable@3060** — fits in 12 GB at Q4/Q8 → verified end-to-end on this box (coherent output,
    and where feasible logits/parity vs a Python/llama.cpp reference).
  - **Build-defer** — architecture + key-mapping + unit/slice parity tests land, but e2e generation
    is marked *pending-hardware*. Applies to Kimi-K2 (1T), DeepSeek-V3 (671B), Mixtral (47B), large
    MoE. (Per decision 2026-06-26: build the arch, defer verification.)
- **Status legend:** `[x]` done+verified · `[~]` built, verification pending · `[ ]` not started.

## Current state (Phase 0 — done)

- [x] `GenericTransformer` dense decoder spine (GQA/MQA/MHA, SwiGLU, pre-norm RMSNorm, tied/untied)
- [x] MoE FFN in spine (softmax/sigmoid routing, shared experts, first-K-dense) — *not yet wired from GGUF*
- [x] RoPE: split-half + interleaved pairing; None/Linear/Yarn/Llama3/DynamicNtk/LongRope scaling
- [x] GGUF load + key-remap (`llama`/`qwen2`/`qwen3`), GGUF-native byte-level BPE tokenizer, Jinja chat templates
- [x] **Qwen2 / Qwen2.5** (0.5B verified Q4_K_M + Q8) — Runnable@3060
- [x] **Qwen3** (0.6B verified, incl. `<think>`) — Runnable@3060
- [x] **Llama-3.x** (3.2-1B verified Q8) + **Mistral / TinyLlama / SmolLM / Yi** (load via `llama` arch) — Runnable@3060
- Bugs fixed during bring-up: Jinja string-blind lexer; `length` filter on strings (+`first`/`last`/`join`/`tojson`);
  Llama3 `rope_freqs` divisor-not-multiplier; Llama GGUF needs interleaved (NORM) RoPE.

### Phase 0 cleanups (done)
- [x] GGUF load log reports the GGUF `general.architecture` + the mapper (`arch=qwen2 (mapper=llama)`)
- [x] CPU GGUF path: `GgufLanguageModel.Load(dequantizeToF32: true)` widens quantized projections to F32 for `CpuBackend`
- [x] Env-gated e2e test (`GgufEndToEndTests`, `HARTSY_TEST_GGUF_MODELS`): loads each GGUF on CPU, greedy-decodes, asserts the known answer; skips when unset

---

## Engine capability matrix (what the spine has / lacks)

| Capability | Status | Needed by |
|---|---|---|
| SwiGLU (SiLU) FFN | have | Llama/Qwen/Mistral |
| Configurable activation (GeGLU/GELU) | **new** | Gemma, StableLM-2 |
| RMSNorm | have | most |
| `(1+w)` RMSNorm, LayerNorm variants | **new** | Gemma (`1+w`), Command-R (LayerNorm) |
| Embedding / residual scalars | **new** | Gemma (√d_model), Granite (residual/attn/logit multipliers) |
| Attention logit softcap, final-logit softcap | **new** | Gemma-2, (Grok) |
| Sliding-window attention (per-layer pattern) | **new** | Gemma-2/3, Mistral-sliding, Phi-3-small |
| Fused QKV projection loader | **new** | Phi-3, StableLM, GPT-2-lineage |
| Partial rotary (rotary_dim < head_dim) | **new** | Phi, StableLM, Persimmon |
| Parallel attn+FFN residual | **new** | Command-R, GPT-NeoX-lineage |
| Attention sinks | **new** | GPT-OSS |
| MoE wired from GGUF metadata | **new** | Mixtral, Qwen-MoE, DeepSeek, OLMoE |
| MLA (latent attention) | **new (large)** | DeepSeek-V2/V3, Kimi-K2 |
| Vision encoder + projector + image plumbing | **partial** (encoders exist) | all VLMs |
| Encoder/embedding mode (no causal mask, pooling) | **new** | BERT/BGE/Nomic embeddings |

## Reusable assets (do not rebuild)

- Tokenizers: `GemmaTokenizer` (SentencePiece), `LlamaTokenizer`, `GptOssTokenizer` (tiktoken/o200k — reuse for Phi-4),
  `BertWordPieceTokenizer`, `ClipTokenizer`, `HfTokenizerJson` (generic `tokenizer.json` loader), `GgufTokenizer`.
- Vision encoders: `SiglipVisionEncoder` (Gemma-3/PaliGemma), `Dinov2VisionEncoder`, CLIP, image preprocessors, `PngDecoder`.
- RoPE: `RopeFrequencyBuilder` (all scaling types incl. LongRope), both pairings.
- MoE: `MoeFeedForward` + `MoeConfig` in the spine.

---

## Phase 1 — Finish dense text decoders (Runnable@3060)

Complete the non-MoE, non-MLA, non-vision text family. Each is a preset + key-mapper + the engine knob noted.

### 1a. Gemma 2 / Gemma 3 (text)  — engine work: medium
- [x] Configurable activation = GeGLU (reuses the existing tanh-approx `Gelu`); embedding scale √d_model; `(1+w)` RMSNorm (config flag; the GGUF path uses llama.cpp's pre-baked norms so it stays off)
- [x] Sandwich norm (post-attn + post-FFN pre-residual norms); query pre-attn scalar (`AttnScale`)
- [x] Gemma-3 dual-RoPE (local 10k / global 1M, 5:1 pattern) — per-layer cos/sin selection
- [x] SPM-from-GGUF tokenizer (`SpmGgufTokenizer`: score-driven merges + ▁ + byte fallback); float-array GGUF metadata; GGUF arch `gemma`/`gemma2`/`gemma3` → `GemmaKeyMapper`
- [x] Jinja parser: `%`/`*`/`//` operators (Gemma template's `loop.index0 % 2`)
- [x] **Gemma-3-1B-it Q4_K_M verified e2e** on CUDA + CPU (coherent, gated test green)
- [x] **Gemma-2 attention-logit soft-cap (50)**: added a `softcap` param to `FlashAttention` (CPU reference +
      `flash_attn_f32.cu` kernel, recompiled PTX); final-logit soft-cap (30) already done.
- [x] **Gemma-2-2B-it Q4_K_M verified e2e** on CUDA (low-VRAM mode — F16 weight cache OOMs the 3060's 12 GB
      otherwise) + CPU. Coherent.
- [ ] Sliding-window attention masking for local layers (correct for context ≤ window today; needed for long context)
- Build-defer siblings: Gemma-2-9B/27B (`[~]`)

### 1b. Phi-3 / Phi-3.5-mini / Phi-4-mini — engine work: medium
- [x] Fused QKV + fused gate/up split at load (`GgufLanguageModel.SplitFusedPhi`, contiguous row-byte copy — works
      on quantized weights, no dequant); `PhiKeyMapper` (arch `phi3`); SPM tokenizer reused (Phi-3 = Llama SP)
- [x] LongRope wired from the `rope_factors_long/short` tensors + `original_context_length` + `attn_factor`
- [x] **Non-power-of-two head dim fix**: Phi-3's head_dim=96 hit a CUDA fallback that recursed infinitely; the
      flash-attn kernel now pads the block to the next power of two (handles any D≤1024), and the CPU reference
      was extracted to `AttentionReference` so the GPU fallback no longer re-dispatches into itself.
- [x] **Phi-3.5-mini (3.8B) Q4_K_M verified e2e** on CUDA (low-VRAM) — coherent.
- [ ] Phi-4-mini (tiktoken/o200k tokenizer, reuse `GptOssTokenizer`); Phi-3 sliding window for long context

### 1c. Cohere Command-R (small), StableLM-2, Granite-3 dense — engine work: medium
- [ ] LayerNorm (no bias) + parallel attn/FFN residual + tied + logit scale (Command-R)
- [ ] Partial rotary + (optional) QKV bias + parallel residual (StableLM-2)
- [ ] Residual/attention/embedding/logit multipliers (Granite-3)
- [ ] arches `command-r`, `stablelm`, `granite` → mappers; verify the ≤3B variants

### 1d. Catch-all llama-lineage verification (no/low engine work)
- [ ] Verify already-loadable models render coherently: SmolLM2, TinyLlama, Yi-6B, InternLM2, OLMo-2, Falcon3, MiniCPM, Nemotron-mini, Ministral-3B
- [ ] Add per-model presets/notes + chat-template checks; file bugs as found (like the Phase-0 RoPE bugs)

---

## Phase 2 — MoE wiring + breadth (mostly Build-defer)

Spine already does MoE; this is GGUF detection + per-arch quirks. Most MoE models exceed 12 GB.

- [ ] `GgufConfigFactory`: read `{arch}.expert_count`, `expert_used_count`, `expert_feed_forward_length`,
      `expert_shared_count`, `expert_weights_scale`, `expert_gating_func` → populate `MoeConfig`
- [ ] MoE expert-tensor key-mapping (stacked `ffn_*_exps` ↔ per-expert) in the llama-family mapper
- [ ] arches: `qwen2moe`, `qwen3moe`, `mixtral`(llama-moe), `olmoe`, `granitemoe`, `phimoe`
- [ ] **Mixtral 8x7B** — `[~]` build-defer (47B)
- [ ] **Qwen3-MoE 30B-A3B / 235B** — `[~]` build-defer
- [ ] Verify a *small* MoE end-to-end if one fits: **OLMoE-1B-7B** (~4 GB active set at Q4) — Runnable@3060 proxy
- [ ] Router parity test (softmax+renorm vs sigmoid+bias) on tensor slices

---

## Phase 3 — MLA: DeepSeek-V2/V3, Kimi-K2 (heavy engine work, Build-defer)

Multi-head **latent** attention is a new attention path, not a knob. Once done, Kimi-K2 is a config of DeepSeek-V3.

- [ ] MLA in spine + backend: KV compression (`kv_lora_rank`), Q compression (`q_lora_rank`),
      decoupled RoPE (`qk_rope_head_dim` + `qk_nope_head_dim`), `v_head_dim`
- [ ] DeepSeek-V3 routing: sigmoid + `e_score_correction_bias`, node/group-limited top-k, 256 experts, shared expert
- [ ] Optional: MTP (multi-token-prediction) heads — can stub for inference
- [ ] arches `deepseek2`, `deepseek-v3`/`kimi-k2` → mappers + MoE wiring from Phase 2
- [ ] **Verification proxy:** DeepSeek-V2-Lite (16B, 2.4B active, has MLA) at Q4 (~10 GB) — attempt Runnable@3060;
      this is the one place a small MLA model exists to validate the math
- [ ] **DeepSeek-V3 671B / Kimi-K2 1T** — `[~]` build-defer (cannot load on 12 GB; verified structurally + via proxy)

---

## Phase 4 — Vision / multimodal VLMs (largest new surface)

Encoder half largely exists (SigLIP/CLIP/DINOv2). New work = projector + image-token interleaving + mmproj GGUF load.

- [ ] Multimodal input plumbing: image preprocess → encoder → projector → image tokens spliced into the decoder sequence
- [ ] mmproj GGUF loader (llama.cpp ships vision weights as a separate `mmproj-*.gguf`)
- [ ] Projector variants: MLP (LLaVA), pixel-shuffle (Gemma-3, InternVL), perceiver/resampler (Qwen-VL), patch-merger (Qwen2.5-VL)
- [ ] Multimodal chat-template handling (image placeholder tokens)
- [ ] **Qwen2.5-VL-3B** (SigLIP-style + patch merger) — Runnable@3060
- [ ] **Gemma-3-4B vision** (SigLIP + pixel-shuffle, reuses Phase-1a Gemma text) — Runnable@3060
- [ ] **LLaVA-1.6 / SmolVLM / MiniCPM-V** — Runnable@3060 (LLaVA-7B tight)
- [ ] **Llama-3.2-11B-Vision / Qwen2.5-VL-7B+** — `[~]` build-defer
- [ ] Per-encoder image-preprocessing parity vs reference (resize/normalize/patch)

---

## Phase 5 — Cross-cutting coverage (interleave as needed)

- [ ] **CPU GGUF** dequant path (also unblocks reference-free CPU parity tests) — overlaps Phase-0 cleanup
- [ ] **More quant formats**: Q4_0/Q5_0/Q4_1, Q2_K/Q3_K, IQ-quants — coverage for the Ollama library
- [ ] **Embedding models** (encoder mode, no causal mask, mean/CLS pooling): nomic-embed, bge, all-MiniLM, `BertWordPieceTokenizer` reuse
- [ ] **GPT-OSS** specifics: attention sinks + `GptOssTokenizer` (o200k) — dense, ~20B build-defer / 3060 if a small variant exists
- [ ] Longer-context tests (rope-scaling correctness at >8k), batch>1 throughput, optional speculative decoding
- [ ] Keep `docs/Checklists/PARITY_VERIFICATION.md` updated with each model's verified/pending status

---

## Ollama-popular coverage map (target → phase → 3060)

| Model | Phase | Runnable@3060 |
|---|---|---|
| Llama 1/2/3.x, Mistral, TinyLlama, SmolLM, Yi | 0/1d | ✅ |
| Qwen2 / Qwen2.5 / Qwen3 (dense) | 0 | ✅ |
| Gemma 2 / 3 (text) | 1a | ✅ verified (Gemma-3-1B, Gemma-2-2B) |
| Phi-3 / Phi-3.5-mini | 1b | ✅ verified (Phi-3.5-mini); Phi-4-mini pending (o200k tokenizer) |
| Command-R, StableLM-2, Granite-3 | 1c | ✅ (small) |
| Mixtral, Qwen-MoE, OLMoE | 2 | OLMoE ✅; rest `[~]` |
| DeepSeek-V2-Lite | 3 | ⚠️ tight |
| DeepSeek-V3, Kimi-K2 | 3 | ❌ build-defer |
| Qwen2.5-VL, Gemma-3-vision, LLaVA, MiniCPM-V | 4 | ✅ (small) |
| Llama-3.2-Vision (11B) | 4 | ❌ build-defer |
| nomic-embed / bge / MiniLM | 5 | ✅ |
| GPT-OSS | 5 | ❌/⚠️ |
