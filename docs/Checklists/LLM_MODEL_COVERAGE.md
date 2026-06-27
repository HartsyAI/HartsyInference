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
- [x] **Phi-4-mini tokenizer**: added the o200k / `gpt-4o` byte-level pre-tokenizer regex to the GGUF tokenizer
      path (`pre == "gpt-4o"/"o200k"`) — tokenization now correct (reusable for GPT-OSS). Config (GQA 24/8, tied)
      detected fine.
- [x] **Phi-4-mini-instruct verified e2e** — gpt-4o tokenizer + partial rotary (96/128) + a fused-QKV-split fix
      (the split must use the real head_dim=128, not `rope.dimension_count`=96; these coincided on Phi-3.5).

### 1c. Cohere Command-R (small), StableLM-2, Granite-3 dense — engine work: medium
- [x] **Granite-3** scalar multipliers: `embedding_scale` (reuses EmbeddingScale), `attention.scale` (direct
      `AttentionMultiplier`, not 1/√d), `residual_scale` (`ResidualMultiplier`, in `PostSublayer`), `logit_scale`
      (divide in `ProjectLogits`). Arch `granite`/`granitemoe` registered to the llama mapper.
- [x] **Granite-3.1-2B-instruct Q4_K_M verified e2e** on CUDA.
- [x] **Command-R7B (cohere2) verified e2e** on CUDA (low-VRAM). Recipe: LayerNorm (zero-bias) + parallel
      residual + **interleaved (NORM) RoPE** (like llama, permuted q/k) + **NoPE on global layers** (1 of every 4
      is full-attention with no positional encoding) + logit-scale-as-multiply. Jinja `break`/`continue` +
      `macro`-tolerance added for its template. Reusable for orig Command-R 35B / GPT-NeoX.
- [x] **Partial rotary** — `RotaryDim` config + `rotaryDim` param on `ApplyRopeSingle` (CPU ref + the shared
      `dit_rope_f32` kernel, defaulted so DiT is byte-identical) + `BuildRope` sizing. Detected from
      `rope.dimension_count < head_dim`. **Verified via Phi-4-mini and StableLM-2.**
- [x] **StableLM-2-1.6B verified e2e** (partial rotary + QKV bias; `stablelm` registered to the llama mapper).

### 1d. Catch-all llama-lineage verification (no/low engine work)
- [x] **Mistral-7B-Instruct-v0.3** verified e2e (llama arch, interleaved RoPE, GQA 32/8) — loads + runs unchanged
- [x] **SmolLM2-1.7B, TinyLlama-1.1B, Yi-1.5-6B** verified e2e (all work as-is via the llama-family path)
- [ ] Remaining sweep (lower priority, expected to work): InternLM2, OLMo-2, Falcon3, MiniCPM, Ministral, Nemotron-mini

---

## Phase 2 — MoE wiring + breadth (mostly Build-defer)

Spine already does MoE; this is GGUF detection + per-arch quirks. Most MoE models exceed 12 GB.

- [x] `GgufConfigFactory`: reads `expert_count`, `expert_used_count`, `expert_feed_forward_length`,
      `expert_shared_feed_forward_length`, `expert_gating_func`, `leading_dense_block_count`, `expert_weights_norm`
      → `MoeConfig`. NormTopKProb arch-aware (OLMoE = false, Mixtral/Qwen-MoE = true).
- [x] Stacked-expert split (`GgufLanguageModel.SplitStackedExperts`): each 3D `ffn_*_exps` [E,·,·] flattened to 2D
      and row-byte-sliced into per-expert weights (dtype-preserving). Router + shared-expert key mappings added to
      the llama-family mapper; arches `olmoe`/`qwen2moe`/`qwen3moe` registered to it.
- [x] **Whole-vector Q/K norm** (OLMoE norms the full Q/K vector, not per-head): config `QkNormFullDim`, detected
      from the q_norm weight length; the projection output is shaped `[1,T,QDim]` so RMSNorm reduces over it.
- [x] **OLMoE-1B-7B-0924-Instruct Q4_K_M verified e2e** on CUDA (low-VRAM) — coherent, detailed.
- [x] **Granite-MoE (granitemoe) verified** — scalars + MoE combined; engine confirmed via raw completion
      ("The capital of France is" → " Paris."). granite-3.0-1b-a400m's weak instruct quality (repetition / early
      EOS on chat prompts) is the 400M-active model, not the engine.
- [x] **Shared-expert + softmax-renorm path validated** by `MoeTests.MoeFeedForward_MatchesReference` against an
      independent HF-Qwen-MoE reference (covers shared_expert + shared_expert_gate) — no 14GB Qwen-MoE GGUF needed.
- [~] **Mixtral 8x7B** (47B) / **Qwen3-MoE 30B-A3B / 235B** — build-defer: config (expert metadata → MoeConfig),
      mapper, and stacked-expert split are all wired; just exceed 12 GB to run. (Mixtral = `llama` arch + experts,
      interleaved RoPE, renorm; Qwen3-MoE = `qwen3moe`, per-head Q/K norm, no shared expert.)

---

## Phase 3 — MLA: DeepSeek-V2/V3, Kimi-K2 (heavy engine work, Build-defer)

Multi-head **latent** attention is a new attention path, not a knob. Once done, Kimi-K2 is a config of DeepSeek-V3.

- [x] **MLA attention path built** (`GenericTransformer.MlaForward`): direct `q_proj` (V2-Lite) → split [nope|rope];
      KV down (`kv_a_proj`) → latent + shared rope-key → `kv_a_norm` (RMSNorm) → KV up (`kv_b_proj`) → per-head
      [k_nope|v]; decoupled RoPE on the rope parts; shared rope-key broadcast across heads (`RepeatKvHeads`);
      V zero-padded to the qk head dim so the equal-dim FlashAttention (non-pow2 D=192 supported) is reused, then
      sliced back. `MlaConfig` + `deepseek2` factory wiring + `DeepSeekKeyMapper`.
- [x] **DeepSeek MoE**: shared expert (size = `expert_shared_count` × expert ffn), leading dense layer
      (`leading_dense_block_count`), softmax routing; `expert_weights_scale` read (1.0 on V2-Lite, no-op).
- [x] **yarn** RoPE scaling wired from the GGUF (`rope.scaling.type=yarn`, factor/orig-ctx); mscale approximate.
- [x] **Memory fixes** (needed even to load): MoE expert split is now a **zero-copy view** over the mmap (was
      copying ~7 GB → host OOM-killed); the embedding table is **not** uploaded to GPU when untied (host-gather
      only) — saves ~0.8 GB.
- [x] **Validated** by `MlaTests` (synthetic 1-layer MLA: prefill + decode, finite/shaped, cache advances) and
      by loading the real DeepSeek-V2-Lite GGUF correctly (arch detect, all MLA/MoE tensors mapped, config right).
- [~] **e2e on 3060 not possible** — DeepSeek-V2-Lite (smallest MLA model, 9.7 GB Q4 + the model's size) exceeds
      the 3060's ~10.5 GB free VRAM during weight preload. **Build-defer confirmed** (per the >12GB policy).
- [ ] DeepSeek-V3 extras (build-defer): sigmoid + `e_score_correction_bias`, node/group-limited top-k, 256
      experts; q-LoRA (`q_a_proj`/`q_b_proj` — mapper has the keys; the MLA forward currently does direct-q only).
- [ ] **DeepSeek-V3 671B / Kimi-K2 1T** — `[~]` build-defer (architecturally a bigger config of the above).

---

## Phase 4 — Vision / multimodal VLMs (largest new surface) — GROUNDED, ready to build

Encoder half exists (SigLIP/CLIP/DINOv2). **Text side is ready**: `ForwardEmbeds` is an embedding-in path, so
spliced image embeddings flow straight through the verified Gemma-3 / Qwen2.5 / Llama decoders.

**First target: Gemma-3-4B-vision** (reuses the verified Gemma-3 text + SigLIP). mmproj inspected
(`ggml-org/gemma-3-4b-it-GGUF` / `mmproj-model-f16.gguf`, 812 MB): `clip.projector_type=gemma3`; vision 1152-dim,
27 blocks, 16 heads, image 896, patch 14 → 64×64=4096 patches. Tensors: `v.patch_embd` (Conv 14²×3→1152),
`v.position_embd` [1152,4096], per block `v.blk.N.{ln1, attn_q/k/v/out+bias, ln2, ffn_up/down+bias}`, `v.post_ln`;
projector `mm.soft_emb_norm` (RMSNorm 1152) + `mm.input_projection` (1152→2560 = text hidden).
**Pipeline:** PNG → resize 896² + normalize → patch+pos embed → 27 ViT blocks → post_ln → avg-pool 4096→256 →
soft_emb_norm → input_projection → splice 256 embeddings at `<image_soft_token>` → `ForwardEmbeds`.

- [x] mmproj GGUF loader + `v.*`/`mm.*` key mapping → vision weight dict (`GgufModelLoader.LoadDequantized`, passthrough mapper)
- [x] Vision-tower forward (`Gemma3VisionEncoder`: patch Conv → +pos → 27 pre-norm ViT blocks, separate q/k/v+bias,
      bidirectional FlashAttention `causal:false`, GELU-tanh → post_ln)
- [x] Gemma-3 projector (`Project`: avg-pool 4096→256 → soft_emb_norm RMSNorm → input_projection Linear)
- [x] Image preprocessing (resize/normalize via SigLIP mean/std) + structured synthetic test images (`VlmRunner`)
- [x] Multimodal pipeline `MultimodalGenerator` (encode → splice 256 embeds between `<start_of_image>`/`<end_of_image>` → `ForwardEmbeds` → greedy)
- [x] CLI `vlm` mode (`hartsyinference-textgen vlm`) end-to-end on the 3060 (loads text 4B Q4 + mmproj, encodes, splices, generates)
- [x] **Gemma-3-4B-vision VALIDATED vs reference — vision tower numerically correct (corr 1.0).** Reference-diffed the
      full tower (`tests/python-reference/dump_gemma3_vision_ref.py`: loads the same mmproj GGUF + same pixel input, runs
      a torch SigLIP tower, diffs each C# stage dump) — **pixels/seq/blk0/postln/embeds all corr=1.0**, maxdiff ~1e-3
      (F32 accumulation through 27 blocks). **Two bugs found & fixed:** (1) llama.cpp's clip names the SigLIP MLP
      projections **swapped** — `ffn_down` is fc1 (up, hidden→intermediate, bias=4304) and `ffn_up` is fc2 (down,
      bias=1152); the bias sizes give it away. The Block now wires them by role, not name. (2) the nn.Linear weights
      (attn q/k/v/out, ffn) only need a **shape relabel** to `[out,in]` (the GGUF bytes are already row-major `[out,in]`),
      NOT a data transpose — only `mm.input_projection` (a raw `[in,out]` param) needs the actual transpose.
      Also verified: embedding scale (image embeds × √hidden), FlashAttention head_dim=72 non-causal (new
      `Flash_NonPow2HeadDim_NonCausal_MatchesSdpa` test), GELU tanh-approx, Conv2D `[out,in,kh,kw]`, position layout,
      soft_emb_norm (+1 pre-baked). **E2E:** generates coherent image-grounded text (`HARTSY_VLM_NOPRELOAD=1` to share
      the GPU with a running SwarmUI). Greedy decode on a 4B-Q4 model + synthetic OOD images is repetition-prone — use
      the sampler + real images for quality. Instrumentation: `HARTSY_VLM_DEBUG=1`, `HARTSY_VLM_DUMP=<dir>`, `HARTSY_VLM_ENCODE_ONLY=1`.
- [x] **SmolVLM2-2.2B VALIDATED + e2e correct.** Generalized the encoder to `SiglipVlmEncoder` (shared SigLIP tower +
      pluggable projector, detected by tensor presence). SmolVLM reuses the validated SigLIP tower verbatim; the only new
      piece is the **idefics3 pixel-shuffle** projector (scale_factor 3: 729 patches → 81 tokens → `mm.model.fc` 10368→2048).
      Reference-validated (`tests/python-reference/dump_smolvlm_vision_ref.py`): seq/blk0/postln/embeds all **corr=1.0**.
      Text side = SmolLM2/llama (already verified; no embedding scale). **E2E correct answers** (`HARTSY_VLM_PATTERN=…`):
      red circle → "a single red circle"; blue-over-green → "top … blue … bottom … green"; blue square → "a blue square".
      `mm.model.fc` is a true nn.Linear (relabel, not transpose); SmolVLM prompt = `<|im_start|>User:<fake_token_around_image><global-img>`
      + [81 img] + `<fake_token_around_image>{q}<end_of_utterance>\nAssistant:` (no separate BOS).
- [x] **LLaVA-1.5-7B VALIDATED + e2e correct.** Extended `SiglipVlmEncoder` to also cover the CLIP ViT (the `v.blk.*`
      dialect is shared): CLS token (`v.class_embd` prepended), pre-LN (`v.pre_ln`), quick-GELU (`x·sigmoid(1.702x)` via
      Scale+Sigmoid+Mul), NO post-LN (mmproj is truncated to the penultimate 23 layers), drop-CLS, and the LLaVA MLP
      projector (`mm.0`→GELU→`mm.2`, both nn.Linear → relabel). Conv patch-embed has no bias (CLIP) → optional.
      Reference-validated (`tests/python-reference/dump_llava_vision_ref.py`): seq/blk0/postln/embeds all **corr=1.0**.
      Text = Vicuna/llama (verified). **E2E correct**: red circle → "a red circle with a white background … a Japanese flag".
      Prompt = Vicuna v1: `<bos>{system} USER: [576 img] \n{q} ASSISTANT:`. Flags auto-detected from mmproj tensors
      (class_embd/pre_ln/post_ln presence, `clip.use_gelu`). Covers the whole LLaVA family.
- [x] **Qwen2.5-VL-3B VALIDATED + e2e correct.** New `Qwen25VlEncoder` (own ViT, doesn't share the SigLIP/CLIP path):
      Conv3D patch embed (2 temporal conv weights, summed since a single image fills both frames), **2D vision RoPE**
      (no position table; per-patch (h,w) freqs, merge-permuted), **window attention** (full only on layers 7/15/23/31 via
      `n_wa_pattern=8`; patches reordered into window-contiguous merge-units, block-diagonal mask), **RMSNorm** (no bias),
      **SwiGLU** MLP, and a **2×2 patch-merger** (`mm.0`→GELU→`mm.2`, 5120→2048). Patchify in merge-block order; rope +
      window reorder/un-reorder host-side. `IVlmImageEncoder` interface lets `MultimodalGenerator` hold either encoder.
      Reference-validated (`tests/python-reference/dump_qwen25vl_vision_ref.py`): embed/embed_win/blk0/postln/embeds all
      **corr=1.0**. Text = Qwen2.5/qwen2 (verified). **E2E correct**: red circle → "a red circle"; blue/green halves →
      "top … blue … bottom … green". Prompt = ChatML `<|im_start|>user\n<|vision_start|> [img] <|vision_end|>{q}<|im_end|>…`.
      Dynamic resolution (grid from pixel size; `HARTSY_VLM_IMGSIZE` for the test).
- [x] **Production wiring (sampler + image preprocessing).** `MultimodalGenerator` now runs through the real
      `SamplerChain` (temperature/top-p/repetition-penalty over decode history; greedy still selectable), default a light
      sampler. Added reusable `VlmImagePreprocessor` (bilinear resize any `[H,W,3]` → `size×size` + per-channel normalize,
      no Vision dependency) so callers feed arbitrary images; CLI gained `HARTSY_VLM_PNG=<file>` (real PNG decode via
      `Vision.Codec.PngDecoder` → preprocess → e2e). Verified: 400×300 PNG → SmolVLM → "a red circle".
- [x] **Qwen2.5-VL-7B — covered by existing code (build-defer verification).** Same `Qwen25VlEncoder` + qwen2 text as the
      validated 3B; only the size differs, so it is build-complete. 7B-Q4 (~4.5 GB) + vision likely fits the 3060 when the
      GPU is free; e2e verification deferred (no functional gap, just hardware/time).
- [~] **Llama-3.2-11B-Vision (mllama) — build-deferred as a DESIGN SPEC (no small proxy → unverifiable; not coding
      unvalidated guesswork).** Unlike every VLM above, mllama does NOT splice image tokens into the sequence — it injects
      vision features via **gated cross-attention layers** interleaved in the text decoder (`cross_attention_layers =
      [3,8,13,18,23,28,33,38]` for 11B). **To build (when an mllama-capable GPU/reference is available):** (1)
      `MllamaVisionEncoder` — its own ViT (tiled 560px, patch 14, 32 local + 8 global gated layers, pre/post tile position
      embeds, `class_embedding`, returns hidden states from multiple layers concatenated → a projector). (2) Core decoder
      change: a **cross-attention layer** variant in `GenericTransformer` where Q comes from text, K/V from the (cached)
      vision features, with a learned `tanh` gate on both the attn and FFN outputs (`cross_attn_attn_gate`,
      `cross_attn_mlp_gate`) and q/k RMSNorm. (3) mllama key-mapper + config flag marking which layers are cross-attn. The
      gated cross-attn decoder layer is the only genuinely new core primitive; the rest reuses the existing decoder + a
      CLIP-family ViT. Deferred because there is no <12 GB mllama variant to validate against (the build-defer policy relies
      on small same-arch proxies; mllama has none) and correctness can't be reference-checked on the 3060.
- [x] Per-encoder image-preprocessing — `VlmImagePreprocessor` (bilinear + normalize); per-model mean/std read from mmproj
      metadata. (Qwen2.5-VL dynamic-resolution smart-resize not replicated; fixed-square is sufficient for the synthetic
      harness and typical square inputs.)

---

## Phase 5 — Cross-cutting coverage (interleave as needed)

- [ ] **CPU GGUF** dequant path (also unblocks reference-free CPU parity tests) — overlaps Phase-0 cleanup
- [x] **Quant formats — codecs present + decode-validated.** `Gguf/Codecs/` already has Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1,
      Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, and IQ4_NL (rarer IQ2/IQ3/IQ1 mapped but no codec yet). **Verified by decoding bge-small
      in each quant and comparing the embedding to the HF F32 reference (cosine):** Q8_0 0.99994, Q5_K_M 0.99800,
      Q4_K_M 0.99709, Q3_K_M 0.99363 — all confirm correct decode. Q2_K 0.641 (output finite/normalized; the codec
      byte-matches ggml `dequantize_row_q2_K`, so this is genuine 2-bit degradation on a 33 M-param model, not a bug —
      Q2_K is not recommended for tiny models). Higher IQ-quants (IQ2_XXS/IQ3_S/etc.) remain unimplemented (rare).
- [x] **Embedding models VALIDATED (cosine = 1.0).** `BertEmbeddingModel` (`HartsyInference.LLM.Embeddings`): bidirectional
      post-norm BERT encoder (token + abs-position + token-type embeds → LayerNorm; per-layer self-attn via FlashAttention
      `causal:false` → +res → LayerNorm → GELU FFN → +res → LayerNorm), then pooling (CLS / mean per `bert.pooling_type`)
      + L2-normalize. Registered `bert`/`nomic-bert` as **passthrough** architectures so the GGUF loader keeps verbatim
      tensor names (the llama key-heuristic was mangling them). nn.Linear weights relabeled `[out,in]` like the VLM path.
      Reference-validated vs HF transformers (`tests/python-reference/dump_bert_embedding_ref.py`, same token ids):
      **bge-small-en-v1.5 (CLS) cosine=1.000000**, **all-MiniLM-L6-v2 (mean) cosine=1.000000**. E2E via the GGUF-vocab
      `BertWordPieceTokenizer` (`embed` CLI mode): cos(cat,kitten)=0.91 > cos(cat,car)=0.78. Covers bge/MiniLM/nomic/e5-BERT.
- [~] **GPT-OSS — build-defer design (20B/120B, no small variant → unverifiable).** Two specifics: (1) **attention sinks** —
      each head has a learned per-head sink logit included in the softmax denominator but not the weighted value sum
      (`softmax([scores, sink])`, drop the sink column from the output). This is a small `FlashAttention` change: an optional
      `float* sinkPerHead` that seeds the running denominator with `exp(sink - rowmax)`. (2) o200k tokenizer — already have the
      `gpt-4o`/o200k pre-tokenizer regex from Phi-4-mini, reuse for GPT-OSS. MoE (sigmoid/top-k) reuses the existing MoE path.
      Deferred: dense 20B+ won't fit the 3060 and there's no small GPT-OSS to reference-check the sink math.
- [x] Long-context / rope-scaling — covered: linear/YaRN/llama3/longrope scaling all wired + verified during Phase 1/3
      (Phi LongRope, DeepSeek YaRN, llama3 rope_freqs divisor fix). Batch>1 throughput + speculative decoding remain optional/future.
- [ ] Keep `docs/Checklists/PARITY_VERIFICATION.md` updated with each model's verified/pending status

---

## Ollama-popular coverage map (target → phase → 3060)

| Model | Phase | Runnable@3060 |
|---|---|---|
| Llama 1/2/3.x, Mistral, TinyLlama, SmolLM, Yi | 0/1d | ✅ |
| Qwen2 / Qwen2.5 / Qwen3 (dense) | 0 | ✅ |
| Gemma 2 / 3 (text) | 1a | ✅ verified (Gemma-3-1B, Gemma-2-2B) |
| Phi-3 / Phi-3.5-mini / Phi-4-mini | 1b | ✅ verified (all three) |
| StableLM-2 | 1c | ✅ verified (1.6B) |
| Granite-3 | 1c | ✅ verified (3.1-2B) |
| Command-R (cohere2) | 1c | ✅ verified (Command-R7B) |
| OLMoE, Granite-MoE | 2 | ✅ verified | 
| Qwen-MoE shared-expert path | 2 | ✅ unit-test-verified (HF reference) |
| Mixtral, Qwen3-MoE | 2 | `[~]` build-defer (wired; >12GB) |
| DeepSeek-V2-Lite | 3 | built + unit-tested; loads but OOMs 12GB GPU at preload |
| DeepSeek-V3, Kimi-K2 | 3 | `[~]` build-defer (MLA + DeepSeek-MoE done; V3 sigmoid/group-routing + q-LoRA TODO) |
| Qwen2.5-VL, Gemma-3-vision, LLaVA, MiniCPM-V | 4 | ✅ (small) |
| Llama-3.2-Vision (11B) | 4 | ❌ build-defer |
| nomic-embed / bge / MiniLM | 5 | ✅ |
| GPT-OSS | 5 | ❌/⚠️ |

---

# Completion plan — what's left to claim FULL LLM support

**Definition of "full LLM support":** every architecture family that Ollama / llama.cpp can run is either
(a) **verified e2e** on available hardware, or (b) **build-complete + slice/reference-validated** with e2e marked
build-defer for >12 GB. That spans six families — dense decoders, MoE, MLA, VLMs, embeddings/rerankers, and the
**non-attention** families (SSM, RWKV, hybrid, encoder-decoder). Phases 0–5 cover the first four for the common /
SOTA models. Phases 6–9 below are the remainder. Status legend: `[ ]` todo · `[~]` build-defer · `[x]` done.

## Phase 6 — Transformer-family completion (tractable + verifiable on the 3060)
- [x] **6a. Decoder-based embeddings — VALIDATED (cosine = 1.0).** `DecoderEmbeddingModel` (`HartsyInference.LLM.Embeddings`):
      runs the verified `GenericTransformer` decoder (`ForwardEmbeds(applyFinalNorm:true)`, headless — no lm_head needed),
      pools the final hidden states (last-token default per `pooling_type`, or mean) + L2-normalize. No new arch — pure reuse.
      Reference-validated vs HF transformers (`tests/python-reference/dump_decoder_embedding_ref.py`, shared ids):
      **Qwen3-Embedding-0.6B (qwen3, last-token) cosine = 1.000000** (maxdiff 7e-5). E2E semantic: cos(cat,kitten)=0.84 >
      cos(cat,car)=0.33. Covers gte-Qwen2 / e5-mistral / LLM2Vec (qwen2/llama-arch + pooling). Also made `GgufLanguageModel.Load`
      **tolerant of chat-template parse failures** (Qwen3-Embedding's template uses Python slicing the Jinja engine rejects →
      falls back to ChatML so embedding/raw-completion loads succeed). `embed` CLI auto-routes bert vs decoder by GGUF arch.
- [x] **6b. nomic-bert VALIDATED (cosine = 1.0).** Generalized `BertEmbeddingModel` to a config-driven encoder covering
      both families (detected from metadata + tensor presence): position = absolute table (bge) vs **rotary** (nomic,
      `rope.freq_base`); QKV = separate vs **fused `attn_qkv`** (split into q/k/v at load); MLP = GELU vs **SwiGLU**
      (`ffn_gate`·`ffn_up`, SiLU gate); biases present (bge) vs **none** (nomic, `Wopt`). Reference-validated vs HF
      (`trust_remote_code`, shared ids): **nomic-embed-text-v1.5 cosine = 1.000000** (maxdiff 0.0; GELU-gate gave 0.97 →
      SiLU/swiglu was the fix). bge/all-MiniLM still cosine 1.0 (unaffected). `jina-bert-v2` (ALiBi) + mxbai remain (add
      an ALiBi position-bias mode); nomic covers the popular rotary-BERT embedder.
- [x] **6c. Rerankers VALIDATED.** bge-reranker-v2-m3 (xlm-roberta, arch `bert`) reuses the `BertEmbeddingModel` encoder
      verbatim + a cross-encoder head: `BertEmbeddingModel.Score` runs the encoder, takes the CLS token, and applies
      `cls` (dense) → tanh → `cls.output` (→1 logit). The out_proj is stored rank-1 `[hidden]` → reshaped to `[1,hidden]`.
      Validated vs HF `XLMRobertaForSequenceClassification` (shared ids): relevant pair logit **4.63 (HF 4.61, FP16)**,
      irrelevant pair **−11.0435 (HF −11.0435, exact)** — correct ranking + magnitudes. The xlm-roberta SPM tokenizer
      (unigram + precompiled_charsmap) for self-contained pair encoding is the remaining e2e-convenience piece; the model
      math is validated. `embed` CLI gained `HARTSY_RERANK=1`.
- [ ] **6d. Exotic IQ-quant codecs**: IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ1_S / IQ1_M / IQ4_XS (only IQ4_NL
      exists). Each is a `Gguf/Codecs/Codec_*.cs` + decode test vs an F32 reference. Rare but needed for "every Ollama quant".

## Phase 7 — New architecture families (genuinely new model code — the real frontier)
The config-driven transformer does NOT cover these; each needs new core primitives. All have small variants that fit the 3060.
- [x] **7a. Mamba-1 (SSM) VALIDATED (cosine = 1.0).** `MambaModel` (`HartsyInference.LLM.Ssm`) — the engine's first
      non-transformer arch. Block = RMSNorm → in_proj → [x,z] → causal depthwise Conv1d → SiLU → x_proj → (dt,B,C) →
      dt_proj+softplus → **selective scan** (hₜ = exp(δA)·hₜ₋₁ + δB·xₜ; yₜ = C·hₜ + D·xₜ) → y·SiLU(z) → out_proj +
      residual. Linear projections via `IBackend`; conv/softplus/scan/gate host-side (the scan is sequential). Registered
      `mamba`/`mamba2` as passthrough. Reference-validated vs HF `MambaForCausalLM` (`tests/python-reference/dump_mamba_ref.py`,
      shared ids): **mamba-130m next-token logits cosine = 1.000000, argmax matches** ("The capital of France is"→" in").
      **Key bug found:** GGUF `ssm_a` already stores `A = −exp(A_log)` (llama.cpp bakes the −exp at conversion) — use it
      directly, don't re-apply −exp (caught by diffing GGUF vs HF weights: ssm_a cosine −0.18 → −exp(A_log) cosine 1.0).
      Mamba-2 + Falcon-Mamba reuse this path (Mamba-2 has scalar-A heads — a small variant).
- [ ] **7b. RWKV v6/v7** — WKV linear-attention recurrence + token-shift. New: WKV op, time/channel-mix blocks,
      `rwkv6`/`rwkv7` mapper. Verify: RWKV-1.6B/3B.
- [ ] **7c. Hybrid SSM+attention** — Jamba, Zamba2, Granite-4.0, Nemotron-H (Mamba blocks interleaved with attention +
      MoE). Composes 7a + the existing attention/MoE once 7a lands. Some fit at Q4.
- [x] **7d. T5/FLAN-T5 seq2seq VALIDATED (cosine = 1.0).** `T5Model` (`HartsyInference.LLM.Seq2Seq`) — full
      encoder-decoder from a `t5` GGUF. Handled T5 quirks: inner attn dim = n_heads·key_length (≠ d_model); **no 1/√d
      scaling** (attention via `ScaledDotProductAttention` with scale=1 + an additive per-head mask carrying the
      **relative-position bias**, bucketed — bidirectional for the encoder, causal/unidirectional for decoder self-attn,
      none for cross-attn; weights on block 0, shared); T5LayerNorm = RMSNorm; GeGLU FFN; untied lm_head (no logit scale).
      Reference-validated vs HF `T5ForConditionalGeneration` (`tests/python-reference/dump_t5_ref.py`, shared ids):
      **flan-t5-small encoder cosine = 1.000000, decoder first-token cosine = 1.000000** (argmax 644 "Das"). **E2E
      greedy translation**: "translate English to German: The house is wonderful." → **"Das Haus ist schön."** BART (same
      encoder-decoder shape, learned abs-pos instead of rel-bias) is a near-variant.

## Phase 8 — Build-defer completions (real code gaps; verify needs >12 GB hardware)
- [~] **8a. DeepSeek-V3 / Kimi-K2 routing + q-LoRA** — add to the MLA path: **sigmoid** router + `e_score_correction_bias`
      + **group-limited top-k** (node-limited routing) + **q-LoRA** (`q_a_proj`/`q_a_norm`/`q_b_proj` split; current MLA is
      direct-q only). Mapper keys exist. The only same-family proxy (DeepSeek-V2-Lite) uses softmax/direct-q, so the V3
      routing itself is verify-deferred.
- [~] **8b. Llama-3.2-Vision (mllama) cross-attention** — `MllamaVisionEncoder` + a **gated cross-attention decoder layer**
      (Q=text, K/V=cached vision, tanh gates, q/k RMSNorm) at layers `[3,8,13,18,23,28,33,38]` + `mllama` mapper. No <12 GB
      proxy. (Full spec in Phase 4.)
- [~] **8c. GPT-OSS** — `FlashAttention` per-head **sink logit** (seed the softmax denominator) + reuse o200k tokenizer +
      existing MoE. 20B+ only, no small variant to reference-check.
- [~] **8d. Verify-on-bigger-GPU (code already done)**: Mixtral 8x7B, Qwen3-MoE, Qwen2.5-VL-7B+, DeepSeek-V2-Lite e2e.

## Phase 9 — Production / serving quality (cross-cutting, mostly optional)
- [ ] **9a. Batch>1 / continuous batching / paged KV** — throughput for a server (`FixedKvCache` is single-sequence today).
- [ ] **9b. Speculative decoding** (draft model + verify) — latency.
- [ ] **9c. Long-context stress** — >32 k correctness, sliding-window-attention masks at scale (scaling math already verified).
- [ ] **9d. CPU GGUF parity tests** + fix the pre-existing `GgufRoundTripTests.RoundTrip_SimpleF32` expectation (declared
      arch vs passthrough — committed-code mismatch, not introduced by VLM/embedding work).
- [ ] **9e. (optional) Tool-calling / grammar-constrained / structured-output decode** — JSON-schema / GBNF guided sampling.

## Remaining-work summary (the short list)
| Item | Kind | Effort | Verifiable@3060 |
|---|---|---|---|
| 6a decoder-embeddings, 6b/6c BERT variants + reranker | reuse | low | ✅ |
| 6d exotic IQ-quant codecs | codec | low–med | ✅ (decode test) |
| 7a Mamba/SSM | **new arch** | high | ✅ (small) |
| 7b RWKV | **new arch** | high | ✅ (small) |
| 7c hybrid SSM+attn | compose | med | partial |
| 7d T5/BART seq2seq | new decode path | med | ✅ (small) |
| 8a DeepSeek-V3 routing+q-LoRA | code gap | med | ❌ defer |
| 8b mllama cross-attn | code gap | high | ❌ defer |
| 8c GPT-OSS sinks | code gap | low | ❌ defer |
| 8d Mixtral/Qwen-MoE/Qwen2.5-VL-7B e2e | verify only | — | ❌ (>12 GB) |
| 9a–9e serving/quality | infra | varies | ✅ |

**Bottom line:** transformer-family LLMs (dense, MoE, MLA, VLM, embeddings) are effectively complete for common/SOTA
models. To claim *full* support the open frontier is: **Phase 7 (Mamba/RWKV/hybrid/seq2seq — the only families the engine
fundamentally cannot run yet)**, the **Phase 6** cheap reuse wins, and finishing the **Phase 8** code gaps (verify when
bigger hardware is available).
