# LLM Architecture Parameter Audit (2026-06-28)

Cross-reference of every architecture hyper-parameter our config-driven LLM path consumes vs the
authoritative GGUF reference (llama.cpp `master`: `src/llama-arch.cpp` LLM_KV table, per-arch
`src/models/*.cpp::load_arch_hparams`, `src/llama-graph.cpp build_moe_ffn`, `tools/mtmd/clip*`).
Our engine is GGUF-metadata-driven, so llama.cpp's per-arch loader is the contract we must match.

Audit target files:
- `src/HartsyInference.LLM/Transformer/GgufConfigFactory.cs` — the only GGUF→config path (dense+MoE+MLA)
- `src/HartsyInference.LLM/Transformer/TransformerConfig.cs` — the config data model
- `src/HartsyInference.LLM/Transformer/GenericTransformer.cs` — forward use (whether a read param is applied)
- `src/HartsyInference.LLM/Transformer/MoeFeedForward.cs` — MoE routing/scaling
- `src/HartsyInference.LLM/Generation/GgufLanguageModel.cs` — remap / fused-split / arch registration
- `src/HartsyInference.LLM/Ssm/{MambaModel,RwkvModel}.cs`, `Seq2Seq/T5Model.cs`,
  `Embeddings/BertEmbeddingModel.cs`, `Multimodal/*`

Legend: **OK** read+used correctly · **PARTIAL** read but hardcoded/incomplete · **WRONG** read or
computed incorrectly · **MISSING** key exists & matters but never read · **UNSUPPORTED** whole arch absent.

---

## Implementation status

**Tier A — FIXED (2026-06-28).** A1–A9 + A10b landed and build clean; 44 LLM unit tests green incl. a new
`Flash_SlidingWindow_MatchesMaskedSdpa` parity test. The flash-attn CUDA kernel
(`native/cuda/lm/flash_attn_f32.cu`) gained a `slidingWindow` arg and the PTX was recompiled
(`src/HartsyInference.Cuda/Ptx/flash_attn_f32.ptx`). **A10a (granite RoPE pairing) is still VERIFY-only** —
not flipped pending a real Granite GGUF check (changing it blind could regress the working Granite path).
**e2e VALIDATED (2026-06-29):** StableLM-2-zephyr-1.6B Q4_K_M (A7/A8 LayerNorm+bias path, `arch=stablelm
bias=True qkNorm=False tied=False`) and Gemma-2-2B-it Q4_K_M (A2 sliding-window kernel) both greedy-decode
"The capital of France is Paris." on **CPU and CUDA**, with **byte-identical token ids across both backends**
— cross-validating the recompiled `flash_attn_f32.ptx` (new `slidingWindow`/`alibiSlopes` args) against the
CPU `AttentionReference`. Long-context (>window) SWA behavior is covered by the unit test, not the short e2e
prompt. A4/A5 (DeepSeek) and A1 (Qwen2-MoE) remain build-defer (>12 GB); covered by unit tests. (Note: loading
two GGUF models in one process still crashes — a pre-existing unmanaged-memory-across-models lifetime issue,
not caused by these changes; each model is correct in isolation.)

**Tier B — primitives + first new arches.**
- B1 (ALiBi: kernel+CPU+config+slopes, parity-tested) — landed.
- B3 (non-gated FFN + ReLU/ReLU² activations; post-norm placement `NormPlacement`) — landed.
- **OLMo-2 VERIFIED e2e (2026-06-29):** OLMo-2-0425-1B-Instruct Q4_K_M greedy-decodes "The capital of France is
  Paris." on CUDA (`arch=olmo2 qkNorm=True tied=False`, post-norm path, whole-vector q/k RMSNorm auto-detected,
  SplitHalf rope). Implemented as the `NormPlacement.PostNorm` primitive + olmo2 registration in
  `LlamaKeyMapper` (+ `post_attention_norm`/`post_ffw_norm` → sandwich-norm slots). 46 LLM unit tests green;
  pre-norm models (StableLM/Gemma-2) re-verified unchanged after the shared-path refactor.
- **GPT-2 lineage VERIFIED e2e (2026-06-29):** gpt2-medium Q4_K_M raw-completes "The capital of France is" →
  fluent coherent text on CUDA (78 tok/s, `arch=gpt2 bias=True`). This landed the **B2 absolute-position**
  primitive + the GPT-2-family infra all other arches in this lineage share: fused-QKV(+bias) split
  (`SplitFusedGpt2`), non-gated GELU FFN **with biases**, attention-output bias, LayerNorm biases everywhere,
  no-RoPE — via a new `Gpt2KeyMapper` + config flags `GatedFfn`/`FfnBias`/`AbsolutePositionEmbeddings`. Reusable
  for StarCoder-1 / GPT-NeoX / BLOOM / MPT / Falcon (each now mostly a preset + that arch's positional/bias quirk).
- **BLOOM VERIFIED e2e (2026-06-29):** bloom-560m Q4_K_M raw-completes "The capital of France is" → "Paris. …
  the city of light." on CUDA (65 tok/s). This **validates B1 (ALiBi) end-to-end** in a real model (the
  recompiled kernel's ALiBi path) and added the BLOOM **word-embedding LayerNorm** (`EmbeddingLayerNorm`);
  everything else reused the GPT-2 infra (`Gpt2KeyMapper` now also covers `bloom`, fused-QKV split generalized
  to read the arch prefix). All three engine primitives B1/B2/B3 are now e2e-verified.
- **InternLM2 VERIFIED e2e:** internlm2_5-1.8B-chat → "Paris is the capital city of France." (the llama.cpp
  converter de-fuses wqkv, so it is the plain llama dialect + interleaved RoPE — only arch registration needed).
- **Nemotron VERIFIED e2e:** Nemotron-Mini-4B-Instruct → "The capital of France is Paris." — validates B3's
  **squared-ReLU** activation + LayerNorm (LayerNorm1p baked in the GGUF) + partial rotary (64/128), non-gated FFN.
- **Verified-working arches so far:** StableLM-2, Gemma-2, OLMo-2, GPT-2, BLOOM, InternLM2, Nemotron (+ the
  pre-existing Qwen2/3, Llama, Phi-3, Gemma-3, Cohere, Granite, Mistral, Yi, SmolLM, etc.). 46 LLM unit tests
  green throughout.
- **Mamba2 (SSD) VERIFIED — cosine 0.999996 (2026-06-29).** New `Mamba2Model` (the engine's 2nd SSM family,
  first SSD/Mamba-2): fused `in_proj`→[z|xBC|dt], causal Conv1d over xBC, per-head scalar-A SSD scan (heads of
  head_dim, grouped B/C), gated grouped RMSNorm, no dt_proj. Validated vs HF `Mamba2ForCausalLM` on
  mamba2-370m (converted to GGUF locally via llama.cpp's `conversion/mamba.py`): last-token logits
  **cosine 0.999996, argmax exact**, and greedy generation **token-for-token identical** to HF ("The capital of
  France is a city of contrasts. It is a city of contrasts because"). Unlocks Codestral-Mamba. `mamba2` CLI mode
  + `Mamba2Runner` added. (Mamba2-groups>1 / hybrid SSM+attn remain.)
- **RWKV7 ("Goose") VERIFIED — coherent e2e (2026-06-30).** New `Rwkv7Model` — the generalized **delta-rule**
  recurrence (the other recurrent gap; far more complex than RWKV6's WKV6). Time-mix: fused token-shift lerp
  (r/w/k/v/a/g) → receptance/key/value + decay `w=exp(−0.606531·σ(w0+tanh(xw·w1)·w2))`, in-context-learning-rate
  `a`, value-residual mix, gate; L2-normalized `kk`, modified `k`; the **WKV7 state recurrence**
  `S=S·w+v⊗k+(a·S)⊗b` (a=−kk, b=kk·iclr), out=S·r; per-head GroupNorm + r·k bonus + gate + out_proj; channel-mix
  = squared-ReLU MLP (no receptance). Recurrence/decay/k-mod taken verbatim from llama.cpp's `rwkv-wkv7` op +
  `rwkv7-base.cpp`. Validated on RWKV7-Goose-0.4B GGUF: "The Eiffel Tower is located in the city of" → **"Paris,
  France. It is a symbol of the city and"** (coherent + factually correct; HF logits-cosine blocked only by the
  missing `fla` package). The engine's **SSM/recurrent family is now complete: Mamba-1, Mamba-2, RWKV-6, RWKV-7.**
  `rwkv7` CLI mode + `Rwkv7Runner` added.
- **StarCoder2 + EXAONE re-confirmed e2e (2026-06-30):** starcoder2-3b (`def fibonacci(n):` → correct recursive
  body) and EXAONE-3.5-2.4B ("The capital of France is **Paris**.") both greedy-decode correctly on CUDA.
- **GPT-NeoX VERIFIED e2e (2026-06-30):** Pythia-410m (converted to GGUF locally) → "The capital of France is
  Paris." on CUDA. This is the GPT-2 infra (fused QKV+bias split, non-gated GELU FFN+biases, LayerNorm+biases,
  untied head) + NEOX (split-half) **partial rotary 16/64** + GPT-NeoX-style **two-norm parallel residual** (attn
  and FFN each read the raw residual through a SEPARATE norm — distinct from Cohere's single-norm parallel
  residual; added `_parFfnNorm` to GenericTransformer). Reuses `Gpt2KeyMapper` (now also covers `gptneox`).
- **MiniCPM VERIFIED e2e (2026-06-30):** MiniCPM-2B-sft (chat template) → "The capital of France is Paris."
  (clean EOS) on CUDA. Llama dialect + the three Granite-style scalar multipliers (`embedding_scale=12`,
  `residual_scale`, `logit_scale`=9 as a divisor) wired via the existing Granite plumbing (`isGraniteLike`);
  NORM rope → Interleaved. Just arch registration in `LlamaKeyMapper` + the scalar reads.
- **GLM-4 VERIFIED e2e (2026-06-30):** GLM-4-9B-0414 Q4_K_M → "The capital of France is Paris. The Eiffel Tower
  is in Paris." (coherent + factually correct) on CUDA. New `Glm4KeyMapper` + config: Gemma-style **sandwich
  norm** but RMSNorm + **Q/K/V biases** + **fused gate+up FFN** (ffn_up=2·ffn, split via the generalized
  `SplitFusedPhi`) + **partial rotary 64/128** + NORM (Interleaved) rope, GQA 32/2. No softcap/SWA.
  - This also drove two reusable engine additions: **(1) Q5_0 + Q4_0 GPU dequant kernels**
    (`native/cuda/dequant/dequant_q{5,4}_0_to_f16.cu`, PTX recompiled, parity-tested vs the CPU codec +
    real-weight-validated on GLM-4's 20 Q5_0 `ffn_down` tensors). These legacy quants appear in most `*_K_M`
    mixes; previously they were force-dequantized to F32 at load (+4.5 GB resident → OOM). Added to
    `GpuSupportedQuant`. **(2) Large-vocab-head F16 logits path** — for a quantized LM head under low-VRAM,
    `ProjectLogits` now casts the (tiny) hidden state to F16 so the head GEMM dequants to F16 (~1.2 GB) instead
    of BF16 (which staged a ~2.4 GB F32 temp → OOM on 12 GB). Together these make 9B-class models with 150k+
    vocab (GLM-4, Qwen) actually run on the 3060.
- **Verified-working arches now:** + StarCoder2, EXAONE, GPT-NeoX, MiniCPM, GLM-4 (joining StableLM-2, Gemma-2,
  OLMo-2, GPT-2, BLOOM, InternLM2, Nemotron, Mamba-2, RWKV-7, and the pre-existing Qwen2/3, Llama, Phi-3,
  Gemma-3, Cohere, Granite, Mistral, Yi, SmolLM). 46 LLM unit tests + 7 GPU dequant tests green.
- **jina-bert-v2 VERIFIED — cosine 1.000000 vs HF (2026-06-30).** jina-embeddings-v2-base-en (mean-pooled,
  bit-exact). This added the **symmetric ALiBi** primitive to the flash kernel: bidirectional encoders use
  `−slope·|k−q|` (vs the causal `slope·(k−q)`), selected by `!causal` so no signature change (the causal/BLOOM
  path is unchanged; reduces to it when k ≤ q). PTX recompiled. Also fixed the encoder GEGLU bug — jina's gated
  MLP is GELU-gated (GEGLU), not nomic's SiLU-gated (SwiGLU) — and skipped the position-table lookup under ALiBi.
- **neo-bert VERIFIED — cosine 1.000000 vs HF (2026-06-30).** chandar-lab/NeoBERT (CLS token, bit-exact). New
  pre-norm RMSNorm encoder path (`NeoBlock`): RMSNorm → RoPE attn → +res → RMSNorm → SwiGLU → +res, single final
  RMSNorm, no embedding/token-type norm. Two non-obvious fixes the cosine harness caught: **(1) interleaved
  (complex `view_as_complex`) RoPE**, not NeoX rotate-half; **(2) per-head interleaved fused QKV**
  `[h0:(q|k|v)|h1:…]` (NeoBERT reshapes to `[heads,3·dim_head]` then chunks), de-interleaved at load — distinct
  from nomic's `[all_q|all_k|all_v]`. SwiGLU fused `ffn_up`=2·ffn split into gate|up. Registered `neo-bert` in
  PassthroughKeyMapper.
- **nomic-bert-moe VERIFIED — cosine 0.999737 vs HF (2026-06-30).** nomic-embed-text-v2-moe (mean-pooled; the
  ~3e-4 gap is the F16 GGUF + F16 MoE accumulation, not a structural error). Added a routed **MoE-in-encoder**
  path to `BertEmbeddingModel`: a dense/MoE interleave (`moe_every_n_layers`, MoE on `i % N == N-1`), an
  `ffn_gate_inp` softmax router selecting top-k of `expert_count` **non-gated GELU** experts, with the stacked
  expert-major `ffn_{up,down}_exps` tensors sliced per-expert at load and the fused `attn_qkv` **bias** split too.
  Key correctness detail from llama.cpp `build_moe_ffn(..., norm_w=false)`: the top-k weights are the **raw**
  softmax values, NOT renormalized to sum 1 (renormalizing gave cos 0.765; raw gave 0.9997).
- **Encoder track (B8) COMPLETE:** all three new encoder variants verified vs HF — jina-bert-v2 (1.000000),
  neo-bert (1.000000), nomic-bert-moe (0.999737) — joining the pre-existing bge / all-MiniLM / nomic-bert /
  xlm-roberta-reranker. The GEGLU-vs-SwiGLU activation bug the audit flagged is fixed (arch-aware gate activation).
- **OLMoE-1B-7B VERIFIED e2e (2026-06-30).** olmoe-1b-7b-0924-instruct Q4_K_M (64 experts, top-8, full-vector
  Q/K norm, no shared expert, no top-k renorm) greedy-decodes "The capital of France is Paris. Paris is the
  largest city in" on CUDA. First real-weight MoE validation (the routed-FFN path was unit-test-only before).
- **Qwen2-MoE expert-FFN bug FIXED (2026-06-30).** Some GGUFs (older Qwen1.5-MoE imatrix quants) omit
  `expert_feed_forward_length`; the code fell back to the dense `feed_forward_length` (the SHARED-expert size,
  5632) which is wrong for the routed experts (1408) → expert-split reshape crash. Fix: derive the routed-expert
  FFN width from the authoritative stacked-expert tensor shape (`mlp.up_exps`/`gate_exps` ggml dim 1) in
  `GgufConfigFactory`. Validated: the split now produces correct `[1408]` experts (Qwen1.5-MoE-A2.7B = 14 B total,
  too large to greedy-decode on the 12 GB 3060 even at Q4 — config/split validated, e2e pending bigger GPU).
- **MoE breadth remaining:** grok/dots1/arctic/bailingmoe/smallthinker etc. are mostly >12 GB (hardware-blocked).
  Engine gaps for them (from the llama.cpp audit): ReLU experts (smallthinker), `expert_gating_func==3`
  SOFTMAX_WEIGHT, correction-bias (dots1), `moe_every_n_layers` for *decoders* (have it for encoders now). These
  are config/routing presets buildable but not e2e-verifiable here.
- **InternVL2.5 VERIFIED e2e (2026-06-30).** InternVL2_5-1B (InternViT-300M + Qwen2-0.5B) on CUDA: a red circle →
  "The shape in the image is red.", a blue square → "The shape is blue." (genuinely reads the image). Added the
  **InternVL tower + projector** to `SiglipVlmEncoder`: InternViT = CLIP ViT + **LayerScale** (per-channel ls1/ls2
  via `AffineBroadcastLastDim` before each residual); projector = drop CLS → pixel-shuffle (same fold as Idefics3)
  → LayerNorm (`mm.model.mlp.0`) → Linear → GELU → Linear; ChatML `<img>…</img>` prompt. Also fixed a latent
  **MLP fc1/fc2 wiring bug**: the code hardcoded clip's SigLIP/CLIP *swapped* ffn names; InternViT uses the
  *natural* names. Now shape-driven (`FfnUpIsFc1`: the fc1/up projection is whichever ffn weight's out-dim equals
  `intermediate`), robust for both conventions.
- **Qwen2-VL VERIFIED correct (2026-06-30).** Qwen2-VL-2B-Instruct (red circle→"The shape is red.", blue
  square→"The shape is blue.") — branched the existing `Qwen25VlEncoder`: Qwen2-VL = **LayerNorm+bias** (vs 2.5's
  RMSNorm), **non-gated GELU** MLP (vs SwiGLU), **full attention everywhere** (no `n_wa_pattern` window), ffn dim
  derived from the tensor (the GGUF stores feed_forward_length=0); shared M-RoPE/Conv3D/2×2-merger/LLaVA-MLP
  projector. Detected structurally (ln1.bias / ffn_gate presence). NOTE: a pre-existing streaming-activation-pool
  race surfaces in the **vision tower** at very large patch counts (560 px → 1600 patches): correct under
  `CUDA_LAUNCH_BLOCKING=1` but flaky without (the pool accumulates transients across 32 ViT blocks; `Sync()` syncs
  the stream but doesn't reclaim the pool). Not Qwen2-VL-specific — affects any 560 px+ ViT on the fast path.
- **MiniCPM-V VERIFIED e2e (2026-06-30).** MiniCPM-V-2.6 (SigLIP tower + Qwen2-7B) — red circle→"The shape is
  red.", blue square→"The shape is blue." Added the **perceiver-resampler** projector to `SiglipVlmEncoder`:
  `query_num`(=64) learnable queries cross-attend the vision features (K = ln_kv(kv_proj(feat)) + 2D-sincos pos,
  V = ln_kv(kv_proj(feat)), Q = ln_q(query); MHA d_head=128 → out_proj → ln_post → proj into the 3584 text hidden)
  + **integer-bucketed** ViT positions (patch (i,j) → 70×70 table row floor(70i/grid)·70+floor(70j/grid), no
  interpolation) + ChatML `<image>…</image>` prompt. Worked first run.
- **VLMs now (9 verified):** Gemma-3, SmolVLM2, LLaVA-1.5, Qwen2.5-VL 3B/7B, Qwen2-VL-2B, mllama-11B,
  InternVL2.5-1B, MiniCPM-V-2.6.
- Remaining: pixtral tower (2D-RoPE ViT + IMG_BREAK; no small GGUF surfaced) + VLM smart-resize/tiling +
  structural MoE presets (>12 GB, code-only) + audio modality. jais (ALiBi) deferred (no small GGUF).

## 0. TOP PRIORITY — confirmed defects on architectures we already claim to support

These are not "exotic arch not registered"; they are wrong/missing params on models marked verified or
build-complete. Ordered by impact.

| # | Defect | Arch(es) affected | File / line | Confidence |
|---|---|---|---|---|
| 1 | **Qwen2-MoE top-k renorm defaults to TRUE; llama.cpp uses `norm_w=false`** | qwen2moe (Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B) | `GgufConfigFactory.cs:146` (`: arch != "olmoe"`) | High (code + llama.cpp source) |
| 2 | **Sliding-window masking never applied** — every `FlashAttention` call is `causal:true` with no window arg. `SlidingWindow` is read & stored but unused | gemma2, gemma3, cohere2, gpt-oss, phi3.5, exaone4, glm4, llama4 | `GenericTransformer.cs:637,772,881` | High (confirmed in code) |
| 3 | **DeepSeek MLA attention scale omits YaRN `mscale²`** and never reads `rope.scaling.yarn_log_multiplier`; uses generic mscale, not DeepSeek's `1+0.1·log_mul·ln(1/freq_scale)` form | deepseek2 (V2/V3 long-context) | `GenericTransformer.cs:770`; `GgufConfigFactory` MLA block | High (llama.cpp source) |
| 4 | **`stablelm` loads but is numerically WRONG** — uses RMSNorm where StableLM uses **LayerNorm with bias**; q/k LayerNorm treated as RMSNorm (biases dropped); `use_parallel_residual`, `attention.layer_norm_epsilon` (as LN) not honored | stablelm / stablelm-2 | `GgufConfigFactory` (only sets `UseLayerNorm` for cohere) | High |
| 5 | **MLA ignores `attention.key_length_mla` / `value_length_mla`** — DeepSeek-V2/V3 set separate MLA head dims (qk=192, v=128); we use `key_length`/`value_length` only | deepseek2 | `GgufConfigFactory` MLA block | Med-High |
| 6 | **`expert_weights_scale` dropped unless group-routing is on**, and `TransformerConfig.ExpertWeightsScale` is **dead** in the MoE forward (only `RoutedScalingFactor` is applied, in `RouteGroupLimited`) | dots1, bailingmoe, deepseek-v1, any scaled non-grouped MoE | `GgufConfigFactory.cs:159`; `MoeFeedForward.cs:227` | High |
| 7 | **Granite/GraniteMoE RoPE pairing is `SplitHalf`** but Granite shares llama's permuted wq/wk dialect → likely needs `Interleaved` | granite, granitemoe | `GgufConfigFactory.cs:188` (granite not in the interleaved list) | Med — VERIFY vs real checkpoint |
| 8 | **GPT-OSS alternating sliding-window pattern not configured** (`SlidingWindowPattern` falls through to 0 = all-global; gpt-oss is swa_period 2) + YaRN beta params from metadata unread | gpt-oss | `GgufConfigFactory` (swPattern default branch) | Med-High |
| 9 | **Mamba `ssm.dt_b_c_rms` flag + dt/B/C RMSNorm not applied** | FalconMamba, Jamba-style Mamba-1 | `MambaModel.cs` | High |
| 10 | **`nextn_predict_layers` (MTP) not read** — DeepSeek-V3 / GLM-4-MoE GGUFs may include an MTP layer in `block_count` that we'd run as a normal decoder layer | deepseek2-v3, glm4moe, bailingmoe2 | `GgufConfigFactory` | Med — verify per-checkpoint |

---

## 1. Dense decoders

### Well covered (OK)
`llama`, `qwen2`, `qwen3` (dense), `gemma`/`gemma2`/`gemma3` (text, modulo SWA mask #2), `phi3`
(LongRope solid; SWA matches llama.cpp's currently-disabled Phi SWA), `command-r`/`cohere2`
(LayerNorm + parallel residual + NoPE-on-global + logit-scale double-inverse all correct),
`granite` scalar multipliers (embedding/attention/residual/logit). Every core key
(`block_count`, `embedding_length`, `head_count`, `head_count_kv`, `feed_forward_length`,
`key_length`, `context_length`, `rope.freq_base`, `rope.dimension_count`, rms eps, rope.scaling) is
read and used for these.

### Defects / gaps (beyond §0)
- `attention.value_length` is only read inside the MLA branch — no dense field. Latent (all in-scope
  dense arches use `key_length == value_length`).
- YaRN `beta_fast`/`beta_slow`/`ext_factor`/`yarn_log_multiplier`/`alpha` are **never read from GGUF
  metadata** — only `RopeScaling` record defaults (32/1) are used. Llama-3 reads the precomputed
  `rope_freqs.weight` tensor (fine); metadata-only YaRN overrides are ignored.
- No `attention.causal`, `rope.scaling.finetuned`, generic `rope.freq_base_swa` (only the Gemma-3-only
  `rope.local_freq_base`), `router_logit_softcapping`, `attn_temp`/`f_attn_temp_scale`,
  `clamp_kqv`, `max_alibi_bias` fields.
- LayerNorm **bias** tensors are never loaded — `GenericTransformer` supplies a zero bias on the
  LayerNorm path. Breaks any arch with real norm biases (gpt2, gptneox, stablelm, falcon).
- `ActivationKind` has only `Silu` + `GeluTanh` — no square-relu (nemotron), plain-gelu, or ALiBi.

### Dense arches llama.cpp runs that we do NOT register at all
`gpt2`, `gptneox`, `falcon`, `mpt` (ALiBi), `starcoder`/`starcoder2`, `refact` (ALiBi), `bloom`
(ALiBi), `internlm2`, `olmo`, `olmo2` (**post-norm** residual layout), `exaone`/`exaone4`,
`nemotron` (square-relu), `glm4` (post-self-attn norm + partial rope + SWA), `bitnet`, `jais`
(ALiBi), `minicpm` (scalar multipliers like granite), `plamo`/`plamo2` (plamo2 = SSM hybrid),
`openelm` (per-layer varying head counts), `deci`/Nemotron (**per-layer variable** n_head_kv/ffn,
some no-attn layers), `dbrx` (MoE, clamp_kqv). Activation/ALiBi/abs-pos/post-norm/per-layer-array
support would be required for several of these — they are not just "a preset".

---

## 2. MoE

### Covered keys (GgufConfigFactory MoE block, lines ~120-161)
`expert_count`, `expert_used_count`, `expert_feed_forward_length`,
`expert_shared_feed_forward_length`, `expert_shared_count`, `expert_gating_func` (1 softmax / 2
sigmoid), `expert_weights_norm`, `leading_dense_block_count`, `expert_group_count`,
`expert_group_used_count`, `e_score_correction_bias` (`exp_probs_b`), `shared_expert_gate`. Stacked
expert split + router mapping work for qwen2moe/qwen3moe/olmoe/mixtral(llama)/granitemoe/deepseek2.

### Defects / gaps (beyond §0 #1, #6)
- `expert_gating_func == 3` (`SOFTMAX_WEIGHT`, softmax **after** top-k — smallthinker) not modeled.
- `nextn_predict_layers` (MTP) — MISSING (§0 #10).
- `interleave_moe_layer_step` / `moe_every_n_layers` — MISSING (llama4, ernie4.5-moe, nomic-bert-moe).
  `IsMoeLayer(i)` is purely `i >= FirstDenseLayers`; interleaved dense/MoE cadence unmodeled.
- `expert_group_scale` (grovemoe) — MISSING.
- `MoeFeedForward.SwiGlu` is **SiLU-only** — ReLU experts (smallthinker) / GELU experts inexpressible.
  One-router + optional-one-shared-expert shape — arctic's parallel dense+MoE and grovemoe chunk
  experts don't fit.

### MoE arches not registered at all
`dots1`, `bailingmoe`/`bailingmoe2`, `grok` (needs attn_output_scale + router/attn/final softcap +
emb scale 78.38), `arctic`, `ernie4_5-moe`, `hunyuan-moe`, `minimax-m1`/`m2` (linear-attn hybrid),
`phimoe` (Phi-3.5-MoE, sparse-mixer routing + LongRope), `glm4moe`, `smallthinker`, `llama4`,
`grovemoe`, `jamba` (SSM hybrid).

---

## 3. MLA (deepseek2)

Covered: `kv_lora_rank`, `q_lora_rank` (0=direct, V2-Lite), `value_length`, `rope.dimension_count`
(qk_rope), derived qk_nope, `kv_a_norm`/`q_a_norm` (by tensor presence),
`rope.scaling.original_context_length`. Q-LoRA + group-limited routing slice-verified.
Defects: §0 #3 (mscale²/yarn_log_multiplier), §0 #5 (key_length_mla/value_length_mla). `beta_fast/slow`
default 32/1 happen to match DeepSeek-V2 but are not read.

---

## 4. SSM / RWKV

| Family | Status |
|---|---|
| Mamba-1 (vanilla `state-spaces/mamba`) | OK — conv/scan/gating/A=−exp all correct |
| Mamba-1 + `ssm.dt_b_c_rms` (FalconMamba/Jamba) | **MISSING** dt/B/C RMSNorm (§0 #9) |
| **Mamba2** (codestral-mamba) | **UNSUPPORTED** — needs `ssm.group_count`, dt_rank-as-n_head, conv-over-BC, grouped `ssm_norm` |
| RWKV6 / Finch (standard) | OK — but `time_mix_extra_dim`/`time_decay_extra_dim` **inferred from tensor shapes**, not read; `token_shift_count` not read (PARTIAL) |
| **rwkv6qwen2** (hybrid) | **UNSUPPORTED** — no `time_mix_first` (u); our `TimeMix` derefs it → crash |
| **RWKV7 / ARWKV7** ("Goose") | **UNSUPPORTED** — delta rule; needs `attention.decay_lora_rank`, `iclr_lora_rank`, `value_residual_mix_lora_rank`, `gate_lora_rank`; structurally different time-mix |
| **plamo2** (SSM+attn hybrid) | **UNSUPPORTED** |

---

## 5. Seq2Seq (T5)

T5/FLAN-T5 symmetric: OK (`relative_buckets_count`, `key_length` inner dim, `decoder_start_token_id`,
rms eps, scale=1, rel-bias on layer 0, GeGLU). Gaps:
- **`decoder_block_count` MISSING** — we use one `NumLayers` for both encoder and decoder loops.
  Fine when enc==dec layers; asymmetric T5 variants would run wrong depth / crash.
- **T5ENCODER** (encoder-only) not arch-detected — `Load` always builds enc+dec.
- BART: N/A (no `LLM_ARCH_BART` in llama.cpp).

---

## 6. BERT / embeddings / rerankers

bge / all-MiniLM (absolute-pos), nomic-bert (RoPE + fused-QKV + SwiGLU/SiLU), decoder-embeddings,
xlm-roberta reranker head: OK & validated cos=1.0. Gaps:
- **FFN activation WRONG when a gate tensor is present on a non-nomic model** — GeGLU path uses
  `backend.Silu` (SwiGLU). Correct for nomic-bert (`LLM_FFN_SILU`); **jina-bert-v2** uses GEGLU
  (GELU gate), plain gated BERT uses `LLM_FFN_GELU`.
- **nomic-bert-moe UNSUPPORTED** — needs `moe_every_n_layers` + expert routing.
- **jina-bert-v2 (ALiBi) UNSUPPORTED** — no pos-embed + no RoPE → our loader requires one or the
  other and would throw at `position_embd.weight`; no ALiBi bias path (`f_max_alibi_bias=8`).
- **neo-bert UNSUPPORTED** — RMSNorm + RoPE + pre-norm; we use LayerNorm unconditionally.
- `attention.causal`, `tokenizer.ggml.token_type_count` not read (defaults OK for embedders).
- `attn_q_norm`/`attn_k_norm` (some jina/nomic) not applied (PARTIAL).

---

## 7. Vision / multimodal (mmproj `clip.*`)

Solid & validated (corr/cos = 1.0) on **single-tile square** inputs: Gemma-3 (SigLIP), Qwen2.5-VL,
mllama, base LLaVA-1.5-336, base SmolVLM. The ViT/projector math and the keys those paths depend on
are read and used correctly. Concentrated gaps:

1. **Smart-resize / dynamic resolution MISSING** (`VlmImagePreprocessor` does naive square bilinear) —
   never reads `image_min_pixels`/`image_max_pixels`/`preproc_*`. Breaks Qwen2.5-VL, Idefics3/SmolVLM,
   LLaVA-1.6, MiniCPM-V, InternVL on real arbitrary-aspect images (token count/positions diverge).
   Self-admitted at `LLM_MODEL_COVERAGE.md:250`.
2. **LLaVA `clip.vision.feature_layer` (−2 penultimate) not honored** — `Encode` runs all blocks
   (only the post-LN is skipped). Latent parity bug.
3. **`image_grid_pinpoints` + `mm_patch_merge_type` MISSING** — no LLaVA-1.6/AnyRes or llama4 tiling.
4. **Qwen2.5-VL `spatial_merge_size` (=2) and `window_size`/`attn_window_size` (=112) hardcoded**, not
   read. Correct for the shipping checkpoint only.
5. **`clip.use_silu` never read** (only `use_gelu`) — SiLU-FFN SigLIP variants mis-activated.
6. **`clip.vision.projector_type` never read** — connector chosen by tensor-name probing; can't
   distinguish qwen2vl vs qwen2.5vl vs qwen3vl, mlp vs mlp_norm vs ldp.
7. **mllama multi-tile MISSING** + `intermediate_layers_indices` hardcoded (single square tile).
8. `clip.vision.attention.head_count_kv` and `feature_layer` (array) never read (latent GQA/multi-tap).

### Vision arches in llama.cpp mtmd not implemented
`pixtral`, `internvl`, `qwen2vl_merger` (Qwen2-VL, distinct from 2.5), `qwen3vl_merger`, `glm_edge`/
`glm4v`, `ldp`/`ldpv2` (MobileVLM), `llama4` vision, `phi4`, `minicpm-v` (resampler;
`clip.minicpmv_version`/`minicpmv_query_num`), plus the entire **audio modality**
(`clip.has_audio_encoder`, `clip.audio.*`: ultravox, voxtral, qwen2a/qwen3a, granite_speech, etc.) —
all keys unread.

---

## Reference

All load semantics verified against llama.cpp `master`: `src/llama-arch.cpp` (LLM_KV),
`src/models/{deepseek2,qwen2moe,qwen3moe,olmoe,llama4,ernie4-5-moe,grok,smallthinker,dots1,
bailingmoe2,grovemoe,gemma3,stablelm,cohere2,openai-moe,phi3,granite,mamba,mamba2,rwkv6,rwkv7,t5,
bert}.cpp`, `src/llama-graph.cpp build_moe_ffn`, `src/llama-hparams.h`, `tools/mtmd/clip.cpp` +
`clip-impl.h`. Items marked **VERIFY** (granite rope pairing #7, gpt-oss YaRN-from-GGUF, MTP in
`block_count` #10) should be confirmed against a real downloaded checkpoint before fixing.
