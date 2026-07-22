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
> **UPDATE (2026-07-10): CUDA-graph decode DONE** for the plain dense GQA/RoPE shape (Llama/Qwen2/Qwen3/Mistral) —
> Qwen3-0.6B 2.57× faster (1.77× off llama.cpp, was 4.5×), byte-identical output verified. Greedy-only, opt-in
> `HARTSY_GRAPH_DECODE=1`. Also fixed a missing fused Q5_0 GEMV kernel (odd-hidden-dim models, e.g. qwen2.5-0.5b,
> were silently using the ~10-20× slower path for most of their weights — 2.5× fix).
>
> **Bug sweep (2026-07-10):** a live-API benchmark across all 22 supported GGUF architectures found and fixed:
> **mamba2/rwkv7/mamba1/rwkv6 were never actually wired** — the classes below were unit-test-verified in
> isolation but `GgufLanguageModel.Load` routed every architecture through the transformer config path
> regardless, which divides by `mamba2.attention.head_count` (0 for SSM archs) → instant crash. Now dispatch
> through a proper `SsmLanguageModel`/`SsmGenerationPipeline`, plus a new incremental per-layer state (was
> O(n²) full-sequence recompute per decode step, now O(1)/token) and a new RWKV-World byte-trie tokenizer
> (rwkv6/7 GGUFs have no BPE merges at all). **exaone and granite/granite-MoE both had the wrong RoPE pairing**
> (Interleaved vs SplitHalf swapped) — wrong pairing is a no-op at position 0, so short/greedy smoke tests
> (like the granite-MoE entry below) don't catch it; only shows up as sequence length grows. **nemotron** hit a
> CUDA double-free (`CudaBackend.Mul`'s finally block frees the same device pointer twice when a==b — nemotron's
> ReLU² activation is the only call site that does `Mul(x,x,x)`). **mllama** crashed on ANY text-only chat
> request — not an mllama bug at all, a Jinja engine bug (`for x in "someString"` should iterate characters,
> matching real Jinja2/Python, but threw `Value is not iterable`); mllama's own official chat template scans
> every message for image parts before checking `content is string`, so it hit this unconditionally.

## CLI catalog-path verification (2026-07-22)

Everything above this section is **engine-internal** parity verification (`MlaTests`/`MoeTests`/direct forward-pass
comparisons against HF) — it does not prove `hartsy text -m <id>` itself works. This pass actually ran
`hartsy text` (built `-c Release -f net10.0`; net8.0 runtime is absent on this box) against real downloaded
GGUFs for every catalog entry added by the prior CLI-wiring session, per
[LLM_CLI_CATALOG_HANDOFF.md](LLM_CLI_CATALOG_HANDOFF.md). Two GPUs used in parallel — RTX 3060 (12GB,
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`) and RTX 4090 (24GB, `=1`), confirmed empirically. Full
per-model prompt/output/dmon logs and exact repro commands are preserved in this session's transcript; the
`ModelCatalog.cs` entries themselves carry the source repo + sha256 for every model that passed.

**Real bugs found and fixed this pass** (all verified against the existing 129-test `HartsyInference.LLM.Tests`
suite, no regressions, plus re-verified against the real GGUF that surfaced each one):

1. **Jinja `is true`/`is false` test was unimplemented** (`ChatTemplates/JinjaExpr.cs` `IsTestExpr.Eval`) —
   threw `NotSupportedException`, crashing `--no-thinking` against Qwen3's real chat template (the prior
   synthetic unit test used a fake template that never exercised this). Fixed; `--thinking`/`--no-thinking` now
   confirmed to produce genuinely different real output (visible `<think>` reasoning trace vs. a direct terse
   answer) against Qwen3-4B.
2. **`ModelResolver.LocateLocal` resolved a Text catalog id to its *directory*, not the `.gguf` file inside it**
   (`Cli/Infra/ModelResolver.cs`) — `TextService.LoadInto` requires a direct file path and throws on a
   directory. This broke `-m <id>` end-to-end for **every** Text catalog entry, even ones with `Assets`
   correctly wired, since `ModelAcquisition.EnsurePresent`'s asset-based path only fills `LocalPath` when it's
   still null, and the (wrong) directory resolution ran first. Fixed: Text modality now only accepts a direct
   file match, or auto-discovers a single `*.gguf` inside the directory, falling through to `null` otherwise so
   Assets-based download can resolve it. Verified: `hartsy text -m qwen3 "9×7?"` → "63", zero `--model-path`.
3. **`BuildVisionSampling` defaulted `RepetitionPenalty` to 1.0** (`Engine/Services/TextService.cs`) — silently
   shadowed `MultimodalGenerator`'s own documented intent of defaulting VLM sampling to 1.1 (small quantized
   VLMs are repetition-prone). Fixed to 1.1.
4. **RWKV-World's `tokenizer.chat_template` GGUF field is a bare format-name sentinel** (`"rwkv-world"`, not
   real Jinja source — some llama.cpp converters store this as a signal for llama.cpp's own hardcoded prompt
   tables) (`Generation/GgufLanguageModel.cs` `BuildTemplate`). The engine "compiled" it as literal text (no
   `{{`/`{%` syntax to fail on), so every render silently produced that same constant regardless of the actual
   prompt — confirmed via 3 different prompts giving byte-identical garbage output before the fix. Fixed:
   detect templates with no substitution syntax and route straight to `ChatMlTemplate`.
5. **`JinjaExpr.cs`'s slice parser only supported 2-part `[start:stop]`** — Qwen3.5's real template uses
   `messages[::-1]` (3-part, negative step = reverse), which failed to compile, silently falling back to
   ChatML — meaning `--thinking`/`--no-thinking` produced *identical* output (exactly the failure mode the
   handoff doc asked to check for). Fixed: `SliceExpr` gained a `step` param with Python-style reverse-walk
   semantics for negative step.
6. **`JinjaChatTemplate.Encode` had no render-time fallback**, only `GgufLanguageModel.BuildTemplate`'s
   compile-time one — a template that compiles but fails on a specific real conversation shape (exposed by
   fixing #5, which surfaced a *separate*, still-unresolved `tojson | safe` filter-chain gap in Qwen3.5's
   tool-calling branch) would have been a hard crash. Fixed: `Encode()` now falls back to ChatML on any
   render-time exception too, logged via `[WRN]` instead of failing generation.

**Found, NOT fixed** (real new-architecture-support or deep-forward-pass work, correctly left out of scope for
a CLI-verification pass — see the `ModelCatalog.cs` comments on each entry for full detail):
- `qwen35`: the `tojson | safe if ... is mapping` filter-chain gap noted above (Bug from #6's fix).
- `glm4`: a wrong-architecture GGUF (`chatglm`, not `glm4`) crashes with `ArgumentOutOfRangeException` via the
  key-heuristic fallback instead of a clear "unsupported architecture" error; separately, the *correct*
  `glm4`-arch checkpoint loads and generates but produces consistently incoherent output (not root-caused).
- `gemma3-vision`: confirmed real, deeply diagnosed (vision-tower math proven numerically correct via a
  from-scratch PyTorch parity replay, cosine ≈ 1.0 at every stage), still consistently hallucinates unrelated
  content. See the VLM section below.
- `gpt-oss`: every available public GGUF (native MXFP4 release and community "Q4_K_M" repacks, which keep the
  MoE expert tensors in MXFP4 regardless of label) fails with `Unsupported GGUF tensor type: 39`
  (`GGML_TYPE_MXFP4`) — `GgufLoader.MapGgufType` has no case for it. An engine gap, not a size/VRAM limit.
- `gemma4-moe`: throws `KeyNotFoundException` on `model.layers.0.mlp.experts.0.gate_proj.weight` — this real
  checkpoint's MoE expert tensor naming doesn't match what `MoeFeedForward.LoadWeights` expects.
- `deepseek-v2-lite`: loads and generates without crashing but output is incoherent garbage — a real,
  unconfirmed MLA-attention-path bug (not a VRAM/size limit — this one fits the 4090 fine).

**Host-glue / GPU-utilization finding** (the specific class of bug this project already found once in
diffusion — see memory `dit-blocks-must-be-gpu-resident`): the dense/MoE `GenericTransformer` per-layer hot
path is **clean** — confirmed by direct code read, every attention/MLP op is a `backend.*` GPU call; the only
CPU touches (token-embedding gather, RoPE cos/sin table precompute) happen once per forward pass, not per
layer, matching the already-accepted diffusion RoPE precedent. **The SSM/hybrid families are a different
story**: `Mamba1Model`/`Mamba2Model`/`RwkvModel`/`Rwkv7Model` run their causal Conv1d AND the actual
selective-scan / WKV recurrence **host-side, every layer, by design** (their own doc comments: "the recurrence
is inherently sequential... run host-side"; only the big linear projections go through `IBackend.Linear`).
Measured real-world cost: `mamba` (2.8B) took **26.6s for ~20 tokens (~0.8 tok/s)** with `nvidia-smi dmon`
showing GPU sm% pinned at 2-14% the *entire* run — vs. 2-7s for comparable/larger dense models. This is a
legitimate architectural tradeoff (a fused parallel-scan CUDA kernel is real specialized engineering, the same
class of justified exception as diffusion's CPU-side interleaved RoPE), not an oversight bug — but it's a real,
substantial, measured performance cost worth a dedicated kernel if Mamba/RWKV throughput becomes a priority.

**Verified PASS via `hartsy text -m <id> "<prompt>"`** (coherent + correct output, judged by reading the actual
text, not just exit code — factual/checkable prompts like "capital of Italy" → "Rome", "12×8?" → "96",
"translate 'good morning' to Spanish" → "¡Buenos días!"): `qwen3` (+ `--thinking`/`--no-thinking` real
difference), `llama3`, `mistral`, `gemma`, `phi`, `granite3`, `olmoe`, `granite-moe`, `gpt2`, `starcoder2`
(correct code completion), `mamba` (correct but slow, see above), `qwen25-vl`/`llava15`/`smolvlm2` (`--image`,
see VLM section). `stablelm2` and `gemma4` pass on a factual prompt but `stablelm2` failed a separate
arithmetic prompt (small-model capability limit, not an engine bug) and `gemma4` was notably slow with low GPU
util (inconclusive — likely load-dominated for such a short completion, not chased further).

**PARTIAL**: `qwen35` (SSM forward pass verified correct in all modes via the ChatML fallback; `--thinking`
against the model's *real* template still blocked by the unresolved tool-calling filter gap above) — `rwkv`
(the engine bug is fixed and prompt-sensitivity restored, but the only downloadable RWKV-6-World checkpoint is
a base multilingual model, not ChatML-tuned, so its answers are often still wrong against this CLI's
ChatML-formatted single-turn prompting — needs an instruction-tuned RWKV GGUF for a true pass) —
`llama32-vision` (see VLM section) — `command-r` (genuine kernel OOM on this 62GB-RAM box even at Q3_K_S,
confirmed via `dmesg`; peak resident RAM was ~3.1× the file size, not the loader's 2.5× headroom estimate —
worth widening that safety margin).

**FAIL**: `gemma3-vision`, `glm4`, `deepseek-v2-lite` (still open — see the follow-up pass below for `gemma4-moe`/`gpt-oss`).

## Follow-up fix + perf pass (2026-07-22, same day)

Went back through the 5 "found, not fixed" bugs above and did a full perf sweep of the rest of the LLM stack.
Split the 5 by what's actually fixable/verifiable on this box rather than treating them as a uniform batch —
see each entry for what changed and what's still open. No regressions: full `HartsyInference.LLM.Tests` (130,
was 129) + `HartsyInference.ModelAssets.Tests` (363, one pre-existing unrelated EnCodec failure confirmed via
`git stash` to predate this session) + the touched `HartsyInference.Cuda.Tests` all green.

**Fixed and e2e-verifiable (mechanical robustness fix):**
- `glm4` bug (1): `GgufModelLoader.Load` now wraps the tensor-remap loop and, when the GGUF's declared
  `general.architecture` has no registered `IGgufKeyMapper` (the `chatglm`-vs-`glm4` case here, but this is
  architecture-agnostic — any declared-but-unregistered arch gets the same treatment), translates whatever the
  heuristic-fallback mapper throws (`ArgumentOutOfRangeException` et al.) into a clear
  `UnsupportedModelException` naming the declared architecture and the heuristic-matched mapper, instead of an
  opaque crash. `glm4` bug (2) — the *correct*-architecture checkpoint's incoherent output — is **not**
  root-caused (needs a reference-logit comparison against real GLM-4, out of scope for this pass); `glm4` stays
  FAIL.

**Fixed (code + unit-tested), NOT e2e-verifiable on this hardware — do not read as "gpt-oss/gemma4-moe now
work end-to-end", only that the specific reported crash is gone:**
- `gpt-oss`: added real MXFP4 (OCP microscaling 4-bit, ggml type 39) decode support —
  `DType.MXFP4` (17 bytes / 32 elements: 1-byte E8M0 scale + 16 bytes of packed E2M1 codewords) +
  `Codecs/Codec_MXFP4.cs` + `GgufLoader.MapGgufType` case 39 + `GgufCodecRegistry` registration. Implementation
  verified byte-for-byte against upstream ggml source (`ggml-common.h`'s `block_mxfp4`/`kvalues_mxfp4` table,
  `ggml-quants.c`'s `dequantize_row_mxfp4`, `ggml-impl.h`'s `ggml_e8m0_to_fp32_half`), not derived from memory —
  fetched live from `ggml-org/llama.cpp` and hand-verified the E8M0→FP32 bit manipulation for both the
  denormal (`e<2`) and normal branches. Two new `GgufCodecRegistryTests` cases pin the exact reconstruction
  formula. Since MXFP4 isn't in `GgufLanguageModel.GpuSupportedQuant`, it automatically routes through the
  existing CPU-dequant-to-F32 fallback at load — no new CUDA kernel needed for correctness, same pattern every
  other unsupported-on-GPU quant type already uses. Still can't verify e2e: every gpt-oss checkpoint exceeds
  this box's VRAM regardless (per the existing build-defer entry).
- `gemma4-moe`: root-caused via the real checkpoint's own header (fetched `unsloth/gemma-4-26B-A4B-it-GGUF`'s
  `gemma-4-26B-A4B-it-UD-Q3_K_M.gguf` — the exact catalog source — over an HTTP range request, no full 12.7GB
  download needed). The checkpoint fuses the MoE gate+up expert projections into one `blk.N.ffn_gate_up_exps`
  tensor (llama.cpp's `LLM_TENSOR_FFN_GATE_UP_EXPS`), not the separate `ffn_gate_exps`/`ffn_up_exps` pair
  `Gemma4KeyMapper` and `GgufLanguageModel.SplitStackedExperts` only handled — the fused tensor had no mapper
  case at all, so it silently dropped, `SplitStackedExperts` no-op'd for gate/up, and `MoeFeedForward.LoadWeights`
  hit `KeyNotFoundException` on the per-expert key that was never created. Fixed: `Gemma4KeyMapper` now maps
  `ffn_gate_up_exps.weight`; `SplitStackedExperts` gained a `SplitFusedGateUpExperts` path (gate = first half
  of each expert's row-block, up = second half — matches ggml `build_moe_ffn`'s `gate_up_exps` view split
  exactly, confirmed by reading `llama-graph.cpp` upstream) alongside the pre-existing separate-tensor path.
  New `GgufMoeExpertSplitTests` regression test (made `SplitStackedExperts` `internal` for testability) pins
  the exact row-slicing math with hand-computed expected values. **Residual gap, deliberately not fixed**: the
  same real checkpoint also carries `ffn_down_exps.scale` (and every UD-quant expert tensor implicitly a
  `w_s`-style per-expert post-matmul scalar in ggml's `build_lora_mm_id`) that `MoeFeedForward` has no concept
  of at all — left unmapped (silently dropped, same as before) rather than half-wired. Numerically, output will
  likely still be off by a per-expert scale factor even once VRAM allows a real run. Flagged in
  `Gemma4KeyMapper.cs` and here, not silently swept under the KeyNotFoundException fix.

**Perf: two cheap levers pulled, both real GPU-kernel reuse (no new kernels written), both verified live:**
- `Qwen25VlEncoder` (Qwen2.5-VL's own ViT, used by `qwen25-vl`) ran its 2D RoPE **host-side, per-layer, twice
  per layer** (Q and K), with a `backend.Sync()` each call — ~64 syncs for a 32-layer tower on a single image.
  The exact same rotate-half math is already a real GPU kernel, `IBackend.ApplyRopeSingle` (the one
  `GenericTransformer`'s own dense-model RoPE uses) — proved algebraically identical formula-for-formula before
  touching anything. Replaced: cos/sin tables now upload once per image (not recomputed per layer) as
  `[1, np, headDim]` tensors, `Block` calls `backend.ApplyRopeSingle` directly, the host loop + its two
  per-layer syncs are gone. **Verified live**, not just theorized: re-ran `hartsy text -m qwen25-vl -i
  tests/HartsyInference.Vision.Tests/TestData/bus.png` on the 4090 before deleting the model again to reclaim
  disk — output unchanged in substance from the documented pre-fix baseline (correctly identifies the bus as
  blue, reads the "cero emisiones" text on it).
- Granite-3 / MiniCPM (`EmbeddingScale != 1.0`) were **unconditionally excluded** from CUDA-graph decode
  (`GenericTransformer.SupportsGraphDecode`) even though nothing else about them is graph-incompatible — the
  actual and ONLY blocker is that `IBackend.EmbedGatherDecodeStep` (the on-device embed-gather kernel the
  captured graph uses) has no scale parameter, unlike `EmbedLookup`'s ordinary host gather which multiplies
  per call. Fixed without touching any kernel/PTX (no `nvcc` in this environment, and none was needed): 
  `EnsureEmbedResidentForGraphDecode` now lazily builds a **separate** GPU copy of the embed table pre-scaled
  once via the existing `backend.Scale` op, used only by the graph-decode gather; the plain `_embed` table
  `EmbedLookup` uses stays unscaled. New `GraphDecodeEmbeddingScaleTests` (real `CudaBackend`, not a fallback)
  verifies both the eligibility flip and that the returned table's actual GPU-resident values equal
  `embed * EmbeddingScale` to 1e-4, while `EmbedLookup`'s own path is unaffected — **run on real hardware**,
  passed.

**Perf: surveyed, real findings, NOT implemented this pass (documented so they aren't re-discovered from
scratch) — dispatched as parallel read-only investigations, not deep-verified beyond what's noted:**
- **Missing fused GEMV kernels for `Q4_1`/`Q5_1`/`Q2_K`/`Q3_K`/`Q8_K`/every `IQ*` type** — same class of bug as
  the already-fixed Q5_0 gap (2.5× win when found). `CudaBackend.LinearImpl`'s fused-GEMV dispatch chain only
  covers `Q4_0`/`Q4_K`/`Q5_0`/`Q5_K`/`Q6_K`/`Q8_0`; anything else silently falls to the ~10-20× slower generic
  cuBLAS dequant-to-F16 path, with **no log line distinguishing fused vs. fallback** (the Q5_0 gap was found by
  comparing relative model speeds by hand, not a log). Currently latent for cataloged models: `command-r`
  (Q3_K_S) and `gemma4-moe` (Q3_K_M) are the only Q3_K-quant catalog entries and both are blocked by other bugs
  before ever reaching decode, so this hasn't cost real wall-clock yet — but it will the moment either unblocks.
- **`starcoder2`, `phi3`, `stablelm2` are already structurally eligible for CUDA-graph decode today** —
  `SupportsGraphDecode` is a pure feature-test on `TransformerConfig`, not an architecture allowlist, and none
  of these three trip any of its exclusions (confirmed by reading `GgufConfigFactory`'s per-arch config for
  each). Nobody has actually run them with `HARTSY_GRAPH_DECODE=1` and confirmed the speedup + byte-identical
  output the way `qwen3` was — free win, just needs the verification pass, not new code.
- **`T5Model.Generate` has no KV cache** — every decode step re-embeds and reruns the *entire* decoder from
  scratch over the growing output (O(steps²)), and `RelBiasMask`'s host-side relative-position-bucket triple
  loop is rebuilt every step (shared correctly across layers within one call, but fully recomputed step to
  step, growing with sequence length). Not "cheap" — a real KV-cache retrofit — but the single biggest lever
  found this pass if T5/FLAN-T5 generation ever gets wired into `hartsy text` (currently not reachable via the
  CLI at all, see the existing `t5` catalog entry).
- **`BertEmbeddingModel`** (nomic-bert/neo-bert rotary variants) has the same host-RoPE-with-per-layer-sync
  pattern as Qwen25VlEncoder had, plus its optional `MoeFfn` path (nomic-bert-moe) does per-token LINQ top-k
  selection, several `new float[]` per call, and a `backend.Sync()` **inside** the per-expert loop. Lower
  severity than the VLM case (single pass per sequence, not the dominant cost), not fixed this pass.
- SSM/RWKV's host-side recurrence (Mamba/RWKV, and `Qwen35Model`'s Gated-DeltaNet layers, confirmed to share
  the identical accepted tradeoff) is **not** a cheap lever — re-confirmed, not chased further; a fused
  parallel-scan kernel is real dedicated engineering, same class of justified exception as diffusion's
  CPU-side interleaved RoPE.

**Build-defer, confirmed too large for this box** (real ungated sources found via HF API, sizes measured):
`mixtral` (Q4_K_M ≈26GB VRAM, exceeds the 24GB 4090 — not downloaded, disk budget spent on the other five),
`qwen3-moe` (Qwen3-30B-A3B Q4_K_M, 18.6GB — downloaded, loaded, then genuine `CUDA_ERROR_OUT_OF_MEMORY` during
weight preload on the 4090 even with `--low-vram-quant`), `deepseek-v3` (671B/1T — real sources confirmed to
exist, not attempted at any quant, not attemptable on a single consumer box). TODO once a rented/larger GPU is
available: re-verify all six build-defer entries (`mixtral`, `qwen3-moe`, `deepseek-v2-lite`, `deepseek-v3`,
`gpt-oss`, `gemma4-moe`) — `gpt-oss` (MXFP4 tensor-type decode) and `gemma4-moe` (fused gate_up_exps key
mapping) got the engine fixes described in the 2026-07-22 follow-up pass above; `gemma4-moe` also needs
`ffn_down_exps.scale` support (not yet built, see that section) before its numerics can be trusted even once
VRAM allows a run.

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
| **Gemma-4** (E2B/E4B mobile) | Gemma-4-E2B-it Q4_K_M | Ported cold from llama.cpp's `src/models/gemma4.cpp` (no local reference model existed) — per-layer embeddings (PLE, Gemma-3n-lineage), per-layer head dim (global 512 vs local/SWA 256 on E2B — the Q/K/V/O weight shapes themselves differ, not just RoPE theta), cross-layer KV-cache sharing (donor-layer formula from llama.cpp's kv-cache `reuse` callback), weightless V RMSNorm, optional per-layer output scale, per-layer FFN width (derived from the loaded weight's own shape, not a config constant — some layers are 2× wider). Found 3 real bugs along the way, none Gemma-4-specific: our GGUF parser silently discarded BOOL-typed arrays (`sliding_window_pattern` is a genuine per-layer array, not a broadcast period — every layer was reading as "global"); our `Linear`'s `Tensor.Shape[0]`=outDim/`Shape[1]`=inDim convention got reversed in my first pass; `tokenizer.ggml.model="gemma4"` wasn't routed to the SentencePiece tokenizer (fell through to generic BPE, dropping the `▁`→space substitution — coherent but "Thecapitalof..."). Also fixed 4 general Jinja chat-template bugs surfaced by Gemma-4's tool-calling template (unary minus, block-form `{% set %}...{% endset %}`, `range()`, `is sequence`). Verified live: coherent, factually correct, properly-spaced multi-sentence generation. |

Qwen-MoE shared-expert path is unit-test-verified against an HF reference (no 14 GB GGUF needed).

## LLM — build-defer (🔧, wired but >12 GB)

| Model | Notes |
|---|---|
| **Mixtral 8x7B** (47B) | `llama` arch + experts, interleaved RoPE, renorm; config + mapper + stacked-expert split wired. |
| **Qwen3-MoE 30B-A3B / 235B** | `qwen3moe`, per-head Q/K norm, no shared expert; wired. |
| **DeepSeek-V2-Lite** | MLA + DeepSeek-MoE built + `MlaTests` pass; loads but OOMs the 3060 at preload. |
| **DeepSeek-V3 671B / Kimi-K2 1T** | MLA + MoE + **V3 node-limited routing (sigmoid + e_score bias + group top-k + routed_scaling) + q-LoRA query** all built & **slice-verified** (`MoeTests` group-routing vs HF `noaux_tc`, `MlaTests` q-LoRA block vs host ref). e2e >12 GB. |
| **GPT-OSS 20B / 120B** | Per-head **attention sinks** built (CPU+CUDA, PTX recompiled) & **slice-verified** (`FlashAttentionTests.Flash_Sink_*`); `gpt-oss` arch/mapper/config wired. MoE + o200k tokenizer reused. MXFP4 tensor-type decode added 2026-07-22 (`DType.MXFP4`/`Codec_MXFP4`) — the load-time crash every public checkpoint hit is fixed. e2e 20B+ still deferred (VRAM, not an engine gap anymore). |
| **Llama-3.2-Vision-11B (mllama)** | ✅ **VERIFIED e2e on the 3060** (leafspark Q4_K_M + mmproj-F16, low-VRAM): red circle→"red", blue square→"a blue square with a white outline…". The only splice-free VLM — vision feeds gated cross-attention layers `[3,8,13,18,23,28,33,38]`. `MllamaVisionEncoder` (560px ViT, class embd, pre/post-tile + dual-gated position embeds, 32 local + 8 gated-global, intermediate-concat `[3,7,15,23,30]`→7680→`mm.0`→4096) **reference-validated cos=1.000000 on every stage** (`dump_mllama_vision_ref.py`). Key finding: Ollama's converter **pre-tanh's all gates** (and `1−tanh` for `position_embd.gate`, splitting HF's single gate into two) so the forward multiplies gates directly; MLP names clip-swapped; q/k permute is a no-op (no vision RoPE). `MllamaCrossAttentionLayer` slice-verified (`MllamaCrossAttentionTests`); `MllamaGenerator` (no token splice, `crossStates` threaded through `ForwardEmbeds` every step). Covers the whole mllama family. |
| **Gemma-4 31B-dense / 26B-A4B-MoE** | Same `gemma4` arch/config/forward path as the verified E2B row above — the 26B-A4B variant additionally exercises `ParallelDenseMoeBranch` (routed-expert FFN running IN PARALLEL WITH the dense/"shared" branch, each own pre/post norm, summed — a genuinely different pattern from every other MoE arch here, which fuses a shared expert into the router output instead) and `ComputeRouterLogits`'s separately-normalized router input. Built and compiles; not e2e verified (>12 GB, exceeds the 3060 — per user directive, no local load attempted). Real checkpoint's fused `ffn_gate_up_exps` MoE tensor now handled correctly (2026-07-22, was a `KeyNotFoundException`) — `ffn_down_exps.scale` still unsupported, see the follow-up pass section above. |

## VLMs (vision-language) — verified end-to-end (✅)

| Model | Notes |
|---|---|
| **Gemma-3-4B-vision** | SigLIP + avg-pool/RMSNorm/Linear projector. Tower corr 1.0 vs reference; e2e coherent on simple synthetic shapes. 2 bugs fixed (swapped SigLIP MLP names; relabel-not-transpose). ⚠️ **CLI pass 2026-07-22 (real photo, not a synthetic shape) found a real, still-unresolved FAIL**: `hartsy text -m gemma3-vision -i <bus photo>` consistently hallucinates an unrelated scene despite the vision-tower math independently re-verified as numerically correct (fresh PyTorch parity replay, cosine ≈1.0 every stage) — see MODEL_STATUS_LLM.md's "CLI catalog-path verification" section above. Simple-shape tests (red circle, blue square) are apparently not sufficient to catch this; a real-photo regression test is worth adding. |
| **SmolVLM2-2.2B** | SigLIP + idefics3 pixel-shuffle projector. Tower corr 1.0; e2e correct. CLI-reverified 2026-07-22 against a real photo (not just a synthetic shape) — still correct. |
| **LLaVA-1.5-7B** | CLIP ViT (CLS token, pre-LN, quick-GELU, penultimate layer) + MLP projector. Tower corr 1.0; e2e "a red circle … a Japanese flag". CLI-reverified 2026-07-22 against a real photo — still correct. |
| **Qwen2.5-VL-3B** | Own ViT — Conv3D patch embed, 2D-RoPE, window attention (full at 7/15/23/31), SwiGLU, 2×2 merger. All stages corr 1.0; e2e correct. |
| **Qwen2.5-VL-7B** | Same `Qwen25VlEncoder` + qwen2 text as the 3B. **Verified e2e on the 3060** (unsloth Q4_K_M + mmproj-F16, low-VRAM): blue→"Blue.", red→"Red." in ~5s. Bring-up fixes: metadata-based Qwen mmproj detection (`clip.projector_type`, not filename) + a CUDA int-overflow in the cast byte-size math that OOM'd the 152k-vocab Q6_K lm_head (`count * SizeInBytes` widened to 64-bit — affected any large-vocab quantized head). CLI-reverified 2026-07-22 on the 4090 against a real photo — best result of the whole VLM CLI pass, correctly read on-image text ("cero emisiones"). |
| **Llama-3.2-Vision-11B (mllama)** | See build-defer table above for the original engine-internal verification (red circle→"red" etc). CLI-reverified 2026-07-22 against a real photo: PARTIAL — correctly identifies the broad scene (bus, street, people, trees) but gets the bus color wrong and doesn't reliably stop at content end (free-runs into a hallucinated follow-up turn past a normal token budget; confirmed the eos token IS registered in StopIds, so this is model behavior, not a stop-token wiring bug — a lower `--max-tokens` truncates cleanly). |

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
| **Qwen3.5 dense** (Gated DeltaNet hybrid) | ✅ verified (0.8B) | `Qwen35Model` (new `HartsyInference.LLM.Ssm.ISsmModel`, not `GenericTransformer` — mixes TWO attention mechanisms per model, not pure-recurrent like the rows above). Every 4th layer (`full_attention_interval`) is regular GQA + partial RoPE + KV cache (Qwen3-style QK-norm, plus a fused query+gate projection unique to this arch — query and a sigmoid gate share one projection, de-interleaved per head); the rest are Gated DeltaNet: causal Conv1d over a fused QKV projection, per-head L2-norm on Q/K, then a sequential delta-rule recurrence (`S_t=α_t·S_{t-1}+β_t(v_t−S_{t-1}k_t)k_t^T`, `o_t=S_t·q_t`) → gated RMSNorm(o, silu(z)) → out_proj. Ported cold from llama.cpp's `src/models/{qwen35.cpp, delta-net-base.cpp}` (no local reference model existed). Text-only M-RoPE degenerates to ordinary partial-rotary RoPE (every section gets the same position when there's no multimodal input) — no M-RoPE machinery built. **Real bug found and fixed via live testing**: missed the `q *= 1/√S_k` scale llama.cpp applies right before the recurrence (present in the reference, easy to miss reading it once) — produced word-salad, not a crash; fixing it flipped straight to coherent, factually correct, grammatically clean output. Verified live: 0.8B, ~100 tokens, stable. 2B/4B/9B share the identical code path, untested (same-tier VRAM as tested, likely fine, just not run). `qwen35moe` (the 35B-A3B/122B-A10B/397B-A17B MoE tier) is a separate GGUF arch — NOT wired, no MoE FFN support in `Qwen35Model` yet. |
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
