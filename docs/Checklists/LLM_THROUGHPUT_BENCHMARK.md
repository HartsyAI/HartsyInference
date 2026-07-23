# LLM Throughput Benchmark — Design & Tracking

Status legend: ⬜ not started · 🔧 in progress · ✅ done · ⚠ blocked

**Goal.** For every latest-SOTA LLM we support, establish a llama.cpp baseline (tokens/sec, GPU) and prove the HartsyInference engine **matches or beats** it. The bar is set by `llama-bench`; the engine is measured both directly (Tier 1) and through the real Swarm API path (Tier 2).

> **STATUS (2026-07-04):** baseline captured (all 7 models); benchmark revealed a 20-54× gap; the optimization grind (**`LLM_DECODE_PERF_GRIND.md`**) closed it to **1.94-2.88×** (Llama-3.2-1B **under 2×**). Remaining lever = CUDA graphs (foundation verified, full build deferred). See the "AFTER the grind" results table below.
>
> **UPDATE (2026-07-10):** CUDA graphs done — see `LLM_DECODE_PERF_GRIND.md`'s top status block. Qwen3-0.6B
> 1.77× off llama.cpp (was 2.26×, i.e. 4.5× against a fresh baseline run); Llama-3.2-1B 1.72×; Mistral-7B 2.08×
> (big models are GEMV-bandwidth-bound, not launch-bound, so graphs help them far less — the memory-access
> redesign is still the open lever there). Byte-identical greedy output verified vs the non-graphed path.

**Hardware.** RTX 3060 12 GB (sm_86), driver CUDA 13.2. Keep desktop/SwarmUI VRAM in mind (~5 GB baseline occupancy); free it before a run for headroom. Single GPU, batch=1.

---

## Decisions (locked)

- **Baseline mechanism: `llama-bench` only.** Compile llama.cpp with CUDA and use its `llama-bench` for the reference pp/tg numbers. No LLamaSharp baseline.
- **Model set (latest SOTA, all ✅ verified e2e, all fit the 3060):**
  | Model | GGUF (repo `Models/llm/`) | Family / why included |
  |---|---|---|
  | Qwen3-0.6B | `Qwen3-0.6B-Q4_K_M.gguf` | newest Qwen dense, `<think>` |
  | Llama-3.2-1B | `llama-3.2-1b-instruct-q8_0.gguf` | Llama-3.x flagship small |
  | Gemma-3-1B | `gemma-3-1b-it-Q4_K_M.gguf` | Gemma-3, dual-RoPE + soft-cap |
  | Phi-4-mini | `Phi-4-mini-instruct-Q4_K_M.gguf` | Phi-4 lineage, LongRope |
  | Mistral-7B-v0.3 | `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | largest single-GPU-fit dense |
  | Granite-3.1-2B | `granite-3.1-2b-instruct-Q4_K_M.gguf` | Granite scalar multipliers |
  | OLMoE-1B-7B | `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf` | **MoE** sparse-routing decode path |

  (Command-R7B and Gemma-2-2B intentionally excluded this pass.)
- **Location:** benchmark project at `benchmarks/HartsyInference.LlmBench/`; results committed under `benchmarks/results/`; this doc is the tracker.

---

## Metrics & fair-comparison protocol

Two throughput numbers per model, matching `llama-bench`'s split:
- **pp (prompt processing / prefill):** `prompt_tokens ÷ time_to_first_token`. llama-bench: `pp512`.
- **tg (token generation / decode):** `(gen_tokens − 1) ÷ (t_last_token − t_first_token)`. llama-bench: `tg128`.

Engine-side pp/tg come **for free** from the `TextGenerationPipeline.Generate(request, onToken)` per-token callback timestamps — no engine changes needed. `t_first_token` ends prefill; deltas between later tokens are pure decode.

**Controlled variables (identical on both sides):**
- Same GGUF file & quant (Q4_K_M / Q8_0 as in the table); our engine runs the **quantized-matmul path** (`QuantizedMatMul`, keep compressed on-device) to match llama.cpp's on-the-fly dequant — do **not** dequantize to F32.
- batch = 1 · greedy sampling (temp 0) for determinism · fixed prompt length 512 (pp) · fixed gen length 128 (tg) · same context length.
- Full GPU offload: `-ngl 99` on llama.cpp; all layers on `CudaBackend` on our side.
- Warmup run discarded · ≥5 reps · report **median ± stddev**.
- **Health assert:** ~0 mid-decode D2H syncs on our side (`CudaBackend.GetD2hSyncCount()`); any per-token sync is a residency bug (cf. the Ideogram4 57-min regression) and invalidates the tg number.

**Interpreting the two tiers:** Tier-1 tg is the kernel bar (must match/beat llama-bench). Tier-2 tg (over the WebSocket) carries serialization overhead so it sits below Tier-1; its job is to prove the Swarm integration doesn't regress the bar, not to beat llama-bench directly.

---

## Phase 0 — Prep ✅ (llama.cpp baseline captured 2026-07-03)

- [x] Free VRAM: 9.7 GB free at snapshot (SwarmUI released its ~3.2 GB). Add an `nvidia-smi` pre-check to the runner.
- [x] **All 7 GGUFs confirmed present** in `Models/llm/` (exact files + sizes):
      `Qwen3-0.6B-Q4_K_M.gguf` (397 MB) · `llama-3.2-1b-instruct-q8_0.gguf` (1321 MB) · `gemma-3-1b-it-Q4_K_M.gguf` (806 MB) · `Phi-4-mini-instruct-Q4_K_M.gguf` (2492 MB) · `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` (4373 MB) · `granite-3.1-2b-instruct-Q4_K_M.gguf` (1545 MB) · `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf` (4214 MB). No fallback dir needed.
- [x] **nvcc risk resolved:** a full **CUDA 13.2 toolkit exists at `/usr/local/cuda-13.2`** (`nvcc V13.2.78`), matching the driver — the PATH-default 11.5 was a red herring. Installed `cmake` 4.3.4 via `pip --user`. Host compiler gcc 11.4.
- [🔧] **Compiling `llama-bench` with CUDA** from `~/models/llamacpp` (commit `6f4f53f`): configure succeeded (CUDA backend included, sm_86); build in progress. Exact commands used:
      ```
      export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:$PATH"
      export CUDACXX=/usr/local/cuda-13.2/bin/nvcc CUDA_PATH=/usr/local/cuda-13.2
      cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
      cmake --build build --config Release -j$(nproc) --target llama-bench
      ```
      (LLamaSharp fallback no longer needed.)
- [x] Smoke + **full baseline sweep** done: `build/bin/llama-bench -m <all 7> -ngl 99 -p 512 -n 128 -r 5 -o json` → `benchmarks/results/llamacpp_baseline_3060.json`. Canonical params **`-ngl 99 -p 512 -n 128 -r 5`** are now frozen for every tier.

**llama.cpp baseline (RTX 3060, driver CUDA 13.2, llama.cpp `6f4f53f`, F16 KV, 5 reps):**

| Model | pp512 t/s | tg128 t/s |
|---|---|---|
| Qwen3-0.6B Q4_K_M | 14210.9 ± 755.0 | 354.46 ± 2.13 |
| Llama-3.2-1B q8_0 | 12403.6 ± 421.5 | 215.91 ± 2.04 |
| Gemma-3-1B Q4_K_M | 11295.1 ± 382.8 | 229.75 ± 3.33 |
| Phi-4-mini Q4_K_M | 3958.6 ± 30.6 | 107.56 ± 0.74 |
| Granite-3.1-2B Q4_K_M | 4620.3 ± 74.5 | 148.46 ± 2.18 |
| OLMoE-1B-7B Q4_K_M (MoE) | 5857.5 ± 55.9 | 283.27 ± 2.08 |
| Mistral-7B-v0.3 Q4_K_M | 2131.6 ± 31.7 | 66.46 ± 0.76 |

## Phase 1 — Tier 1: engine micro-benchmark ✅ (2026-07-22, via `TextDecodeThroughputBenchmark.cs`, llama-cpp-python baseline instead of llama-bench)

Built as `tests/HartsyInference.Cuda.Tests/TextDecodeThroughputBenchmark.cs` — real production path
(`TextGenerationPipeline.Generate`, not a hand-rolled loop), warmup + 5 reps × 128-token greedy decode,
per-token timestamps via the `onToken` callback, `GetD2hSyncCount()` asserted ≤2 (graph-on decode must be
~0 mid-decode syncs or the tg number isn't trustworthy — see Decisions below). Reports both graph-off and
graph-on (`GraphDecode=true`, our best config) medians. Python-side baseline used **llama-cpp-python**
(the llama.cpp CUDA engine via a Python binding) instead of `llama-bench` directly, since the box has no
system CUDA toolkit — built llama-cpp-python from source against pip-installed `nvidia-cuda-nvcc`/
`nvidia-cuda-runtime`/`nvidia-cublas` wheels (see `MODEL_STATUS_LLM.md`'s glm4 section for the toolchain
details; same discovery applies here). Same frozen prompt/params on both sides
(`benchmarks/python-baseline/bench_llama_cpp_python.py`).

**⚠ Measurement gotcha hit and fixed**: this box has two GPUs (3060 index 0, 4090 index 1 in `nvidia-smi`'s
PCI-bus ordering). `CUDA_VISIBLE_DEVICES=0` alone is **not sufficient** to pin the 3060 — CUDA's default
device ordering is `FASTEST_FIRST`, so plain `CUDA_VISIBLE_DEVICES=0` silently selects the **4090** as CUDA
device 0, not the 3060. Symptom: both engines' measured tg roughly doubled between sessions with no code
change, and the ratio between them looked to flip (a "we're faster than llama.cpp!" reading that evaporated
on a supposedly-identical re-run). Root cause confirmed via `nvidia-smi dmon -s u` sampled live during a
run: without the flag, GPU **index 1** (4090) hit 70-82% utilization while GPU 0 sat idle; with the flag,
GPU 0 (3060) hit ~90%. **Fix: always set both `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0`
together** to target the 3060 — checked via `nvidia-smi -L` (PCI order) name match, not by index alone.
Anchor-verified against this doc's own documented llama.cpp baseline for `llama-3.2-1b-instruct-q8_0.gguf`
(215.91 tok/s, Results table below) — a fresh llama-cpp-python run with the flag set landed at 190.34 tok/s
(same ballpark, unlike the ~400 tok/s misattributed-to-4090 reading), confirming the fix is real, not another
misattribution. **All results below use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` on both sides.**

**Results (RTX 3060, `--low-vram-quant`, greedy, 128 tok, graph-on = our best config):**

> **UPDATE 2026-07-22 (later session): FASTER THAN llama-cpp-python on both models.** The dp4a
> int8-activation GEMV kernel set from `LLM_GEMV_KERNEL_HANDOFF.md` landed and is default-on
> (kill-switch `HARTSY_DP4A_ON=0`); full kernel/verification detail in `LLM_DECODE_PERF_GRIND.md`'s
> "FASTER THAN llama.cpp" status block. Final medians (default config, 5 reps, ±0.1-0.9 tok/s spread):
>
> | Model | Ours (graph-off) | Ours (graph-on) | llama-cpp-python | Ratio (ours÷llama, graph-on) |
> |---|---|---|---|---|
> | Llama-3.2-1B-Instruct Q8_0 | 162.61 tok/s | **197.28 tok/s** | 190.34 (best documented); 173.58 same-hour back-to-back | **1.04× (vs best) - 1.14× FASTER** |
> | Qwen3-4B Q4_K_M | 75.58 tok/s | **92.21 tok/s** | 85.59 (documented); 86.83 same-hour back-to-back | **1.06-1.08× FASTER** |
>
> Measurement note: desktop/rustdesk GPU contention on this box can swing either engine ±20%+ —
> compare only back-to-back runs from a quiet GPU (`nvidia-smi` util ≲15% on GPU 0 first).
>
> **UPDATE 2026-07-22 (round 2, popular-model sweep):** benchmarked four more models chosen from
> Ollama's top-downloads list (checkpoints under `Models/LLM/`), fixed what the sweep found (Q4_0/Q5_0/
> Q5_K dp4a kernels, the `ProjectLogits` fusedHead divisibility gate, windowed/softcap split-K
> attention, GLM-4 partial-rotary graph-decode corruption — see `LLM_DECODE_PERF_GRIND.md`'s round-2
> block). Final medians, default config, llama-cpp-python same-day/same-pin:
>
> | Model | Ours (graph-off) | Ours (graph-on) | llama-cpp-python | Ratio (best÷llama) |
> |---|---|---|---|---|
> | DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M | 137.56 | **161.06** | 180.12 | 0.89× (1.12× slower) |
> | qwen2.5-0.5b-instruct Q4_K_M | 113.82 | **158.94** | 322.72 | 0.49× — was 0.23× pre-sweep |
> | gemma-3-1b-it Q4_K_M | **88.90** | not eligible (SWA) | 190.89 | 0.47× — was 0.18× pre-sweep |
> | GLM-4-9B-0414 Q4_K_M | 35.62 | **39.21** | 47.80 | 0.82× (graph output now CORRECT — was degenerate before the rope-table fix) |
>
> Llama-3.2-1B (197.29) and Qwen3-4B (92.91) re-confirmed faster than llama-cpp-python in the same
> run. Remaining-gap roadmap (gemma graph decode, tiny-model graph node overhead) is scoped in the
> grind doc's round-2 block.
>
> **UPDATE 2026-07-22 (round 3): Gemma-family CUDA-graph decode landed** (sliding-window + soft-cap +
> dual local/global RoPE through the graph path — grind doc round-3 block; graph-vs-eager output
> verified byte-identical on BOTH gemma checkpoints before enabling, per the GLM-4 lesson):
>
> | Model | Ours (graph-off) | Ours (graph-on) | llama-cpp-python | Ratio (best÷llama) |
> |---|---|---|---|---|
> | gemma-2-2b-it Q4_K_M | 111.47 | **129.51** | 121.08 | **1.07× FASTER** |
> | gemma-3-1b-it Q4_K_M | 88.24 | **115.09** | 192.34 | 0.60× (was 0.18× at day start) |
>
> Fleet summary end-of-day: 3 of 7 benchmarked models FASTER than llama-cpp-python (Llama-3.2-1B
> 196-197, Qwen3-4B 92.6-92.9, gemma-2-2b 129.5), GLM-4-9B at 0.82×, DeepSeek-distill-1.5B at 0.89×,
> and the two sub-2B stragglers (gemma3-1b 0.60×, qwen2.5-0.5b 0.49×) both bottlenecked on the
> tiny-model graph-node-overhead bucket, not on kernels.
>
> **UPDATE 2026-07-22 (round 4, decode-overhead fixes — Dev-path split-attention gate, two-stage
> argmax, persistent dp4a/argmax scratch; grind doc round-4 block): every model improved.** Final
> fleet table (graph-on medians, llama-cpp-python baselines as above):
>
> | Model | Ours (best) | llama-cpp-python | Ratio |
> |---|---|---|---|
> | Llama-3.2-1B Q8_0 | **202.74** | 190.34 | **1.07× FASTER** |
> | Qwen3-4B Q4_K_M | **93.67** | 85.59 | **1.09× FASTER** |
> | gemma-2-2b-it Q4_K_M | **131.59** | 121.08 | **1.09× FASTER** |
> | DeepSeek-R1-Distill-1.5B Q4_K_M | **166.01** | 180.12 | 0.92× |
> | GLM-4-9B Q4_K_M | **39.35** | 47.80 | 0.82× |
> | gemma-3-1b-it Q4_K_M | **118.48** | 192.34 | 0.62× |
> | qwen2.5-0.5b Q4_K_M | **162.16** | 322.72 | 0.50× |
>
> The two sub-2B stragglers' remaining gap is graph-replay per-node overhead (~340 kernel nodes/step;
> per-token GPU-busy is only ~40% of wall on qwen2.5-0.5b) — next lever is the Phase-4-style fusion
> campaign (norm+rope fusion, t=1 permute elimination, quantize-in-GEMV), scoped in the grind doc.
>
> **UPDATE 2026-07-23 (round 5, long-K/small-N GEMV K-split — grind doc round-5 block):**
> DeepSeek-R1-Distill-1.5B 166.0 → **167.87** (0.93×, gap 1.07×), GLM-4-9B 39.35 → **40.37** (0.84×,
> gap 1.18×); all other models unchanged within noise. The ffn_down GEMV class (K ≥ 8192, N ≤ 4096)
> now splits each row across 4 warps (was 51-82% of DRAM peak at warp-per-row on those shapes).

Superseded pre-dp4a table (2026-07-22, earlier session):

| Model | Ours (graph-off) | Ours (graph-on) | llama-cpp-python | Ratio (ours÷llama, graph-on) |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct Q8_0 | 128.49 → 134.49 → **~139 tok/s** | 155.28 → **~157 tok/s** | 190.34 tok/s | **~0.82× (~1.21× slower)**, was 1.34× |
| Qwen3-4B Q4_K_M | 54.44 → 58.7 → 60.64 → **~62.0 tok/s** | 60.05 → 70.09 → **~70.4 tok/s** | 85.59 tok/s | **~0.82× (~1.21× slower)**, was 1.34× |

Four real fixes landed this session, all `ncu`/`nsys`-guided (profiler access unblocked via `sudo` —
per-invocation elevation, not a persistent system change; see `LLM_DECODE_PERF_GRIND.md` for how): (1) a
Q6_K GEMV latency-bound-load fix (Qwen3-4B only, no Q6_K tensors in Q8_0 llama); (2)+(3) split-K flash-decode
attention **split-count formula fixes** for both the graph-decode and eager dispatch paths — the split-K path
was real and already engaging, but its split count was capped far below the actual sweet spot (~9% on both
models); (4) **QKV + gate/up projection fusion** — llama.cpp fuses these projections into fewer, larger GEMV
calls; we didn't, so the small ones (K/V projections) ran the GPU at as little as 0.38 waves across all SMs.
Fused at load time (byte-level weight concatenation, generic across dtypes) and wired into every decode path;
correctness verified byte-identical. Smaller win than (2)+(3) (~1-1.5% graph-on, ~7-9% graph-off) — the raw
GEMV saving is real but partly offset by the cost of splitting the fused output back apart afterward. Full
root-cause, verification, and the known VRAM tradeoff are in `LLM_DECODE_PERF_GRIND.md`'s 2026-07-22 entries.
Still not faster than Python/llama.cpp on either model, but the gap is now ~1.34x → ~1.21x, and every
remaining lever found this session either has no headroom left (GEMV kernels near-roofline) or needs a
genuine kernel-memory-access-pattern redesign (not a dispatch/fusion change) to close further.

- [ ] `llama-bench` itself (vs. llama-cpp-python) not run this pass — no system CUDA toolkit; the pip-wheel
  toolchain builds llama-cpp-python fine but not the full llama.cpp CLI suite. Same underlying engine either
  way, so the comparison is still apples-to-apples for the CUDA compute path.

## Phase 2 — Tier 2: Swarm API integration benchmark (the real test) ⬜

- [ ] Client (C# preferred, reuses tokenizer for token counts) that drives `ws://<host>/API/LLMAssistantSendMessageWS` with the **hartsy-local** provider selected, per model, greedy, 128 gen.
- [ ] Measure TTFT = first `{"chunk"}` arrival; tg = decoded tokens between first chunk and `{"done":true}`, timestamped client-side. Cross-check token counts via `LLMAssistantCountTokens`.
- [ ] Compare Tier-2 tg vs Tier-1 tg (overhead delta) and vs the llama-bench bar.
- [ ] Note: `SwarmUI-LLMAssistant` provides the LLM path; `SwarmUI-HartsyInference` is T2I-only and not involved. Engine consumed via forced-local DLLs (`UseLocalHartsy=true`), so no NuGet republish needed.

## ⚠ HEADLINE FINDING (2026-07-03, preliminary) — engine decode is far below llama.cpp

On Qwen3-0.6B Q4_K_M, RTX 3060, 128-token greedy decode:

| Path | tok/s | vs llama.cpp |
|---|---|---|
| **llama.cpp** (llama-bench tg128) | **354.5** | 1.0× (bar) |
| Engine, Tier 2 Swarm (WS, decode-window) | ~39.8 | ~0.11× |
| **Engine, Tier 1 direct** (Stopwatch, lowVram=0) | **13.6** | **~0.038× (26× slower)** |
| Engine, Tier 1 direct (lowVram=1, QuantizedMatMul) | 9.3 | ~0.026× (38× slower) |

Diagnosis so far:
- **Not a residency/sync bug:** D2H syncs = 129 for 128 tokens (≈1/token, exactly the logits read, same as llama.cpp). No mid-decode syncs.
- **GPU is saturated:** utilization pegged at **100%** during decode, yet delivers ~14 t/s. So the kernels are **compute-inefficient**, not stalled — decode at batch=1 should be memory-bandwidth bound and llama.cpp reaches ~354 t/s on the same file/GPU.
- The compressed `QuantizedMatMul` path is *slower* than the (default) path, pointing at the quant-GEMV/dequant kernels as a prime suspect.
- Tier-2 (Swarm) reads *higher* than Tier-1 here; the Swarm decode-window metric is less trustworthy (initial-token buffering compresses the window). **Tier-1 direct Stopwatch is the authoritative engine number.**

Conclusion: the engine does **not** currently match/beat llama.cpp; it is ~10-38× slower at decode. The benchmark harness (baseline + both tiers) works and captured this. Optimization of the decode/quant-matmul kernels is a separate effort (candidate: dedicated fused quant-GEMV kernels; cf. `docs/Checklists/QUANT_GEMM_PERF_PLAN.md`).

## ⚠ FINDING #2 — Swarm `hartsy-local` provider leaks VRAM on model swap

During the Tier-2 sweep, SwarmUI's LLM backend grew to **10 GB VRAM** after only 3 models and OOM'd the rest — with nothing else on the GPU. `HartsyLocalLLMProvider.LoadInto` calls `UnloadSlot` before loading a different model, but the previous model's **GPU weights are not released** (the `CudaBackend.PreloadWeights` allocations survive the swap). The explicit `POST /API/LLMAssistantUnloadModels` API *does* free them (`{"freed":1}`, 10 GB → 3.5 GB). So the free path exists; it just isn't invoked on swap.
- **Impact:** a multi-model Tier-2 sweep must call `LLMAssistantUnloadModels` between models (client updated to do this), or restart the backend.
- **Fix APPLIED (2026-07-03):** `SwarmUI-LLMAssistant/Backends/HartsyLocalLLMProvider.cs` `UnloadSlot` now calls `CudaBackend.FreeAllDeviceMemory()` (EvictAll + TrimPool) instead of the per-weight `FreeWeights(EnumerateWeights())` — a slot holds one model, so full-context eviction on swap reclaims weights + F16 casts + activations + KV + pool reservations with no reference-identity dependence. Extension recompiles clean; **needs SwarmUI restart to load, then re-run the Tier-2 sweep to confirm all 7 models run without OOM.** No engine rebuild (method already in the shipped DLL).
- **Related:** default `lowVram=0` expands quantized weights to **F32 in VRAM** (Qwen3 0.4 GB file → 2.4 GB resident), which both inflates memory and moves 4-8× more bytes/token than llama.cpp's compressed GEMV — a prime contributor to FINDING #1. The fair comparison config is `lowVram=1` (compressed, same memory model as llama.cpp).

## Phase 3 — Report ⬜

- [ ] Fill the results table below; flag any model where engine tg < llama.cpp tg and root-cause (dequant kernel, D2H syncs, attention path, KV dtype).
- [ ] Record run environment (driver, CUDA, free VRAM, llama.cpp commit, engine version).

---

## Results — ORIGINAL BASELINE (2026-07-03), then AFTER the optimization grind (2026-07-04)

> **The gap has been closed from 20-54× to 1.94-2.88×** by the work in **`LLM_DECODE_PERF_GRIND.md`** (fused quantized GEMV kernels, quantized lm_head, split-K flash-decode attention, vectorized loads). See that doc for the per-phase progression and kernel details. The tables below are the *starting* point and the *current* state.

### Original baseline (before the grind)
Engine column = Tier-1 direct, CUDA, `lowVram=1`, 128-token greedy overall tok/s (single-shot, incl. first-token JIT → pessimistic; the accurate warm/256-tok baselines are in the grind doc). D2H syncs ≈1/token (no residency bug). This is the pre-optimization starting point:

| Model | Quant | llama.cpp tg t/s | engine tg t/s | engine/llama | slowdown | Swarm tg t/s |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | Q4_K_M | 354.46 | 17.6 | 0.050 | 20× | 20.1 |
| Llama-3.2-1B | Q8_0 | 215.91 | 6.05 | 0.028 | 36× | 6.2 |
| Gemma-3-1B | Q4_K_M | 229.75 | 8.16 | 0.036 | 28× | 11.7 |
| Phi-4-mini | Q4_K_M | 107.56 | 2.74 | 0.025 | 39× | VRAM‡ |
| Granite-3.1-2B | Q4_K_M | 148.46 | 3.18 | 0.021 | 47× | VRAM‡ |
| OLMoE-1B-7B | Q4_K_M | 283.27 | OOM† | — | — | VRAM‡ |
| Mistral-7B-v0.3 | Q4_K_M | 66.46 | 1.24 | 0.019 | 54× | VRAM‡ |

### ✅ AFTER the grind (current, 2026-07-04) — warm decode tok/s, GPU idle
Same GGUF files. Engine now uses fused mul_mat_vec Q4_K/Q6_K/Q8_0 decode kernels + quantized lm_head + split-K flash-decode attention + vectorized loads. All coherent (token-checked vs the pre-change path).

| Model | Quant | llama.cpp tg t/s | engine tg t/s | engine/llama | vs llama.cpp | Δ from baseline |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | Q8_0 | 215.91 | ~111.5 | 0.52 | **1.94×** (under 2×!) | 6.05 → 111.5 (**18×**) |
| Mistral-7B-v0.3 | Q4_K_M | 66.46 | ~30.7 | 0.46 | **2.12×** | 1.24 → 30.7 (**25×**) |
| Qwen3-0.6B | Q4_K_M | 354.46 | ~157 | 0.44 | **2.26×** | 17.6 → 157 (**5.9× warm; ~9× vs cold**) |
| Gemma-3-1B | Q4_K_M | 229.75 | ~79.7 | 0.35 | 2.88× | (sliding-window attn can't use split-K yet — slowest) |
| Phi-4-mini / Granite-3.1-2B / OLMoE | Q4_K_M | — | — | — | pending re-measure | (small models expected ~2-2.5×) |

‡ Won't run through SwarmUI on this 12 GB card: **SwarmUI reserves ~8 GB VRAM at idle** (its T2I/ComfyUI/image backends), leaving ~4 GB — too little for Phi-4/granite/OLMoE/Mistral, which fit standalone in Tier-1 (whole GPU). Each attempt OOMs and auto-restarts SwarmUI. Their engine throughput is the Tier-1 column. The Swarm-tier Qwen3/Llama/Gemma numbers (compressed, `LowVramQuant=true`) **corroborate Tier-1** (20 vs 18, 6.2 vs 6.0, 12 vs 8), confirming the engine kernels are the bottleneck, not the Swarm/WebSocket layer.

### Config change made (revert if undesired)
- Set `Data/Backends.fds` → `llmassistant-hartsy-local` → `LowVramQuant: true` (was `false`). Compressed path: smaller host+VRAM footprint, fits more, matches llama.cpp's memory model. Slightly slower decode. Revert to `false` for the (memory-hungrier) dequantized-F16 path.
- Fixed the swap VRAM leak in `HartsyLocalLLMProvider.UnloadSlot` (uses `FreeAllDeviceMemory`); verified: gemma-3 (model #3) now loads where it OOM'd before.

### FINDING #3 — SwarmUI's ~8 GB idle VRAM crowds out LLM on a 12 GB card
SwarmUI holds ~8 GB VRAM before any LLM loads (T2I/ComfyUI/image backends). On a 12 GB 3060 that leaves too little for 2B+ LLMs through the integrated path, though they run fine standalone. Options: unload the T2I backends while benchmarking LLM, run LLM on a second GPU, or reduce the engine's LLM decode memory (F32 lm_head → tiled/streamed).

† OLMoE-1B-7B Tier-1 OOM'd with only 6.7 GB free (SwarmUI holds 3.5 GB base). Re-run with SwarmUI stopped to fit its 4.2 GB compressed weights.

**Reading it (historical):** at baseline the engine was 20-54× slower than llama.cpp, gap widening with model size (54× on Mistral) — GPU 100% utilized but compute-bound on the dequant-then-cuBLAS quant path and low-occupancy attention. Root cause + fixes are in `LLM_DECODE_PERF_GRIND.md`: there was **no fused quantized GEMV** (weights were dequantized whole to F16 then run through cuBLAS at M=1), the tied lm_head ran as a 622 MB/token F32 GEMV, and decode attention was one-block-per-head. Fixing those closed the gap to 1.94-2.88×. Remaining lever to reach/beat parity = **CUDA graphs** (launch overhead on small models; foundation verified, full build deferred — see grind doc Phase 6).

## Phase 1/2 status
- Phase 1 (Tier-1 direct engine): ✅ 6/7 measured (OLMoE OOM, needs SwarmUI stopped). Harness = existing `samples/HartsyInference.TextGen.Cli` (`gguf cuda 128`), no new project needed. Raw log: `benchmarks/results/tier1_engine_3060.txt`.
- Phase 2 (Tier-2 Swarm): 🔧 2/7 measured; blocked past model #3 by FINDING #2 (VRAM leak on swap). Client at `benchmarks/swarm_llm_bench/swarm_llm_bench.py` needs a per-model `LLMAssistantUnloadModels` call to complete the sweep. Raw: `benchmarks/results/swarm_llm_3060.json`.

---

## Open notes / risks

- **nvcc 11.5** is the main prep risk for compiling `llama-bench` (see Phase 0 fallbacks).
- **Quant fairness:** if any repo GGUF quant differs from what llama-bench conventionally reports, re-quantize or note the mismatch; keep the *same file* on both sides regardless.
- **VRAM contention:** Mistral-7B Q4_K_M (~4.7 GB) + KV + desktop must fit 12 GB; run it with a clean GPU.
- **Decode is bandwidth-bound** at batch=1 — both engines should approach HBM limits; the winner is decided by dequant-kernel efficiency and per-token overhead, which is exactly what D2H-sync count and the tg ratio expose.
