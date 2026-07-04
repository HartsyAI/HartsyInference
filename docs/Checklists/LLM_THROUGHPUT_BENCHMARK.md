# LLM Throughput Benchmark — Design & Tracking

Status legend: ⬜ not started · 🔧 in progress · ✅ done · ⚠ blocked

**Goal.** For every latest-SOTA LLM we support, establish a llama.cpp baseline (tokens/sec, GPU) and prove the HartsyInference engine **matches or beats** it. The bar is set by `llama-bench`; the engine is measured both directly (Tier 1) and through the real Swarm API path (Tier 2).

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

## Phase 1 — Tier 1: engine micro-benchmark ⬜

- [ ] New project `benchmarks/HartsyInference.LlmBench/` (console, references `HartsyInference.LLM` + `HartsyInference.Cuda`; copies `Ptx/` to output).
- [ ] Per model: `GgufLanguageModel.Load(path, lowVramQuant:true)` → `PreloadWeights` → warmup → N reps of prefill(512)+decode(128), capturing per-token timestamps + `GetD2hSyncCount()`.
- [ ] Compute pp/tg median ± stddev; assert D2H≈0; emit `benchmarks/results/llmbench_<host>_<date>.json` + a CSV.
- [ ] Run `llama-bench` on the same 7 GGUFs with identical params; parse its JSON.
- [ ] Join into a comparison table: `model | quant | llama.cpp pp/tg | engine pp/tg | ratio (engine÷llama)`.

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

## Results (to be filled)

Engine column = Tier-1 direct, CUDA, `lowVram=1` (compressed, matches llama.cpp memory model), 128-token greedy, **overall tok/s (decode-dominated, single-shot incl. first-token kernel JIT → slightly pessimistic for small models)**. D2H syncs = 129/128 tokens on every model (≈1/token; no residency bug anywhere). Verdict vs the "match or beat" goal: all **FAIL** by 20-54×.

| Model | Quant | llama.cpp tg t/s | engine tg t/s | engine/llama | slowdown | Swarm tg t/s | D2H/tok | Verdict |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | Q4_K_M | 354.46 | 17.6 | 0.050 | 20× | 20.1 | ~1.0 | FAIL |
| Llama-3.2-1B | Q8_0 | 215.91 | 6.05 | 0.028 | 36× | 6.2 | ~1.0 | FAIL |
| Gemma-3-1B | Q4_K_M | 229.75 | 8.16 | 0.036 | 28× | 11.7 | ~1.0 | FAIL |
| Phi-4-mini | Q4_K_M | 107.56 | 2.74 | 0.025 | 39× | VRAM‡ | ~1.0 | FAIL |
| Granite-3.1-2B | Q4_K_M | 148.46 | 3.18 | 0.021 | 47× | VRAM‡ | ~1.0 | FAIL |
| OLMoE-1B-7B | Q4_K_M | 283.27 | OOM† | — | — | VRAM‡ | — | — |
| Mistral-7B-v0.3 | Q4_K_M | 66.46 | 1.24 | 0.019 | 54× | VRAM‡ | ~1.0 | FAIL |

‡ Won't run through SwarmUI on this 12 GB card: **SwarmUI reserves ~8 GB VRAM at idle** (its T2I/ComfyUI/image backends), leaving ~4 GB — too little for Phi-4/granite/OLMoE/Mistral, which fit standalone in Tier-1 (whole GPU). Each attempt OOMs and auto-restarts SwarmUI. Their engine throughput is the Tier-1 column. The Swarm-tier Qwen3/Llama/Gemma numbers (compressed, `LowVramQuant=true`) **corroborate Tier-1** (20 vs 18, 6.2 vs 6.0, 12 vs 8), confirming the engine kernels are the bottleneck, not the Swarm/WebSocket layer.

### Config change made (revert if undesired)
- Set `Data/Backends.fds` → `llmassistant-hartsy-local` → `LowVramQuant: true` (was `false`). Compressed path: smaller host+VRAM footprint, fits more, matches llama.cpp's memory model. Slightly slower decode. Revert to `false` for the (memory-hungrier) dequantized-F16 path.
- Fixed the swap VRAM leak in `HartsyLocalLLMProvider.UnloadSlot` (uses `FreeAllDeviceMemory`); verified: gemma-3 (model #3) now loads where it OOM'd before.

### FINDING #3 — SwarmUI's ~8 GB idle VRAM crowds out LLM on a 12 GB card
SwarmUI holds ~8 GB VRAM before any LLM loads (T2I/ComfyUI/image backends). On a 12 GB 3060 that leaves too little for 2B+ LLMs through the integrated path, though they run fine standalone. Options: unload the T2I backends while benchmarking LLM, run LLM on a second GPU, or reduce the engine's LLM decode memory (F32 lm_head → tiled/streamed).

† OLMoE-1B-7B Tier-1 OOM'd with only 6.7 GB free (SwarmUI holds 3.5 GB base). Re-run with SwarmUI stopped to fit its 4.2 GB compressed weights.

**Reading it:** the engine is 20-54× slower than llama.cpp at decode, and the gap *widens with model size* (54× on Mistral-7B). GPU is 100% utilized throughout, so it is compute-bound on inefficient quant-GEMV/dequant + attention kernels, not stalled. Swarm-path (Tier-2) numbers read higher than Tier-1 for the two models that ran, but the Swarm decode-window metric over-counts throughput (initial-token buffering) — treat Tier-1 as authoritative.

## Phase 1/2 status
- Phase 1 (Tier-1 direct engine): ✅ 6/7 measured (OLMoE OOM, needs SwarmUI stopped). Harness = existing `samples/HartsyInference.TextGen.Cli` (`gguf cuda 128`), no new project needed. Raw log: `benchmarks/results/tier1_engine_3060.txt`.
- Phase 2 (Tier-2 Swarm): 🔧 2/7 measured; blocked past model #3 by FINDING #2 (VRAM leak on swap). Client at `benchmarks/swarm_llm_bench/swarm_llm_bench.py` needs a per-model `LLMAssistantUnloadModels` call to complete the sweep. Raw: `benchmarks/results/swarm_llm_3060.json`.

---

## Open notes / risks

- **nvcc 11.5** is the main prep risk for compiling `llama-bench` (see Phase 0 fallbacks).
- **Quant fairness:** if any repo GGUF quant differs from what llama-bench conventionally reports, re-quantize or note the mismatch; keep the *same file* on both sides regardless.
- **VRAM contention:** Mistral-7B Q4_K_M (~4.7 GB) + KV + desktop must fit 12 GB; run it with a clean GPU.
- **Decode is bandwidth-bound** at batch=1 — both engines should approach HBM limits; the winner is decided by dequant-kernel efficiency and per-token overhead, which is exactly what D2H-sync count and the tg ratio expose.
