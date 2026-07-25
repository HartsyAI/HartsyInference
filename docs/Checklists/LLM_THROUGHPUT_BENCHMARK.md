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
>
> **UPDATE 2026-07-23 (round 6, node-fusion campaign — permute-free graph step + fused GLU epilogue;
> grind doc round-6 block). Final fleet table (graph-on medians; llama-cpp-python baselines):**
>
> | Model | Ours (best) | llama-cpp-python | Ratio |
> |---|---|---|---|
> | Llama-3.2-1B Q8_0 | **208.85** | 190.34 | **1.10× FASTER** |
> | Qwen3-4B Q4_K_M | **95.27** | 85.59 | **1.11× FASTER** |
> | gemma-2-2b-it Q4_K_M | **135.35** | 121.08 | **1.12× FASTER** |
> | DeepSeek-R1-Distill-1.5B | **175.34** (172.2 back-to-back) | 180.12 (177.7 back-to-back) | 0.97× (gap ~1.03×) |
> | GLM-4-9B Q4_K_M | **40.14** | 47.80 (47.33 back-to-back) | 0.84× |
> | gemma-3-1b-it Q4_K_M | **119.44** | 192.34 | 0.62× |
> | qwen2.5-0.5b Q4_K_M | **163.82** | 322.72 | 0.51× |
>
> **UPDATE 2026-07-23 (round 7, deep-fusion pass — QKV rope-scatter + fused add-RMSNorm; grind doc
> round-7 block). FINAL table (clean GPU — a stuck 6.9 GB test-host process was found and killed
> mid-round; measure with `nvidia-smi --query-compute-apps` clean, and measure GLM solo):**
>
> | Model | Ours (graph-on) | llama-cpp-python | Ratio |
> |---|---|---|---|
> | Llama-3.2-1B Q8_0 | **211.49** | 190.34 | **1.11× FASTER** |
> | Qwen3-4B Q4_K_M | **96.29** | 85.59 | **1.13× FASTER** |
> | gemma-2-2b-it Q4_K_M | **136.56** | 121.08 | **1.13× FASTER** |
> | DeepSeek-R1-Distill-1.5B | **178.66** | 180.8/180.7 same-window sandwich | **0.988× — parity within 1.2%** |
> | GLM-4-9B Q4_K_M | **40.80** (solo) | 47.33 | 0.86× |
> | gemma-3-1b-it Q4_K_M | **119.97** | 192.34 | 0.62× |
> | qwen2.5-0.5b Q4_K_M | **167.45** | 322.72 | 0.52× |

> **UPDATE 2026-07-23 (round 8, launch-overhead round — partial [q|k] fusion (decode-only) + GEMV
> WARPS_PER_BLOCK 8→4; grind doc round-8 block). FIVE of seven now faster than llama-cpp-python
> (fresh same-window baselines, quiet GPU, GLM solo):**
>
> | Model | Ours (graph-on) | llama-cpp-python | Ratio |
> |---|---|---|---|
> | Llama-3.2-1B Q8_0 | **212.93** | 191.59 | **1.11× FASTER** |
> | Qwen3-4B Q4_K_M | **97.58** | 87.99 | **1.11× FASTER** |
> | gemma-2-2b-it Q4_K_M | **138.30** | 123.44 | **1.12× FASTER** |
> | DeepSeek-R1-Distill-1.5B | **195.96** | 183.29 | **1.07× FASTER — parity broken** |
> | GLM-4-9B Q4_K_M | **43.42** (solo) | 48.08 | 0.90× (was 0.86×) |
> | gemma-3-1b-it Q4_K_M | **124.49** | 197.35 | 0.63× |
> | qwen2.5-0.5b Q4_K_M | **173.56** | 326.92 | 0.53× |

> **UPDATE 2026-07-23 (rounds 9-10, gemma3 deep-dive: fused QK-norm epilogue + quantize-at-producer;
> grind doc rounds 9-10). Final same-window sandwich, quiet GPU, GLM solo:**
>
> | Model | Ours (graph-on) | llama-cpp-python | Ratio |
> |---|---|---|---|
> | Llama-3.2-1B Q8_0 | **212.15** | 192.96 | **1.10× FASTER** |
> | Qwen3-4B Q4_K_M | **98.76** | 88.73 | **1.11× FASTER** |
> | gemma-2-2b-it Q4_K_M | **137.05** | 123.07 | **1.11× FASTER** |
> | DeepSeek-R1-Distill-1.5B | **195.72** | 181.55 | **1.08× FASTER** |
> | GLM-4-9B Q4_K_M | **42.78** (solo) | 47.93 | 0.89× |
> | gemma-3-1b-it Q4_K_M | **127.13** | 193.96 | 0.66× (was 0.62 at day start) |
> | qwen2.5-0.5b Q4_K_M | **173.69** | 324.70 | 0.53× |
>
> gemma3/qwen2.5's remaining gap has ~3 ms/token of switch-invariant unattributed graph time —
> see the grind doc round-10 block for the quantified budget and the pending `ncu` action.

> **UPDATE 2026-07-23 (round 11 — the graph node purge: per-capture arena + headroom-guarded weight
> preload; grind doc round-11 block. The "unattributed 3 ms" was 1881 non-kernel graph nodes —
> alloc/free pairs and per-token PCIe re-uploads of never-promoted weights. SIX of seven now
> FASTER than llama-cpp-python:**
>
> Column convention (all tables from here on): "ours ÷ llama.cpp" is the plain speed ratio —
> 1.46 means we generate 46% more tokens per second than llama-cpp-python, NOT 2.46×; below 1.00
> means we are slower by that percentage.
>
> | Model | Ours tok/s | llama.cpp tok/s | ours ÷ llama.cpp | Plain English |
> |---|---|---|---|---|
> | Llama-3.2-1B Q8_0 | **213.20** | 190.74 | 1.12 | **12% faster than llama.cpp** |
> | Qwen3-4B Q4_K_M | **100.68** | 88.70 | 1.13 | **13% faster** |
> | gemma-2-2b-it Q4_K_M | **139.61** | 122.52 | 1.14 | **14% faster** |
> | DeepSeek-R1-Distill-1.5B | **224.78** | 182.42 | 1.23 | **23% faster** |
> | gemma-3-1b-it Q4_K_M | **246.99** | 193.81 | 1.27 | **27% faster** (was 38% slower at day start) |
> | qwen2.5-0.5b Q4_K_M | **434.25** | 296.80 | 1.46 | **46% faster** (was 48% slower at day start) |
> | GLM-4-9B Q4_K_M | **43.80** (solo) | 47.72 | 0.92 | 8% slower (best yet) |

> **UPDATE 2026-07-23 (prefill/TTFT campaign opened — decode is won, first-token latency is NOT).**
> TTFT medians, same ~50-token prompt, same-window llama-cpp-python (ms, lower is better):
>
> | Model | Ours TTFT | llama.cpp TTFT | Plain English |
> |---|---|---|---|
> | qwen2.5-0.5b | 77 | 4 | 19× slower to first token |
> | DeepSeek-1.5B | 113 | 7 | 16× slower |
> | gemma-3-1b | 130 | 7 | 19× slower |
> | Llama-3.2-1B | 144 | 6 | 24× slower |
> | gemma-2-2b | 198 | 10 | 20× slower |
> | Qwen3-4B | 270 | 13 | 21× slower |
> | GLM-4-9B | 639 | 22 | 29× slower |
>
> Attribution (profile + code): prefill Linear averages ~1.8 ms/call. (1) low-vram models route
> prefill through QuantizedMatMul → cacheWeightCast:false → the ENTIRE weight set is transiently
> dequantized to F16 (with an F32 staging pass at the F32-activation BF16 resolve) EVERY prefill;
> (2) non-low-vram models' cast cache sits behind a free-VRAM budget gate (2 GB floor) that the
> now-fully-resident decode state likely trips, silently forcing the same per-call dequant.
> llama.cpp prefills straight from quant weights (MMQ kernels) and has neither cost. NEXT:
> instrument which branch fires in a warm benchmark prefill; tune/bypass the gate for prefill;
> long-term the real fix is quantized-GEMM prefill kernels (MMQ-class).
> The benchmark harness now prints `ttft median` alongside tg for every model/mode.

> **UPDATE 2026-07-23 (TTFT round 1: prefill cast-cache fix — llama 3.2× and Qwen3 1.9× faster to
> first token; the small-model prefill floor is NOT dequant and needs its own attribution).**
> Root causes fixed: (1) the benchmark loaded EVERY model `lowVramQuant:true`, routing all prefill
> through QuantizedMatMul which hard-disabled weight-cast caching → the entire weight set was
> re-dequantized every prefill (now: low-vram only for >2 GB GGUFs); (2) QuantizedMatMul now
> participates in the budget-gated cast cache (quantized weights use a stricter 4 GB free-VRAM
> floor); (3) the cast cache is now SAFE-BY-CONSTRUCTION: CudaMemory's OOM retry evicts all cached
> casts (pure, rebuildable) before failing — this un-broke gemma2's 2.25 GB scaled-embed alloc.
>
> | Model | TTFT before | TTFT now | llama.cpp | Verdict |
> |---|---|---|---|---|
> | Llama-3.2-1B | 144 ms | **45 ms** | 6 ms | 3.2× better; dequant WAS the cost (Q8_0) |
> | Qwen3-4B | 270 ms | **144 ms** | 13 ms | 1.9× better (partial cache within budget) |
> | gemma-2-2b | 198 ms | 110 (graph-off) / 229 (graph-on!) | 10 ms | OOM fixed; graph-on REGRESSION = suspected cast-eviction thrash (scaled-embed alloc evicts the casts prefill just built, every generation) — investigate |
> | DeepSeek-1.5B | 113 ms | 121-125 ms | 7 ms | flat — prefill NOT dequant-dominated |
> | gemma-3-1b | 130 ms | 145-149 ms | 7 ms | flat-to-worse — same |
> | qwen2.5-0.5b | 77 ms | 85-87 ms | 4 ms | flat-to-worse — same |
> | GLM-4-9B | 639 ms | 597-624 ms | 22 ms | marginal (casts don't fit its budget) |
>
> NEXT: profile a WARM cached prefill on a small Q4_K model to name its ~85-150 ms floor (it is
> not weight dequant); fix the gemma2 graph-on eviction thrash (order the scaled-embed alloc
> before prefill, or exempt in-use casts from eviction); GLM's real fix remains MMQ-class
> quantized-GEMM prefill kernels. Decode is UNAFFECTED by all of this (verified same tg medians).

> **UPDATE 2026-07-23 (TTFT rounds 2-3: constructor weight-preload + last-row-only prefill logits —
> first-token latency down 4-12× across the fleet; GLM parked pending MMQ kernels).**
> Fixes: (a) headroom-guarded weight preload moved from graph setup to PIPELINE CONSTRUCTION —
> auto-promotion's size floor had left all small weights host-side FOREVER on eager paths (5271
> H2D misses per 7 generations on qwen2.5; this also lifted qwen2.5 GRAPH-OFF decode 126→257
> tok/s); (b) prefill now projects logits for the LAST prompt position only (GatherRows before
> ProjectLogits): skips (n−1)/n of the vocab GEMM AND takes the fused dp4a head at t=1 — no more
> 16-bit head cast, whose F32 staging OOMed gemma2 (first-token numerics now match the decode
> path's dp4a math — a numerics-class change, sanity-verified); (c) preload headroom reserves the
> graph embed table (+2 GB extra for low-VRAM giants); (d) the graph arena's release is now a
> SYNCHRONOUS free on a drained stream — its async free raced the pool across backends
> (intermittent CudaGraph_RepeatedReplay failure, 0 recurrences in 10 suite runs since).
>
> | Model | TTFT campaign start | TTFT now (solo, graph-on) | llama.cpp | Plain English |
> |---|---|---|---|---|
> | qwen2.5-0.5b | 77 ms | **19 ms** | 4 ms | 4.1× better, 4.7× behind llama.cpp |
> | Llama-3.2-1B | 144 ms | **22 ms** | 6 ms | 6.5× better, 3.7× behind |
> | DeepSeek-1.5B | 113 ms | **22 ms** | 7 ms | 5.1× better, 3.1× behind |
> | gemma-3-1b | 130 ms | **25 ms** | 7 ms | 5.2× better, 3.6× behind |
> | gemma-2-2b | 198 ms (+OOM) | **48 ms** | 10 ms | 4.1× better, OOM fixed |
> | Qwen3-4B | 270 ms | **146 ms** | 13 ms | 1.8× better (layer dequants dominate) |
> | GLM-4-9B | 639 ms | **595 ms** | 22 ms | parked — needs MMQ-class quant-GEMM prefill (F16 cast set can never fit 12 GB) |
>
> Decode throughput unaffected throughout (re-verified per model). SUITE CAVEAT: the W8A8/Sage
> PERF-ASSERTION tests (the other workstream's) fail intermittently whenever another process
> shares the 3060 — check `nvidia-smi` compute-apps before trusting a red suite.

> **UPDATE 2026-07-23 (round 12 — model-family expansion: 14 new catalog models downloaded + benchmarked,
> +2 more structural-status models (llama32-vision, qwen35) grabbed for completeness. Same harness/params
> as round 11 (5 reps, 128-token greedy, warmup discarded). Ran on 3060 (Tier-1 + python baseline) while
> Swarm's own hartsy-local backend ran concurrently on the 4090 with zero contention — confirmed via
> per-GPU `nvidia-smi -i <n> --query-compute-apps` that Swarm's process lives entirely on GPU 1 (4090);
> our `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` pin targets GPU 0 (3060) exclusively, so the
> two workloads never shared VRAM. This means Tier-1 dotnet-test work and Tier-2 Swarm-path work can run
> in parallel on this box going forward — no need to serialize them.**
>
> | Model | Ours (best) | llama-cpp-python | Ratio | Plain English |
> |---|---|---|---|---|
> | Mistral-7B-Instruct-v0.3 Q4_K_M | **57.61** | 53.15 | 1.08 | 8% faster |
> | gemma-3-4b-it Q4_K_M | **96.16** | 87.54 | 1.10 | 10% faster |
> | stablelm-2-1.6b-chat Q4_K_M | **232.27** | 202.26 | 1.15 | 15% faster |
> | granite-3.0-2b-instruct Q4_K_M | **131.54** | 117.36 | 1.12 | 12% faster |
> | SmolVLM2-2.2B-Instruct Q4_K_M | **216.21** | 197.19 | 1.10 | 10% faster |
> | Qwen2.5-VL-7B-Instruct Q4_K_M | **63.33** | N/A — llama-cpp-python 0.3.34 fails `Failed to create llama_context` on this GGUF | — | ours WORKS, no valid python baseline exists on this build |
> | Phi-3-mini-4k-instruct Q4 | **76.80** (graph-off; not graph-eligible at its best) | 93.93 | 0.82 | 18% slower |
> | olmoe-1b-7b-0924 Q4_K_M (MoE) | **21.97**⚠ | 255.68⚠ (noisy, stdev 62.98) | 0.086 | ~11.6× slower — ⚠ D2H syncs=2193/rep, fails the doc's own health-assert; number likely NOT representative of best-case, see Issue #5 below |
> | granite-3.0-1b-a400m Q4_K_M (MoE) | **51.54**⚠ | 276.32⚠ (noisy, stdev 45.61) | 0.187 | ~5.4× slower — same D2H-sync caveat (3225/rep) |
> | gemma-4-E2B-it Q4_K_M | **68.84** (not graph-eligible) | 100.07 (noisy, stdev 10.02) | 0.688 | 31% slower |
> | llava-v1.5-7b Q4_K_M | CRASH — see Issue #2 | 52.88 | — | Tier-1 harness can't measure it; Swarm's own chat endpoint generates fine (2.8 tok/s incl. reload) |
> | gpt2-medium Q4_K_M | CRASH — see Issue #2 | 402.12 | — | same as above; Swarm generates fine (11.1 tok/s incl. reload) |
> | starcoder2-3b Q4_K_M | CRASH — see Issue #2 | 117.60 | — | same as above; Swarm generates fine (10.1 tok/s incl. reload) |
> | mamba-2.8b-hf Q4_K (SSM) | N/A — wrong harness, needs `SsmLanguageModel` (by design, see Issue #6) | 108.72 | — | Swarm generates fine (2.0 tok/s incl. reload; consistent order-of-magnitude with the catalog's documented ~0.8 tok/s host-bound decode once reload is subtracted) |
> | Qwen3.5-0.8B Q4_K_M (Gated DeltaNet, SSM-routed) | N/A — same SSM-path exclusion | not attempted | — | Swarm generates fine (1.0 tok/s incl. reload) |
> | Llama-3.2-11B-Vision Q4_K_M | not run (size/time budget) | not run | — | Swarm OOM'd twice — VRAM-leak accumulation, see Issue #4 |
>
> Five of six architecturally-eligible dense/VLM models beat llama-cpp-python (8-15% faster), consistent
> with the round-11 fleet pattern. Phi-3-mini is the one dense-model exception (18% slower) — not
> root-caused this session, candidate for a future grind round. The two MoE models and gemma4 (mobile,
> per-layer-embedding architecture) remain the known weak spots, consistent with pre-existing catalog
> comments about MoE's host-side routing and gemma4's PLE-gather path.

> **UPDATE 2026-07-23 (gemma-4 E2B added to the fleet — BASELINE, graph bring-up pending).**
> `Models/LLM/gemma4/gemma-4-E2B-it-Q4_K_M.gguf`. Ours (eager-only): tg **91.7** tok/s, TTFT
> **36 ms**; llama-cpp-python: tg 117.2, TTFT 10 — 22% behind on decode, 3.6× on TTFT. Decode gap
> = the graph-decode exclusion (structural gate: HeadDimSwa per-layer head dim, PLE, KV-sharing/
> V-norm); the eager path already carries the day's fleet work (dp4a, ctor preload, last-row
> prefill logits). Bring-up plan (round-3 playbook, byte-A/B before widening the gate): dual-DIM
> rope tables per layer → PLE per-token pinned-buffer feed → donor-cache reads + V-norm in the
> graph step. Expected landing ~125-135 tok/s (ahead of llama.cpp).

> **UPDATE 2026-07-24 (rounds 12-13 close-out: gemma-4 graph bring-up + fleet-wide residency fixes.
> EIGHT of nine text models now at-or-ahead of llama-cpp-python; GLM reached effective parity.)**
> New this round: (1) **gemma-4 E2B graph decode** — dual-head-dim rope tables, per-layer dims,
> weightless V-norm, device-side Q5_K PLE row gather by device token id (`lm_ple_gather_q5k_f32`),
> and KV-sharing via a q-only donor-cache path; byte-identical graph-vs-eager, gate verifiably
> engaged. (2) **Redundant-split preload exclusion** — EnumerateWeights was double-loading fused
> weights AND their split originals (~2.9 GB dead VRAM on GLM), crowding real weights out of
> residency; the preloader now excludes splits (their one consumer, the batch scheduler's split
> path, lazily promotes). This took GLM to its best-ever decode. (3) mllama: ctor preload,
> template-token decode-skip fixup, decode-rate logging (template-encoding root cause scoped).
> (4) Qwen3.5 SSM: weight residency wired — 40.3 s → 18.4 s for an identical run; full SSM
> bring-up scoped. Suites 194/194 + 132/132.
>
> | Model | Ours tok/s (graph-on) | llama.cpp tok/s | Plain English |
> |---|---|---|---|
> | qwen2.5-0.5b | **435.6** | 328.9 | 32% faster than llama.cpp |
> | gemma-3-1b | **251.3** | 196.8 | 28% faster |
> | DeepSeek-1.5B | **224.5** | 183.4 | 22% faster |
> | Llama-3.2-1B | **213.7** | 192.0 | 11% faster |
> | gemma-2-2b | **141.8** | 123.8 | 15% faster |
> | gemma-4-E2B | **137.8** | 124.7 | **11% faster (new bring-up; was 22% slower)** |
> | Qwen3-4B | **100.7** | 89.0 | 13% faster |
> | GLM-4-9B | **47.07** (solo) | 48.13 | **2% slower — effective parity (was 18% slower at day start)** |
> | Llama-3.2-11B-Vision | works e2e, eager-only | n/a | cross-attn graph bring-up scoped |
>
> TTFT highlights this pass: DeepSeek 21.6 ms, llama 20.5, gemma3 27.6, qwen2.5 16.7, qwen3 112.7
> (improved by the residency fix), gemma-4 50.4, GLM 563 (MMQ prefill kernels remain its fix).

> **UPDATE 2026-07-24 (Qwen3.5-0.8B SSM first head-to-head).** After phases A/B/B2 (quant-resident
> weights, device recurrence, device attention step — grind doc): ours **54.6 tok/s** decode vs
> llama-cpp-python's **234.6** (4.3× behind; was ~25× behind at the 40 s baseline). Outputs
> identical to the host reference throughout. The remaining gap is the phase-C orchestration
> (per-token host sampling round-trip + eager launches): the transformer pipeline's device-argmax
> + graph-capture design applies directly and is the scoped finish line.

> **UPDATE 2026-07-24 (Qwen3.5-0.8B SSM phase C landed: graph-captured decode).** Ours is now
> **~145 tok/s** decode (496 tokens in 3.35–3.56 s, delta-timed 512- vs 16-token greedy runs to
> exclude load+prefill) vs llama-cpp-python's **234.6 tok/s** — llama.cpp is now 62% faster (was
> 330% faster before phase C, 25× faster at the original baseline). Overall campaign: the original
> 40.3 s reference generation now takes ~3.3 s wall including model load. Bring-up surfaced and
> fixed a shared-infrastructure bug: `BuildRopeTableDevice`'s partial-rotary table layout only
> matched the interleaved rope kernels; split-half partial rotary (Qwen3.5's 64-of-128 dims) read
> identity entries and left upper-half pair elements unrotated. Fixed with an explicit
> `splitHalfPartial` layout switch — full-rotary and interleaved (GLM-4) tables are byte-identical
> to before. Details + A/B methodology in `LLM_DECODE_PERF_GRIND.md` (phase C entry).
> **Same day, delta-kernel round:** row-parallel `lm_ssm_delta_step_rows_f32` (bit-exact schedule
> change, byte-verified, suites green) lifts Qwen3.5-0.8B to **217 tok/s** steady decode
> (4.61 ms/token, per-token onToken timing over 512 greedy tokens) vs llama-cpp-python's 234.6 —
> llama.cpp now 8% faster (was 62% faster this morning, 330% before phase C). Remaining levers
> scoped in the grind doc (warp-per-row delta kernel built but unverified/opt-in; SSM
> quantize-at-producer + sandwich fusions unwired; Q6_K head at 63% of peak).

> **FINAL 2026-07-24: Qwen3.5-0.8B BEATS llama-cpp-python — ours 254.5 tok/s (254.9/254.4/254.2
> reps, 3.93 ms/token) vs llama.cpp 232.8 same-window sandwich (232.3 mean, σ=1.0): 9.3% faster.**
> Landed after the 217 entry: SSM sidecar/fused-add-norm wiring (+1.4%), per-head delta gate
> scalar hoisting (+0.8%), and the decisive one — the warp-per-row delta kernel default-on (its
> earlier "no gain" e2e verdict was an artifact of pre-hoist transcendental masking; isolated
> micro: 20.5 vs 55 µs/call). All byte-identical vs kill-switched builds; suites 201/201 CUDA +
> 132/132 LLM on a quiet GPU. Every SSM-path change is kill-switched: HARTSY_SSM_GRAPH,
> HARTSY_SSM_DEVICE_STEP, HARTSY_SSM_DELTA_V2, HARTSY_SSM_DELTA_WARPROW, HARTSY_QUANT_AT_PRODUCER,
> HARTSY_SANDWICH_FUSION. Known remaining weakness: first-token latency (~0.8 s incl.
> per-generation graph capture vs llama.cpp's 15 ms) — unoptimized, scoped in the grind doc.

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

## Phase 2 — Tier 2: Swarm API integration benchmark (the real test) 🔧

### Tier 2 (2026-07-23, real Swarm API benchmark via HTTP LLMAssistantSendMessage)

**Test Setup**: Real-world prompt ("Explain machine learning in one sentence"), generated via Swarm HTTP API, `noCache=true` to bypass response caching, empty instructionId for neutral system prompt. 7 unique models × 2 invocations each = 14 total requests.

**Results Summary** (RTX 4090):

| Model | Type | Tokens | Time (s) | Throughput | Quality |
|---|---|---|---|---|---|
| **Llama-3.2-1B-Instruct** | Q8_0 | 26 | 2.44–2.68 | 9.7–10.6 tok/s | ✅ Answers correctly |
| **Qwen2.5-0.5B** | Q4_K_M | 22 | 2.27–2.30 | 9.6–9.7 tok/s | ✅ Answers correctly |
| **DeepSeek-R1-Distill-1.5B** | Q4_K_M | 84 | 4.57–5.17 | 16.3–18.4 tok/s | ✅ Reasoning + correct |
| **Qwen3-4B** | Q4_K_M | 81 | 5.98–6.29 | 12.9–13.5 tok/s | ✅ Answers correctly |
| **Gemma-3-1B-IT** | Q4_K_M | 18 | 3.15–3.72 | 4.8–5.7 tok/s | ⚠️ System prompt override |
| **Gemma-2-2B-IT** | Q4_K_M | 12 | 4.90–5.14 | 2.3–2.5 tok/s | ⚠️ System prompt override |
| **GLM-4-9B** | Q4_K_M | — | — | — | ❌ API response parsing error |

**Key Findings**:
- ✅ 12/14 requests successful (85.7%)
- ✅ Real generations from 6 models with diverse outputs
- ⚠️ Gemma models hit system-prompt override (small models unable to override default SwarmUI persona)
- ❌ GLM-4-9B returns HTTP 200 but JSON parsing fails (format/encoding issue)
- ✅ Throughput 2.3–18.4 tok/s; DeepSeek fastest due to reasoning/Q4 quantization
- ⚠️ Gemma-3/2 slower due to persona-generation (not model speed)

**Known Issues Fixed This Session**:
1. ✅ **Response Caching Bug**: API cached by `(message, instructionId)`, not by model. Fix: pass `noCache=true`.
2. ⚠️ **System Prompt Override**: Small models (Gemma-3, Gemma-2) respond with assistant persona instead of answering. Workaround: use empty or model-specific system prompts.
3. ❌ **GLM-4-9B**: Needs debugging (response format / encoding).

### Tier 2 round 2 (2026-07-23, 14 new catalog models + 2 extra, same noCache=true/empty-instructionId setup)

Ran on the 4090 (Swarm's actual GPU — see round 12 note above) while Tier-1 ran concurrently on the 3060.

| Model | Time (incl. reload) | Tokens | Status |
|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 9.62s | 23 | ✅ |
| gemma-3-4b-it | 9.63s | **0** | ⚠️ empty response — see Issue #7 |
| Phi-3-mini-4k-instruct | 11.85s | 80 | ✅ |
| stablelm-2-1.6b-chat | 9.28s | 78 | ✅ |
| granite-3.0-2b-instruct | 8.64s | 23 | ✅ |
| olmoe-1b-7b-0924 | 14.33s | 33 | ✅ |
| granite-3.0-1b-a400m | 6.50s | 25 | ✅ |
| gemma-4-E2B-it | 18.24s | 21 | ✅ |
| SmolVLM2-2.2B-Instruct | — | — | ❌ `Jinja filter 'capitalize'` — see Issue #3 |
| llava-v1.5-7b | 7.46s | 21 | ✅ (works via Swarm despite Tier-1 crash — see Issue #2) |
| Qwen2.5-VL-7B-Instruct | — | — | ❌ `CUDA_ERROR_OUT_OF_MEMORY` — VRAM leak, see Issue #4 |
| mamba-2.8b-hf | 38.38s | 78 | ✅ |
| gpt2-medium | 7.93s | 88 | ✅ (works via Swarm despite Tier-1 crash) |
| starcoder2-3b | 7.64s | 77 | ✅ (works via Swarm despite Tier-1 crash) |
| Llama-3.2-11B-Vision | — | — | ❌ `CUDA_ERROR_OUT_OF_MEMORY` — VRAM leak, see Issue #4 |
| Qwen3.5-0.8B | 25.91s | 25 | ✅ |

**13/16 succeeded.** The 3 failures are 2 recurrences of the VRAM-leak (Issue #4, previously "fixed" in
FINDING #2 below but reproduced this session) and 1 new Jinja-renderer gap (Issue #3). Calling
`POST /API/LLMAssistantUnloadModels` between models works around the leak (used mid-run to unblock the
sweep) but the underlying leak itself is not re-fixed.

### Next Steps for Tier 2
- [ ] Investigate GLM-4-9B JSON parsing error (log raw bytes)
- [ ] Test WebSocket streaming endpoint (`LLMAssistantSendMessageWS`) for comparison
- [ ] Measure TTFT (first-token latency) separately from total generation time
- [ ] Tune system prompts to prevent Gemma override on small models
- [ ] Compare vs llama-bench synthetic 128-token greedy to understand Swarm overhead
- [ ] Re-fix the VRAM-leak-on-swap regression (Issue #4) — FINDING #2's fix has regressed or was incomplete

**Note**: `SwarmUI-LLMAssistant` v2.0.0-alpha.2 provides the LLM path; `SwarmUI-HartsyInference` is T2I-only and not involved. Engine consumed via `HartsyInference.LLM.dll` (in-process).

## Issues & bugs found during the round-12 model-family expansion (2026-07-23) — debug guide for the next agent

Numbered so other sections can cross-reference. Each entry: what broke, exact repro, where to start looking, and severity.

### Issue #1 — Response cache keyed by `(message, instructionId)`, not by model [FIXED — workaround, not a code fix]
- **Symptom:** Calling `LLMAssistantSendMessage` for 7 different models with the same prompt text returned byte-identical output for all of them after the first call.
- **Root cause:** `ChatEndpoints.cs` `LLMAssistantSendMessage` — `Cache.GetOrCreate(message, instructionId, ...)` — the cache key omits `model`/`assistantId` entirely.
- **Where to fix for real:** `SwarmUI-LLMAssistant/WebAPI/ChatEndpoints.cs` and whatever `Cache` class backs `GetOrCreate` (find via `grep -rn "class Cache" src/Extensions/SwarmUI-LLMAssistant`). Add `model` (and probably `assistantId`) to the cache key tuple.
- **Workaround used this session:** pass `"noCache": true` on every benchmark request. Fine for benchmarking; a real product-code fix is still needed so the chat UI doesn't silently serve another model's cached answer when a user switches models mid-conversation with an identical follow-up message.
- **Severity:** Medium — silent wrong output in the actual product, not just benchmarking.

### Issue #2 — `ChatMlTemplate.Encode` throws instead of falling back for tokenizers with no `<|im_start|>` token
- **Symptom:** `System.InvalidOperationException: Tokenizer has no <|im_start|> token; ChatML template not applicable.` Hit on llava-v1.5-7b, gpt2-medium, and starcoder2-3b when driven through the Tier-1 harness (`TextDecodeThroughputBenchmark.cs`, which builds a `GenerationRequest` with a bare `Prompt` string via `ExtendedLLMInput`-style construction).
- **Important nuance:** these same three checkpoints generate correctly through Swarm's actual `LLMAssistantSendMessage` endpoint (confirmed: gpt2-medium 88 tokens, starcoder2-3b 77 tokens, llava-v1.5-7b 21 tokens, all coherent). So Swarm's real request path already has *some* working fallback or different template-resolution logic that the raw Tier-1 pipeline invocation doesn't use.
- **Where to look:** `src/HartsyInference.LLM/ChatTemplates/ChatMlTemplate.cs` (`Encode`, the throw site) vs `src/HartsyInference.LLM/Generation/PromptBuilder.cs` (`BuildPromptIds`, which picks `ChatMlTemplate` as a fallback) vs whatever `SwarmUI-LLMAssistant`'s `ChatEndpoints.cs` does differently before calling `LLMDispatcher.Generate` (it may resolve a template per-model up front and skip straight to a raw-string path when none exists, or it may catch this exact exception — grep for `catch` near the `LLMDispatcher.Generate` call sites).
- **Fix direction:** `PromptBuilder.BuildPromptIds` should detect "no ChatML tokens" *before* calling `ChatMlTemplate.Encode` and fall back to a plain-text/raw-completion prompt format (what base models like GPT-2/StarCoder2 actually expect) instead of throwing. This would let `TextDecodeThroughputBenchmark.cs` (and any other direct-pipeline caller) measure these models' real decode throughput, which is currently impossible.
- **Severity:** Medium — doesn't block the product (Swarm path works), but blocks direct-engine testing/benchmarking of any non-instruction-tuned checkpoint, and is a landmine for any future caller that uses `GenerationRequest.Prompt` directly instead of going through Swarm.

### Issue #3 — Swarm's Jinja template engine is missing the `capitalize` filter
- **Symptom:** SmolVLM2-2.2B-Instruct fails via `LLMAssistantSendMessage` with `Jinja filter 'capitalize'` (not found/implemented).
- **Where to look:** the Jinja template interpreter used by `SwarmUI-LLMAssistant` (per other catalog comments this repo has its own Jinja subset implementation — search for `JinjaChatTemplate.cs` / `JinjaExpr.cs`, mentioned elsewhere in this doc's catalog re: qwen35's tool-calling filter chain gap). Find the filter dispatch table (likely a `switch`/dictionary mapping filter names to implementations) and confirm `capitalize` is simply unimplemented.
- **Fix direction:** add a `capitalize` filter (first-letter-uppercase, matching Jinja2's built-in semantics) to the filter table. Low-risk, mechanical fix — this is the class of gap the doc already predicted ("SEPARATE, deeper 'Unsupported Jinja call expression' failure" was noted for qwen35's template; this is the same class of issue, different filter, different model).
- **Severity:** Low-effort fix, but currently a hard blocker for SmolVLM2 (and any other model whose real chat template uses `capitalize`) via the actual product surface.

### Issue #4 — VRAM leak on Swarm model swap (REGRESSION of previously-"fixed" FINDING #2 below)
- **Symptom:** Swarm's `hartsy-local` backend process (4090) grew from ~2.9 GB baseline to **14.6 GB** after just ~10 sequential model loads in this session's Tier-2 sweep, then OOM'd on Qwen2.5-VL-7B and again on Llama-3.2-11B-Vision — both of which fit comfortably in 24 GB on their own.
- **This is the same failure mode as FINDING #2 (below in this doc)**, which claims a fix was applied 2026-07-03 (`HartsyLocalLLMProvider.UnloadSlot` calling `CudaBackend.FreeAllDeviceMemory()`). That fix has either regressed, was incomplete, or doesn't cover every code path that swaps models (e.g. maybe only covers the explicit unload API, not the implicit swap-on-next-generate path).
- **Where to look:** `SwarmUI-LLMAssistant/Backends/HartsyLocalLLMProvider.cs` — confirm `UnloadSlot`/`LoadInto` are still calling `CudaBackend.FreeAllDeviceMemory()` on every swap (not just explicit `/API/LLMAssistantUnloadModels` calls). Check whether backend id 5 (added via `AddNewBackend` API this session) and backend id 4 (from `Backends.fds`) are BOTH holding independent resident models simultaneously — this session added a *second* backend instance pointed at the same model folder, which could mean two provider instances each keep their own slot resident, doubling the effective leak. Verify via `nvidia-smi -i 1 --query-compute-apps` growth after each single model swap, isolating whether it's a per-swap leak or a per-extra-backend-instance issue.
- **Workaround used this session:** `POST /API/LLMAssistantUnloadModels` between models. Confirmed working (`{"success":true,...}`, memory dropped from 14.6 GB → 4.3 GB after one call).
- **Severity:** High — this blocks real usage of large models (7B+) after a chat session has touched several other models, which will be a common real-world pattern (a user comparing models in the UI).

### Issue #5 — MoE models (olmoe, granite-moe) show pathological D2H sync counts, invalidating their Tier-1 numbers
- **Symptom:** `olmoe-1b-7b-0924-instruct` measured 2193 D2H syncs per 128-token rep (vs ~129/rep — 1 per token — for every dense model); `granite-3.0-1b-a400m` measured 3225. Both are ~17-25× the expected sync count. Per this doc's own "Controlled variables" section, "~0 mid-decode D2H syncs... any per-token sync is a residency bug... invalidates the tg number."
- **Measured throughput:** olmoe 21.97 tok/s (llama-cpp-python: 255.68, though noisy at stdev 62.98), granite-moe 51.54 tok/s (llama-cpp-python: 276.32, noisy at stdev 45.61). Neither is graph-decode-eligible.
- **Where to look:** the MoE expert-routing/dispatch code path — likely `src/HartsyInference.LLM/**/MoeFeedForward.cs` or similar (grep `MoeFeedForward`). Each expert selection per-token is probably reading a routing decision back to host (`D2H`) to decide which expert weights to gather/launch, which is exactly the kind of per-token host-round-trip the doc flags as a residency bug elsewhere (cf. "Sync-H2D stream drain" and "CPU-glue async race" prior-session memory entries in this repo's broader history).
- **Fix direction:** keep the top-k expert selection and gather entirely on-device (no D2H readback of routing indices before launching the expert GEMVs) — likely needs a device-side argmax/top-k + indirect/gather-dispatch kernel instead of reading indices back to the CPU to decide which kernel to launch next.
- **Severity:** High for these two specific architectures — the current numbers likely understate the model's real achievable throughput AND llama-cpp-python's own numbers for these are also noisy/unreliable (high stdev), so this needs its own dedicated, cleaner remeasurement once the sync count is fixed, not just a perf optimization.

### Issue #6 — SSM / hybrid architectures can't be measured via `TextDecodeThroughputBenchmark.cs` [BY DESIGN, not a bug]
- **Symptom:** `GgufLanguageModel.Load` throws `NotSupportedException: 'mamba' is a state-space (non-transformer) architecture — load it via HartsyInference.LLM.Ssm.SsmLanguageModel, not GgufLanguageModel.` Same for `qwen35` (Gated DeltaNet hybrid, routed as SSM).
- **This is intentional** — the catalog already documents that mamba/rwkv/qwen3.5 route through `SsmGenerationPipeline`, not `TextGenerationPipeline`. It is NOT a bug, but it does mean **no Tier-1 throughput number exists for these architectures from this session** (mamba's ~0.8 tok/s host-bound figure cited elsewhere in the catalog predates this session and wasn't re-verified here).
- **Fix direction (feature work, not a bug fix):** extend `TextDecodeThroughputBenchmark.cs` (or write a sibling `SsmDecodeThroughputBenchmark.cs`) to detect the architecture up front and dispatch to `SsmLanguageModel`/`SsmGenerationPipeline` instead of hardcoding `GgufLanguageModel.Load` + `TextGenerationPipeline`, so SSM models get measured with the same rigor as transformers.
- **Severity:** Low (no incorrect behavior, just a measurement gap).

### Issue #7 — gemma-3-4b-it returns an empty (0-token) response via Swarm
- **Symptom:** `LLMAssistantSendMessage` for `gemma-3-4b-it-Q4_K_M.gguf` returned `success: true` with an empty `response` string (0 tokens) in 9.63s. The exact same checkpoint architecture (gemma3) at the 1B size worked fine via Swarm in the round-1 sweep, and the 4B size worked fine at 69-96 tok/s in the direct Tier-1 test — so the underlying decode is provably fine; this is specific to the Swarm request path for this particular model size/variant.
- **Where to look:** could be a stop-token/EOS-on-first-token issue specific to gemma-3-4b's larger vocab or a different `<end_of_turn>`-handling path than the 1B checkpoint, or a response-parsing truncation bug in `ChatEndpoints.cs` for longer prefill times (9.63s prefill+generate for a 4B model is plausible if most of that time is prefill and the decode loop then immediately hits an early stop condition). Add logging around the raw token stream before it's assembled into the `response` string for this specific model to see whether tokens were generated and dropped, or never generated at all.
- **Severity:** Medium — a real model silently produces nothing via the product's main entry point, which would look like the model is broken to an end user, when the model itself works fine directly.

### Issue #8 — Qwen2.5-VL-7B's GGUF fails to even construct a `llama_context` in llama-cpp-python 0.3.34 [not our bug, informational]
- **Symptom:** `internals.LlamaContext(...)` → `ValueError: Failed to create llama_context` when loading `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` via `llama_cpp.Llama(...)` (pip package `llama-cpp-python==0.3.34`), confirmed reproducible with a clean/idle GPU (2.2 GB used, not an OOM).
- **This is llama.cpp's limitation, not ours** — our engine loads and decodes this exact file successfully at 56-63 tok/s via a heuristic key-mapper fallback (`declared architecture 'qwen2vl' has no registered mapper; falling back to key-heuristic detection` → routes to `LlamaKeyMapper`). No fair Python-baseline comparison is possible for this model on this box with this llama-cpp-python build; a newer llama-cpp-python build or a different quant/repack might succeed where this one doesn't.
- **Action for future sessions:** if a Python baseline for this specific model becomes important, try upgrading `benchmarks/.bench-venv`'s `llama-cpp-python` to a newer release, or try `unsloth`'s non-VL text-only variant if one exists.
- **Severity:** None (not an engine bug) — recorded for context so a future session doesn't waste time assuming our engine is broken here.

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
