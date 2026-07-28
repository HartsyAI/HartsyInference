# LLM decode throughput — HartsyInference vs llama.cpp scoreboard

Canonical, single-source-of-truth scoreboard for LLM token-generation (decode) throughput.
Consolidates the round-by-round tables in
[`docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) and
[`docs/Checklists/LLM_DECODE_PERF_GRIND.md`](../../docs/Checklists/LLM_DECODE_PERF_GRIND.md) into one
table. Where the same model was re-measured across multiple rounds, **the freshest dated result wins**
(rounds are dated inline in both source docs); see Notes for the couple of cases where an older number
is the only one that exists.

**Hardware:** RTX 3060 12GB only. A separate RTX 4090 pass exists
(`LLM_THROUGHPUT_BENCHMARK.md` "Tier 2", Swarm HTTP API) but it measures HartsyInference-only wall-clock
through SwarmUI with no llama.cpp baseline run on that GPU, so it isn't comparable head-to-head data and
is excluded from this table.
**Methodology:** batch=1, greedy (temp 0) decode, CUDA, warm/resident weights, same GGUF file and quant
on both sides. Most rows are the Tier-1 in-process micro-benchmark
(`TextDecodeThroughputBenchmark.cs`) against a same-window `llama-cpp-python` sandwich baseline,
128-token generation — this is what the source docs call **tg128**, matching `llama-bench`'s own
tg128 definition (`(gen_tokens−1) ÷ (t_last_token−t_first_token)`). **Ratio = HartsyInference ÷
llama.cpp** throughout (e.g. 1.32× means the engine generates 32% more tokens/sec than llama.cpp, not
2.32×; below 1.00× means the engine is slower — same convention the source doc explicitly calls out to
avoid misreading). A raw `llama-bench` baseline for 7 models also exists
(`benchmarks/results/llamacpp_baseline_3060.md`) and was used
only to sanity-check the llama.cpp-side numbers below, not as a separate set of rows: for the two
models it shares with the table (Llama-3.2-1B Q8_0 and gemma-3-1b Q4_K_M), raw `llama-bench` reads
~8-13% above the `llama-cpp-python` baselines used here (212-216 vs 192.0 for Llama-3.2-1B; 225-230 vs
196.8 for gemma-3-1b) — the same gap the source doc itself already anchored and explained (a fresh
llama-cpp-python run landing "in the same ballpark, same tool family" as the documented llama-bench
number, `LLM_THROUGHPUT_BENCHMARK.md` lines 106-109), not a discrepancy in this reconciliation. See
[`../../docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md) for the engine's default performance profile
and [`../../docs/Checklists/LLM_DECODE_PERF_GRIND.md`](../../docs/Checklists/LLM_DECODE_PERF_GRIND.md)
for full optimization history/methodology.

**pp512 (prefill) is not in this table** — the engine has no direct prefill-throughput comparison vs
llama.cpp; only time-to-first-token (TTFT) was measured, and it is a known weak spot (engine trails
llama.cpp by roughly 2–30× depending on model, see Notes). llama.cpp's own solo pp512 numbers (no
HartsyInference counterpart) are in `LLM_THROUGHPUT_BENCHMARK.md`'s Phase 0 baseline table.

## Results — tg128 decode, tokens/sec (higher is better)

| Model | Quant | GPU | Metric | HartsyInference (tok/s) | llama.cpp (tok/s) | Ratio | Date | Source |
|---|---|---|---|---:|---:|---:|---|---|
| qwen2.5-0.5b-instruct | Q4_K_M | RTX 3060 | tg128 | **435.6** | 328.9 | 1.32× faster | 2026-07-24 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) rounds 12-13 close-out |
| gemma-3-1b-it | Q4_K_M | RTX 3060 | tg128 | **251.3** | 196.8 | 1.28× faster | 2026-07-24 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) rounds 12-13 close-out |
| DeepSeek-R1-Distill-Qwen-1.5B | Q4_K_M | RTX 3060 | tg128 | **224.5** | 183.4 | 1.22× faster | 2026-07-24 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) rounds 12-13 close-out |
| stablelm-2-1.6b-chat | Q4_K_M | RTX 3060 | tg128 | **232.27** | 202.26 | 1.15× faster | 2026-07-23 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) round 12 |
| gemma-2-2b-it | Q4_K_M | RTX 3060 | tg128 | **141.8** | 123.8 | 1.15× faster | 2026-07-24 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) rounds 12-13 close-out |
| Qwen3-4B | Q4_K_M | RTX 3060 | tg128 | **100.7** | 89.0 | 1.13× faster | 2026-07-24 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) rounds 12-13 close-out |
| granite-3.0-2b-instruct | Q4_K_M | RTX 3060 | tg128 | **131.54** | 117.36 | 1.12× faster | 2026-07-23 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) round 12 |
| Llama-3.2-1B-Instruct | Q8_0 | RTX 3060 | tg128 | **213.7** | 192.0 | 1.11× faster | 2026-07-24 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) rounds 12-13 close-out |
| gemma-4-E2B-it | Q4_K_M | RTX 3060 | tg128 | **137.8** | 124.7 | 1.11× faster | 2026-07-24 | [LLM_DECODE_PERF_GRIND.md](../../docs/Checklists/LLM_DECODE_PERF_GRIND.md) rounds 12-13 (graph bring-up) |
| gemma-3-4b-it | Q4_K_M | RTX 3060 | tg128 | **96.16** | 87.54 | 1.10× faster | 2026-07-23 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) round 12 |
| SmolVLM2-2.2B-Instruct | Q4_K_M | RTX 3060 | tg128 | **216.21** | 197.19 | 1.10× faster | 2026-07-23 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) round 12 |
| Qwen3.5-0.8B (Gated DeltaNet / SSM) | Q4_K_M | RTX 3060 | tg128 | **254.5** | 232.8 | 1.09× faster | 2026-07-24 | [LLM_DECODE_PERF_GRIND.md](../../docs/Checklists/LLM_DECODE_PERF_GRIND.md) "FINAL 2026-07-24" (Qwen3.5 SSM campaign) |
| Mistral-7B-Instruct-v0.3 | Q4_K_M | RTX 3060 | tg128 | **57.61** | 53.15 | 1.08× faster | 2026-07-23 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) round 12 |
| GLM-4-9B-0414 | Q4_K_M | RTX 3060 | tg128 | 47.07 | **48.13** | 0.98× — effective parity | 2026-07-24 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) rounds 12-13 close-out |
| Phi-3-mini-4k-instruct | Q4_K_M | RTX 3060 | tg128 | 76.80 (graph-off; not graph-eligible) | **93.93** | 0.82× — 18% slower | 2026-07-23 | [LLM_THROUGHPUT_BENCHMARK.md](../../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) round 12 |
| Qwen3-0.6B | Q4_K_M | RTX 3060 | tg128 | 190.8 | **337.6** | 0.57× — 1.77× slower | 2026-07-10 | [README.md](../../README.md) "LLM decode vs llama.cpp" (pre-dp4a, oldest number still in use — not re-benchmarked since) |

Row count: 16, sorted by Ratio descending.

## Excluded / not comparable (no reliable head-to-head number)

- **olmoe-1b-7b-0924 (MoE, Q4_K_M)** and **granite-3.0-1b-a400m (MoE, Q4_K_M)** — measured 0.086× and
  0.187× respectively (round 12), but both fail the checklist's own health-assert (D2H syncs 2193/rep
  and 3225/rep vs the ~0 target) — the source doc explicitly flags these numbers as "likely NOT
  representative," so they're left out rather than reported as real MoE performance.
- **Qwen2.5-VL-7B-Instruct** — HartsyInference measures 63.33 tok/s but no valid llama.cpp baseline
  exists (llama-cpp-python 0.3.34 fails to construct a `llama_context` for this GGUF).
- **llava-v1.5-7b, gpt2-medium, starcoder2-3b** — crash the Tier-1 benchmark harness (works fine through
  the Swarm chat endpoint, 2.8–11.1 tok/s incl. model reload, but that's not a clean decode-only number).
- **mamba-2.8b-hf, Llama-3.2-11B-Vision** — no llama.cpp-comparable number was ever captured (SSM
  harness gap / VRAM-leak-on-load OOM respectively).

## Notes

- **9 "core fleet" text models, 8 of 9 at-or-ahead of llama.cpp as of 2026-07-24** (qwen2.5-0.5b,
  gemma-3-1b, DeepSeek-1.5B, Llama-3.2-1B, gemma-2-2b, gemma-4-E2B, Qwen3-4B, Qwen3.5-0.8B all beat
  llama.cpp by 9–32%; GLM-4-9B is the only non-win at 0.98× — effective parity, up from 0.82× at the
  start of the optimization campaign). Counting GLM's parity as "at-or-ahead," that's 9/9.
- The round-12 catalog expansion (Mistral-7B, gemma-3-4b, stablelm-2-1.6b, granite-3.0-2b,
  SmolVLM2-2.2B, Phi-3-mini) adds 6 more dense/VLM models with clean numbers: 5 of 6 beat llama.cpp
  (8–15% faster); **Phi-3-mini-4k-instruct is the one loss** (18% slower, not graph-eligible at its
  best config) — not root-caused as of the source doc's last update.
- **GLM-4-9B** is bounded by its square 4096² Q4_K attention-projection GEMV sitting at ~68% of DRAM
  bandwidth peak, resistant to the K-splitting trick that helped other models — the source doc calls
  0.92–0.98× "the practical frontier without structural work" (SoA repack and/or graph fork/join).
- **Qwen3-0.6B and the original Gemma-3-1B/Llama-3.2-1B CUDA-graph-era numbers** (README.md's small
  4-row table, dated 2026-07-10) reflect an early state before the dp4a int8-GEMV kernels and the
  residency/preload fixes landed. Gemma-3-1B and Llama-3.2-1B were re-benchmarked in later rounds and
  the newer numbers are used above; **Qwen3-0.6B was never re-benchmarked** after 2026-07-10, so its
  row here (0.57×, i.e. 1.77× slower) is almost certainly stale/pessimistic relative to the engine's
  current state — flagged, not re-measured, per the no-fabrication rule.
- **Prefill/TTFT is the current known weak point**, not decode: time-to-first-token trails llama.cpp by
  roughly 2–6× after three rounds of fixes (was 16–29× pre-fix), worst on GLM-4-9B (595 ms vs
  llama.cpp's 22 ms — parked pending MMQ-class quantized-GEMM prefill kernels). Decode throughput
  (the table above) is unaffected by this gap.
- **MoE models (olmoe, granite-moe) and three crash-prone architectures (llava, gpt2, starcoder2)**
  don't have a trustworthy Tier-1 number yet — see the Excluded section above.
- A Tier-2 Swarm-API integration pass exists on **RTX 4090** (`LLM_THROUGHPUT_BENCHMARK.md` "Phase 2"),
  confirming ~13-16 catalog models generate correctly end-to-end through SwarmUI, but it has no
  llama.cpp baseline on that GPU and measures wall-clock-incl.-reload rather than steady-state decode,
  so it's referenced here for completeness but not folded into the ratio table.
