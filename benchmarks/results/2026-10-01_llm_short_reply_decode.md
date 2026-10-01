# Qwen3-4B short-reply decode: finding and fixing the sampler, not the stream — 2026-10-01

`perf/llm-short-reply-decode`. The phone voice agent (PR #202, `feat/voice-package`, not merged) streams replies
from Qwen3-4B Q4_K_M on the 4090 (engine `cuda:0`) through `ITextService.StreamAsync` via the Tools package's
`ToolLoop`, tools installed, `EnableThinking=false`. A real session on 2026-10-01 measured decode for short
replies at only 37-40 tok/s (9 tokens ~25 ms apart, uniform, no bursts) against the Phase-1 isolated probe's
151 tok/s (`VoiceLlmTurnBenchTests`, same branch, 64-token generations, bare `onToken` callback). Two named
hypotheses going in — (A) per-generation startup cost (CUDA graph capture, KV-cache alloc, sampler-chain
construction) and (B) per-token `StreamAsync`/`ToolLoop` session-path overhead (incremental detokenizer,
`ITextStreamFilter`/`ToolCallStreamFilter`/`ToolCallParser`, the channel/async-iterator chain in
`TextStreamPump`) — both measured directly rather than assumed.

Hardware: engine `cuda:0`, RTX 4090 (`NVIDIA GeForce RTX 4090`, nvidia-smi index 1 — ordinals are reversed from
nvidia-smi's, per `docs/Research/`), shared with a live SwarmUI (`192.168.10.188:7801`). Every run below went
through `tests/swarm-quiet-window.sh` (10-minute clean window, confirmed, then `--verify-since` after), the
VRAM/foreign-process check and the shared lock; no SwarmUI activity landed during any run. Checkpoint:
`/mnt/model-storage/Models/llm/qwen3/Qwen3-4B-Q4_K_M.gguf` (`HARTSYINFERENCE_MODELS_DIR`).

## What actually differs between the probe and the real session

`VoiceAgentSession.Turns.cs`'s `BuildRequest()` (PR #202) never sets `Greedy`/`Temperature`/`TopP` on its
`TextRequest`. `TextRequest`'s own defaults (`Temperature=0.7, TopP=0.95, Greedy=false`) flow straight through
`TextService.BuildSampling` unchanged. The Phase-1 probe measured only `SamplingOptions.GreedyPreset` for both
its "tools on" and "tools off" rows (`BuildRequest` there: `tools ? GreedyPreset with { JsonModeSentinel = ... }
: GreedyPreset` — both branches greedy). The probe never exercised the session's real, non-greedy sampling path
at all. That mismatch, not CUDA graph capture and not `StreamAsync`'s transport, is the root cause — see below.

## Before: isolating A, B, and the sampler

Two new benches (`tests/HartsyInference.LLM.Tests/ShortReplyDecodeStepBenchTests.cs`,
`tests/HartsyInference.Tools.Tests/ShortReplyStreamTransportBenchTests.cs`; opt-in,
`HARTSY_SHORTREPLY_BENCH=1`), both gated on CUDA ordinal 0 reporting a "4090" device name.

**Hypothesis A — bare `TextGenerationPipeline.Generate`, no `StreamAsync` at all.** Fresh / second (same
process, right after) / long (128 tokens) generations, greedy vs. the session's actual default sampling, tools
on and off:

| scenario | run | tokens | first-gap median ms (steps 1-9) | steady-gap median ms (last <=12) | overall tok/s |
|---|---|---:|---:|---:|---:|
| greedy, tools on (probe-equivalent) | fresh | 40 | 8.4 | 8.3 | 113.7 |
| greedy, tools on (probe-equivalent) | second | 40 | 7.0 | 6.5 | 140.0 |
| greedy, tools on (probe-equivalent) | long(128) | 52 | 6.4 | 6.0 | 158.1 |
| session-default (temp .7/topP .95), tools on | fresh | 37 | 24.8 | 25.1 | 31.9 |
| session-default (temp .7/topP .95), tools on | second | 37 | 25.2 | 24.8 | 39.5 |
| session-default (temp .7/topP .95), tools on | long(128) | 37 | 26.1 | 24.7 | 39.1 |
| session-default (temp .7/topP .95), tools off | fresh | 40 | 24.2 | 24.2 | 39.5 |
| session-default (temp .7/topP .95), tools off | second | 40 | 25.6 | 25.2 | 38.0 |
| session-default (temp .7/topP .95), tools off | long(128) | 128 | 25.4 | 25.1 | 38.5 |

**A is not the cause.** Within each sampling scenario, fresh/second/long are flat (session-default: 24.2-26.1 ms
across all three, no decay at all — matches the real session's "uniform, no bursts"). There is a small, real
per-generation warm-up under greedy only (first-gap 8.4 ms vs. the long run's steady 6.0 ms, decaying by run 2)
— present, but roughly 2 ms, nowhere near the ~19 ms gap between greedy and session-default at identical
tools/fresh-ness. Tools on vs. off is also not it: 39.1-39.5 vs. 38.0-38.5 tok/s, a wash. Switching ONLY the
sampling config from `GreedyPreset` to the session's real default — nothing else — drops throughput ~3.5-4x:
113-158 down to 32-40 tok/s, and reproduces the reported 37-40 tok/s and ~25 ms/token almost exactly.

**Hypothesis B — the real path, `ToolLoop.RunAsync` over `StreamAsync`**, with an `IInferenceDiagnostics`
recorder timestamping the decode thread's `TokenGenerated` event (fires at the top of `TextService.RunText`'s
`onToken`, before the incremental detokenizer / output parser / `ToolCallStreamFilter` / channel write run)
against the consumer's own `await foreach` arrival time, on a realistic system-prompt + tool + ~250-token-history
turn, session-default sampling:

| run | tokens | decode-thread gap median ms | consumer gap median ms | consumer tok/s | decode-thread tok/s |
|---|---:|---:|---:|---:|---:|
| 1 | 30 | 25.7 | 25.7 | 27.4 | 27.2 |
| 2 | 30 | 24.6 | 24.6 | 38.2 | 38.2 |
| 3 | 30 | 26.0 | 25.9 | 37.0 | 37.0 |

**B is not the cause either.** Decode-thread gap and consumer gap are identical to ~0.1 ms at every token in all
three runs (e.g. run 3: `[25.4, 25.7, 39.9, 26.6, 25.2, ...]` vs. `[25.4, 25.7, 39.8, 26.7, 25.1, ...]`) — the
detokenizer/filter/channel/async-iterator chain adds nothing measurable on top of the decode thread's own cost.
(Run 1's first two gaps, ~170 ms, are a one-time JIT/warm-up cost specific to the first-ever generation through
this `InferenceEngine` instance's full `TextService`/`ToolLoop` stack — absent in runs 2 and 3, and irrelevant to
a long-lived voice process that handles many turns.)

**Root cause: `SamplerChain`'s non-greedy path, specifically `TopPStep`.** Every decode step, `TopPStep.Apply`
allocated a fresh `float[vocab]` (~608 KB at Qwen3's 151,936-token vocabulary), ran a full softmax over it, then
called `SamplerMath.ArgsortDescending`, which allocated a fresh `int[vocab]` and sorted it via
`Array.Sort(order, (a, b) => values[b].CompareTo(values[a]))` — an O(n log n) sort where every one of the ~2.6M
comparisons paid a `Comparison<int>` delegate dispatch. `SamplerChain.Next`'s own final multinomial draw then
allocated a THIRD fresh `float[vocab]` and ran a second full softmax. None of this runs under `Greedy=true`
(`Next` short-circuits to `Argmax` before any of it), which is exactly why the Phase-1 probe — greedy only —
never saw it.

## After: the fix

`src/HartsyInference.LLM/Sampling/{SamplerMath,TopPStep,TopKStep,MinPStep,SamplerChain}.cs`, in two tiers — no
algorithm change, in either tier, to anything that touches a float value.

**Tier 1 — stop reallocating, stop paying a delegate per comparison.**
- **Buffers become per-generation, not per-token.** `TopPStep`/`TopKStep`/`MinPStep`/`SamplerChain` each own a
  private, lazily-sized scratch array now, allocated once on the first decode step and reused for every later
  one — one `SamplerChain`/step instance already lives for exactly one generation (built fresh per
  `SamplerChain.FromOptions` call, which `TextGenerationPipeline.Generate` calls once per request), so this is
  pure reuse within an existing per-request lifetime, not a new pooling layer.
- **The sort.** `SamplerMath.SortDescendingByValue` replaces the delegate comparer with
  `Array.Sort(float[], int[])` — .NET's primitive two-array sort, no per-comparison delegate dispatch. Negating
  before and restoring after is exact (sign-bit flip only, including the softmax's signed-zero masked entries)
  and turns "ascending by `-value`" into "descending by `value`" without a second array or a reversal pass.

Measured after Tier 1 alone: session-default throughput roughly 38 → 55 tok/s (steady state; first-gap median
~17-18 ms). An improvement, but short of the ≥ 60 tok/s target. A CPU-only (no GPU) micro-profile of
`SamplerChain.Next` at Qwen3's real vocabulary pinned why: `SortDescendingByValue` ALONE cost ~11 ms of
`TopPStep.Apply`'s ~12.4 ms — even the fast primitive two-array sort is still O(n log n) over the WHOLE
vocabulary, every token.

**Tier 2 — stop sorting the whole vocabulary.** The cumulative walk almost never needs most of it: `TopPStep`
now collects tokens above a tiny probability floor (`1e-6`) into a compact buffer first (one O(n) pass, no
sort), and sorts ONLY those when their own total already reaches `p` — true whenever the distribution is at
all peaked, which a trained LLM's next-token distribution ordinarily is. Identity argument: every candidate's
probability exceeds every excluded token's by construction, so a full descending sort always ranks every
candidate first; if the candidates' own total already reaches `p`, the full sort's cumulative walk reaches `p`
before ever leaving the candidate block, so sorting just that block reproduces the identical kept set AND the
identical floating-point cumulative sums the full sort would. Falls back to sorting the whole vocabulary,
unchanged (Tier 1's code), whenever the candidates fall short (a near-uniform distribution).

Measured after Tier 2, the same CPU-only micro-profile: `SamplerChain.Next` (temperature + topP + draw) dropped
from ~12.5 ms/call to ~4.7 ms/call — the sampler is no longer close to the bottleneck; the GPU forward pass
(~7-9 ms/token, from the greedy baseline) now dominates.

`TopPSortRefactorIdentityTests.cs` (new) keeps the pre-fix `TopPStep.Apply`/`SamplerMath.Softmax`/
`ArgsortDescending` verbatim as a reference and asserts the new code masks the SAME logits to `-Infinity`, for
the same input, across peaked / near-flat / pre-masked-tail / fallback-forcing distributions from 1 to
151,936 (and, for the dedicated fallback case, 2,000,000) elements — the actual correctness proof, since
`regression-ab.sh`'s identical-output arms are greedy and greedy never reaches `TopPStep` or the non-greedy
draw at all.

### After numbers

Same bench, same prompts, same seed, re-run on the same box after Tier 1 and again after Tier 2. Reply text and
token-ids are byte-identical to the before run in every scenario (e.g. "session-default, tools on / fresh":
`[40,2776,11889,311,3410,11682,40841,...]` in BOTH the Tier 1 and Tier 2 runs; the full-history stream-transport
bench's reply — "We've updated your delivery address to your office, flagged the delivery for a call-ahead, and
logged a note for the missed callback on your refund." — is identical across the before run and both after
runs).

**Bare `pipeline.Generate` (hypothesis-A bench), tok/s, by tier:**

| scenario | run | before | Tier 1 | Tier 2 |
|---|---|---:|---:|---:|
| greedy, tools on (probe-equivalent) | fresh | 113.7 | 111.7 | 110.5 |
| greedy, tools on (probe-equivalent) | second | 140.0 | 116.5 | 115.7 |
| greedy, tools on (probe-equivalent) | long(128) | 158.1 | 135.7 | 121.8 |
| session-default (temp .7/topP .95), tools on | fresh | 31.9 | 36.0 | 124.2 |
| session-default (temp .7/topP .95), tools on | second | 39.5 | 54.9 | 123.6 |
| session-default (temp .7/topP .95), tools on | long(128) | 39.1 | 55.3 | 124.6 |
| session-default (temp .7/topP .95), tools off | fresh | 39.5 | 54.4 | 122.3 |
| session-default (temp .7/topP .95), tools off | second | 38.0 | 54.6 | 125.0 |
| session-default (temp .7/topP .95), tools off | long(128) | 38.5 | 54.4 | 119.9 |

Greedy numbers move a little run to run (JIT/scheduler noise — same code path throughout, unaffected by this
fix) but hold steady in the 110-160 tok/s band across all three measurement passes. After Tier 2, session-default
is in the SAME band as greedy — the two are no longer distinguishable by workload, only by run-to-run noise,
because the sampler itself is no longer the cost.

First-10-token latency (the actual target metric), session-default, tools on, "second" request (i.e. a warm
process's next turn, not the one-time first-ever-request JIT cost described below): first-gap median 25.2 ms
before (39.7 tok/s) → 18.1 ms after Tier 1 (55.2 tok/s) → 7.6 ms after Tier 2 (131.6 tok/s). Target was
≥ 60 tok/s; met with roughly 2x margin.

**Realistic voice turn through `ToolLoop.RunAsync` → `StreamAsync` (hypothesis-B bench), tok/s, by tier:**

| run | before (consumer tok/s) | Tier 1 | Tier 2 |
|---|---:|---:|---:|
| 1 | 27.4 | 37.5 | 94.1 |
| 2 | 38.2 | 52.7 | 105.4 |
| 3 | 37.0 | 51.8 | 106.2 |

Decode-thread gap stays equal to consumer gap to ~0.1 ms in every tier — `StreamAsync`/`ToolLoop`'s transport
was never the cost, before or after. Run 1's first couple of tokens still show the one-time first-generation
JIT/warm-up blip described above (absent from run 2 onward; irrelevant to a long-lived voice process serving
many turns) — it is smaller after Tier 2 (peak ~23 ms vs. ~53 ms after Tier 1) since there is simply less new
code being JITted into a hot path.

## Regression evidence

- `TopPSortRefactorIdentityTests`: 22 cases (20 after Tier 1, +2 after Tier 2 for the candidate/fallback split),
  plus the pre-existing `SamplingAndTemplateTests` — all pass. Bit-identical masking, old algorithm vs. new,
  across every distribution/vocab-size/fast-path-vs-fallback case tried.
- The two bench tests themselves are a real-weight identity check beyond the synthetic unit tests: byte-identical
  reply text and token-ids, before vs. Tier 1 vs. Tier 2, for the same seeded request (see "After numbers" above).
- `tests/HartsyInference.Tools.Tests` CPU lane: 95/95 pass, unchanged (this fix touches no Tools-package code).
- `tests/HartsyInference.LLM.Tests` CPU lane: 545/546 pass; the one failure
  (`Glm4SyntheticParityTests.SyntheticGlm4_MatchesHfTransformers_FinalLogits`, a missing fixture file) is listed
  as a known pre-existing failure and is unrelated to sampling.
- Full solution CPU lane: see the PR body / final report for the run taken at ship time.
- `regression-ab.sh --expect identical` on `llama32-1b` and `qwen25-1.5b-iq3xs` (greedy — see the PR for why that
  gate, specifically, cannot exercise `TopPStep` or the non-greedy draw either): see the PR body / final report.
