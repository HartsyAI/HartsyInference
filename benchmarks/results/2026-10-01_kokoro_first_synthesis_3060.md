# Kokoro first synthesis of new text — RTX 3060 (2026-10-01)

**Question.** The voice session e2e (PR #202) showed the first synthesis of new text running 60-80 ms slower than
repeats: a 15-word sentence at 254 ms the first time against 150-191 ms after, "Okay." at 131 then 53 ms, and turn 1's
first chunk at 261 ms against the 250 ms budget. On a call nearly every sentence is new, so the honest gate is a
first-time 15-word sentence ≤ 250 ms.

**Answer.** Each new sentence length rebuilt about 39 cuDNN convolution plans, and ~97 % of a plan build is cuDNN's
heuristic query (~2 ms each). Choosing the conv engine once per power-of-two length bucket (alpha.232) brings new text
to within ~4 ms of a repeat: p50 / p95 180.1 / 220.3 ms back to back and 195.3 / 227.9 ms with 2-5 s gaps, against
212.9 / 255.6 and 245.4 / 296.1 ms before. The change is bounded, not byte-identical (log-spectral correlation
≥ 0.99995), and order-independent.

## Setup

- RTX 3060 12 GB (`GPU-b2198524…`), driver 595.91.07, cuDNN 9; persistence mode off, idle P8 at 210 / 405 MHz; host
  i7-6900K (8C/16T, `intel_cpufreq` + `schedutil`, 1.2-4.0 GHz, C-states to C6).
- Bench: `KokoroFirstSynthesisBenchTests` (opt-in `HARTSY_KOKORO_FIRST=1`), one process per arm,
  `CUDA_VISIBLE_DEVICES=1`. Each process: one warm-up synthesis (the session's `WarmAsync`), "Okay." twice, 20 distinct
  15-word phone-call sentences (the gate pass), then the same 20 again (lengths already seen). G2P is inside the timer,
  as the synthesizer lease runs it; `FreeActivations()` runs after each synthesis, outside the timer, as the voice
  session's GPU worker did. Gap mode sleeps a fixed 2-5 s sequence before every sentence of both passes.
- Per sentence: wall time, tokens and frames, device→host syncs, cuDNN plans built (with heuristic and finalize time,
  and how many came from a length bucket), cuBLASLt plans, pool reserve; nvidia-smi (P-state, SM / memory clock,
  utilization), host CPU frequency and vmstat sampled every 100 ms / 1 s alongside.
- Hygiene: 10-minute Swarm quiet window before each session (`tests/swarm-quiet-window.sh`), the shared bench lock
  taken before waiting for foreign jobs, a pre-arm gate (no foreign process on the 3060, no foreign dotnet build or
  test, host CPU < 15 % busy), and `--verify-since` after every arm (all exit 0). SwarmUI held an idle 430 MiB context
  on the 3060 during the validation session; the quiet-window script resets on any movement of it.

## Diagnosis (main + per-shape counters, build `77713614`)

| Arm | New text p50 / p95 ms | Same text again | "Okay." first → second | Conv plans per new sentence |
|---|---|---|---|---|
| a: back to back | 212.9 / 255.6 | 175.5 / 199.2 | 132.8 → 60.0 | 33 (60.9 ms building) |
| b: 2-5 s gaps | 245.4 / 296.1 | 197.5 / 226.7 | 131.5 → 62.5 | 33 (72.7 ms) |
| c: pool kept (no trim per job) | 216.5 / 242.6 | 175.6 / 200.6 | 164.0 → 66.0 | 33 (60.1 ms) |
| d: direct F32 conv kernels, no cuDNN | 607.2 / 665.7 | 601.2 / 660.9 | 174.9 → 171.1 | 0 |

- **The first-time cost is cuDNN plan building.** A plan is keyed by the exact shape with the time extent included, so
  a new frame count builds one per conv shape: 38-39 plans per new sentence. Per plan: heuristic 1.7-2.5 ms, finalize
  ~0.01 ms, graph descriptors negligible, no runtime-compiled engine (engines 28, 46, 55, 56, 58). "Okay." spends 61 of
  its 73 extra ms building plans. The three sentences whose frame count had already occurred (13, 16, 19) built 0-1
  plans and ran at repeat speed.
- **Ruled out.** cuBLASLt planning (11 plans per new length, 0.3 ms in total); the per-job pool trim (arm c matches arm
  a; ~86 MB re-mapped per synthesis costs nothing measurable); GPU clocks (1506 of 1533 samples in arm b were P2 at
  1807 MHz, P8 only before the process opened its context); G2P (no per-word cache; 0.03-0.35 ms).
- **The gap penalty is on the host.** With 2-5 s gaps the CPU-only heuristic time rose from 61 to 73 ms per sentence
  (sentence 1: 64 → 108 ms) while the GPU sat in P2. `schedutil` plus C6 clock the cores down during the gap
  (sampled in the validation runs: 1.3-1.6 GHz before a sentence after a gap, ~1.9 GHz back to back). That is an
  operator setting (governor, root).
- **Direct kernels are out.** Arm d is ~3× over budget; its audio differs from cuDNN's by TF32 rounding only
  (log-spectral correlation 0.9979-0.9997, same lengths).
- Output is deterministic: first and repeat passes byte-identical (20/20) in every arm, and two processes byte-identical.

## Fix

`CudnnConv` keeps its exact-shape plan cache, and for a 1D conv (H = 1, forward or transposed) takes the engine
configuration — engine global index and every knob choice — from `(conv family, power-of-two length bucket)`. The
heuristic runs once per bucket at the bucket's top length minus one; each new exact length builds its graph and
finalizes a plan from that configuration (~0.1 ms). A configuration that does not finalize for a length (or wants more
than the workspace cap) falls back to that length's own heuristic, never disabling cuDNN for the session. The choice
cache is a `ConcurrentDictionary<string, Lazy<EngineChoice?>>` (one heuristic per bucket under a race, as `CudnnSdpa`
guards its plans). Kill switch: `numerics.audioConvLengthBuckets=false`.

The first version ran the heuristic at the power-of-two length itself. There the heuristic picked edge-free tiles that
only finalize for aligned lengths: 119 fallbacks over 40 sentences, all in the 128-channel generator stage, and new
text at p50 185.8 ms. An odd reference length is a multiple of no tile, so the chosen configuration handles a ragged
final tile and fits every length in the bucket: 0 fallbacks.

## Validation (build `3f63901b`; the buckets-off control ran on `7f514cb3`, whose per-length path is the same code)

| Arm | New text p50 / p95 ms | Same text again | Conv plans per new sentence | Against main's audio |
|---|---|---|---|---|
| fix, back to back | **180.1 / 220.3** | 176.3 / 202.2 | 33, all from buckets (6.2 ms) | 0/20 identical, floor below |
| fix, reverse order | 179.9 / 200.4 | 178.3 / 205.8 | 33 (6.4 ms) | 20/20 identical to the forward fix run |
| fix, 2-5 s gaps | **195.3 / 227.9** | 203.4 / 213.8 | 33 (6.9 ms) | 0/20 identical, same floor |
| fix with buckets off | 219.0 / 251.7 | 173.6 / 187.6 | 33 (62.3 ms, heuristic) | 20/20 identical to main |

Per sentence, first pass (ms, conv plans built):

| # | frames | main, back to back | fix, back to back | main, gaps | fix, gaps |
|---:|---:|---:|---:|---:|---:|
| 1 | 214 | 203.3 (39) | 228.0 (39) | 315.0 (39) | 221.8 (39) |
| 2 | 222 | 246.9 (39) | 199.8 (39) | 295.1 (39) | 230.6 (39) |
| 3 | 231 | 224.2 (38) | 211.3 (38) | 275.9 (38) | 199.2 (38) |
| 4 | 249 | 254.8 (39) | 199.4 (39) | 264.2 (39) | 218.2 (39) |
| 5 | 226 | 232.1 (39) | 186.8 (39) | 245.5 (39) | 204.6 (39) |
| 6 | 243 | 230.8 (39) | 219.9 (39) | 278.9 (39) | 227.8 (39) |
| 7 | 201 | 196.9 (38) | 189.0 (38) | 228.5 (38) | 189.2 (38) |
| 8 | 241 | 272.0 (39) | 206.6 (39) | 230.2 (39) | 211.3 (39) |
| 9 | 215 | 202.5 (39) | 173.3 (39) | 245.4 (39) | 184.3 (39) |
| 10 | 232 | 241.6 (39) | 198.5 (39) | 282.7 (39) | 211.8 (39) |
| 11 | 210 | 192.7 (39) | 181.8 (39) | 273.5 (39) | 186.1 (39) |
| 12 | 186 | 207.5 (38) | 170.2 (38) | 233.5 (38) | 181.6 (38) |
| 13 | 201 | 174.6 (1) | 178.4 (1) | 189.0 (1) | 181.9 (1) |
| 14 | 212 | 216.1 (39) | 154.5 (39) | 247.5 (39) | 188.6 (39) |
| 15 | 185 | 223.3 (39) | 160.9 (39) | 229.1 (39) | 178.4 (39) |
| 16 | 212 | 171.6 (0) | 173.8 (0) | 197.6 (0) | 191.8 (0) |
| 17 | 177 | 211.5 (38) | 141.4 (38) | 220.6 (38) | 178.9 (38) |
| 18 | 207 | 214.3 (38) | 171.2 (38) | 239.3 (38) | 191.2 (38) |
| 19 | 226 | 146.3 (0) | 139.6 (0) | 189.2 (0) | 205.8 (0) |
| 20 | 208 | 208.1 (39) | 176.7 (39) | 254.6 (39) | 198.8 (39) |

In the fix columns every plan came from a bucket; the first sentence also paid the reference builds of the buckets
the warm-up had not touched.

- **Gate met** for new text in both modes, at p50 and p95. Repeats are unchanged (176.3 against 175.5 ms; two runs of
  main differed by ~2 ms), so the bucket configurations are not slower kernels.
- **A cold bucket still pays once.** "Okay." with the fix: 123.8 → 64.3 ms (its 32-frame bucket is new). The voice
  session's warm-up across the 32-512-frame buckets removes that for in-call sentences.
- **Bounded, not byte-identical.** With buckets off the per-length heuristic itself used different engines for
  different lengths of one family (for example 46, 55 and 56 for the 512-channel k3 convs), so no single choice per
  bucket can reproduce it. Floor against main, both modes: log-spectral correlation ≥ 0.999952, max-abs ≤ 5.22e-3,
  identical lengths, Whisper-tiny recall unchanged (min 71 % / mean 93 %).
- **Order-independent.** The 20 sentences in reverse order give byte-identical audio (20/20); the CUDA test pins it on
  two lengths of one bucket.

### Piper (`en_US-ryan-medium`, seed 1234, each arm on a fresh backend)

| # | samples | buckets off ms | on ms | on, reverse ms | on vs off |
|---:|---:|---:|---:|---:|---|
| 1 | 6400 | 124.6 | 96.4 | 75.9 | log-spec 1.000000, max-abs 3.29e-5 |
| 2 | 28928 | 241.2 | 170.6 | 146.7 | log-spec 1.000000, max-abs 5.74e-5 |
| 3 | 88576 | 629.8 | 575.6 | 371.5 | log-spec 0.994974, max-abs 0.290 |
| 4 | 89600 | 576.9 | 413.2 | 387.8 | log-spec 1.000000, max-abs 1.07e-4 |
| 5 | 106752 | 726.6 | 541.9 | 548.4 | log-spec 1.000000, max-abs 4.00e-4 |
| 6 | 140800 | 875.9 | 724.3 | 731.3 | log-spec 0.999932, max-abs 1.22e-2 |

Buckets on and on in reverse order are byte-identical on all six. First synthesis of the six took 3175 → 2522 ms
(heuristic time 313 → 185 ms). Sentence 3 is analyzed against exact math in the spot check below.

### CUDA tests on the 3060

`CudnnConvPlanStatsTests` 16/16 (on these test shapes the bucketed forward and transposed outputs are byte-identical to
the per-length heuristic); the `GpuIntegration` category in four batches, 99 / 101 / 104 / 9 passed (the two-GPU NCCL
test and the opt-in soak tests return early with one GPU visible); the untagged cuDNN conv classes (`CudnnConvTests`,
`Conv1dKernelTests`) 22/22 with the plan-cache tests. No failures.

## Spot check: four more vocoder families, buckets off vs on (engine `6d0e717a`, 2026-10-01 07:27-08:32Z)

Each arm is a fresh process and backend on the 3060, with the same text, seed (1234) and reference audio. Every sentence
is synthesized twice: first (per-length setup) and repeat (steady state, every plan cached). The repeat against the first
is the model's own run-to-run noise floor. Piper also has a no-TF32 arm (direct F32 conv kernels, full-precision
GEMMs) as an exact-math reference. The LM models cannot have that one, because changing GEMM precision can flip a
sampled token; CosyVoice 2 instead has a conv-only F32 arm (direct F32 conv kernels, GEMMs untouched). Scored on the
CPU: Whisper small.en, log-spectral correlation, max-abs, and where on and off differ beyond rounding the alignment lag
and the difference distribution. Every bucketed arm built its plans from buckets with no fallback.

| Model (codec) | # | on vs off | Noise floor (repeat vs first), off / on | Recall off / on | Transcripts |
|---|---:|---|---|---|---|
| CosyVoice 2 (HiFT) | 1 | log-spec 0.987337, max-abs 1.10 | ≥ 0.999999, ≤ 8.5e-5 / same | 100 % / 100 % | same |
| CosyVoice 2 (HiFT) | 2 | log-spec 0.979200, max-abs 1.98 | ≥ 0.999999, ≤ 1.2e-4 / same | 100 % / 100 % | same |
| Kyutai TTS (Mimi) | 1 | identical | identical / identical | 100 % / 100 % | same |
| Kyutai TTS (Mimi) | 2 | log-spec 0.999657, max-abs 8.5e-5 | identical / identical | 100 % / 100 % | same |
| Orpheus (SNAC) | 1 | log-spec 0.999959, max-abs 1.1e-4 | identical / identical | 100 % / 100 % | same |
| Orpheus (SNAC) | 2 | identical | identical / identical | 100 % / 100 % | same |
| Piper (VITS) | 1-6 | ≥ 0.999932 except #3 (below) | identical (deterministic) | 0 / 100 / 100 / 100 / 100 / 85 %, both arms | same |

- **Transcripts match for every sentence of every model.** Recall is identical between arms. Piper's "Okay." scores
  0 % in every arm, including F32, so that is Whisper on a 0.3 s clip, not the change.
- **Piper sentence 3 against exact math.** Buckets off vs F32: log-spectral correlation 0.995503, max-abs 0.380.
  Buckets on vs F32: 0.996205, max-abs 0.320. The bucketed arm is no further from exact math than the per-length arm;
  here it is slightly closer. Every Piper sentence sits 0.9965-0.9992 from F32 in both arms, which is Piper's own TF32
  band, and on vs off is inside it.
  - **What moved in sentence 3:** no time shift (best alignment lag 0 samples, correlation 0.9975 either way). The
    difference is local: 89 % of its energy sits in the top 5 % of 10 ms windows, peaking at 2.85 s of 4.02 s, while
    the waveforms agree to ~1e-3 elsewhere. A TF32-level difference upstream of the vocoder (the flow) reshaped one
    short stretch of the waveform; lengths and the transcript are unchanged.
- **CosyVoice 2 moves most, and its two arms are equally far from exact conv math.** On vs off is 0.987 and 0.979,
  far above its run-to-run floor (≈ 1.0). A conv-only F32 arm (`numerics.audioConvCudnn=false`, GEMMs untouched, so
  the sampled tokens did not move; same lengths) gives CosyVoice's own band against exact conv math:

  | Sentence | off vs conv-F32 | on vs conv-F32 | on vs off |
  |---:|---|---|---|
  | 1 | 0.983683, max-abs 1.66 | 0.981989, max-abs 1.95 | 0.987337, max-abs 1.10 |
  | 2 | 0.981261, max-abs 1.93 | 0.985641, max-abs 1.85 | 0.979200, max-abs 1.98 |

  - Both TF32 arms sit 0.982-0.986 from exact conv math. That is CosyVoice's TF32 band: its flow ODE carries any conv
    rounding into the mel, and the bucket choice moves it no more than the per-length choice does.
  - **What moved, per sentence.** Sentence 1: no time shift (best lag −1 sample, correlation 0.884 against 0.873 at
    lag 0), and the difference is spread across the clip (the top 5 % of 10 ms windows hold only 32 % of its energy).
    Sentence 2: a sub-millisecond phase shift (best lag −15 samples, −0.6 ms, lifts the correlation from −0.22 to 0.65),
    which is what HiFT's NSF source does when the F0 it integrates moves slightly.
  - Transcripts and recall (100 %) are the same in all three arms.
  - Incidentally, the conv-only F32 arm is deterministic run to run while both cuDNN arms vary by ≤ 1.2e-4, and the
    direct kernels ran CosyVoice faster here (6535 / 10431 ms first synthesis against 7350-7492 / 11811-11866 ms).
    That is outside this change; it is noted for a follow-up look at CosyVoice's conv routing.

**Timing, buckets off → on** (sum over the model's sentences; first = per-length setup, repeat = steady state):

| Model | First synthesis | Repeat |
|---|---|---|
| Piper (6 sentences) | 3021.4 → 2815.5 ms (−6.8 %) | 2495.3 → 2502.0 ms (+0.3 %) |
| CosyVoice 2 (2) | 19357.9 → 19160.7 ms (−1.0 %) | 18983.5 → 18983.0 ms (0.0 %) |
| Kyutai TTS (2) | 7485.6 → 7450.9 ms (−0.5 %) | 7319.6 → 7407.4 ms (+1.2 %) |
| Orpheus (2) | 15481.0 → 14890.0 ms (−3.8 %) | 15229.1 → 14838.5 ms (−2.6 %) |

There is no regression beyond run-to-run noise. Per sentence the arms move up to ±10 % in both directions. For
example, Kyutai's buckets-on repeat of sentence 2 (4987.4 ms) was slower than its own first synthesis (4922.6 ms),
which also paid the plan builds.

## Reproduce

```bash
# One arm (CUDA_VISIBLE_DEVICES=1 puts the 3060 at ordinal 0):
CUDA_VISIBLE_DEVICES=1 HARTSY_KOKORO_FIRST=1 HARTSY_KOKORO_FIRST_CUDA_ORDINAL=0 \
  HARTSY_KOKORO_FIRST_MODE=burst HARTSY_KOKORO_FIRST_OUT_DIR=out/fix HARTSY_KOKORO_FIRST_REF_DIR=out/main \
  dotnet test tests/HartsyInference.Audio.Tests -c Release -f net10.0 \
  --filter "FullyQualifiedName~KokoroFirstSynthesisBenchTests.Kokoro_DistinctSentences_3060_FirstSynthesis"
# HARTSY_KOKORO_FIRST_MODE=gaps, HARTSY_KOKORO_FIRST_ORDER=reverse, HARTSY_KOKORO_FIRST_PROFILE=1 (stage timer),
# HARTSY_KOKORO_FIRST_KNOBS='{"settings": {"numerics.audioConvLengthBuckets": false}}' for the per-length arm.
```
