# Audio (TTS / STT / Music / VC / Fx) — canonical scoreboard

Consolidates the audio-modality benchmark data scattered across `README.md`, `docs/Checklists/AUDIO_THROUGHPUT_BENCHMARK.md`,
`benchmarks/results/audio_tts_stt_2026-07-12.md`, and 14 dated per-model result files into one place. For methodology
(warm/cold definition, GPU pinning, profiler flags) see [`README.md`](README.md).

**GPUs.** RTX 3060 (primary; audio is usually pinned here via `HARTSY_AUDIO_CUDA_DEVICE=1`) and RTX 4090 where noted.
A couple of CPU baselines exist (Kyutai TTS). Most rows have only one GPU measured, not both.

**Baselines.** Audio has no shared engine like ComfyUI (image) or `llama-cpp-python` (LLM) — each architecture has
its own bespoke upstream Python package, or none published at all. Where a real, timed Python reference exists
(`moshi` for Kyutai, official `qwen_tts`, a `CAMPAIGN_SCORECARD.md` harness), it's cited in the Baseline column.
Everywhere else, "Baseline" is **self-comparison** (this repo's own before→after perf-pass number) — there is no
external ground truth to compare against, only whether the model got faster than it used to be.

**RTF convention (important — sources disagree).** This file reports **RTF = generated-audio-seconds ÷
wall-clock-seconds (higher is faster; RTF > 1× means faster than real time)**, matching `README.md`'s definition.
Several source files (`AUDIO_THROUGHPUT_BENCHMARK.md`'s own tables, `vibevoice_tts`, `cosyvoice2_tts`, `bark_tts`,
`chatterbox_tts`, `qwen3_tts_perf`, `fishspeech_tts`, `neutts_tts`, the 2026-07-26 Dia/MusicGen/AudioGen/ACE-Step
table) instead report the inverse, standard ASR-style **RTF = wall ÷ audio (lower is faster)**. Where a row's number
was converted, the cell shows the computed ×-value **and** the raw wall/audio pair it was derived from, so it can be
re-derived independently. Two autoregressive-decode models (HeartMuLa, Zonos) use a third, genuinely different unit
(**ms/frame**, lower is better) because they're judged frame-by-frame, not clip-by-clip — see the second table.

**Deploy-path column.** Several perf-pass fixes are explicitly **engine-only, not yet packed/deployed to the Swarm
extension** as of the date of this file (Qwen3-TTS, Kyutai TTS, Kyutai STT). Others were measured in a **standalone
harness** driving the pipeline directly, not through Swarm (Bark, Chatterbox, Fish-Speech, NeuTTS, Zonos, VibeVoice,
CosyVoice2). This matters: the full-fleet Tier-3 Swarm sweep (2026-07-25, `AUDIO_THROUGHPUT_BENCHMARK.md`) still
shows several of these models at their *old, pre-fix* speed because the fix hadn't shipped to Swarm yet by that
date — e.g. Kyutai STT reads ~0.82× in the Tier-3 sweep vs. 6.6–10.1× in its dedicated engine-level perf pass, same
day range. Both numbers are real; they measure different things.

---

## One-shot / streaming TTS + STT + Music + VC + Fx

| Model | Type | GPU | RTF (audio÷wall, higher=faster) | Baseline | Ratio | Date | Path | Source |
|---|---|---|---:|---|---:|---|---|---|
| Piper (VITS) | TTS | 3060 / 4090 | **8.6× / 8.3×** | self-comparison | — | 2026-07-12 | Swarm | `audio_tts_stt_2026-07-12.md` |
| Kokoro-82M (StyleTTS2) | TTS | 3060 / 4090 | **4.5× / 5.2×** | self-comparison | — | 2026-07-12 | Swarm | `audio_tts_stt_2026-07-12.md` |
| StyleTTS2 (LibriTTS) | TTS (clone) | 4090 | **~1.3×** | self-comparison | — | 2026-07-15 | Swarm | `audio_tts_stt_2026-07-12.md` |
| MeloTTS (en-v3) | TTS | 3060 / 4090 | **1.4× / 1.4×** | self-comparison (still host/GPU-flat; no perf pass yet) | — | 2026-07-12 | Swarm | `audio_tts_stt_2026-07-12.md` |
| F5-TTS (v1 base) | TTS (clone) | 3060 | **~0.4×** | self before→after (174.6s → 6.4s host-conv fix); Python ref RTF 0.729 (wall÷audio, not cross-compared — different harness/GPU) | 34× (self) | 2026-07-13 | Swarm | `audio_tts_stt_2026-07-12.md`, `tests/python-reference/CAMPAIGN_SCORECARD.md` |
| Qwen3-TTS (1.7B) | TTS | 3060 / 4090 | **2.01× / 3.77×** (1.87s/3.76s; 1.14s/4.32s) | official `qwen_tts` (PyTorch): 0.42× (3060) / 0.43× (4090) | **4.8× / 8.8×** faster than official | 2026-07-18 | engine-only | `qwen3_tts_perf_2026-07-18.md` |
| VibeVoice-1.5B | TTS | 3060 | **1.28×** (6.47s/8.27s) | self before→after (RTF 0.061× → 1.28×, 20.8× cumulative) | 20.8× (self) | 2026-07-17 | standalone harness | `vibevoice_tts_2026-07-17.md` |
| CosyVoice 2 (0.5B) | TTS (clone) | 4090 | **0.75×** (7.97s/6.0s) | self before→after (RTF 0.217× → 0.75×, 3.4× cumulative; gallery gen 28.9s→5.1s = 5.7×) | 5.7× (self) | 2026-07-17 | standalone harness | `cosyvoice2_tts_2026-07-17.md` |
| Chatterbox | TTS (clone) | 3060 | **1.45×** short / **1.11×** long (3.51s/5.08s; 12.36s/13.80s) | self-comparison (inherited CosyVoice2 S3Gen fix for free; "no perf pass needed") | — | 2026-07-18 | standalone harness | `chatterbox_tts_2026-07-18.md` |
| Bark (Suno) | TTS | 3060 | **0.44×** (12.23s/5.32s) | self before→after (RTF 0.064× → 0.44×, 6.75× cumulative) | 6.75× (self) | 2026-07-18 | standalone harness | `bark_tts_2026-07-18.md` |
| Kyutai TTS (DSM, tts-1.6b) | TTS | 3060 / 4090 | **1.09× / 1.47×** | `moshi` (Python, bf16+CUDA graph): 2.25× (3060) | **0.48×** of moshi (moshi still ~2.1× faster) | 2026-07-18 | engine-only | `kyutai_tts_perf_2026-07-18.md` (supersedes `kyutai_tts_2026-07-16.md`'s 0.51×) |
| NeuTTS Air | TTS (clone) | 3060 | **0.52×** decode (7.52s/3.92s); encode (one-time/ref) **0.67×** | self-comparison, verification only (no perf pass run) | — | 2026-07-18 | standalone harness | `neutts_tts_2026-07-18.md` |
| Fish-Speech 1.5 | TTS | 3060 | **~0.95×** (near real-time; 3.56s/3.44s, 6.86s/6.55s) | self-comparison, verification only (no perf pass needed) | — | 2026-07-18 | standalone harness | `fishspeech_tts_2026-07-18.md` |
| Spark-TTS-0.5B | TTS | 4090 | **0.167×** (22.3s/~3.7s) | none | — | 2026-07-25 | Swarm (Tier 3, incl. cold-load) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| GPT-SoVITS v2 | TTS (clone) | 4090 | **0.146×** (39.54s/~5.76s) | none | — | 2026-07-24 (Tier 2, stale) | Swarm (legacy `ProcessTTS`) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| Orpheus | TTS | 4090 | **0.314×** (11.4s/~3.6s) | no comparable baseline published (the only upstream figure is a self-comparison with no absolute numbers) | — | 2026-07-25 | Swarm (Tier 3, incl. cold-load) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| ZipVoice | TTS | 4090 | **0.055×** (99.1s/~5.4s) | none — known-slow, unoptimized (~11 min/10s clip documented elsewhere) | — | 2026-07-25 | Swarm (Tier 3, incl. cold-load) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| Dia 1.6B | TTS | 4090 | **0.142×** (44.3s/6.3s, warm, seed=42) | self before→after (350–800s+ / non-terminating → 44.3s) | large (self, not a clean multiplier — old baseline was inconsistent-seed) | 2026-07-26 | Swarm | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) (07-26 SDPA-batching entry; supersedes the 07-12 RTF 0.036× and the Tier-3 "hang") |
| CSM-1B (Sesame) | TTS | 4090 | **0.024×** (173.72s/~4.2s) | none — first-load-dominated, near 180s timeout | — | 2026-07-24 (Tier 2, stale/not a clean number) | Swarm (legacy) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| Moonshine-streaming | STT | 4090 | **13.7×** (0.81s/11.1s) | none | — | 2026-07-25 | Swarm (Tier 3) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| Moonshine | STT | 3060 / 4090 | **6.5× / 6.5×** | self-comparison (word-perfect on real speech) | — | 2026-07-12/13 | Swarm | `audio_tts_stt_2026-07-12.md` |
| Whisper (base) | STT | 3060 / 4090 | **~10× / 5.4×** — 3060 number is fresher (07-18) than the 4090 one (07-12); not re-measured together, flagged not directly conflicting but not confirmed either | self-comparison | — | 2026-07-18 (3060) / 2026-07-12 (4090) | Swarm | `stt_profiling_2026-07-18.md`, `audio_tts_stt_2026-07-12.md` |
| Whisper Streaming | STT | 4090 | **10.2×** (1.08s/11.0s) | none | — | 2026-07-25 | Swarm (Tier 3) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| Distil-Whisper | STT | 3060 / 4090 | **3.28× / 2.79×** | self-comparison | — | 2026-07-18 | Swarm | `stt_profiling_2026-07-18.md` (supersedes the 2.27×-equivalent same-day number in `stt_gaps_2026-07-18.md`) |
| Kyutai STT (1B) | STT | 3060 / 4090 | **6.6× / 10.1×** | self before→after (1.24×/1.37× → 6.6×/10.1×) | 5.3×/7.4× (self) | 2026-07-18 | engine-only | `kyutai_stt_perf_2026-07-18.md` |
| Stable Audio Open Small | Music | 4090 | **2.18×** (4.58s/10.0s) | none | — | 2026-07-25 | Swarm (Tier 3) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| MusicGen | Music | 4090 | **0.70×** (28.4s/20.0s, warm, seed=42) | self-comparison (cold 273s dominated by load, not steady-state) | — | 2026-07-26 | Swarm | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| AudioGen | SFX | 4090 | **0.64×** (46.7s/30.0s produced) | self-comparison | — | 2026-07-26 | Swarm | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) — duration-cap bug: 45s requested reproducibly yields only 30.0s |
| ACE-Step turbo | Music | 4090 | **6.45×** (3.1s/20.0s) | none | — | 2026-07-26 | Swarm | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) — vocals confirmed unintelligible (by design, 8-step no-CFG distillation) |
| ACE-Step sft | Music | 4090 | **4.44×** (4.5s/20.0s) | none | — | 2026-07-26 | Swarm | same — vocals confirmed intelligible |
| ACE-Step xl-turbo | Music | 4090 | **1.49×** (13.4s/20.0s) | none | — | 2026-07-26 | Swarm | same |
| ACE-Step xl-sft | Music | 4090 | **1.20×** (16.6s/20.0s) | none | — | 2026-07-26 | Swarm | same |
| ACE-Step xl-base | Music | 4090 | **1.47×** (13.6s/20.0s) | none | — | 2026-07-26 | Swarm | same — was a hard timeout pre-fix (case-sensitive dir bug) |
| YuE | Music | 4090 | **0.108×** (92.38s/~10s) | none | — | 2026-07-25 | Swarm (Tier 3) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) — real sung vocals confirmed (Issue #E fixed) |
| HeartMuLa (3b-base) | Music | 4090 | **0.059×** full Swarm e2e (169.98s/~10s, includes load) | see AR-decode table below for the clean steady-state number | — | 2026-07-25 | Swarm (Tier 3, incl. cold-load) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| OpenVoice V2 | VC | 4090 | **2.75×** (3.99s/~11s) | none | — | 2026-07-25 | Swarm (legacy `ProcessAudio`) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| Demucs (stem separation) | Fx | 4090 | n/a (225.3s wall, not a duration ratio) | none | — | 2026-07-25 | Swarm (legacy, CPU-forced backend) | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| RVC v2 | VC | — | n/a — no trained voice checkpoint exists on this box (not a bug) | — | — | — | — | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |
| Resemble-Enhance | Fx | — | n/a — blocked, weight file 404s / architecture mismatch (Issue #G, open) | — | — | — | — | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) |

---

## Autoregressive decode models (ms/frame — a different shape, don't compare to the RTF table above)

HeartMuLa and Zonos decode codec frames one at a time; the meaningful unit is **milliseconds per frame**
(lower is better), not a clip-level RTF. Mirrors the two tables already in
[`README.md`](README.md).

| Model | GPU | Config | ms/frame | Ratio vs baseline | Date | Source |
|---|---|---|---:|---:|---|---|
| HeartMuLa-oss-3B | RTX 3060 | bf16 eager (baseline) | 91.5 | 1.0× | — | `PERFORMANCE.md` (retired) |
| HeartMuLa-oss-3B | RTX 3060 | + CUDA-graph decode (default on) | ~85–90 | ~1.05× | — | `PERFORMANCE.md` (retired) |
| HeartMuLa-oss-3B | RTX 3060 | Q8_0 disk-quant | **64.8** | **1.41×** | — | `PERFORMANCE.md` (retired) |
| HeartMuLa-oss-3B | RTX 4090 | dual-stream B=2 graph decode (round 3, marginal) | **~17.6** (0.220 s/audio-s) | **5.2×** vs Python `heartlib` (1.14 s/audio-s marginal); d=60 e2e 34.1s vs 82.2s = 2.4× | 2026-07-25 | [`AUDIO_THROUGHPUT_BENCHMARK.md`](../../docs/Checklists/ROADMAP.md) round-3 entry (freshest — supersedes both `heartmula_music_e2e_2026-07-11.md` and `PERFORMANCE.md` (retired) own 3060 table on this exact number; different GPU, not a direct conflict) |
| Zonos-v0.1 (transformer) | RTX 4090 | host-glue decode (baseline) | 203 | 1.0× | 2026-07-17 | `zonos_tts_2026-07-17.md`, `PERFORMANCE.md` (retired) |
| Zonos-v0.1 (transformer) | RTX 4090 | GPU-resident decode (`FixedKvCache` + GQA FlashAttention) | **32** | **~6.3×**; still ~2.9× *slower* than real-time (real-time = 11.6 ms/frame) | 2026-07-17 | same |

---

## Notes / caveats

- **RTF-direction conflicts across sources are real, not a sign of a wrong number** — see the convention note above.
  Every converted cell in the main table carries its raw wall/audio pair so the direction can be checked.
- **Kyutai STT and Kyutai TTS both lose to their Python baseline** (moshi 2.25× vs our 1.09–1.47× for TTS; no
  Python STT baseline was measured) despite large *self*-speedups — don't read "self-improved 5×" as "beats the
  reference."
- **Tier-3 Swarm-sweep numbers (`AUDIO_THROUGHPUT_BENCHMARK.md`, 2026-07-25) conflate model-load time with steady-state
  generation** — the file says so explicitly ("time to a working generation today, not a clean steady-state RTF").
  Used here only for models with no dedicated perf-pass file; treat these as pessimistic lower bounds, not clean RTF.
- **Engine-only fixes not yet in Swarm**: Qwen3-TTS, Kyutai TTS, Kyutai STT perf passes are explicitly unshipped to
  the extension as of their write-up date — this is *why* the Tier-3 sweep still shows old, slower numbers for
  Kyutai STT (~0.82×) despite a 6.6–10.1× engine-level fix landing the same week.
- **NeuTTS Air and GPT-SoVITS v2** were unregistered/unselectable in the Tier-3 sweep (Issue #H) but that
  registration bug was reported fixed 2026-07-25/26 — no fresh speed number has been taken since the fix, so the
  rows above still cite the last real measurement (standalone harness for NeuTTS, stale Tier-2 for GPT-SoVITS).
- **F5-TTS vs. Python**: `CAMPAIGN_SCORECARD.md` cites a Python reference RTF of 0.729 (its own wall÷audio
  convention). No ratio is computed against our number here — different harness, different GPU, and our own
  post-fix number wasn't captured in that same table, so a computed ratio would be fabricated.
- **HeartMuLa's two AR-decode rows are different GPUs** (3060 vs 4090) at different points in a multi-round perf
  campaign — not a contradiction, but not directly comparable either.
- Two models are separation/enhancement tools, not fixed-duration generators (**Demucs**, and a source-audio-in
  case for **Resemble-Enhance**) — RTF doesn't apply; wall time only.
- Every number in this file is a **warm** number unless stated otherwise; cold/first-load numbers are cited only
  where a source made the warm number unavailable.
