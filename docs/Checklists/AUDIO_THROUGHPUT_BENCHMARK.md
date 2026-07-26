# Audio Throughput Benchmark — Design & Tracking

Status legend: ⬜ not started · 🔧 in progress · ✅ done · ⚠ blocked

> ## 🏁 2026-07-26 perf pass: Issue #I root cause + fix, shared-SDPA batching, ACE-Step follow-up
> **Issue #I was never a hang.** Direct instrumentation (a log line immediately before/after every
> `cublasGemmEx` call inside `CudaBackend.ScaledDotProductAttention`) proved every individual GEMM call
> returned in 0.0ms — the AR decode loop was making continuous forward progress the whole time. The real
> cost: Dia's cross+self attention issues one `cublasGemmEx` launch **per head, per CFG stream, per decoder
> layer** — `16 heads × 2 CFG × 18 layers × 2 (QK^T + attn·V) = 1152` individual driver calls **per decode
> step**, and at `Sq=1` (single-token decode) each is a GEMV dressed as a GEMM — launch-overhead-bound, not
> compute-bound. A request that used to read as "hung" (client timeouts at 130s/480s with zero progress
> visible) was actually ~900-1200+ decode steps of accumulated per-call launch overhead.
>
> **Fix (landed, shared code path — affects every model that calls `ScaledDotProductAttention`'s materialized
> path, not just Dia):** `CudaBackend.cs`'s QK^T and attn·V loops now issue ONE `cublasGemmStridedBatchedEx`
> call (`batchCount = totalHeads`) instead of `totalHeads` sequential `cublasGemmEx` calls — mathematically
> identical (each batch element is independent, offset by the existing per-head stride), ~32× fewer launches
> per attention call. Verified against the CPU-vs-GPU SDPA parity suite (`SdxlGenerationTests`, 12/12 pass)
> and the broader CUDA suite (198/201 pass; the 3 failures are pre-existing/unrelated — 2 are FP8 cuBLASLt
> heuristic gaps on Ampere hardware the test class's own doc comment says to expect, 1 is a 0.001-tolerance
> rounding difference in an unrelated multi-GPU isolation test, plausible from cuBLAS choosing a different
> internal algorithm for batched vs. sequential calls).
>
> **Result — Dia (fixed seed=42, same prompt, cold-then-warm):** 66.2s → 44.3s wall for a 6.3s clip (RTF
> 10.5× → 7.0×). A major reduction from the pre-fix state (350-800+ seconds, or simply never finishing
> within an 8-minute test window) but **still far from realtime** — Dia has no CUDA-graph decode path
> (unlike MusicGen/HeartMuLa) and the per-step compute itself (not just launch overhead) is substantial at
> 18 layers × 2 CFG streams. A full fix to realtime-class speed needs the same scale of dedicated perf work
> as the HeartMuLa rounds (graph-captured decode, CFG batching) — out of scope for this pass. Also
> confirmed and NOT fixed: `DiaPipeline.Generate` takes no `CancellationToken` anywhere in its call chain, so
> a client-side timeout does not stop server-side generation — the request keeps running (and holding
> `AudioRuntime`'s shared single-slot `_genLock`) until it naturally completes.
>
> **Result — MusicGen/AudioGen (same batched-SDPA path, fixed seed=42):** MusicGen's cold call (273s for
> 10s audio, RTF 27×) vs warm (28.4s for 20s audio, RTF 1.42×) shows the huge first-call cost is model
> load/compile, not steady-state generation — steady-state MusicGen is a bit above realtime. AudioGen warm:
> 46.7s wall but only **30.0s of audio produced for a 45s request** — a real, reproducible duration cap
> (RTF 1.56× against what it actually produced), not a hang; this reproduces the discrepancy first noticed
> 2026-07-25 and is a separate, not-yet-root-caused bug (not Issue #J — see below).
>
> **Issue #J reclassified: it was never an independent AudioGen bug.** Re-tested AudioGen 45s in complete
> process isolation (fresh restart, first and only request) — completed cleanly in 95.6s, HTTP 200, real
> audio, no hang. The original "hard hang, needed SIGKILL twice" reports are fully explained by Issue #I:
> `AudioRuntime._genLock` is a global `SemaphoreSlim(1,1)` serializing ALL audio generation process-wide: a
> stuck-looking (in fact just very slow) Dia request from earlier in the same session held that lock for its
> entire multi-minute run, and any request issued afterward — AudioGen included — simply queued behind it
> and eventually looked identically "hung" from the outside. One root cause, not two.
>
> **ACE-Step: both open items closed.** `xl-base`'s "timeout" was the same case-sensitivity directory bug
> that hit Demucs (`AudioWeights.WeightsDirectory()` built paths from `ModelPrefix` — e.g. `AceStep` — but
> the real weight file sat in a lowercase `acestep/` dir; fixed with a case-insensitive fallback lookup,
> see Issue #H) — now completes in 13.6-16.4s. Turbo vocal intelligibility: **confirmed real via a
> controlled, same-seed/same-lyrics A/B**, not a fluke — `sft`'s Whisper round-trip recovers clearly
> recognizable lyrics ("Walking down the city street... tonight... burnin'... so bright... music playing
> all the time"); `turbo`'s returns only `"(upbeat music)"`, no words, at the same seed/prompt/duration.
> This tracks with turbo's own documented design (8 steps, no CFG) vs. sft's (50 steps, CFG≈7) — an
> inherent speed/quality tradeoff of the distilled checkpoint, not a code bug. Nothing to fix; if
> vocal clarity matters more than speed, use `sft`/`base`/`xl-sft`/`xl-base` instead of a turbo variant, or
> try `turbo-shift1`/`turbo-shift3` as an untested middle ground.
>
> | Model | Wall (cold) | Wall (warm) | Produced | RTF (warm) | Notes |
> |---|---:|---:|---:|---:|---|
> | Dia 1.6B | 66.2s | 44.3s | 6.3s | 7.0× | seed=42; was 350-800s+/non-terminating pre-fix |
> | MusicGen medium | 273.4s | 28.4s (d=20) | 20.0s | 1.42× | cold cost = load/compile, not steady-state |
> | AudioGen medium | 30.6s (d=10) | 46.7s (d=45 req) | 30.0s | 1.56× | duration cap bug, separate from Issue #J |
> | ACE-Step turbo | — | 3.1s | 20.0s | 0.15× | vocals confirmed unintelligible (by design) |
> | ACE-Step sft | — | 4.5s | 20.0s | 0.23× | vocals confirmed intelligible |
> | ACE-Step xl-turbo | — | 13.4s | 20.0s | 0.67× | |
> | ACE-Step xl-sft | — | 16.6s | 20.0s | 0.83× | |
> | ACE-Step xl-base | — | 13.6s | 20.0s | 0.68× | was: timeout/never completed, pre-fix |
>
> Full raw output: [`benchmarks/swarm_audio_bench/sdpa_batch_perf_results.json`](../../benchmarks/swarm_audio_bench/sdpa_batch_perf_results.json);
> harness: [`sdpa_batch_perf_pass.py`](../../benchmarks/swarm_audio_bench/sdpa_batch_perf_pass.py).

> ## 🏁 HeartMuLa vs Python head-to-head (2026-07-25, RTX 4090, solo, duration-verified outputs)
> **Ours is 3.6× faster than upstream `heartlib` per audio-second** (marginal: 0.31 vs 1.14 s per
> audio-second; d=60 end-to-end: 43.7 s vs 82.2 s = 1.9× incl. load). Same weights (HeartMuLa-oss-3B bf16 +
> HeartCodec F32), same defaults (cfg 1.5, topk 50). Method: d=10/d=60 pairs, wall-clock slope; heartlib
> venv at `scratchpad/heartlib-venv` (torchaudio save patched to stdlib WAV — no ffmpeg on box; needs
> `PYTORCH_ALLOC_CONF=expandable_segments:True` or it OOMs 24 GB at d=60 in its own whole-song codec decode).
> On the 4090 our bf16 beats our q8 marginally (0.314 vs 0.375 s/audio-s — q8's GPU time is LOWER, so the
> q8 gap is host-side; on the bandwidth-starved 3060 q8 still wins). Why we win: CUDA-graph AR decode
> (heartlib is eager PyTorch).
>
> **Perf-pass round 1 landed:** HeartCodec estimator attention routed from the LM FlashAttention API to
> `ScaledDotProductAttention` (Sage/cuDNN dispatch, D=64/128) — the 300 monolithic full-sequence calls
> (~5.7 ms each, 1.71 s GPU per 10 s song = the codec's largest cost) collapsed to **28 ms** of cuDNN fused
> kernels (~60×); same-length e2e wall −8% (28.0→25.8 s at d=10). Output-equivalence: kill-switch A/B
> (HARTSY_SDPA_CUDNN=0 HARTSY_SAGE_ATTN=0) produces the IDENTICAL sampled song, so the swap changes speed
> only. (A d=60 EOS shift — 60.0→43.2 s of audio at seed 7 — predates the swap and is invariant to it:
> tolerance-class sampling sensitivity from earlier allocation-order changes, quality unaffected.)
>
> **Perf-pass round 2 (2026-07-25, all landed):**
> - **Codec convs → cuDNN (~11×)**: all 54 direct `conv1d_f32` calls were the CAUSAL (asymmetric-pad)
>   vocoder convs — cuDNN's graph API takes pre/post pads separately, so the symmetric-pad gate was
>   unnecessary; transposed convs routed via convolution-backward-data. Conv path 1.21 s → 104 ms per 10 s
>   song. Per-plan resident workspaces caused a device OOM at d=60 — now per-execution pool workspaces +
>   512 MB engine cap. Kill-switch HARTSY_AUDIO_CONV_CUDNN=0.
> - **Chunked ScalarModel vocoder decode**: fixed 256-latent-frame chunks (64L/2R context, uniform windows
>   share one plan set) — BITWISE identical to monolithic, host peak bounded (d=60 peak == d=10 peak; the
>   flow-matching 29.76 s windowing already matched upstream). HARTSY_HEARTCODEC_SCALAR_CHUNK=0 reverts.
>   The remaining decode-phase +5.7 GB host step is O(model) (codec F32 weight materialization), NOT
>   O(song) — the memory story is closed.
> - **CFG cond+uncond batched backbone step** (CsmModel.StepFrame → ForwardBatchDecode, B=2, streams are
>   position-aligned by construction) + **multi-M row-reuse in the bf16/f16 GEMV kernels** (M∈[2,4]: one
>   warp dots each weight row against all M activations — the old grid.y=M layout re-streamed the matrix
>   per batch row, costing 1.27× instead of ~1.0×; M=1 path untouched after an early over-eager version
>   regressed it). Kill-switch HARTSY_CSM_CFG_BATCH=0. Marginal decode 0.39 → 0.31 s/audio-s; costs ~2.2 s
>   fixed at d=10 (graph replay traded for eager launches — crossover ≈20 s of audio, so default ON).
> - **Final clean matrix (4090, seed 7, all changes)**: d=10 26.7 s, d=60 42.2 s (full 60.0 s audio) vs
>   python heartlib 25.4/82.2 s → **1.95× faster end-to-end at 60 s; 3.7× marginal (0.31 vs 1.14)**.
>
> **Perf-pass round 3 (2026-07-25, landed): dual-stream B=2 graph decode — marginal 0.220 s/audio-s.**
> New `GenericTransformer.ForwardGraphDecodeStepDualEmbeds` + `Layer.ForwardGraphStepDual` (copy-adapted
> from the B=1 graph step so the TEXT fleet's path is untouched — only new methods added): batched parts
> (norms, fused-QKV/o/FFN as M=2 GEMVs via the multi-M kernels) stream each weight ONCE per frame for
> both CFG rows; rope/KV-scatter/attention run per row against each stream's own cache off ONE shared
> devicePos (the streams are position-aligned by construction, checked at runtime with an eager
> fallback). CSM side: dual GraphStream ([1,2,bh] fixed buffers), warmup-then-capture, one replay per
> frame. Kill-switches: HARTSY_CSM_CFG_GRAPH=0 → eager batched; HARTSY_CSM_CFG_BATCH=0 → two-stream
> graphed; HARTSY_CSM_GRAPH=0 → fully eager. VERIFIED: **bit-identical** to eager-batched at d=10 (max
> diff 0 over 480k samples) AND to the shipped two-stream graph over all 540 frames at d=60; suites
> 201/201 CUDA + 132/132 LLM; nsys shows per-frame graph replay (~13 cuGraphLaunch/frame incl. depth).
> Three-arm marginal (4090): **graph-on 0.220** / eager-batched 0.354 / two-stream-graphed 0.274.
> **vs Python heartlib: 5.2× faster marginal (0.220 vs 1.14); d=60 end-to-end 34.1 s vs 82.2 s = 2.4×.**
> Caveat: the dual step's composed-QKV and per-head-QK-norm sub-paths are written but exercised by no
> current model (HeartMuLa's Llama layers take fused-QKV); gated by SupportsDualGraphDecode.

> **KEY REFRAME for round 3 (context for the numbers above): the GPU is now mostly idle during decode.** With attention 60× and convs 11×
> down, GPU-busy is a small fraction of the ~15.5 s marginal at d=60 — HOST orchestration (eager per-op
> launches + tensor bookkeeping in the batched path, ~17 ms/frame vs the graphed path) is the dominant
> cost. Round-3 levers, in order: (1) graph-capture the BATCHED B=2 step (recover replay + keep the
> traffic halving — needs a batched GraphStream with fixed [1,2,bh] buffers); (2) host-side op overhead
> (the LLM grind's launch-count discipline applies); (3) q8 host anomaly (same class).
>
> **Suite status (2026-07-25)**: audio suite 335 passed / 1 skipped / 5 FAILED — all 5 pre-existing
> (EnglishG2P ×2, AudioTextFrontend BPE ×3; files untouched by any of this work, failing at HEAD) — plus
> the F5 generation tests (TtsBenchTests.Bench_F5, F5CorrectnessTests.GenMatchedInput_SttAndDumpMel)
> exceed a 10-minute blame-hang timeout and abort the run — they need a Slow/Bench trait quarantine, and
> the G2P/BPE failures need their own investigation (likely environment/data drift).

> **Scoped next levers (nsys-attributed, not started):** (1) CFG cond+uncond batching through the backbone
> as M=2 GEMVs — halves the LM's weight traffic (bf16 GEMV measured AT roofline: the whole 3B streams per
> stream-frame ≈ 11.4 ms; batching is the only way down). (2) Codec vocoder convs: 54 `conv1d_f32` calls
> ≈ 0.95 s per 10 s song at 48 kHz sample-domain shapes; the cuDNN conv path skips them (causal/asymmetric
> pads and grouped shapes) — pre-pad + symmetric-conv would make them eligible. (3) Chunked codec decode —
> bounds the +64 MB-per-audio-second host spike (the old 49 GB OOM driver; upstream has the same flaw and
> OOMs 24 GB VRAM at d=60) and shrinks codec latency. (4) q8 host-side overhead (GPU is faster than bf16,
> wall is slower). (5) Cosmetic: 4× "OutHidden dispose failed" warnings per run = benign double-free in
> GraphStream teardown, root-cause pending.

**Goal.** Same spirit as [`LLM_THROUGHPUT_BENCHMARK.md`](LLM_THROUGHPUT_BENCHMARK.md), for the audio fleet: run
every catalog model through the real Swarm API, verify it still generates correctly today, and report speed
against a Python reference wherever one honestly exists.

**Why this doc looks different from the LLM one.** LLMs share one universal Python reference
(`llama-cpp-python`) that runs every GGUF — a clean, mechanical head-to-head. Audio has no such thing: TTS,
STT, music, voice-conversion, and Fx each have 30+ distinct architectures, most with their own bespoke
upstream Python implementation (or none published at all). Building a fresh timed Python harness for all of
them in one pass would take many hours and fail on a large fraction the same way 5+ LLM models failed their
`llama-cpp-python` load (missing packages, architecture drift, GPU-only weights needing bespoke loader code)
— confirmed by trying it on the LLM fleet first. **User-approved scope for this pass:** verify every model
fresh through Swarm's real API (the part that finds real bugs and can't be faked), and report the engine-side
RTF/timing already measured in prior dedicated perf-grind sessions (documented in
[`MODEL_STATUS_AUDIO.md`](MODEL_STATUS_AUDIO.md) and `benchmarks/results/*.md`) rather than re-deriving all
of them from scratch. Python comparisons are included **only** where a real, timed number already exists —
labeled with its source and date — not invented to fill the column.

**Hardware.** RTX 4090 (Swarm's AudioLab backend — confirmed via `nvidia-smi -i 1 --query-compute-apps`,
same box as the LLM campaign) + RTX 3060 for prior perf-grind sessions (mixed, noted per row where relevant).

---

## Methodology

**Tier 3 (canonical-path, 2026-07-25) — supersedes Tier 2 below.** The 2026-07-24 Tier 2 pass used
`ProcessTTS`/`ProcessSTT`/`ProcessAudio` with a raw args dict. That path **bypasses `BuildEngineArgs`**
(`DynamicAudioBackend.cs`) — the layer that maps a model's real params (e.g. ACE-Step/YuE/HeartMuLa's
`genre` = style vs. a *separate* `lyrics` field) onto engine kwargs. Root-caused this pass: ACE-Step's
"mostly silence, no vocals" complaint was exactly this — a style sentence sent as `prompt` landed in the
engine's *lyrics* slot with `genre` empty, defaulting toward an instrumental/near-silent result.

New harness, [`swarm_audio_bench_v2.py`](../../benchmarks/swarm_audio_bench/swarm_audio_bench_v2.py), drives
every local (non-cloud) provider through the same path a real generation hits — `POST
/API/GenerateText2Image` with `model="Audio Models/<Engine>/<variant>"` — so Swarm resolves params through
the real per-model logic and **auto-saves the output to `/Output`** itself (no manual base64-decode-and-write
on our side, matching the image/video gen convention):
- **TTS** (21 models) → `GenerateText2Image`, `prompt` = test sentence, `referenceaudio`/`referencetext` (JFK
  clip) on every call so clone-capable models exercise their real path.
- **STT** (6 models) → stays on `POST /API/ProcessSTT` (no output *file* to auto-save — text out, not audio).
- **Music/SFX** (6 models) → `GenerateText2Image`, `textaudioduration=10`. ACE-Step/YuE/HeartMuLa get `prompt`
  = style/genre tags **and** a separate `lyrics`/`yuelyrics`/`heartliblyrics` field with real lyric text —
  the fix for the silence/no-vocals bug. AudioGen gets an SFX-appropriate prompt, not a music-style one (an
  AudioCraft SFX model, not a vocal/music model — an earlier ad-hoc sample-generation pass wrongly reused the
  same music prompt for it).
- **Voice Conversion** (2) / **Fx** (2) → `GenerateText2Image` with `sourceaudio`/`targetvoice`/`fxinput` as
  `data:audio/wav;base64,...` — **found broken for all 4** (see Issue #H); fell back to legacy `ProcessAudio`
  for these specifically, documented as an exception, not silent.

Every returned WAV is fetched and content-quality-gated, not just HTTP-200-gated: RMS/peak checked for
near-silence or full-scale noise-clipping, and TTS/music outputs are round-tripped through `whisper_stt` to
confirm actual transcribable content came out (the 2026-07-24 pass only checked "non-zero bytes," which is
exactly how the ACE-Step/AudioGen quality bugs shipped unnoticed).

Each model call is independently try/excepted (a crash on one must never abort the rest — the LLM campaign's
first Tier-1 pass aborted the whole batch on one bad model and had to be re-run four times; this harness
avoids that by construction, same pattern as the LLM Swarm-path script). Wall time includes any first-load
weight-decode cost (no separate warmup call), so treat it as "time to a working generation today," not a
clean steady-state RTF — clean RTFs are the prior-session numbers in the "Documented engine RTF" column.

**Documented engine RTF column** — pulled from `MODEL_STATUS_AUDIO.md` and the per-model
`benchmarks/results/*_2026-07-*.md` write-ups from dedicated perf-grind sessions (each already reports a
clean warm RTF, not conflated with model-load time). Cited inline per row.

**Python reference column** — included only where a real timed number exists from a prior session
(`tests/python-reference/CAMPAIGN_SCORECARD.md`, or a `benchmarks/results/*.md` write-up that ran the actual
upstream Python package). Cells marked "—" mean no timed Python baseline has been captured for that model —
this is an honest gap, not a claimed win or loss.

---

## Known hazards applied from the LLM campaign

- **VRAM accumulates across sequential different-model loads** on this box (reproduced in the LLM sweep as a
  documented regression, and independently flagged for audio in prior memory —
  `audiolab-vram-eviction-threshold`: AudioLab's automatic eviction only fires below a ~3GB-free threshold,
  too conservative for back-to-back 5GB+ models). AudioLab does NOT expose a per-model unload API call (only
  `AudioLabRemoveAllModels`, which deletes weights from disk — too destructive for a benchmark loop); the only
  lever is the engine's own pressure-triggered eviction. Confirmed during this pass: VRAM climbed to ~21 GB
  after 7 sequential TTS models, then the pressure-triggered eviction fired and dropped it back to ~11 GB
  before the 8th — so the mechanism works, just late. If a large model OOMs deep into the sweep, that is this
  known issue recurring, not a new bug.
- **Per-model isolation.** Every provider call is wrapped individually (see harness code) — one bad model
  can't take down the batch.
- **Concurrent dev-mode rebuilds restart the shared Swarm instance out from under a running sweep.** This
  pass hit two mid-sweep restarts (another agent's `--environment dev` session hot-reloading on file edits)
  and one accidental **second SwarmUI process** from a naive restart attempt (`launch-linux.sh` binds the
  next free port instead of replacing an existing instance — it does **not** kill-then-relaunch). A second
  process against the same `Data/` LiteDB is a real corruption risk (`dual-swarm-litedb-corruption` in prior
  memory); the duplicate was killed within seconds with no observed corruption, but the correct restart
  procedure is: explicitly `kill <pid>` the existing process **first**, confirm it's gone, *then* relaunch.
- **AudioGen hangs, not just slows, well before its full generation range.** 10s generates fine (37.6s wall);
  20s generates but scales worse than linearly (51.8s wall, vs. a ~75s linear extrapolation from 10s — so
  still within range, if degrading); **45s hard-hung the entire Swarm process** (pegged at 100% CPU / ~5%
  GPU util indefinitely, unresponsive to `SIGTERM`, required `SIGKILL` — twice, across two separate attempts).
  This is a genuine, reproducible defect, independent of the params fix below — flagged as **Issue #J**.
- **"Installed" ≠ "unusable."** `AudioLabListEngines` reports several providers as `installed: false` despite
  their weights already being present in `~/.cache/hartsyinference/` (e.g. `csm_tts`, `neutts_tts`, several
  STT ids) — this flag tracks the extension's own registry step, not weight presence; generation was
  attempted directly regardless of this flag.

---

## Results — Tier 3 canonical-path sweep (2026-07-25)

**31/37 generated successfully today**, up from 27/37 on 2026-07-24 (several 07-24 failures were the user's
own perf-pass fixes landing: Chatterbox, Zonos, YuE, Distil-Whisper, and HeartMuLa/Demucs — see below — all
now generate real audio). Raw data: `benchmarks/swarm_audio_bench/swarm_audio_results_final.json`. Ran on the
RTX 4090 (confirmed via `nvidia-smi -i 1 --query-compute-apps`), on a SwarmUI dev instance shared with another
agent's concurrent work (their CSS hot-reloads triggered two mid-sweep restarts — see hazards above).

TTS transcripts below are near word-perfect via the `whisper_stt` round-trip built into the new harness (not
a separate manual check) — a real, automated confirmation each model said the right words, not just that it
returned bytes.

### Text-to-Speech (21 models) — `POST /API/GenerateText2Image`

`prompt="Hello, this is a test of the text to speech system."`, `referenceaudio`/`referencetext` = an 11s
JFK clip on every call.

| Model | Wall time | RTF | STT round-trip | Result |
|---|---:|---:|---|---|
| Piper | 1.44s | **0.5×** | "Hello this is a Test of the Text to Speech System." | ✅ |
| Qwen3-TTS (1.7B-Base) | 1.44s | **0.367×** | — | ✅ |
| Kokoro-82M | 3.07s | **0.83×** | "Hello, this is a test of the Text2 Speech System." | ✅ |
| PocketTTS | 3.51s | **1.22×** | "Hello, this is a test of the text to speech system." | ✅ |
| StyleTTS2 | 4.34s | **0.808×** | "Hello, this is a test of the Tag Hut and Speech system." | ✅ |
| MeloTTS | 5.34s | **1.784×** | "Hello, this is a test of the text to speech system." | ✅ |
| Chatterbox | 16.22s | 2.55× | "Hello, this is a test of the text to speech system." | ✅ — **Issue #A fixed** |
| Kyutai TTS 1.6B | 15.42s | 4.943× | "Hello, this is a test of the text to speech system." | ✅ |
| VibeVoice-1.5B | 18.71s | 0.872× | — | ✅ |
| CosyVoice 2 | 19.1s | 3.161× | "Hello! This is a test of the text to speech system." | ✅ |
| Spark-TTS-0.5B | 22.3s | 5.994× | "Hello, this is a test of the text to speech system." | ✅ |
| Fish-Speech 1.5 | 22.87s | 3.468× | "Hello! This is a test of the text to speech system." | ✅ |
| Zonos-v0.1 (transformer) | 24.29s | 7.314× | "Hello, this is a test of the text to speech system." | ✅ — **Issue #B fixed** |
| F5-TTS | 16.42s | 3.275× | "Hello, this is a test of the text to speech system." | ✅ |
| Orpheus | 11.4s | 3.18× | — | ✅ |
| Bark | 21.01s | 1.693× | "Hello, this is a test of the text-to-speech system." | ✅ |
| ZipVoice | 99.1s | 18.324× | "Hello, this is a test of the text to speech system." | ✅ |
| Dia-1.6B | — | n/a | — | ❌ **Issue #I** — hangs regardless of path |
| CSM-1B (Sesame) | — | n/a | — | ⬜ weights not installed, expected |
| NeuTTS Air | 0.03s | n/a | — | ❌ **Issue #H** — installed but unregistered |
| GPT-SoVITS v2 | 0.01s | n/a | — | ❌ **Issue #H** — installed but unregistered |

### Speech-to-Text (6 models) — `POST /API/ProcessSTT`

11s JFK clip, `language=en`. No change of path from Tier 2 (no output *file* to auto-save). All 6 now work —
Distil-Whisper's Issue #C is fixed.

| Model | Wall time | RTF | Result |
|---|---:|---:|---|
| Moonshine-streaming | 0.81s | **0.073×** | ✅ |
| Whisper | 1.07s | **0.097×** | ✅ |
| Whisper Streaming | 1.08s | **0.098×** | ✅ |
| Moonshine | 1.23s | **0.112×** | ✅ |
| Distil-Whisper | 6.21s | 0.564× | ✅ — **Issue #C fixed** |
| Kyutai STT | 13.35s | 1.213× | ✅ |

### Music & SFX (6 models) — `POST /API/GenerateText2Image`

`textaudioduration=10`. ACE-Step/YuE/HeartMuLa: `prompt` = genre/style tags, separate lyrics field = real
lyric text (the actual fix for "mostly silence, no vocals"). AudioGen: SFX-appropriate prompt, not a music
style. STT round-trip is the closest automatable check for "has real vocals," not a certainty — see notes.

| Model | Wall time | RTF | STT round-trip | Result |
|---|---:|---:|---|---|
| Stable Audio Open Small | 4.58s | **0.458×** | "(dramatic music)" — instrumental, expected | ✅ |
| MusicGen | 8.62s | **0.862×** | "(upbeat music)" — instrumental, expected | ✅ |
| ACE-Step (turbo) | 10.48s | 1.048× | "(upbeat music)" | ✅ silence FIXED; vocal intelligibility unconfirmed (see below) |
| AudioGen | 37.56s | 3.756× | "(upbeat music)" | ✅ static FIXED with correct SFX prompt; **Issue #J** duration scaling |
| YuE | 92.38s | 9.238× | "Hello, this is the test of the haze of hula kula..." | ✅ **Issue #E fixed** — real words transcribed, confirms actual sung vocals |
| HeartMuLa (3b-base) | 169.98s | 16.998× | "♪ ♪ ♪ ♪ ♪ ..." (legacy path) | ✅ **Issue #D fixed**; ❌ **Issue #H** — installed but unregistered |

**ACE-Step/HeartMuLa vocal honesty note:** the silence bug (empty `genre`, style text misrouted into the
lyrics slot) is conclusively fixed — peak amplitude went from near-zero to full-scale (29204/32767) once
`genre`/`lyrics` were split correctly. Whether the *singing itself* is intelligible is a separate, harder
question this pass can only partially answer: ACE-Step-turbo's STT round-trip returns a generic non-verbal
tag both before and after the fix (Whisper's usual response to buried/unclear vocals in a fast mix), while
YuE and HeartMuLa's round-trips returned actual attempted transcriptions (YuE: real, if garbled, words; 
HeartMuLa: Whisper's "♪" pattern, its typical response to sung-but-untranscribable melodic content) — both
positive signals of real vocal content that ACE-Step-turbo didn't produce. **Also tested `Audio
Models/AceStep/xl-sft`** (the "ACE-Step large" the user asked about — 9 variants total: turbo/turbo-shift1/
turbo-shift3/turbo-continuous/sft/base/xl-turbo/xl-sft/xl-base, all shown installed via `AudioLabListEngines`)
— generated successfully (20.4s wall for 10s audio); `xl-turbo` failed and `xl-base` timed out at 200s in a
single quick check, not yet root-caused. If un-tagged vocal clarity matters, the non-turbo/`sft`/`base`/`xl-*`
variants (more inference steps, presumably less distilled) are the next thing to try, not a params change.

### Voice Conversion & Fx (4 models)

All 4 were found **not registered as a selectable `Model` for `GenerateText2Image`** this pass (Issue #H) —
routed through the legacy `ProcessAudio` dispatch instead, documented as an exception, not a silent fallback.

| Model | Wall time | RTF | Result |
|---|---:|---:|---|
| OpenVoice V2 | 3.99s (legacy) | 0.363× | ✅ **Issue #H** — installed but unregistered |
| Demucs | 225.3s (legacy) | n/a | ✅ **Issue #F fixed** — real distinct stems returned; **Issue #H** |
| RVC v2 | 0.01s | n/a | ❌ weights genuinely missing (confirmed via legacy path too) — not Issue #H |
| Resemble-Enhance | timeout (legacy, 280s) | n/a | ❌ **Issue #G** persists, unconfirmed either way this pass |

---

## Results — Tier 2 Swarm-path sweep (2026-07-24) — superseded, kept for history

**27/37 generated successfully today.** Raw data: `benchmarks/swarm_audio_bench/swarm_audio_results.json`.
Ran on the RTX 4090 (Swarm's AudioLab backend, confirmed via `nvidia-smi -i 1 --query-compute-apps`) — this
ran concurrently-safe alongside that day's LLM Tier-1 work on the RTX 3060 with zero contention, same
discovery as the LLM campaign (Swarm always runs on GPU 1 on this box).

### Text-to-Speech (21 models) — `POST /API/ProcessTTS`

| Model | Wall time | RTF today | Documented engine RTF (prior session) | Result |
|---|---:|---:|---|---|
| Piper | 1.02s | **0.354×** | — | ✅ |
| Kokoro-82M | 2.41s | **0.651×** | — | ✅ |
| StyleTTS2 | 2.77s | **0.515×** | ~0.8× warm (2026-07-15, clone) | ✅ |
| PocketTTS | 2.92s | **0.986×** | — | ✅ |
| MeloTTS | 4.50s | **1.481×** | 1.4–1.8× (2026-07-12, no perf pass) | ✅ |
| VibeVoice-1.5B | 16.46s | 1.603× | **0.78× warm** (2026-07-17) | ✅ — today's number includes cold model load |
| Bark | 13.09s | 2.922× | **2.30× warm** (2026-07-18) | ✅ |
| Qwen3-TTS | 11.29s | 2.476× | **~0.50× warm** (2026-07-18) | ✅ — today's number includes cold model load |
| F5-TTS | 14.27s | 2.846× | RTF 0.729 *(Python, `CAMPAIGN_SCORECARD.md`)* | ✅ |
| CosyVoice 2 | 15.23s | 3.591× | **1.34× warm**, 5.1s/clip (2026-07-17) | ✅ — today's number includes cold model load |
| Fish-Speech 1.5 | 13.89s | 3.739× | RTF 1.95 (2026-07-18) | ✅ |
| NeuTTS Air | 47.07s | 4.561× | encoder ~RTF 1.5 (2026-07-18) | ✅ — clone path, heavier |
| Spark-TTS-0.5B | 19.79s | 5.856× | — | ✅ |
| GPT-SoVITS v2 | 39.54s | 6.865× | — | ✅ |
| Kyutai TTS 1.6B | 29.30s | 8.516× | **0.51× warm** vs moshi 2.25× (2026-07-16) | ✅ — today's number includes cold model load |
| ZipVoice | 113.68s | 21.0× | ~11 min/10s clip documented, no perf pass | ⚠ ok, known-slow perf target |
| CSM-1B (Sesame) | 173.72s | 41.0× | — | ⚠ ok, first-load-dominated (near 180s timeout) |
| Dia-1.6B | — | n/a | EOS-stops at 11.4s warm (2026-07-15) | ❌ TIMEOUT >180s |
| Orpheus | — | n/a | 6.5× speedup vs baseline (2026-07-14) | ❌ TIMEOUT >180s |
| Chatterbox | 3.13s | n/a | RTF 0.69–0.90× no-reference (2026-07-18) | ❌ **Issue #A** — clone path not wired |
| Zonos-v0.1 | 0.18s | n/a | ~32ms/frame decode (2026-07-17) | ❌ **Issue #B** — weights file missing |

### Speech-to-Text (6 models) — `POST /API/ProcessSTT`

11s JFK clip, `language=en`. All 5 working models transcribed correctly (word-perfect or near-exact).

| Model | Wall time | RTF | Transcript | Result |
|---|---:|---:|---|---|
| Moonshine-streaming | 0.85s | **0.077×** | exact | ✅ |
| Moonshine | 1.20s | **0.109×** | exact (minor punctuation) | ✅ |
| Whisper Streaming | 1.63s | **0.148×** | exact | ✅ |
| Whisper | 1.65s | **0.150×** | exact | ✅ |
| Kyutai STT | 13.33s | 1.212× | exact | ✅ |
| Distil-Whisper | 0.04s | n/a | — | ❌ **Issue #C** — bare id resolves to unsupported repo |

### Music & SFX (6 models) — `POST /API/ProcessAudio`

`prompt="An upbeat electronic dance track with synths and a steady beat"`, `duration=10s`.

| Model | Wall time | RTF | Result |
|---|---:|---:|---|
| Stable Audio Open Small | 4.10s | **0.410×** | ✅ |
| MusicGen | 7.26s | **0.726×** | ✅ |
| ACE-Step (turbo) | 9.42s | **0.942×** | ✅ |
| AudioGen | 34.30s | 3.430× | ✅ |
| HeartMuLa (oss-3B) | — | n/a | ❌ **Issue #D** — OOM-killed the entire Swarm process |
| YuE | 0.35s | n/a | ❌ **Issue #E** — checkpoint never downloaded |

### Voice Conversion & Fx (4 models) — `POST /API/ProcessAudio`

JFK clip as source/target — functional smoke test only, not a separation/enhancement quality eval.

| Model | Wall time | RTF | Result |
|---|---:|---:|---|
| OpenVoice V2 | 10.52s | **0.955×** | ✅ |
| RVC v2 | 0.05s | n/a | ⬜ no trained voice available — expected, not a bug |
| Demucs | — | n/a | ❌ **Issue #F** — Swarm path doesn't force CPU backend |
| Resemble-Enhance | — | n/a | ❌ **Issue #G** — weight file 404s from HuggingFace |

---

## Bugs found this session (numbered for cross-reference)

**Status update (2026-07-25): Issues #A, #B, #C, #D, #E, #F are FIXED** (confirmed generating real audio via
this pass's canonical-path sweep — see the Tier 3 tables above), consistent with the user's own 2026-07-25
perf-pass work. Left in place below for history/context. **New issues this pass: #H, #I, #J.**

### Issue #A — Chatterbox's reference-voice cloning isn't wired [FIXED 2026-07-25]
Clean, well-worded error (not a crash): "Chatterbox reference-voice cloning is not wired yet — it needs a
PCM→40-bin-mel front-end for the voice encoder." Falls back cleanly when no reference is supplied. A real
feature gap, not a defect — every other clone-capable TTS model in this sweep accepted the same reference
args without complaint.

### Issue #B — Zonos-v0.1's actual weights file is missing [FIXED 2026-07-25]
`Could not find file '.../Zyphra--Zonos-v0.1-transformer/model.safetensors'`. The cache directory exists (from
the 2026-07-17 verification session) but the real weights file inside it doesn't — `MODEL_STATUS_AUDIO.md`
documents Zonos as fully verified with real perf numbers from that date, so the file has since been deleted
or the download never fully completed. **Fix: re-download** (`AudioLabInstallEngine` for `zonos_tts`, or
delete the stale directory and let it re-fetch).

### Issue #C — Distil-Whisper's bare provider id resolves to an unsupported repo [FIXED 2026-07-25]
Matches a pre-existing, already-documented gap (`MODEL_STATUS_AUDIO.md`): the bare id defaults to
`distil-whisper/distil-large-v3.5`, which `WhisperPipeline.InferConfig`'s repo switch doesn't recognize (only
v2/v3/medium.en/small.en). A variant suffix (`distilwhisper:v3`) works around it. **Fix direction:**
`SttCatalog.ResolveDistilWhisperRepo`'s no-match default should point at a repo `WhisperPipeline.InferConfig`
actually supports.

### Issue #D — HeartMuLa generation OOM-killed the entire Swarm process [CRITICAL, FIXED 2026-07-25]
Confirmed via kernel log (`journalctl -k`):
```
.NET Tiered Com invoked oom-killer: ... oom_score_adj=200
Out of memory: Killed process 737038 (SwarmUI) total-vm:465219276kB, anon-rss:49056928kB, ...
```
This is **host RAM**, not VRAM — the process was using ~49 GB resident memory on a 62 GB-RAM box (with only
2 GB swap, already 95% full) when the kernel killed it. Took the entire Swarm server down mid-sweep,
aborting the last 4 planned models (VoiceConversion + Fx) until a manual restart. `HeartMuLa-oss-3B`'s
on-disk checkpoint is ~15 GB — needing 3×+ that in transient RAM during `PytorchPickleLoader` load is a real
red flag. **Where to look:** whether the loader holds more than one live copy of the full tensor set at once
during pickle deserialization + framework conversion (a classic "loaded state, converting to target dtype,
haven't freed the source" pattern). **Workaround for future benchmark runs:** skip `heartlib_music` unless
free RAM is confirmed >55 GB immediately beforehand, or test it in isolation (nothing else loaded) with a
RAM-watchdog script (the same pattern already used for Dia per `MODEL_STATUS_AUDIO.md`'s STT reality-check
section: "Heavy runs go through a RAM-watchdog script that hard-kills below 1.5 GB free").

### Issue #E — YuE's real checkpoint was never actually downloaded [FIXED 2026-07-25]
`YuE checkpoint folder not found: '.../Models/audio/music/yue/yue'`. Only an empty stub directory exists
despite a catalog `Assets` entry declaring the ~12.5 GB checkpoint. Low-risk fix: just run the download
(`hartsy music -m yue` with confirm, or `AudioLabInstallEngine` for `yue_music`).

### Issue #F — Demucs fails through Swarm's generic audio path [FIXED 2026-07-25]
`CUDA STFT not supported - use CPU backend for audio`. The CLI (`hartsy fx separate`) already knows to force
the CPU backend for Demucs — documented in `MODEL_STATUS_AUDIO.md`: "`FxSeparateCommand` now always forces
the CPU backend itself so the default invocation just works." The Swarm/AudioLab `ProcessAudio` path doesn't
apply the same override. **Fix direction:** apply the same CPU-backend force in
`AudioEngineBridge.BuildSpec`/`ProcessAsync` for the `Separate` service, matching the CLI's existing fix.

### Issue #G — Resemble-Enhance's weight file 404s from HuggingFace
`HuggingFace file not found: ResembleAI/resemble-enhance/pytorch_model.bin @ main`. A different, *earlier*
failure than the previously-documented one — `MODEL_STATUS_AUDIO.md` describes a real forward-pass/
module-composition architecture mismatch once weights load; this run couldn't even fetch the file. Doesn't
change the model's `ValidationPending` status (it was never going to work end-to-end regardless), but is
worth noting as a second, independent blocker — the exact repo/filename may have moved upstream.

### Issue #H — 7 models with weights on disk and functional providers are unselectable via `GenerateText2Image` [FIXED 2026-07-25/26]
**Resolution:** three distinct bugs, not one. (1) 5 of the 7 (`neutts_tts`/`gptsovits_clone`/
`heartlib_music`/`openvoice_clone`/`resemble_enhance_fx`, plus `demucs_fx` below) were simply never run
through `InstallAndRegisterEngine` — added to `Data/AudioLabInstalledEngines.json` directly. (2) The
`requires_docker` flag on `gptsovits_clone`/`resemble_enhance_fx`/`rvc_clone` was stale — all three are
confirmed present in `AudioEngineBridge`'s in-process-engine binding table (engine-backed, not
Docker-only); removed `.WithRequiresDocker()` from their provider definitions. (`realtimestt_stt` also
carries this flag and was deliberately left alone — it genuinely has no `AudioEngineBridge` binding, so
removing the flag would surface a model that fails on generation.) (3) `demucs_fx` had real weights on disk
but `AudioWeights.WeightsDirectory()` built its path from `ModelPrefix` (`"Demucs"`, capitalized) while the
actual directory was lowercase (`demucs/`) — case-sensitive Linux mismatch. Fixed with a case-insensitive
directory-lookup fallback in `WeightsDirectory()`, which also turned out to fix ACE-Step's `xl-base` (see
the 2026-07-26 perf-pass entry above). `rvc_clone` was deliberately left uninstalled — no checkpoint exists
anywhere on this box, and per the repo's own status docs RVC is inherently bring-your-own-trained-voice
(there is no single canonical "RVC v2 weights" to install, unlike Demucs/ACE-Step). All 6 confirmed live via
`AudioLabListEngines` (`installed: true`) and the actual `Model` dropdown after rebuild+restart.

<details><summary>Original 2026-07-25 investigation (root cause found, not yet fixed)</summary>

`neutts_tts`, `gptsovits_clone`, `heartlib_music` (`3b-base`), `openvoice_clone`, `rvc_clone`, `demucs_fx`,
`resemble_enhance_fx` all return `Invalid value for parameter Model: ... are you sure that model name is
correct?` from `/API/GenerateText2Image`, even though `AudioLabListEngines` reports their specific model
variant as `installed: true`, and — for at least `heartlib_music`, `openvoice_clone`, `demucs_fx` — the
provider is confirmed genuinely functional (produced real, correct audio) via the legacy `ProcessTTS`/
`ProcessAudio` dispatch. **Root cause, isolated this pass:** `DynamicAudioBackend.Init` only calls
`RegisterModelsForProvider` (which populates `Program.MainSDModels`, the list `GenerateText2Image`'s `Model`
param validates against) for provider IDs in the **engine-level** `InstalledEngines` list. Re-querying
`AudioLabListEngines` live confirmed all 7 of these providers report **`installed: false` at the engine
level**, while their individual `models[].installed` field says `true` — the two flags are computed/tracked
independently and can disagree. So the model's weights and provider logic are real and working, but the
provider is never registered at startup, and `GenerateText2Image` can't select it.
Three of the seven (`rvc_clone`, `gptsovits_clone`, `resemble_enhance_fx`) are additionally flagged
`requires_docker: true` in the catalog — `DynamicAudioBackend.Init` explicitly skips docker-flagged providers
("Legacy Docker/Python engines aren't ported to the in-process C# engine ... skipping"). This may be correct
for `rvc_clone` (its weights are genuinely missing — `RVC voice model not found` — even via the legacy path),
but is suspect for `gptsovits_clone`, which responded to the legacy path with a real param-validation error
("Invalid TTS request parameters"), not an "unsupported" error, suggesting it may already run in-process
despite the stale-looking flag.
**Not yet found:** the actual *producer* of the engine-level `installed` boolean (i.e. what sets
`InstalledEngines`, and whether/how it checks on-disk weight presence vs. just recording "install flow
completed"). This is the next thing to locate before planning a fix — needed to tell apart "a state-tracking
bug to reconcile against on-disk reality" from "the install/download flow itself never records completion for
these 7." **User-confirmed real-world context:** the user has generated real HeartMuLa songs through Swarm
before, so the engine/weights are known-good — this is specifically about the `Model`-dropdown/
`GenerateText2Image` selection path, not the model's functionality. **Not fixed this pass** — flagged for a
dedicated, separately-planned pass per explicit user instruction (extension source, shared dev tree, hot-reload
risk to a concurrently-running second agent's session — plan the edit/verify loop deliberately).
</details>

### Issue #I — Dia-1.6B hangs regardless of path [ROOT-CAUSED + partially fixed 2026-07-26 — see perf-pass entry above]
**Was never a hang** — a genuine, severe, launch-overhead-bound slowness (~1152 individual `cublasGemmEx`
calls per decode step). Batched via `cublasGemmStridedBatchedEx`; wall time dropped from 350-800s+/never-
finishing to 44-66s for a short clip. Still far from realtime — full remediation needs CUDA-graph decode +
CFG batching, same scale of work as the HeartMuLa perf rounds, not attempted this pass. Also confirmed:
`DiaPipeline.Generate` has no `CancellationToken`, so a client-side timeout never stops server-side
generation, and the request holds `AudioRuntime._genLock` (global, single-slot) for its full duration — see
Issue #J's reclassification below, which was this issue's downstream symptom, not a separate bug.

<details><summary>Original 2026-07-24/25 investigation (confirmed hang, not root-caused)</summary>

Timed out at 180s via the 2026-07-24 legacy path, then again at 90s and 240s via this pass's canonical path —
three independent hangs across two different code paths and three different timeout ceilings. Not a
params/path issue; a genuine model-level hang or pathological slowness. Not root-caused this pass.

</details>

### Issue #J — AudioGen hard-hangs the entire Swarm process at 45s duration [RECLASSIFIED 2026-07-26: not an independent bug]
**Resolution:** re-tested AudioGen 45s in complete process isolation (fresh restart, first and only
request) — completed cleanly in 95.6s, HTTP 200, real audio, no hang, no elevated CPU. The original
hard-hangs were Issue #I's downstream symptom: `AudioRuntime._genLock` is a global single-slot
`SemaphoreSlim` serializing all audio generation process-wide, and a prior Dia request (genuinely just very
slow, not hung — see Issue #I) held that lock for its entire multi-minute run; any request issued while it
was still running — AudioGen included — queued behind it and looked identically "hung" from the outside,
including surviving `SIGTERM` (the lock-holder, not the queued request, was what needed killing). One root
cause, not two. **Separately confirmed, still open:** AudioGen genuinely does not honor the requested
duration past some point — a 45s request reproducibly produces only 30.0s of audio (see the 2026-07-26
perf-pass entry above) — a real, minor, not-yet-root-caused bug, distinct from the hang.

<details><summary>Original 2026-07-25 investigation (misattributed to AudioGen itself)</summary>

10s generates in 37.6s (RTF 3.76×); 20s generates in 51.8s (RTF 2.59× — worse than linear, but completes);
45s pegs the whole SwarmUI process at 100% CPU / ~5% GPU utilization **indefinitely** — unresponsive to
`SIGTERM`, required `SIGKILL`. Reproduced twice, independently, both times only at 45s (not at 20s). Given
AudioGen is an autoregressive transformer over audio-codec tokens (MusicGen-arch + T5-large text encoder),
this shape — fine at short durations, disproportionately slow as duration grows, then a hard wall — is
consistent with a host-side loop or an attention/KV-cache path that scales worse than the codec's own
frame count would predict (same class of issue as several other host-glue-bound audio models — see
`cpu-rope-bottleneck`/`dit-blocks-must-be-gpu-resident`-type findings elsewhere in this codebase). Not
root-caused this pass; profile with `HARTSY_PROFILE=1` at a duration between 20s and 45s to find the actual
cliff, same method as the DiT host-overhead investigations.

*(In hindsight: both "45s only" reproductions likely coincided with a slow Dia/other request from earlier
in the same test session still holding the generation lock — see the reclassification above.)*

</details>

---

## Next steps

- [ ] Build a real, timed Python baseline for any model where one is cheap to add (an HF `transformers`/
  `diffusers` pipeline a few lines long) — prioritize models currently marked "—" in the Python column that
  are also currently slow, since that's where a real comparison would be most informative.
- [ ] Expose a proper per-provider unload endpoint in AudioLab (mirroring `LLMAssistantUnloadModels`) instead
  of relying on pressure-triggered eviction — would remove the VRAM-climb risk on any future multi-model sweep.
- [ ] `zipvoice_tts` has no GPU-residency perf pass yet (documented ~11 min for a 10s clip) — a real
  optimization target, same class of host-glue issue already fixed for VibeVoice/Bark/F5/Kyutai.
- [ ] `resemble_enhance_fx` remains a genuine forward-pass architecture mismatch (not a quick fix, see
  `MODEL_STATUS_AUDIO.md`) — expect it to fail in this sweep; that's the known, tracked state, not new.
- [x] **Issue #H** — fixed 2026-07-25/26: install-state gap + stale `requires_docker` flags + a
  case-sensitive directory bug, three separate causes. `rvc_clone` deliberately left uninstalled (no
  checkpoint exists, no single canonical one to fetch).
- [x] **Issue #I** — root-caused 2026-07-26: never a hang, ~1152 unbatched GEMM launches/decode-step.
  Fixed via `cublasGemmStridedBatchedEx` (350-800s+ → 44-66s for a short clip). Still not realtime; a
  full fix needs CUDA-graph decode + CFG batching (HeartMuLa-perf-round scale of work) — not attempted.
  `DiaPipeline.Generate` still has no `CancellationToken` — a client timeout still doesn't stop
  server-side generation or release `AudioRuntime._genLock` early.
- [x] **Issue #J** — reclassified 2026-07-26: was Issue #I's zombie-thread-on-the-shared-lock symptom,
  not an independent AudioGen bug. AudioGen alone, in isolation, completes cleanly.
- [ ] AudioGen's duration cap (45s requested → 30.0s produced, confirmed reproducible 2026-07-26) — real,
  minor, not yet root-caused. Check whatever computes `cap`/`tTotal` in `MusicGenPipeline.Synthesize` for
  AudioGen's specific catalog entry.
- [x] ACE-Step-turbo's sung-vocal intelligibility — confirmed 2026-07-26 via controlled same-seed A/B
  (Whisper round-trip): genuinely unintelligible vs. `sft`'s clearly recognizable lyrics. Traced to
  turbo's own documented design (8 steps, no CFG) — an inherent tradeoff, not a bug. Untested: whether
  `turbo-shift1`/`turbo-shift3` do meaningfully better at turbo speed.
- [ ] A fixed-seed, controlled Dia before/after A/B (this pass used a fixed seed for the *after* number
  only — the *before* baseline, 350-800s+, came from earlier unseeded/inconsistent-seed runs) would give a
  cleaner speedup figure than "44-66s vs. 350-800s+."
