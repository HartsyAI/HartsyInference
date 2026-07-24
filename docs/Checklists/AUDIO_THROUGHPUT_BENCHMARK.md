# Audio Throughput Benchmark — Design & Tracking

Status legend: ⬜ not started · 🔧 in progress · ✅ done · ⚠ blocked

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

**Tier 2 (Swarm-path, this pass) — the spine of this document.** A new harness,
[`benchmarks/swarm_audio_bench/swarm_audio_bench.py`](../../benchmarks/swarm_audio_bench/swarm_audio_bench.py),
drives every local (non-cloud) AudioLab provider through the real product API:
- **TTS** (21 models) → `POST /API/ProcessTTS` — `text="Hello, this is a test of the text to speech system."`,
  plus a reference clip (`reference_audio`/`ref_text`, an 11s JFK excerpt) on every call so clone-capable
  models exercise their real path; non-clone models ignore the extra fields harmlessly.
- **STT** (6 models) → `POST /API/ProcessSTT` — same JFK clip, `language=en`.
- **Music/SFX** (6 models) → `POST /API/ProcessAudio` — `prompt="An upbeat electronic dance track with synths
  and a steady beat"`, `duration=10`, `task_type=text2music`.
- **Voice Conversion** (2 models) → `POST /API/ProcessAudio` — JFK clip as both `source_audio` and
  `target_voice` (identity-ish smoke test; RVC has no bundled trained voice to convert to, expected N/A).
- **Fx** (2 models) → `POST /API/ProcessAudio` — JFK clip as `audio_data` (functional smoke test, not a
  quality eval — Demucs/Resemble-Enhance are evaluated for separation/enhancement quality elsewhere, not RTF).

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
- **"Installed" ≠ "unusable."** `AudioLabListEngines` reports several providers as `installed: false` despite
  their weights already being present in `~/.cache/hartsyinference/` (e.g. `csm_tts`, `neutts_tts`, several
  STT ids) — this flag tracks the extension's own registry step, not weight presence; generation was
  attempted directly regardless of this flag.

---

## Results — Tier 2 Swarm-path sweep (2026-07-24)

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

### Issue #A — Chatterbox's reference-voice cloning isn't wired
Clean, well-worded error (not a crash): "Chatterbox reference-voice cloning is not wired yet — it needs a
PCM→40-bin-mel front-end for the voice encoder." Falls back cleanly when no reference is supplied. A real
feature gap, not a defect — every other clone-capable TTS model in this sweep accepted the same reference
args without complaint.

### Issue #B — Zonos-v0.1's actual weights file is missing
`Could not find file '.../Zyphra--Zonos-v0.1-transformer/model.safetensors'`. The cache directory exists (from
the 2026-07-17 verification session) but the real weights file inside it doesn't — `MODEL_STATUS_AUDIO.md`
documents Zonos as fully verified with real perf numbers from that date, so the file has since been deleted
or the download never fully completed. **Fix: re-download** (`AudioLabInstallEngine` for `zonos_tts`, or
delete the stale directory and let it re-fetch).

### Issue #C — Distil-Whisper's bare provider id resolves to an unsupported repo
Matches a pre-existing, already-documented gap (`MODEL_STATUS_AUDIO.md`): the bare id defaults to
`distil-whisper/distil-large-v3.5`, which `WhisperPipeline.InferConfig`'s repo switch doesn't recognize (only
v2/v3/medium.en/small.en). A variant suffix (`distilwhisper:v3`) works around it. **Fix direction:**
`SttCatalog.ResolveDistilWhisperRepo`'s no-match default should point at a repo `WhisperPipeline.InferConfig`
actually supports.

### Issue #D — HeartMuLa generation OOM-killed the entire Swarm process [CRITICAL]
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

### Issue #E — YuE's real checkpoint was never actually downloaded
`YuE checkpoint folder not found: '.../Models/audio/music/yue/yue'`. Only an empty stub directory exists
despite a catalog `Assets` entry declaring the ~12.5 GB checkpoint. Low-risk fix: just run the download
(`hartsy music -m yue` with confirm, or `AudioLabInstallEngine` for `yue_music`).

### Issue #F — Demucs fails through Swarm's generic audio path
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
