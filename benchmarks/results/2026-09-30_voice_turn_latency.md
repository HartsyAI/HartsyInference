# Voice-turn latency — 2026-09-30

PR3 of the phone-call voice-agent plan (`bench/voice-turn-latency`, PR #195). Two phases, kept apart on
purpose: **Phase 0** goes through the live SwarmUI (AudioLab / LLMAssistant over HTTP + base64, whatever
Whisper variant and card those extensions pick) and is bounding context only; **Phase 1** is the in-process,
real-weight, named-checkpoint measurement on the card the plan assigns, and is the ONLY input to the gates.

## Header

| Field | Value |
|---|---|
| Scripts / results commit | `0c62b9bf` on `bench/voice-turn-latency` — the bench scripts exactly as they produced every number below (sentence-count fix, fresh-thread-per-call fix and the quiet-window `--max-minutes` fix included). Raw per-trial JSON: `benchmarks/swarm_audio_bench/swarm_voice_results.json` (STT/TTS) and `benchmarks/results/2026-09-30_voice_turn_latency_llm.json` (LLM) |
| Repo `main` this branch is based on | `79209dc5` (`origin/main`, engine `VersionSuffix` alpha.216 — NOT what Swarm runs, see next row) |
| Engine actually serving Phase 0 | the live install's extension NuGet pins: AudioLab `2.0.0-alpha.183`, LLMAssistant `2.0.0-alpha.130`, HartsyInference-Backend `2.0.0-alpha.202-local.2` (`/home/hartsy/Desktop/Swarm/SwarmUI.not too old`, user unit `swarmui.service`, MainPID 144195) |
| Driver | 595.91.07 (`nvidia-smi`) |
| GPU, nvidia-smi index 0 | NVIDIA GeForce RTX 3060 12 GB — `GPU-b2198524-73ad-f200-16ef-6fc368d0599f` — engine ordinal `cuda:1` — Phase 1 audio card |
| GPU, nvidia-smi index 1 | NVIDIA GeForce RTX 4090 24 GB — `GPU-03eac124-76d8-22a2-1ff1-2be8e2fa098d` — engine ordinal `cuda:0` — Phase 1 LLM card; Swarm's card |
| Card per phase, measured | Phase 0 STT + TTS: 4090 (every request, +272 MiB delta on that UUID). Phase 0 LLM: **3060** (LLMAssistant's dispatcher picked backend 5 = `cuda:1`; the single-reply WS path cannot pin a backend). Phase 1: not run yet |
| Swarm status at start (`GetCurrentStatus`) | `{"waiting_gens": 0, "loading_models": 0, "waiting_backends": 0, "live_gens": 0}`; `ListBackends`: "HartsyInference GPU1 (3060)" (id 8) disabled throughout |
| Quiet window (`tests/swarm-quiet-window.sh --gpu 4090 --gpu 3060`) | OK at 16:11:20Z: 616 s continuous idle on both cards, 41 polls since 15:50:59Z. The first 9 min were blocked by a foreign `/usr/lib/dotnet/dotnet` process (pid 2286259, 430→1008 MiB) on the 4090 — another agent's GPU run; the gate held until it exited. Total wait ≈ 20 min |
| Thermal | 3060: 58 °C at start (idle), 66 °C after the LLM run (it held the Qwen3 model); 4090: 32 °C start and end. `clocks_throttle_reasons.active` = 0x1 (idle) on both at start and end; no throttle sampled. Ambient not recorded |
| Host | i7-6900K, 64 GB, Ubuntu kernel 6.8; Python 3.12 without numpy (narrowband filter is pure Python) |

## Phase 0 (through Swarm — bounding context only, NOT gate input)

Protocol: 2 warm-up + 5 timed calls per case; median / p95 (linear interpolation between order statistics) / min
of the HTTP wall clock measured by the client. **Wall includes the HTTP round trip and the JSON/base64 payload
parse on both sides** (request base64 is built before the timer). The STT response's server-side
`processing_time` is shown beside it; the gap is the transport share. Card attribution: `nvidia-smi
--query-compute-apps` sampled by GPU UUID every 150 ms during each call (used_memory delta / new SwarmUI row on
a card). Between every call `GetCurrentStatus` was re-checked and any foreign GPU pid would have aborted the
batch; none appeared.

**Whisper variant AudioLab resolved: unknown.** The request asked for `model=small` (AudioLab's WhisperProvider
ids are tiny/base/small/medium/large-v2/large-v3/turbo — no `.en` variant is reachable through it); the
`ProcessSTT` response carries no model field, and no Swarm log line during the run (ListRecentLogMessages diffed
from the pre-run cursor, plus the `swarmui.service` journal for the window) named a checkpoint — the model was
already resident from before the run. Derived from source, NOT observed: AudioLab sends the engine token
`whisper:small`, which `SttCatalog.ResolveWhisperRepo` maps to `openai/whisper-small` (multilingual), i.e. not
the `small.en` the gate names. That is one of the reasons Phase 0 is not a gate input.

### STT — `whisper_stt` `model=small`, JFK clip slices (16 kHz mono), 4090

Two runs are shown because the first run's STT half completed before the script stopped on a TTS sentence
word-count assertion (fixed in the PR); the numbers moved a lot between runs, which is itself the finding:
through Swarm, the same request varies by up to 2× run to run.

| Case | run 2 wall median ms | p95 | min | server processing median ms | run 1 wall median ms (p95 / min) | transcript (run 2) |
|---|---:|---:|---:|---:|---|---|
| 2 s, 16k | 2105 | 2623 | 1610 | 2100 | 867 (912 / 783) | And so my fellow Americans |
| 2 s, narrowband 8k→16k | 1169 | 1230 | 724 | 1165 | 1001 (1408 / 948) | And so my fellow Americans |
| 5 s, 16k | 2203 | 2872 | 1148 | 2171 | 1876 (1962 / 1203) | And so, my fellow Americans, ask not! |
| 5 s, narrowband 8k→16k | 2706 | 3053 | 2615 | 2697 | 2431 (2512 / 1764) | And so, my fellow Americans, ask not. |
| 10 s, 16k | 3186 | 4013 | 2273 | 3177 | 4920 (5455 / 4269) | And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. |
| 10 s, narrowband 8k→16k | 3934 | 4012 | 3606 | 3917 | 1994 (2195 / 1743) | And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. |

Reading: transport (HTTP + base64) is ≈ 5–15 ms of every row — the wall is the server. Every row is far above
the 350 ms per-utterance gate, on the faster card, with a different (multilingual) checkpoint, through two
extensions' plumbing; it bounds the problem, it does not measure it. Whisper pads to 30 s so the encoder cost
should not depend on slice length — the growth with length is decode steps (more words) plus run-to-run noise.
Narrowband transcripts were word-identical to 16k at every length (this clip is clean studio speech; the
recall gate is measured in Phase 1 with the engine's own resampler).

Narrowband method here: 63-tap Hamming-windowed sinc, cutoff 4 kHz (0.25 cycles/sample at 16 kHz), decimate ×2
to 8 kHz, zero-stuff ×2 and the same low-pass with gain 2 back to 16 kHz; pure Python. Phase 1 uses the engine's
polyphase `Resampler` — the two narrowband rows are not comparable across phases.

### TTS — `kokoro_tts` voice `af_heart`, 4090

| Case | wall median ms | p95 | min | audio s | RTF (median wall / audio) |
|---|---:|---:|---:|---:|---:|
| 5 words — "Please hold while I check." | 672 | 848 | 519 | 2.10 | 0.320 |
| 15 words — "Thanks for calling, I can see your appointment is booked for Tuesday afternoon at three." | 2917 | 3094 | 1567 | 5.60 | 0.521 |
| 30 words — "I have updated the delivery address on your order, the driver will call you when they are ten minutes away, and you will receive a message with the tracking link." | 3732 | 3869 | 3474 | 10.18 | 0.367 |

Reading: the 15-word sentence takes 2.9 s through Swarm against a 250 ms gate (first-chunk, in-process, 3060).
The returned WAV is base64 in the JSON (5.6 s at 24 kHz ≈ 360 KiB), so a few tens of ms of each row is payload;
the rest is AudioLab → engine → Kokoro, and Kokoro's known host sync loops (`KokoroOps.cs`,
`KokoroIStftNetDecoder.cs`) are the plan's named lever. The prior catalog-sweep baseline for `kokoro_tts` in
`swarm_audio_results.json` is 2.41 s for a 3.7 s clip (RTF 0.65); these rows are in the same band.

### LLM — `Qwen3-4B-Q4_K_M.gguf` via LLMAssistant WS, **served on the 3060** (not the 4090)

| Prompt tokens (LLMAssistantCountTokens) | max tokens | TTFT median ms | p95 | min | decode chunk/s median | p95 | min | reply chunks | thinking |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 553 | 64 | 3944 | 3952 | 3926 | 22.4 | 22.5 | 22.3 | 55 (empty `<think>` block took the other ~9 tokens; extension strips the tags from `full_text`) | off via `/no_think` (the WS API has no thinking field); no reasoning text in the reply |

Deviations that make this row context, not evidence: (1) the LLMAssistant dispatcher owns model→backend
choice and put Qwen3-4B on backend 5 (`cuda:1` = 3060; +8054 MiB on that UUID at the warm-up load); the
single-reply WS path has no `backendId`, and compare mode fans ≥2 lanes out concurrently, so the 4090 cannot be
pinned from the API. (2) TTFT includes LLMAssistant's thread/tool plumbing and the chat template. (3) A first
attempt reused one thread for all 7 calls and TTFT climbed 9.1 → 34.6 s across reps because the WS path re-sends
the whole thread history — the script now creates a thread per call; the numbers above are from that run. The
4090 TTFT/tok-s the gate needs come from Phase 1 only.

## Phase 1 (in-process, 3060 — gate input)

Not run in this PR (in-process GPU runs are the orchestrator's; PR3 only writes the probes). Run each suite
alone, after `tests/swarm-quiet-window.sh` exits 0 for the card in question, from a checkout of this branch
with `~/.config/hartsyinference/settings.json` `paths.modelsRoot` pointing at the RAID (it does on this box:
`/mnt/model-storage/Models`), so `AudioModelCache`, the cmudict and the RNNoise path resolve there. Both tests
print the device name, models root and every resolved path before loading anything; a device-name mismatch
FAILS, a missing mandatory asset SKIPS (or fails with `HARTSY_REQUIRE_REAL_WEIGHTS=1`).

```bash
# audio probes — 3060 (engine ordinal 1); ~3 min after model load
tests/swarm-quiet-window.sh --gpu 3060
HARTSY_VOICE_BENCH=1 HARTSY_VOICE_BENCH_OUT=/tmp/voice_phase1_audio.md \
  dotnet test tests/HartsyInference.Audio.Tests -c Release --filter "FullyQualifiedName~VoiceTurnBenchTests" --logger "console;verbosity=normal"

# LLM probe — 4090 (engine ordinal 0); the Swarm card, so the full 10-min window applies
tests/swarm-quiet-window.sh --gpu 4090
HARTSY_VOICE_BENCH_LLM=1 HARTSY_VOICE_BENCH_OUT=/tmp/voice_phase1_llm.md \
  dotnet test tests/HartsyInference.LLM.Tests -c Release --filter "FullyQualifiedName~VoiceLlmTurnBenchTests" --logger "console;verbosity=normal"
```

Assets on this box today: `openai/whisper-small.en` cached (mandatory); `openai/whisper-medium.en` NOT cached
(optional row → SKIPPED); Kokoro repack + `config.json` + `voices/af_heart.bin` cached; `cmudict.dict` present;
RNNoise weights (`{models}/audio/wake/denoise/rnnoise.safetensors`) NOT present (optional row → SKIPPED unless
placed first); `Qwen3-4B-Q4_K_M.gguf` present.

### Probe A — Whisper `.en` per-utterance wall (3060, in-process, 30 s pad)

| Model | Slice | Variant | median ms | p95 ms | min ms | transcript |
|---|---:|---|---:|---:|---:|---|
| openai/whisper-small.en | 2 s | 16k | | | | |
| openai/whisper-small.en | 2 s | narrowband 8k→16k | | | | |
| openai/whisper-small.en | 5 s | 16k | | | | |
| openai/whisper-small.en | 5 s | narrowband 8k→16k | | | | |
| openai/whisper-small.en | 10 s | 16k | | | | |
| openai/whisper-small.en | 10 s | narrowband 8k→16k | | | | |
| openai/whisper-medium.en (optional) | — | — | | | | |

### Probe B — Kokoro af_heart per sentence (3060, in-process; G2P timed separately)

| Words | synth median ms | p95 ms | min ms | audio s | RTF (median/audio) | G2P ms | Whisper-verify recall | verify transcript |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | | | | | | | | |
| 15 | | | | | | | | |
| 30 | | | | | | | | |

### Probe C — narrowband content-word recall vs the 16 kHz baseline (small.en, full 11 s clip)

| Input | recall | Δ vs 16k (pts) | STT median ms | p95 ms | front-end median ms | transcript |
|---|---:|---:|---:|---:|---:|---|
| 16 kHz baseline | | 0 | | | — | |
| narrowband 8k→16k | | | | | — | |
| narrowband + RNNoise (optional) | | | | | | |

### LLM probe — Qwen3-4B Q4_K_M (4090, in-process, thinking off, greedy, 64 max tokens)

| Variant | prompt tokens (templated) | TTFT median ms | p95 | min | prefill+first-sample median ms | decode tok/s median | p95 | min | tokens | stopped on EOS | reply head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| tools on (sentinel grammar, graph/spec OFF) | | | | | | | | | | | |
| no tools (knob defaults) | | | | | | | | | | | |

Note on the LLM columns: `OnPrefillCompleted` fires after the first sample, so "prefill+first-sample" is not a
pure prefill time; TTFT is the first streamed token from request start; decode tok/s = (n − 1) / (t_last −
t_first).

## Gates

From the plan's "Model bring-up gates" table, restricted to the rows PR3 evaluates; each compound gate is split
into one row per measured quantity so the Measured column holds one number. The plan's other rows (Qwen3 tool
template — PR6; Silero + RNNoise ≤ 2 ms per frame — PR8; Piper digest-identical — PR5) are checked in those PRs
and are not evaluated here. Evaluated on Phase 1 only; any miss stops the plan at that gate and the named task
lands before any PR that touches that model.

| Gate | Where checked | Measured (Phase 1) | Status |
|---|---|---|---|
| Whisper small.en per-utterance ≤ 350 ms on the 3060 | PR3 probe A, PR8 e2e | | unmeasured |
| Whisper narrowband recall ≥ 16k baseline − 10 pts | PR3 probe C, PR8 e2e | | unmeasured |
| Kokoro 15-word sentence ≤ 250 ms on the 3060 | PR3 probe B, PR8 e2e | | unmeasured |
| Kokoro Whisper-verify recall ≥ 80 % | PR3 probe B, PR8 e2e | | unmeasured |
| Qwen3-4B TTFT ≤ 150 ms with tools on (graph/spec decode off) on the 4090 | PR3 LLM probe | | unmeasured |
| Qwen3-4B ≥ 60 tok/s with tools on (graph/spec decode off) on the 4090 | PR3 LLM probe | | unmeasured |

On failure (from the plan): Whisper → task `perf/whisper-short-window-encoder` (the fixed 30 s pad,
`WhisperPipeline.cs`) and/or narrowband quality work; Kokoro → `perf/kokoro-host-sync-loops`; Qwen3 →
`perf/llm-tool-turn-decode` (make the sentinel grammar step graph-decode compatible). The stack composition does
not change to dodge a gate.
