# Voice-turn latency — 2026-09-30 (Phase 1 audio re-run 2026-10-01)

PR3 of the phone-call voice-agent plan (`bench/voice-turn-latency`, PR #195). Two phases, kept apart on
purpose: **Phase 0** goes through the live SwarmUI (AudioLab / LLMAssistant over HTTP + base64, whatever
Whisper variant and card those extensions pick) and is bounding context only; **Phase 1** is the in-process,
real-weight, named-checkpoint measurement on the card the plan assigns, and is the ONLY input to the gates.

## Header

| Field | Value |
|---|---|
| Scripts / probes commit | Every number below was produced by the pre-rebase commits `b9df6302` (probes + scripts) and `0c62b9bf` (sentence-count and fresh-thread-per-call fixes, quiet-window `--max-minutes` fix) on `bench/voice-turn-latency` based on `79209dc5`; those shas were rewritten by the rebase onto `d3b850f8` and now exist as `5071a5d2` / `d632dad0`, byte-identical in every script and probe file (`git diff` between the two bases touched none of them). Raw per-trial JSON for Phase 0: `benchmarks/swarm_audio_bench/swarm_voice_results.json` (STT/TTS) and `benchmarks/results/2026-09-30_voice_turn_latency_llm.json` (LLM). **The Phase 1 audio re-run (2026-10-01, the gate input) was produced by `d64638d0`**: the probes with each case's first call reported beside the warm median and a primer before the loop, and Probe C's RNNoise pass on a `CpuBackend` |
| Repo `main` this branch sits on | `c1a180c4` (alpha.229 — NOT what Swarm runs, see next row): #196 Whisper `.en` token layout, #199 Kokoro host-sync perf, #204 Whisper voice latency, #200/#203 RNNoise and the voice front end, #205 model-folder case. Phase 0 was measured on `79209dc5` (alpha.216); the first Phase 1 runs on that base and on `d3b850f8` (#193), whose changes did not touch the probes' dependencies |
| Engine actually serving Phase 0 | the live install's extension NuGet pins: AudioLab `2.0.0-alpha.183`, LLMAssistant `2.0.0-alpha.130`, HartsyInference-Backend `2.0.0-alpha.202-local.2` (`/home/hartsy/Desktop/Swarm/SwarmUI.not too old`, user unit `swarmui.service`, MainPID 144195) |
| Driver | 595.91.07 (`nvidia-smi`) |
| GPU, nvidia-smi index 0 | NVIDIA GeForce RTX 3060 12 GB — `GPU-b2198524-73ad-f200-16ef-6fc368d0599f` — engine ordinal `cuda:1` — Phase 1 audio card |
| GPU, nvidia-smi index 1 | NVIDIA GeForce RTX 4090 24 GB — `GPU-03eac124-76d8-22a2-1ff1-2be8e2fa098d` — engine ordinal `cuda:0` — Phase 1 LLM card; Swarm's card |
| Card per phase, measured | Phase 0 STT + TTS: 4090 (every request, +272 MiB delta on that UUID). Phase 0 LLM: **3060** (LLMAssistant's dispatcher picked backend 5 = `cuda:1`; the single-reply WS path cannot pin a backend). Phase 1 audio (first run and re-run): 3060, engine ordinal 1 (device name asserted by the probe, no `CUDA_VISIBLE_DEVICES`). Phase 1 LLM: 4090, engine ordinal 0 (asserted) |
| Swarm status at start (`GetCurrentStatus`) | `{"waiting_gens": 0, "loading_models": 0, "waiting_backends": 0, "live_gens": 0}`; `ListBackends`: "HartsyInference GPU1 (3060)" (id 8) disabled throughout. **Caveat found 2026-10-01:** SwarmUI answers `GetCurrentStatus` per session (it counts only the calling session's own gens), so this and every between-call re-check in these runs was blind to anyone else's generations; the scripts now use `GetGlobalStatus`. What establishes that no one else generated is the retroactive journal check in the next row |
| Retroactive journal check (2026-10-01) | `journalctl --user -t swarmui-run.sh` over each window, as `tests/swarm-quiet-window.sh --verify-since` reads it: **no T2I request, gen, backend request or sampler step** in the Phase 0 window (15:50:59–16:11:20Z), either Phase 0 audio run (16:11:20–16:17:30Z), either Phase 0 LLM run (16:17:30–16:24:30Z), the Phase 1 audio first run (16:55–17:25Z) or the Phase 1 LLM window and run (17:58–18:25Z); the only AudioLab TTS requests in those windows are this PR's own 21 kokoro calls (3 sentences × 7) in the recorded Phase 0 audio run; no Swarm restart inside any window (launcher pid 144195 throughout Phase 0 and Phase 1 audio, 2543665 throughout Phase 1 LLM). AudioLab STT calls and LLMAssistant chats write no journal line, so those two kinds cannot be excluded retroactively |
| Quiet windows (`tests/swarm-quiet-window.sh`) | Phase 0: `--gpu 4090 --gpu 3060` OK at 16:11:20Z, 616 s continuous idle on both cards, 41 polls since 15:50:59Z; the first 9 min were blocked by a foreign `/usr/lib/dotnet/dotnet` process (pid 2286259, 430→1008 MiB) on the 4090 — another agent's GPU run; the gate held until it exited (total wait ≈ 20 min). Phase 1 audio: run ~17:05–17:20Z. Phase 1 LLM: run ~18:15Z after a 605 s window on both cards. These windows ran the script's first version, which polled the per-session `GetCurrentStatus` (blind to other sessions' gens — see the caveat above) and the card's compute apps as 30 s snapshots; the server-wide `GetGlobalStatus` counters, the continuous journal request check and the SwarmUI-memory check on the target cards apply from the Phase 1 audio re-run on, and `--verify-since` confirms each arm afterwards. **Phase 1 audio re-run:** `--gpu 3060` OK at 03:31:09Z after 615 s (22 polls since 03:20:54Z) with the fixed gate — global queue 0/0/0 on every poll, nothing on the 3060, no request line in the journal; every probe was then confirmed with `--verify-since` from its own start (all OK) |
| Bench lock and host load (Phase 1 audio re-run) | The shared bench lock (`orch_bench.lock`, which other agents' builds and CPU jobs wait on) was held as `pr195` from 03:59:56Z to 04:01:42Z, covering Probes A and C. Probe B ran at 03:38:48Z, before the lock was taken, in a gap between other agents' builds. Every probe launched only after an 8 s sample of the whole box showed ≤ 3 cores busy and no foreign `dotnet` build/test: B 0.69, A 0.66, C 0.69 of 16 cores busy. The in-run samples (B and C at t+10 s; A finished in 9 s, before its first sample) found no foreign build/test and nothing else on the 3060. Between B and A the gate waited out other agents' builds and a CPU Python reference job (03:39–04:00Z). The monitor's own "foreign CPU" figure is not used: it counted the probe's own test host, a bug since fixed |
| Thermal | Phase 0: 3060 58 °C at start (idle), 66 °C after the LLM run (it held the Qwen3 model); 4090 32 °C start and end; `clocks_throttle_reasons.active` = 0x1 (idle) on both at start and end, no throttle sampled. Phase 1: not sampled by the probes (they record wall clock only) — in particular **no per-run GPU temperature or clock sample exists for the LLM row**, so the 152-vs-150 ms boundary call below is made without thermal evidence (a 4090 warm from an earlier gen moves TTFT by more than that margin); a concurrent `dotnet build` on the host during the first Probe B is the known contention, see that section. Phase 1 audio re-run (3060, sampled by the run monitor): B 61 °C at SM 1950 MHz, C 58 °C at SM 1807 MHz, no thermal / power / HW slowdown bits; A finished before its first sample (54 °C before launch). Both cards drive displays: the 3060 idles at ~27 % utilisation at 210 MHz from the desktop compositor and a rustdesk session. Ambient not recorded |
| Host | i7-6900K, 64 GB, Ubuntu kernel 6.8; Python 3.12 without numpy (narrowband filter is pure Python); .NET SDK 10.0.112 |

## Phase 0 (through Swarm — bounding context only, NOT gate input)

Protocol: 2 warm-up + 5 timed calls per case; median / p95 (linear interpolation between order statistics) / min
of the HTTP wall clock measured by the client. **Wall includes the HTTP round trip and the JSON/base64 payload
parse on both sides** (request base64 is built before the timer). The STT response's server-side
`processing_time` is shown beside it; the gap is the transport share. Card attribution: `nvidia-smi
--query-compute-apps` sampled by GPU UUID every 150 ms during each call (used_memory delta / new SwarmUI row on
a card). Between every call the script re-checked `GetCurrentStatus` — which, it later turned out, counts only
the calling session's own gens, so that re-check could not see anyone else; the retroactive journal check in the
header is what shows no other T2I or TTS request landed during the run — and any foreign GPU pid would have
aborted the batch; none appeared.

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

## Phase 1 (in-process — gate input)

The audio gates are evaluated on the **2026-10-01 re-run** below (alpha.229: after #196 Whisper `.en` token
layout, #199 Kokoro host-sync perf and #204 Whisper voice latency). The 2026-09-30 first run follows it as
history: its Whisper rows were invalid (token-layout bug) and its Kokoro row predates #199 and ran under a
concurrent build. The LLM probe ran once (2026-09-30, 4090) and stands.

**`HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models` is required**: `TestPaths.ModelsDir` defaults to
`<repo>/Models` (which in a worktree holds only `Tokenizers/`), and the probes resolve the cmudict, RNNoise and
Qwen3 paths through it; the `paths.modelsRoot` knob in `~/.config/hartsyinference/settings.json` does not cover
it. Both tests print the device name, models root and every resolved path before loading anything; a device-name
mismatch FAILS, a missing mandatory asset SKIPS (or fails with `HARTSY_REQUIRE_REAL_WEIGHTS=1`, which the re-run
set).

```bash
# audio probes — 3060 (engine ordinal 1), one probe per process, each confirmed afterwards
tests/swarm-quiet-window.sh --gpu 3060
HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models HARTSY_VOICE_BENCH=1 HARTSY_REQUIRE_REAL_WEIGHTS=1 \
HARTSY_VOICE_BENCH_OUT=<file> dotnet test tests/HartsyInference.Audio.Tests -c Release --no-build \
  --filter "FullyQualifiedName~VoiceTurnBenchTests.ProbeB_Kokoro_Sentences_Via_EnglishG2P"   # then ProbeA_…, ProbeC_…
tests/swarm-quiet-window.sh --verify-since @<probe start epoch>   # exit 0 = keep the probe, 4 = redo it

# LLM probe — 4090 (engine ordinal 0); the Swarm card, so the full 10-min window applies
tests/swarm-quiet-window.sh --gpu 4090
HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models HARTSY_VOICE_BENCH_LLM=1 HARTSY_VOICE_BENCH_OUT=/tmp/voice_phase1_llm.md \
  dotnet test tests/HartsyInference.LLM.Tests -c Release --filter "FullyQualifiedName~VoiceLlmTurnBenchTests" --logger "console;verbosity=normal"
```

Assets at re-run time:
- `openai/whisper-small.en` cached (mandatory).
- `openai/whisper-medium.en` NOT cached, so its optional row is SKIPPED.
- Kokoro repack, `config.json` and `voices/af_heart.bin` cached; `cmudict.dict` present.
- RNNoise weights now present at `{models}/audio/wake/denoise/rnnoise.safetensors`: xiph `rnnoise10Ga_12`,
  converted by the engine's `RnnoiseCheckpoint`, with keys matching the loader. The optional row ran.
- `Qwen3-4B-Q4_K_M.gguf` present.

### Phase 1 audio re-run — 2026-10-01, alpha.229 (GATE INPUT)

Tree `d64638d0`, driver 595.91.07, RTX 3060 `GPU-b2198524-73ad-f200-16ef-6fc368d0599f` at engine ordinal 1
(asserted by the probe), `CUDA_VISIBLE_DEVICES` unset. Every probe passed clean. The table summarizes how each was
guarded; the header rows "Quiet windows" and "Bench lock and host load" have the detail. Raw evidence (the guard
lines and each probe's test output): `benchmarks/results/2026-10-01_voice_turn_latency_phase1_rerun.log`.

| Probe | Window (UTC) | Bench lock | Box busy before launch (8 s) | During the run | `--verify-since` | 3060 during the run |
|---|---|---|---:|---|---|---|
| B (Kokoro) | 03:38:48–03:38:58 | not yet held | 0.69 of 16 cores | no foreign build/test or 3060 process at t+10 s | OK | 61 °C, SM 1950 MHz, no clock-cut bits |
| A (Whisper) | 04:00:17–04:00:26 | held | 0.66 | finished before the first sample | OK | 54 °C before launch |
| C (recall) | 04:01:01–04:01:16 | held | 0.69 | no foreign build/test or 3060 process at t+10 s | OK | 58 °C, SM 1807 MHz, no clock-cut bits |

Method change since the first run: Probes A and B run a primer of different content first (the clip's last 3 s;
an unrelated 9-word sentence), so one-time costs stay out of every row. Each case's first call — new input on a
warm model, which is all a live call ever sends — is reported beside the warm median. The 2 warm + 5 timed
protocol is unchanged; the first call is the first of the two warm-ups.

#### Probe B — Kokoro af_heart per sentence (re-run)

| Words | first-run ms (new text) | warm median ms | p95 ms | min ms | audio s | RTF (median/audio) | G2P ms | Whisper-verify recall | verify transcript |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 130.6 | 86.0 | 116.1 | 83.7 | 2.10 | 0.041 | 0.0 | 100% | Please hold while I check. |
| 15 | **238.2** | **156.6** | 170.5 | 141.6 | 5.60 | 0.028 | 0.0 | 90% | Thanks for calling. I can see your appointment is booked for Tuesday afternoon at 3. |
| 30 | 329.4 | 276.9 | 279.5 | 232.9 | 10.18 | 0.027 | 0.1 | 95% | I have updated the delivery address on your order. The driver will call you when they are 10 minutes away and you will receive a message with the tracking link. |

Reading:
- **Gate met.** The 15-word warm median is 156.6 ms against 250 ms (p95 170.5, min 141.6).
- The first synthesis of that new text, 238.2 ms, is also inside the gate, but only by 12 ms.
- A new text's first synthesis costs more than a repeat: +45 / +82 / +53 ms at 5 / 15 / 30 words. This is the open
  item being worked: the voice-session e2e saw 254 ms against 150–191 ms for 15 words. A live call only ever
  synthesizes new text, so the first-run column is what a caller hears.
- Whisper-verify recall is 100 / 90 / 95 %. Both misses are numbers the reference spells out ("three" heard as
  "3", "ten" as "10"), not mispronunciation; content-word recall does not normalize numerals.

#### Probe A — Whisper small.en per-utterance wall (re-run, 30 s pad)

| Model | Slice | Variant | first-call ms | median ms | p95 ms | min ms | transcript |
|---|---:|---|---:|---:|---:|---:|---|
| openai/whisper-medium.en | — | — | SKIPPED | — | — | — | weights not cached |
| openai/whisper-small.en | 2 s | 16k | 139.3 | 115.1 | 128.2 | 108.9 | And so, my fellow Americans, |
| openai/whisper-small.en | 2 s | narrowband 8k→16k | 115.0 | 93.7 | 107.4 | 93.1 | And so, my fellow Americans, |
| openai/whisper-small.en | 5 s | 16k | 113.5 | 111.0 | 113.1 | 109.7 | And so, my fellow Americans, ask not. |
| openai/whisper-small.en | 5 s | narrowband 8k→16k | 111.4 | 112.1 | 116.6 | 109.3 | And so, my fellow Americans, ask not. |
| openai/whisper-small.en | 10 s | 16k | 222.1 | **185.9** | **191.5** | 182.3 | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. |
| openai/whisper-small.en | 10 s | narrowband 8k→16k | 183.3 | 185.0 | 187.5 | 179.8 | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. |

Reading:
- **Gate met** at every length, warm and first call. Warm medians are 93.7–185.9 ms, the worst p95 is 191.5 ms
  and the worst first call is 222.1 ms (all at 10 s, 16 kHz), against 350 ms.
- Transcripts are correct and identical at 16 kHz and narrowband for each length.
- The 30 s pad is still in place: #204 sped up the padded path rather than shortening it. Latency follows the
  number of decoded tokens (≈ 110 ms at 2–5 s, ≈ 185 ms at 10 s).

#### Probe C — narrowband content-word recall vs the 16 kHz baseline (re-run, small.en, full 11 s clip)

| Input | recall | Δ vs 16k (pts) | STT median ms | p95 ms | RNNoise (CPU, whole clip) median ms | transcript |
|---|---:|---:|---:|---:|---:|---|
| 16 kHz baseline | 100% | 0 | 203.8 | 272.9 | — | And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country. |
| narrowband 8k→16k | 100% | 0 | 187.6 | 198.5 | — | And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country. |
| narrowband + RNNoise | 100% | 0 | 187.7 | 200.5 | 794.6 | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. |

Reading:
- **Gate met:** narrowband recall is 100 % against a 100 % baseline (Δ 0). The clip is clean studio speech, so
  this checks the band-limiting round trip, not phone-line noise.
- The RNNoise row is data, not a gate: the front-end gate (≤ 2 ms per frame) is open and awaits a user decision.
  - Recall is unchanged at 100 % after denoising.
  - The whole-clip pass on the CPU backend took a median 794.6 ms for 11 s of audio. That runs at the backend's
    default threading, where the wake worker runs it, which works out to ≈ 1.4 ms per 20 ms of audio.
  - That is not the single-core, serial, per-frame measurement the front-end gate is defined on.

### Phase 1 audio first run — 2026-09-30, before #196 and #199 (historical, superseded)

Kept as the record of the two failure modes that stopped the plan; not gate input.

#### Probe A (first run) — Whisper `.en` per-utterance wall — **INVALID (token-layout bug, fixed in #196)**

**Every transcript came back empty and every row sits at ≈ 11.2 s.** That is not a latency of small.en: the
English-only checkpoints use a different special-token layout (SOT 50257 / EOT 50256 / vocab 51864) from the
multilingual constants the pipeline assumed, so the decoder prompt was built without SOT and greedy decode never
produced EOT — each row is the 30 s-padded encoder pass plus a decode that ran to `MaxNewTokens = 224` and was
then dropped by the tokenizer. Engine bug, not a measurement; fixed by PR #196
(`fix/whisper-english-only-token-layout`). The numbers are kept as evidence of the failure mode only; the re-run
above, on a build with #196 and #204, is the measurement.

| Model | Slice | Variant | median ms | p95 ms | min ms | transcript | status |
|---|---:|---|---:|---:|---:|---|---|
| openai/whisper-small.en | 2 s | 16k | 11192.0 | 11223.5 | 10996.9 | (empty) | INVALID — re-run after #196 |
| openai/whisper-small.en | 2 s | narrowband 8k→16k | 18302.1 | 22762.1 | 11083.5 | (empty) | INVALID — re-run after #196 |
| openai/whisper-small.en | 5 s | 16k | 12983.2 | 14619.4 | 11349.4 | (empty) | INVALID — re-run after #196 |
| openai/whisper-small.en | 5 s | narrowband 8k→16k | 11060.7 | 12111.9 | 10838.8 | (empty) | INVALID — re-run after #196 |
| openai/whisper-small.en | 10 s | 16k | 11172.0 | 11239.9 | 11067.0 | (empty) | INVALID — re-run after #196 |
| openai/whisper-small.en | 10 s | narrowband 8k→16k | 11254.8 | 12451.7 | 11051.3 | (empty) | INVALID — re-run after #196 |
| openai/whisper-medium.en (optional) | — | — | SKIPPED | — | — | weights not cached | SKIPPED |

#### Probe B (first run) — Kokoro af_heart per sentence — **MISSED under a concurrent build (superseded: met on the re-run after #199)**

Synthesis wall is the gate input and does not depend on Whisper. The Whisper-verify recall column is INVALID for
the same reason as Probe A (verification used small.en, which returned empty text) — it says nothing about the
audio; re-run after #196. Contention caveat: a concurrent `dotnet build` on the host overlapped this probe and
may have inflated the medians (the 15-word p95 3259 ms vs min 2659 ms spread is consistent with that); the
cleanest floor in the set, the 5-word min of 491 ms, already misses the 250 ms gate on its own, so the miss stands
regardless of the caveat. The 30-word row being faster than the 15-word one is the same noise — RTF 0.20 vs 0.50
on the same model in one run is not a property of sentence length.

| Words | synth median ms | p95 ms | min ms | audio s | RTF (median/audio) | G2P ms | Whisper-verify recall | verify transcript |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 1314.1 | 1766.0 | 491.3 | 2.10 | 0.626 | 3.0 | 40 % — INVALID (#196) | Please hold |
| 15 | **2788.0** | 3258.6 | 2658.9 | 5.60 | 0.498 | 0.1 | 0 % — INVALID (#196) | (empty) |
| 30 | 2052.6 | 2190.7 | 2029.3 | 10.18 | 0.202 | 0.1 | 0 % — INVALID (#196) | (empty) |

This run missed the gate "Kokoro 15-word sentence ≤ 250 ms on the 3060" by ≈ 10× (2788 ms median, 2659 ms min),
which stopped the plan at that gate. `perf/kokoro-host-sync-loops` (#199) fixed the cost: per-timestep LSTM
launches and a direct DFT computing cos/sin per element, as well as host glue. The re-run above meets the gate.

#### Probe C (first run) — narrowband content-word recall — **INVALID (same token-layout bug)**

Both recalls are 0 % with empty transcripts, so Δ = 0 is an artefact of the same English-only token-layout bug,
not evidence that narrowband holds recall. The re-run above is the measurement.

| Input | recall | Δ vs 16k (pts) | STT median ms | p95 ms | front-end median ms | transcript | status |
|---|---:|---:|---:|---:|---:|---|---|
| 16 kHz baseline | 0 % | 0 | 11192.5 | 11422.5 | — | (empty) | INVALID — re-run after #196 |
| narrowband 8k→16k | 0 % | 0 | 11318.0 | 11355.7 | — | (empty) | INVALID — re-run after #196 |
| narrowband + RNNoise (optional) | SKIPPED | — | — | — | — | weights not found at `/mnt/model-storage/Models/audio/wake/denoise/rnnoise.safetensors` | SKIPPED — RNNoise weights not hosted (task filed) |

### LLM probe — Qwen3-4B Q4_K_M (4090, in-process, thinking off, greedy, 64 max tokens) — gate input

| Variant | prompt tokens (templated) | TTFT median ms | p95 | min | prefill+first-sample median ms | decode tok/s median | p95 | min | tokens | stopped on EOS | reply head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| tools on (sentinel grammar, graph/spec OFF) | 631 | **152.0** | 160.3 | 150.7 | 152.0 | **151.1** | 152.6 | 150.1 | 39 | True | The caller wants to change the delivery address for an order placed last week and confirm if the courier can call ahead.… |
| no tools (knob defaults) | 577 | 131.7 | 134.3 | 130.9 | 131.7 | 150.6 | 151.0 | 144.1 | 39 | True | The caller wants to change the delivery address for an order placed last week and confirm if the courier can call ahead.… |

Reading: the 20 ms tools-on delta is the sentinel grammar step plus the 54 extra templated prompt tokens the
tool definition adds (631 vs 577); decode rate is unchanged by the grammar (151.1 vs 150.6 tok/s). Both variants
stopped on EOS at 39 tokens with the same reply.

Notes on the columns: `OnPrefillCompleted` fires after the first sample, so "prefill+first-sample" is not a pure
prefill time (it equals TTFT here to the 0.1 ms, which says the first `onToken` follows the first sample with no
measurable work between). TTFT is the first streamed token from request start; decode tok/s = (n − 1) / (t_last −
t_first). Request shape as measured: the tool JSON is rendered into the system-prompt TEXT and attached as
`ChatMessage.Tools` (which the template ignored at the time), with `JsonModeSentinel = "<tool_call>"` armed the
way `TextService` arms it. `main` (#193, merged after this measurement) now renders `GenerationRequest.Tools`
through the Jinja template natively; the probe deliberately keeps the measured shape so this row stays comparable,
and switching it to `GenerationRequest.Tools` is the next probe revision.

## Gates

From the plan's "Model bring-up gates" table, restricted to the rows PR3 evaluates; each compound gate is split
into one row per measured quantity so the Measured column holds one number. The plan's other rows (Qwen3 tool
template — PR6; Silero + RNNoise ≤ 2 ms per frame — PR8; Piper digest-identical — PR5) are checked in those PRs
and are not evaluated here. Evaluated on Phase 1 only; any miss stops the plan at that gate and the named task
lands before any PR that touches that model.

| Gate | Where checked | Measured (Phase 1) | Status |
|---|---|---|---|
| Whisper small.en per-utterance ≤ 350 ms on the 3060 | PR3 probe A, PR8 e2e | re-run 2026-10-01 (alpha.229): warm medians 93.7–185.9 ms, worst p95 191.5 ms, worst first call 222.1 ms (10 s, 16 kHz) | **met** — the first run was invalid (token-layout bug, fixed in #196); #204 brought the latency in |
| Whisper narrowband recall ≥ 16k baseline − 10 pts | PR3 probe C, PR8 e2e | re-run: 100 % vs 100 % (Δ 0), full 11 s clip | **met** (the first run was invalid, same bug) |
| Kokoro 15-word sentence ≤ 250 ms on the 3060 | PR3 probe B, PR8 e2e | re-run: warm median 156.6 ms (p95 170.5, min 141.6); first synthesis of the new text 238.2 ms | **met** — warm by 93 ms; the first run of new text is inside by only 12 ms (open item: new text costs +45–82 ms over a repeat). The first run (2788 ms, pre-#199, contended) stopped the plan until #199 |
| Kokoro Whisper-verify recall ≥ 80 % | PR3 probe B, PR8 e2e | re-run: 100 / 90 / 95 % at 5 / 15 / 30 words (misses are numerals: "three" → "3", "ten" → "10") | **met** |
| Qwen3-4B TTFT ≤ 150 ms with tools on (graph/spec decode off) on the 4090 | PR3 LLM probe | 152.0 ms median (p95 160.3, min 150.7), 631 prompt tokens | **met at the boundary** — the 152.0 ms median is 2.0 ms (1.3 %) over the 150 ms target, within run-to-run noise (p95 160.3); recorded as met by the plan owner, no perf task opened |
| Qwen3-4B ≥ 60 tok/s with tools on (graph/spec decode off) on the 4090 | PR3 LLM probe | 151.1 tok/s median (p95 152.6, min 150.1) | **met** |

**No PR3 gate is open.** Whisper and Kokoro meet their gates on the 2026-10-01 re-run, after their stop-gate
bring-ups merged (#196 + #204 for Whisper, #199 for Kokoro), and Qwen3 meets both of its gates (TTFT at the
boundary). The stack composition did not change to get there. RNNoise is not a PR3 gate; Probe C's with/without
rows are data for the open front-end decision.
