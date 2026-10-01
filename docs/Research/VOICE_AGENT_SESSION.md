# Voice agent session (`HartsyInference.Voice`)

> Source snapshot: 2026-09-30. This date does not establish current build or verification status.

## Summary

One phone call's voice agent: caller audio in (16 kHz PCM), spoken replies out, with endpointing, barge-in, tool calls
and per-turn latency metrics. `VoiceModelSet` holds the models every call shares for the host's lifetime;
`VoiceAgentSession` is one call. The package is opt-in (references Audio, Engine and Tools; not in the meta package),
and its only language-model dependency is `ITextService`, so an in-process engine and a remote adapter both work.
Telephony tools (hang up, DTMF, transfer) are registered by the host; the session only dispatches through the
`ToolRegistry` it is given.

`VoiceModelSet.WarmAsync` prepares every model before the first call:

- When the caller passes the host's tool definitions, it also runs `WarmToolMaxTokens` (8) real tokens with tools
  offered, through `ITextService.StreamAsync` rather than `GenerateAsync`, so the tool-call grammar sampler, the
  `ToolCallStreamFilter`/`ToolCallParser` it drives, the Jinja template's tools branch and `WarmToolMaxTokens` real
  decode steps are all hot before the first caller. Without tools (the default; also what a session with none
  registered gets) the request is unchanged from before: one token, no grammar. The warm-up's own `Messages` never
  touch a session's conversation or the sentence splitter, so it changes nothing about what the first real turn
  generates. This is what the Qwen3-4B [measurement](#qwen3-4b-on-the-rtx-4090-audio-on-the-3060-both-cards-visible)
  below is re-measured against; `VoiceHost` (PR9) must be updated to pass its own tool set at boot, or production
  warm-up stays on the cold one-token path.
- It synthesizes five texts, each as its own GPU job: "Okay.", a 3-word, a 6-word, a 13-word and a 30-word sentence.
- On the RTX 3060 these gave 58, 70, 81, 195 and 407 of Kokoro's 25 ms alignment frames. That is the power-of-two
  buckets 64, 128, 128, 256 and 512, every bucket a sentence reaches:
  - even "Okay." carries about a second of edge audio, so nothing reaches the 32 bucket;
  - `MaxSentenceChars` keeps a sentence within the 512 bucket.
- Kokoro chooses its convolution plans per length bucket (alpha.232), so the first sentence of a call finds its bucket
  already built. The 256 bucket holds the 15-word sentences.
- It recognizes a second of silence on the GPU thread, and generates one token on the language model's device.
- It logs one `[Voice] Warm-up` line with each job's time and each synthesis's audio length. On the 3060 the
  syntheses took 595.2 / 122.7 / 94.9 / 229.0 / 363.3 ms. The first carries Kokoro's one-time first-call cost. The
  recognition took 287.7 ms.

## Threads and locks

```
PushInbound (host, one thread) ─► SpscRing<float> 30 s ─► T1 voice-audio (dedicated thread, CpuParallel inline, private CpuBackend)
                                                           RNNoise (×32768 in, ÷32768 out) → Silero 512-sample windows → capture ring
                                                           endpoint ─► turn channel          barge-in ─► flush epoch, turn CancelAsync
T3 turn loop (async, one turn at a time)
   utterance ─► T2: Whisper lease ─► VoiceConversation (token-trimmed) ─► ToolLoop (thinking off, Device = LlmDevice)
   deltas ─► SentenceChunkedSynthesis (run = post to T2, 2 in flight) ─► StreamingResampler ─► VoiceOutbound
T2 voice-gpu (dedicated thread, owned by VoiceModelSet): one DeviceGate hold per job, FreeActivations(trimPool: false)
   per job, TrimMemoryPool queued once per turn on the return to listening, skipped if the next turn's work is behind it
ReadOutbound (host, one thread) ◄─ VoiceOutbound: SpscRing<float> 30 s, applies flushes, zero-fills, never blocks
Events: channel ─► one pump task on the pool (never T1 or T2), in order
```

Lock order: the session's state lock (never held across an await) → the event channel, the turn channel and the
rings, which are leaves. T1 never takes the state lock: its per-frame path touches only the inbound ring (lock-free)
and its doorbell, and it hands off through the turn channel once per utterance. The device gate is taken only on T2,
for one job, never across an await and never together with the language model's device, so the ascending-ordinal rule
of `DeviceGate.AcquireAll` cannot be broken from here. `AudioRuntime`'s generation lock is never touched: the leases
run their runners directly.

The two long-lived tasks (turn loop, event pump) are single loops, not fan-out; CPU fan-out happens only inside
`CpuParallel`, and T1 opts out of it for its whole life.

## Scale and rates

Inbound ±1 at 16 kHz → RNNoise at int16 scale (its silence floor and log offsets are absolute; ±1 audio sits under the
floor and passes through untouched) → Silero ±1 → Whisper ±1 → synthesizer ±1 at its own rate → `StreamingResampler`
in 20 ms frames → outbound ±1 at `OutboundSampleRate`. The recognizer is asked for English (`Language = "en"`): an
English-only checkpoint drops the language token itself, while an empty language on a multilingual checkpoint means
no language token at all, which Whisper tiny answered with `S-N-S-N-…` loops on JFK.

## Endpointing and barge-in

- One clock: samples pushed through the VAD. The capture ring, the hangover, the hold-off and the barge-in run are all
  counted in it, so decisions do not depend on how fast audio arrives. With `Denoise` on (the default), RNNoise sits
  ahead of the VAD in the per-frame pipeline and its output lags its input by `RnnoiseStream.LatencySamples` (640
  samples, 40 ms at 16 kHz: a 2-frame resampler round trip to RNNoise's native 48 kHz plus one analysis window). The
  clock counts samples at the VAD, after that lag, so every figure below measured against it — the 736 ms endpoint
  hangover, the barge-in stop time — is 40 ms later in wall-clock terms than the sample count alone suggests whenever
  denoising runs; nothing in the pipeline subtracts it back out today.
- `SileroVadStream(minSpeechMs 250, minSilenceMs EndOfTurnSilenceMs, speechPadMs 30)`: the segment it closes is the
  endpoint. With 512-sample windows the decision lands 736 ms after the end of speech for the 700 ms default (plus
  RNNoise's 40 ms when denoising). A segment still open at `MaxUtteranceMs` is flushed and answered.
- Barge-in: while a reply is audible and past `BargeInHoldoffMs` (counted from the first window the audio thread sees
  the reply), `BargeInMinMs` of consecutive windows at `BargeInProbability` or above. One compare-and-swap on the
  speaking turn decides it, so a reply that finished playing cannot be flushed and a barge-in cannot be lost to a
  finishing turn. The turn is cancelled with `CancelAsync`, which flips the token at once and runs its callbacks on
  the pool. Its cancellation source is never disposed, because the audio thread may cancel it while the turn ends.
- **Rule added here (accepted by the orchestrator):** an utterance whose speech both began and ended while the reply
  was audible, and that never became a barge-in, is not answered. It is backchannel or the reply's own echo, and
  answering it would talk over the reply with a reply to itself. Both ends are judged when the speech happens, not
  when the endpoint is decided a hangover later. The reply counts as audible for `BargeInHoldoffMs` after it leaves,
  because the echo of its last words arrives late. Speech that began before the reply (a caller talked over by a
  prompt) or outlasted it is answered, as is the interrupting speech of a real barge-in. Discards are counted
  (`DiscardedUtterances`, `UtteranceDiscarded` event), and so is an utterance the recognizer hears no words in.
- History: two user or two plain assistant messages in a row are merged into one (a turn cut off before the model
  said anything, two spoken prompts back to back), because templates that enforce alternating turns reject them.

## Outbound queue and flushes

`SpscRing.DiscardAll` is consumer-only, so a flush is a request: T1 bumps `FlushEpoch`, and `ReadOutbound` discards and
publishes the epoch it applied. Two rules keep a flush from eating the wrong audio:

- The producer re-checks the epoch after each write and, if it moved, bumps it again, so a frame that landed after the
  reader's discard is dropped too.
- A new turn never starts writing until the reader has applied every pending flush, so a late discard cannot take the
  next reply's opening.

A concurrent-flush stress test holds the reader to never hearing an older turn after a newer one. The producer waits
for space rather than dropping.

The reader never blocks or allocates, also while the producer waits:

- The waiter carries what it waits for: a played position, free space, or an applied flush epoch. The reader completes
  it once, on the read that reaches it. A turn waiting for its playback costs the reader nothing per read.
- The producer awaits the waiter's task directly, and the cancellation token completes the waiter through
  `UnsafeRegister`. The producer's own continuation is then the only one, so the reader's single completion queues it
  and allocates nothing. Awaiting `Task.WaitAsync(token)` instead cost the reader 32 B per read: a new invoker for the
  cancellation promise on every wake.
- A writer that finds the queue full waits for its remaining write or a quarter of the ring, whichever is smaller. A
  full queue then costs one wake per quarter rather than one per read.
- Measured on a dedicated reader thread (tests), the reader allocates 0 B:
  - over 1000 reads with a cancellable playback wait pending, and on the read that wakes it;
  - over 1000 reads while a writer waits for space (250 wakes);
  - while the reader plays a whole turn.
- The one-time cost is per thread, and it is the runtime's: the first continuation a thread queues to the pool
  allocates once on that thread (32 B, or 192 B on a freshly started thread). The sender thread pays it on its first
  wake and never again.

Turn tags: a player that forwards audio elsewhere (the voice host, over PhoneLink) reads with
`ReadOutbound(destination, out turnId)`, which returns one turn's samples at most and names the turn that wrote them.
Before a turn's first write it publishes a mark (turn id and write position); the reader takes the ring's fill level
before it reads the marks, so every sample it returns is covered by a mark it has seen, and it stops at the next mark.
The flush rules above leave one window: a write that lands after the reader's discard can be read before the
producer's re-check bumps the epoch again. Tagged, those samples carry the cancelled turn's id, so a remote player drops
them by id; the untagged `ReadOutbound(destination)` reads across marks as before.

## Inbound backlog

The ring drops the newest samples when full, so the audio thread enforces drop-oldest itself: when it is more than
30 s behind it discards the oldest excess, resets the front-end and counts it (`InboundDroppedSamples`); the turn loop logs it, since the audio thread does no logging.

## Lease revocation

Any engine release (dispose, free memory, backend switch) revokes the runner leases; the next call throws
`ObjectDisposedException`. The GPU thread then reopens both leases once, outside the gate (opening takes the gate
itself), and retries the job once. If that fails the job fails, the turn ends with an `Error` event and the session
keeps listening.

## Options

| Option | Default | Scope |
|---|---|---|
| `LlmModel`, `LlmDevice` | `qwen3`, `cuda:0` | session; every request carries `Device` (an engine on the audio card would otherwise load the LLM there) |
| `AudioDevice`, `SttModel`, `TtsModel` | `cuda:1`, `whisper:openai/whisper-small.en`, `kokoro:af_heart` | model set; the engine must be built on `AudioDevice` |
| `OutboundSampleRate` | 16000 | session |
| `EndOfTurnSilenceMs`, `MaxUtteranceMs` | 700, 15000 | session |
| `BargeInEnabled`, `BargeInProbability`, `BargeInMinMs`, `BargeInHoldoffMs` | true, 0.6, 200, 300 | session |
| `Denoise` | **true** | model set; loads RNNoise at int8 (the front-end gate was only met at that precision — see [Measured](#measured)); fails at load when either the F32 weights or the int8 tables beside them are missing, never a passthrough. Voice only: the wake stack's own `WakeModelSet.LoadDenoiser()` call always stays Float. Adds 640 samples (40 ms) of algorithmic delay ahead of both endpointing and barge-in detection (`RnnoiseStream.LatencySamples`), which the sample-clock figures below do not yet subtract out |
| `SystemPrompt`, `MaxToolRoundsPerTurn`, `MaxHistoryTokens`, `MaxReplyTokens` | phone prompt, 4, 3000, 200 | session |
| `FirstSentenceMinChars`, `MaxSentenceChars` | 12, 180 | session |
| `CpuThreadCap` | 0 | model set; sets `numerics.cpuThreads` while loaded and restores the previous value on dispose |
| `PartialTranscripts` | false | `true` is rejected until a streaming recognizer is wired |

## Metrics

One `[Voice] turn N (kind): …` line per turn, every key always present (`-` when the stage did not run):
`voice.frontend.ms.{p50,p99,max}` (denoise plus VAD per 20 ms frame since the previous endpoint; percentiles are
`LatencyHistogram` bucket bounds), `voice.endpoint.ms` (sample clock, plus the denoiser's algorithmic lag —
`VoiceAudioFrontend.DenoiserLatencySamples`, 0 when `Denoise` is off — so the figure is wall-clock honest instead of
40 ms short whenever denoising runs), `voice.stt.ms`, `voice.llm.ttft_ms`,
`voice.llm.first_sentence_ms`, `voice.tts.first_chunk_ms`, `voice.transport.ms`, `voice.turn.total_ms` (sample-clock
hangover plus the wall time from the endpoint to the first reply audio queued) and `voice.bargein.stop_ms` (decision
to the reader's discard). The gateway's stages (`voice.rtp.*`) are measured by the gateway.

## Measured

| Stage | Where | Result |
|---|---|---|
| Silero per 20 ms frame, audio-thread path | i7-6900K, CPU, JFK | mean 0.57 ms, p99 1.68 ms, max 1.75 ms; 749 managed bytes per frame, all from the CPU kernels' dispatch closures (the session's own per-frame path allocates 0 bytes over 1000 frames) |
| RNNoise + Silero per 20 ms frame, F32 (historical) | same, weights from the in-flight RNNoise bring-up | mean 6.2 ms, p99 20.7 ms, max 30.8 ms; 101.6 KB managed per frame. **Missed the original 2 ms back-to-back gate.** The 960-point FFTs took `Fft`'s Bluestein path, which allocates two 2048-float arrays per transform and locks the plan cache per call |
| RNNoise + Silero per 20 ms frame, **int8 (current, `Denoise` default)** | i7-6900K, CPU 0, `schedutil`, spinning 20 ms cadence (the redefined gate; see `CHANGELOG.md` alpha.234) | quiet p50 1.07-1.08 ms, p99 1.20-1.59 ms (gate: p50 ≤ 3, p99 ≤ 5 ms — **met**); 4 streaming threads p50 1.80-2.11, p99 3.56-4.25 ms, none late. `TurnEndpointerRealVadTests.RnnoiseAndSileroSplitJfkWithinTheFrameBudget` (this package, back-to-back over the JFK clip, no live-cadence pacing or thread contention — that variant is the Audio package's int8 bench above) confirms the wiring: p50/p99 well inside the same budget. The wake stack's own loader stays Float; only the voice front end asks for int8 |
| JFK endpointing, 700 ms | Silero | 3 utterances (0.32-2.27, 3.27-4.45, 5.38-11.04 s), hangover 736 ms each |
| Whisper tiny through the session, CPU | JFK | content-word recall 100 %; narrowband (16k→8k→16k, gateway resamplers) 91 %; narrowband through RNNoise 91 % |

On the RTX 3060 (`CUDA_VISIBLE_DEVICES=1`, after a 10-minute Swarm quiet window, under the bench lock),
`VoiceSessionEndToEndTests` gave the rows below **with `Denoise` off** (its default at the time). That build had
Kokoro's length-bucketed convolution plans (alpha.232), the bucket warm-up, and per-job frees that keep the memory
pool. The language model is scripted, so the LLM stages are not measured here. `Denoise` now defaults on: RNNoise's
640-sample (40 ms) lag sits ahead of the endpoint and barge-in decisions these rows measure, so the endpoint hangover
and everything downstream of it are about 40 ms later with denoising on than the `voice.endpoint.ms` row below shows;
a rerun with `Denoise` on is the next entry below once taken.

| Stage | Budget | Result |
|---|---|---|
| Whisper small.en per utterance (`voice.stt.ms`) | ≤ 350 ms | 139 / 89 / 156 ms for JFK's three utterances (1.9 / 1.2 / 5.7 s) |
| Kokoro af_heart, 15-word sentence on the GPU thread | ≤ 250 ms | median 195 ms over five runs (198 ms the first time that text was synthesized, then 156-228 ms) |
| Kokoro first sentence of a turn (`voice.tts.first_chunk_ms`) | ≤ 250 ms | 241 ms for an 18-word sentence synthesized for the first time; 78 ms, then 58 ms, for "Okay." |
| Endpoint hangover (`voice.endpoint.ms`) | 700 ms (tune 500-800) | 736 ms: 700 ms of silence, the 30 ms pad and 32 ms Silero windows |
| Resample and queue (`voice.transport.ms`) | ≤ 50 ms | 9.7 / 0.5 / 0.2 ms |
| Turn total, end of speech to first reply audio queued (`voice.turn.total_ms`) | ≤ 1.3 s, stretch 1.0 s | turn 1: 1195 ms = endpoint 736 + STT 139 + scripted LLM 17 + Kokoro 241 + transport 10, plus 52 ms of turn overhead; 1179 ms without the LLM, which leaves 121 ms for a real model's first sentence within 1.3 s. The endpoint hangover dominates; tuning it within 500-800 ms is a product decision, and the default stays 700 ms. Turns 2 and 3 (8047 and 9626 ms) waited behind the reply before them, because the test pushes the whole clip at once |
| Front-end per 20 ms frame during the call, Silero only (`voice.frontend.ms`) | ≤ 2 ms | p99 bucket ≤ 2 ms in every turn; max 1.5 / 1.2 / 0.8 ms. The test pushes the whole clip at once, so the audio thread works through later turns' frames while earlier turns' recognition and synthesis run; the serial gate is the CPU row above |
| Whisper-verify, both directions | ≥ 80 % | caller (JFK) 11/11 content words; reply 8/9 ("tomorrow" heard as "row") |

The run before (pool trimmed after every job, plans chosen per exact length) put a turn's first sentence at 261 ms,
over the 250 ms budget, and the 15-word median at 187 ms with its first synthesis at 254 ms.

The other two classes ran on the build before the rebase onto the Whisper and Kokoro speed-ups. `BargeInEndToEndTests`:
the reader read no reply audio after the barge-in decision (0.0 ms) and applied the flush 4.1 ms after it (gate 100
ms). `PinnedRunnersSurviveEvictionTests`: the pinned Whisper small.en and Kokoro leases survived three
memory-pressure switches with no reopen, and each transcription stayed word-perfect.

### Re-run with `Denoise` on (int8), RTX 3060

Same test, same card, after a clean 10+-minute Swarm quiet window verified clean afterwards (`--verify-since`),
under the bench lock, one run. `VoiceSessionEndToEndTests` passed; turn 1 (the only turn not waiting behind a
previous reply's playout):

| Stage | Budget | Result |
|---|---|---|
| `voice.frontend.ms` (RNNoise int8 + Silero, live call) | ≤ 2-5 ms (p50/p99) | p50 1.00, p99 2.00, max 1.11 ms |
| Whisper small.en (`voice.stt.ms`) | ≤ 350 ms | 133.26 ms |
| Kokoro 15-word sentence, median of 5 | ≤ 250 ms | 195.4 ms (195.7, 154.7, 170.5, 200.6, 195.4) |
| `voice.tts.first_chunk_ms` | ≤ 250 ms | 259.35 ms (first synthesis of the turn's full reply sentence — the open Kokoro first-synthesis item, not a regression from denoising) |
| `voice.endpoint.ms` | 700 ms (tune 500-800) | 736.00 ms — **unchanged from Denoise off.** This metric counts samples at the VAD, and RNNoise's 640-sample delay shifts every sample's wall-clock arrival uniformly, so a sample-counted interval (close decision minus last-speech sample) is insensitive to it. The 40 ms is real but invisible here: it is paid once, before the 736 ms count even starts, so the caller's true wall-clock wait is nearer 776 ms. |
| `voice.transport.ms` | ≤ 50 ms | 5.92 ms |
| `voice.turn.total_ms` | ≤ 1.3 s, stretch 1.0 s | **1209.39 ms** (endpoint 736 + STT 133.26 + scripted LLM 26.71 + Kokoro 259.35 + transport 5.92, plus ~47 ms overhead). Without the scripted LLM: 1182.68 ms, leaving **~117 ms** for a real model's first sentence within 1.3 s (was 121 ms with `Denoise` off — the ~4 ms narrowing is noise at this precision, not the full 40 ms, consistent with the metric's blind spot above) |
| Whisper-verify, both directions | ≥ 80 % | caller 11/11; reply 8/9 ("tomorrow" heard as "row", same known gap) |

Warm-up: 585.5 / 144.2 / 98.4 / 239.2 / 314.9 ms for the five bucket texts, recognized 1 s of silence in 275.9 ms —
consistent with the pre-int8 numbers; RNNoise adds no measurable warm-up cost of its own.

### Qwen3-4B on the RTX 4090, audio on the 3060 (both cards visible)

`VoiceSessionQwen3EndToEndTests`, `HARTSY_VOICE_LLM_GPU=1`, after a clean quiet window on **both** cards, under the
bench lock. `qwen3` now resolves to `/mnt/model-storage/Models/llm/qwen3/Qwen3-4B-Q4_K_M.gguf` (the model-folder-case
fix, #205, landed on main since this was last tried). Warm-up (speech on the 3060 and one Qwen3 token on the 4090, in
parallel, including the cold GGUF load): 10.0 s. Turn 1, the JFK clip's first utterance, tools installed:

| Stage | Budget | Result |
|---|---|---|
| `voice.frontend.ms` | ≤ 2-5 ms | p50 1.00, p99 5.00, max 2.46 ms (at the p99 ceiling on this single live call; still inside budget) |
| `voice.stt.ms` | ≤ 350 ms | 117.09 ms |
| `voice.llm.ttft_ms` | — (plan's isolated probe: ~150 ms) | **357.36 ms** |
| `voice.llm.first_sentence_ms` | — | **949.65 ms** (one short sentence, "Hello, how can I assist you today?") |
| `voice.tts.first_chunk_ms` | ≤ 250 ms | 123.97 ms |
| `voice.transport.ms` | ≤ 50 ms | 14.52 ms |
| `voice.turn.total_ms` | ≤ 1.3 s, stretch 1.0 s | **1974.72 ms — over budget**, almost entirely from `llm.first_sentence_ms` (everything else sums to ≈1025 ms, already most of the 1.3 s on its own) |
| Caller recall ("fellow Americans") | — | 100 % |
| Reply recall | — | 100 % ("Hello, how can I assist you today?" heard back verbatim) |

This is the first live inference call right after the cold warm-up, with tool calling installed and the real system
prompt and history — not the plan's isolated, pre-warmed TTFT/throughput probe (152 ms TTFT, 151 tok/s, 500-token
synthetic prompt, no tools). Reported as measured; endpoint and LLM-side tuning are the orchestrator's call, per the
plan's decision log.

#### Re-measured after the `WarmAsync` fix, 4 turns, both cards visible

Root cause of the row above: `WarmAsync` warmed the LLM with a one-token, no-tools request, so the grammar sampler,
the stream filter/parser and every decode step past the first were cold on the real first turn. Fixed: `WarmAsync`
now takes the host's tool definitions and, when given any, runs `WarmToolMaxTokens` (8) tokens through `StreamAsync`
instead (see [above](#summary)). Re-run of `VoiceSessionQwen3EndToEndTests`, now driving 4 turns of the same
caller utterance back to back (history grows each turn), clean quiet window on both cards, verified clean after:

| Turn | `llm.ttft_ms` (≤ 150) | `llm.first_sentence_ms` (≤ 200) | `tts.first_chunk_ms` | `voice.endpoint.ms` | `turn.total_ms` (≤ 1.3 s) |
|---|---:|---:|---:|---:|---:|
| 1 (cold) | 65.2 | 291.3 | 142.3 | 776.0 | 1366.3 |
| 2 | 77.1 | 308.5 | 104.4 | 776.0 | 1295.2 |
| 3 | 71.7 | 307.2 | 95.5 | 776.0 | 1284.7 |
| 4 | 77.7 | 318.7 | 105.1 | 776.0 | 1301.4 |

- **TTFT 357 → 65-78 ms, meets budget on every turn, cold or warm.** `CudaBackend.LtGemmPlanStats` read `(0,0,0,0)`
  before and after every turn: this GGUF-quantized model's decode never reaches cuBLASLt's fused-epilogue GEMM path,
  so plan-cache priming is not the mechanism — the fix is warming the real request shape (206 templated tokens vs.
  the old request's ~20), the grammar sampler and real decode steps together; nothing here isolates which of the
  three the remaining credit belongs to.
- **`voice.endpoint.ms` reads 776.00 on every turn = 736 + RNNoise's 40 ms lag**, confirming the endpoint-metric fix
  below in a real session with `Denoise` on (the CPU unit test can only prove the arithmetic in isolation: a real
  RNNoise instance ahead of the harness's level-scripted fake VAD suppresses the fake "speech" as noise, so that
  combination cannot drive a turn at all).
- **`turn.total_ms`**: turns 2 and 3 meet the 1.3 s budget; turn 4 misses by 1.4 ms (noise); turn 1 misses by 66 ms,
  all of it `tts.first_chunk_ms` (142 ms — Kokoro's first-synthesis-of-new-text cost, a known item, not an LLM or
  warm-up residue: turn 1's LLM numbers are indistinguishable from turns 2-4's).
- **`llm.first_sentence_ms` still missed its 200 ms budget by 90-120 ms, every turn, cold or warm — not a warm-up
  problem.** The reply is one sentence ("Hello, how can I assist you today?"); `StreamingSentenceSplitter` cannot
  know a sentence is complete until the stream ends (nothing later confirms the boundary), so for a one-sentence
  reply `first_sentence` = TTFT + the full decode + its EOS step. Turn 1's per-token timeline confirmed it exactly:
  9 tokens roughly 25 ms apart (65 TTFT + 8×25 decode + 25 for the EOS step ≈ 290 ms, measured 291.34); turns 3-4
  were not timelined per token, only by their aggregate rate (37-38 tok/s), consistent with the same shape. The
  lever was decode throughput: the session streamed at **38-40 tok/s**, steady spacing, no bursts (nothing was held
  by the tool-call filter — confirmed separately: Hermes markers are `<tool_call>` / `{"name"` / line-start `{`, so
  a reply starting "Hello" matches none of them and is forwarded a character at a time, see
  `ToolCallParserTests.PlainTextIsForwardedUnchangedWithoutCopying`). The plan's own isolated probe
  (`VoiceLlmTurnBenchTests`, `benchmarks/results/2026-09-30_voice_turn_latency.md`) measured **151.1 tok/s with the
  same sentinel grammar armed** (tools-on vs. tools-off differ by 0.5 tok/s there, ruling out the grammar step as
  the cost) — but that probe samples greedily, and the voice session's `TextRequest` defaults to
  `Temperature=0.7, TopP=0.95, Greedy=false`. **Root cause, found and fixed on `main`** (alpha.236,
  `perf/llm-short-reply-decode`, #215): `TopPStep`'s nucleus filter argsorted the full ~152K-token vocabulary
  through a `Comparison<int>` delegate and allocated three vocab-sized arrays, every decode step, only on that
  non-greedy path — not session-path transport (detokenization, the filter/parser, the channel), which is what the
  per-token timeline's steady spacing had pointed at. Fixed there: reused scratch buffers, a delegate-free sort,
  sorting only the candidate subset when it already reaches `p`.
- **Open for PR9**: `VoiceHost`'s boot-time `WarmAsync` call must pass its own tool set, or production warm-up
  stays on the cold one-token path this fix only helps when a caller opts in.

#### Re-measured after the sampler fix (main alpha.236, #215), same 4 turns

Rebased onto `main` 553de075 and re-ran the same 4-turn scenario, same gates (quiet window on both cards, VRAM
wait-loop, bench lock, `--verify-since` clean after):

| Turn | `llm.ttft_ms` (≤ 150) | `llm.first_sentence_ms` (≤ 200) | decode tok/s | `tts.first_chunk_ms` | `voice.endpoint.ms` | `turn.total_ms` (≤ 1.3 s) |
|---|---:|---:|---:|---:|---:|---:|
| 1 (cold) | 63.7 | 152.2 | 101.5 | 132.8 | 776.0 | 1226.6 |
| 2 | 46.4 | 120.3 | 121.7 | 91.7 | 776.0 | 1092.4 |
| 3 | 62.4 | 148.7 | 102.5 | 98.4 | 776.0 | 1124.0 |
| 4 | 81.9 | 172.1 | 99.8 | 81.4 | 776.0 | 1133.5 |

**Every stage met its budget on every turn, cold or warm, this run** (one run, as the plan's GPU rules keep 4090
time short and few; the margin is tightest on turn 1's total, 73 ms, and turn 4's first sentence, 28 ms — a noisier
run could cross either). Requires the engine at alpha.236 or later (#215's sampler fix); below that, the real
first-sentence number is the ~290 ms the first re-measurement above found, regardless of this PR. TTFT 357 → 46-82
ms (unchanged from the warm-up fix,
as expected — the sampler fix is a decode-step change, not a prefill one). `first_sentence` 950 → 120-172 ms,
inside its 200 ms budget for the first time, consistent with decode at 99.8-121.7 tok/s (was 37-40) — close to the
sampler fix's own cited 94-106 tok/s for a realistic `ToolLoop` turn. `turn.total` 1975 → 1093-1227 ms, inside its
1.3 s budget on every turn (turn 1's 1226.6 ms is the highest, still 73 ms under). `voice.endpoint.ms` reads
776.00 again, unaffected (a decode-rate fix has nothing to do with the front end). Recall 100 % both directions on
turn 1, no `<think>` text in any turn. `CudaBackend.LtGemmPlanStats` stayed `(0,0,0,0)` through this run too,
confirming it was never going to show either the warm-up or the sampler effect for this model's quantized GEMM
path.

## Voice host (`HartsyInference.VoiceHost`)

The phone deployment's model process: a generic-host exe (net10.0, not packed) that the phone gateway dials over
[PhoneLink](PHONE_LINK_PROTOCOL.md). At start it builds `InferenceEngine` on `AudioDevice` (`cuda:1`, the RTX 3060)
with `ToolCalling.Install` (format detected from `LlmModel`), loads and warms the `VoiceModelSet`, then listens on the
socket. Every call gets its own `VoiceAgentSession` on that model set; the models outlive calls. The engine reaches the
language model only through each request's `Device = LlmDevice` (`cuda:0`, the 4090): a request without it would load
the LLM onto the audio card, and the session never sends one. Units and install steps: [deploy](../../deploy/README.md);
end-to-end checks: [runbook](../Checklists/VOICE_AGENT_VERIFICATION.md).

```
voice-link-accept   accept() only; each connection's handshake runs on its own reader
voice-link-reader   per connection: Hello (first frame, version 1, 16 kHz, token compared in constant time on SHA-256
                    digests) → HelloAck or Error + close; then every frame, sequence checked (a gap closes the link):
                    CallStart → session (started on the pool) · InboundAudio PCM16 → ±1 (×1/32768) → PushInbound ·
                    DtmfEvent → PushDtmf · ToolResult → pending request · Ping → Pong · CallEnd → end the session
voice-link-sender   per connection, the only writer after the handshake; absolute 20 ms deadlines (MonotonicClock),
                    no spin, no FIFO; per tick: queued control frames first, then each call's audio:
                    ReadOutbound(out turnId), one turn and at most 20 ms per frame → OutboundAudio(turnId)
pool                each session's turn loop and event pump; session events → Event / Flush / OutboundEnd items;
                    telephony tools awaiting the gateway; call start and end
watchdog timer      nothing received for 20 s, or one write stuck for 20 s → close the link
```

Locks: the server's connection lock and a connection's call-map lock are leaves (nothing is called while holding
them); a call's lifecycle lock may take the process-wide GC-mode lock, a leaf. The control queue is a
`ConcurrentQueue` drained by the sender. The sender reads its call list as a published array, so its audio path takes
no lock and allocates nothing once warm (`AudioPathAllocatedBytes`, asserted 0 over 1200 frames).

Outbound pacing: one 20 ms frame per tick, plus 40 ms (`link.prebufferMs`) at the start of each burst of reply audio,
because the gateway plays what it has at its own 20 ms tick and has no cushion of its own; a late wake-up sends the
missed frames at once (up to five), a later one re-bases the clock. The session therefore counts audio as played about
one prebuffer ahead of the caller's ear.

Barge-in across the socket: the session's `BargeIn(T)` event becomes `Flush(T)`, written first on the sender's next
tick; from then on the sender drops any audio tagged T or lower that the session still hands it (the window in
[Outbound queue and flushes](#outbound-queue-and-flushes)), and the gateway drops anything at or below T that was
already on the socket. `OutboundEnd(T)` goes out before the next turn's first frame, or once the session reports T
finished and nothing of it is left, because the gateway's 16k→8k resampler holds T's last frame until then; a flushed
turn gets none.

Telephony tools are per call, each an `IToolHandler` bound to that call: `send_dtmf`, `transfer`, `hold`, `unhold` and
`play_prompt` (`name` only: `one-moment`, `goodbye`; the gateway's `file` form is not offered to the model) send
`ToolRequest` and return the gateway's `ToolResult` as `{"status","message"}`, or `Failed` after `tools.timeoutMs` or at
the end of the call. `get_time` is answered on the host. **`hangup` is deferred:** the gateway sends its BYE the moment
it runs the tool, while the model's goodbye is usually still being synthesized, so the handler answers `Ok` at once and
the host sends the request when that turn's reply has finished playing, then `CallEnd(Completed)`. A caller who barges
into the goodbye keeps the call.

Events: state changes, final transcripts and turn latencies go to the gateway as `Event` frames. The gateway logs every
host event at Info, so its journal holds what callers said; treat it like a recording (see the gateway's consent note).

Calls: `CallStart` with `resume=true` (the gateway re-attaching after the link dropped) gets a fresh session and
`agent.resumeApology` spoken; a new call gets `agent.greeting` when set. `CallEnd` from the gateway ends and disposes the
session without echoing. A lost link ends that connection's sessions (no `CallEnd`: the gateway re-announces live calls).
Any session failure (creation, start, its audio thread, a throw on a link thread) ends only that call with
`CallEnd(Failed)`. SIGTERM ends every call with `CallEnd(LocalHangup)`, lets the sender put those frames on the wire,
closes the link, releases the models and removes the socket file. A socket file left by a killed host is removed at
start after a connect probe finds nobody listening; a live listener there fails the start.

Process: workstation concurrent GC; `GCLatencyMode.SustainedLowLatency` while at least one call is up, restored after
the last. The pool's minimum worker threads are `numerics.cpuThreads + 8`: `CpuParallel` fans out on the shared pool, up
to `numerics.cpuThreads` workers at once across all callers, so a synthesis or a decode can hold that many; the turn
loops, event pumps, tool calls, GPU job continuations and socket completions need a few more runnable at the same
moment, and above the minimum the pool adds threads too slowly for a live turn.

`/etc/hartsyinference/voice.json` (template `src/HartsyInference.VoiceHost/voice.example.json`; `{}` is valid; an
unknown key fails the start):

| Section | Settings |
|---|---|
| `link` | `socketPath` (`/run/hartsyinference/phone.sock`), `socketMode` (0600-0660), `tokenFile` (absolute, 0600/0400, a `LoadCredential=` path; empty = no token, which the gateway must match), `prebufferMs` (40), `livenessTimeoutSeconds` (20) |
| `models` | `llmModel`, `llmDevice`, `audioDevice`, `sttModel`, `ttsModel`, `denoise`, `wakeModelRoot`; defaults are the session's |
| `agent` | `systemPrompt`, `greeting`, `resumeApology`, `outboundSampleRate` (a PhoneLink rate), and the session's turn, barge-in and history settings |
| `tools` | `enabled` (all seven by default), `timeoutMs` (10000) |
| `engine` | `cpuThreadCap` (`numerics.cpuThreads` for the host's life; 14 under the unit), `settingsFile` (an engine settings file instead of the service user's) |
| `logging` | `level` |

## Open

- Partial transcripts need a streaming recognizer; `PartialTranscripts = true` is rejected.
- After a barge-in the history keeps the text generated before the cut, flagged `Interrupted` in the transcript, not
  the part the caller actually heard (that needs sentence-to-played-sample bookkeeping).
- DTMF during a reply is queued behind it rather than interrupting it.
- No hallucination filter beyond "only VAD-closed segments reach the recognizer" and the no-words discard.
- The CPU kernels' per-call dispatch closures make real Silero allocate on the audio thread; fixing them is a
  cross-model Cpu change with its own A/B.
- `VoiceHost` (PR9)'s boot-time `WarmAsync` call needs updating to pass its own tool set, or production warm-up
  stays on the cold one-token path.
- Host: outbound calls start like inbound ones (same prompt, same greeting); per-call instructions (why the agent is
  calling) are not wired. The host keeps one gateway connection; per-call and per-link summaries go to the log only (no
  metrics endpoint on the host side).
