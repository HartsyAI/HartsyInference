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

- It synthesizes five texts, each as its own GPU job: "Okay.", a 3-word, a 6-word, a 13-word and a 30-word sentence.
  They are one per power-of-two bucket of Kokoro's 25 ms alignment frames, from 32 to 512, estimated at about 15
  frames a word.
- Kokoro builds its convolution plans per length, or per length bucket once the engine buckets them. The first
  sentence of a call therefore finds its bucket already built; the 256 bucket holds the 15-word sentences.
- It recognizes a second of silence on the GPU thread, and generates one token on the language model's device.

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
  counted in it, so decisions do not depend on how fast audio arrives.
- `SileroVadStream(minSpeechMs 250, minSilenceMs EndOfTurnSilenceMs, speechPadMs 30)`: the segment it closes is the
  endpoint. With 512-sample windows the decision lands 736 ms after the end of speech for the 700 ms default. A segment
  still open at `MaxUtteranceMs` is flushed and answered.
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
| `Denoise` | false | model set; true fails at load when RNNoise weights are missing, never a passthrough |
| `SystemPrompt`, `MaxToolRoundsPerTurn`, `MaxHistoryTokens`, `MaxReplyTokens` | phone prompt, 4, 3000, 200 | session |
| `FirstSentenceMinChars`, `MaxSentenceChars` | 12, 180 | session |
| `CpuThreadCap` | 0 | model set; sets `numerics.cpuThreads` while loaded and restores the previous value on dispose |
| `PartialTranscripts` | false | `true` is rejected until a streaming recognizer is wired |

## Metrics

One `[Voice] turn N (kind): …` line per turn, every key always present (`-` when the stage did not run):
`voice.frontend.ms.{p50,p99,max}` (denoise plus VAD per 20 ms frame since the previous endpoint; percentiles are
`LatencyHistogram` bucket bounds), `voice.endpoint.ms` (sample clock), `voice.stt.ms`, `voice.llm.ttft_ms`,
`voice.llm.first_sentence_ms`, `voice.tts.first_chunk_ms`, `voice.transport.ms`, `voice.turn.total_ms` (sample-clock
hangover plus the wall time from the endpoint to the first reply audio queued) and `voice.bargein.stop_ms` (decision
to the reader's discard). The gateway's stages (`voice.rtp.*`) are measured by the gateway.

## Measured

| Stage | Where | Result |
|---|---|---|
| Silero per 20 ms frame, audio-thread path | i7-6900K, CPU, JFK | mean 0.57 ms, p99 1.68 ms, max 1.75 ms; 749 managed bytes per frame, all from the CPU kernels' dispatch closures (the session's own per-frame path allocates 0 bytes over 1000 frames) |
| RNNoise + Silero per 20 ms frame | same, weights from the in-flight RNNoise bring-up | mean 6.2 ms, p99 20.7 ms, max 30.8 ms; 101.6 KB managed per frame. **Misses the 2 ms gate.** The 960-point FFTs take `Fft`'s Bluestein path, which allocates two 2048-float arrays per transform and locks the plan cache per call |
| JFK endpointing, 700 ms | Silero | 3 utterances (0.32-2.27, 3.27-4.45, 5.38-11.04 s), hangover 736 ms each |
| Whisper tiny through the session, CPU | JFK | content-word recall 100 %; narrowband (16k→8k→16k, gateway resamplers) 91 %; narrowband through RNNoise 91 % |

On the RTX 3060 (`CUDA_VISIBLE_DEVICES=1`, after a 10-minute Swarm quiet window, under the bench lock),
`VoiceSessionEndToEndTests` gave the rows below. The language model is scripted: the LLM stages are not measured here.
At that point every GPU job also trimmed the device's memory pool, through the parameterless `FreeActivations`. Since
then each job keeps the pool and only the turn trims it, so the next GPU run should show faster Whisper and Kokoro
calls than these.

| Stage | Budget | Result |
|---|---|---|
| Whisper small.en per utterance (`voice.stt.ms`) | ≤ 350 ms | 110 / 88 / 167 ms for JFK's three utterances (1.9 / 1.2 / 5.7 s) |
| Kokoro af_heart, 15-word sentence on the GPU thread | ≤ 250 ms | median 187 ms over five runs (254 ms the first time that text was synthesized, then 150-191 ms) |
| Kokoro first sentence of a turn (`voice.tts.first_chunk_ms`) | ≤ 250 ms | **Over budget:** 261 ms for an 18-word sentence synthesized for the first time; 131 ms, then 53 ms, for "Okay.". The first synthesis of a new text runs 60-80 ms slower than repeats; open with the Kokoro bring-up |
| Endpoint hangover (`voice.endpoint.ms`) | 700 ms (tune 500-800) | 736 ms: 700 ms of silence, the 30 ms pad and 32 ms Silero windows |
| Resample and queue (`voice.transport.ms`) | ≤ 50 ms | 14 / 0.4 / 2.0 ms |
| Turn total, end of speech to first reply audio queued (`voice.turn.total_ms`) | ≤ 1.3 s, stretch 1.0 s | turn 1: 1191 ms = endpoint 736 + STT 110 + scripted LLM 27 + Kokoro 261 + transport 14, plus 43 ms of turn overhead; 1164 ms without the LLM, which leaves 136 ms for a real model's first sentence within 1.3 s. The endpoint hangover dominates; tuning it within 500-800 ms is a product decision, and the default stays 700 ms. Turns 2 and 3 (8064 and 9571 ms) waited behind the reply before them, because the test pushes the whole clip at once |
| Front-end per 20 ms frame during the call, Silero only (`voice.frontend.ms`) | ≤ 2 ms | p99 bucket ≤ 2 ms in every turn; max 1.9 / 21.7 / 2.4 ms. The test pushes the whole clip at once, so the audio thread works through turn 2's frames while turn 1's recognition and synthesis run; the serial gate is the CPU row above |
| Whisper-verify, both directions | ≥ 80 % | caller (JFK) 11/11 content words; reply 8/9 ("tomorrow" heard as "row") |

The other two classes ran on the build before the rebase onto the Whisper and Kokoro speed-ups. `BargeInEndToEndTests`:
the reader read no reply audio after the barge-in decision (0.0 ms) and applied the flush 4.1 ms after it (gate 100
ms). `PinnedRunnersSurviveEvictionTests`: the pinned Whisper small.en and Kokoro leases survived three
memory-pressure switches with no reopen, and each transcription stayed word-perfect.

## Open

- Partial transcripts need a streaming recognizer; `PartialTranscripts = true` is rejected.
- After a barge-in the history keeps the text generated before the cut, flagged `Interrupted` in the transcript, not
  the part the caller actually heard (that needs sentence-to-played-sample bookkeeping).
- DTMF during a reply is queued behind it rather than interrupting it.
- No hallucination filter beyond "only VAD-closed segments reach the recognizer" and the no-words discard.
- The CPU kernels' per-call dispatch closures make real Silero allocate on the audio thread; fixing them is a
  cross-model Cpu change with its own A/B.
