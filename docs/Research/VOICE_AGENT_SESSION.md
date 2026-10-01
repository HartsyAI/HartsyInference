# Voice agent session (`HartsyInference.Voice`)

> Source snapshot: 2026-09-30. This date does not establish current build or verification status.

## Summary

One phone call's voice agent: caller audio in (16 kHz PCM), spoken replies out, with endpointing, barge-in, tool calls
and per-turn latency metrics. `VoiceModelSet` holds the models every call shares for the host's lifetime;
`VoiceAgentSession` is one call. The package is opt-in (references Audio, Engine and Tools; not in the meta package),
and its only language-model dependency is `ITextService`, so an in-process engine and a remote adapter both work.
Telephony tools (hang up, DTMF, transfer) are registered by the host; the session only dispatches through the
`ToolRegistry` it is given.

## Threads and locks

```
PushInbound (host, one thread) ─► SpscRing<float> 30 s ─► T1 voice-audio (dedicated thread, CpuParallel inline, private CpuBackend)
                                                           RNNoise (×32768 in, ÷32768 out) → Silero 512-sample windows → capture ring
                                                           endpoint ─► turn channel          barge-in ─► flush epoch, turn CancelAsync
T3 turn loop (async, one turn at a time)
   utterance ─► T2: Whisper lease ─► VoiceConversation (token-trimmed) ─► ToolLoop (thinking off, Device = LlmDevice)
   deltas ─► SentenceChunkedSynthesis (run = post to T2, 2 in flight) ─► StreamingResampler ─► VoiceOutbound
T2 voice-gpu (dedicated thread, owned by VoiceModelSet): one DeviceGate hold per job, FreeActivations per job,
   TrimMemoryPool once per turn when the queue is empty
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
- **Rule added here (veto-able):** an utterance that ends while the reply is still audible and never became a
  barge-in is not answered. It is backchannel or the reply's own echo, and answering it would talk over the reply with
  a reply to itself. It is counted (`DiscardedUtterances`, `UtteranceDiscarded` event). The interrupting speech of a
  real barge-in is answered. An utterance the recognizer hears no words in is counted the same way.

## Outbound queue and flushes

`SpscRing.DiscardAll` is consumer-only, so a flush is a request: T1 bumps `FlushEpoch`, and `ReadOutbound` discards and
publishes the epoch it applied. Two rules keep a flush from eating the wrong audio:

- The producer re-checks the epoch after each write and, if it moved, bumps it again, so a frame that landed after the
  reader's discard is dropped too.
- A new turn never starts writing until the reader has applied every pending flush, so a late discard cannot take the
  next reply's opening.

A concurrent-flush stress test holds the reader to never hearing an older turn after a newer one. The producer waits
for space rather than dropping. The reader never blocks or allocates; it completes the producer's waiter after reads
that made progress.

## Inbound backlog

The ring drops the newest samples when full, so the audio thread enforces drop-oldest itself: when it is more than
30 s behind it discards the oldest excess, resets the front-end and counts it (`InboundDroppedSamples`).

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

GPU rows (Whisper small.en per utterance, Kokoro per 15-word sentence, turn total, barge-in stop) come from
`VoiceSessionEndToEndTests`, `BargeInEndToEndTests` and `PinnedRunnersSurviveEvictionTests` on the RTX 3060.

## Open

- Partial transcripts need a streaming recognizer; `PartialTranscripts = true` is rejected.
- After a barge-in the history keeps the text generated before the cut, flagged `Interrupted` in the transcript, not
  the part the caller actually heard (that needs sentence-to-played-sample bookkeeping).
- DTMF during a reply is queued behind it rather than interrupting it.
- No hallucination filter beyond "only VAD-closed segments reach the recognizer" and the no-words discard.
- The CPU kernels' per-call dispatch closures make real Silero allocate on the audio thread; fixing them is a
  cross-model Cpu change with its own A/B.
