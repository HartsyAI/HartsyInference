# Wake Word Detection — Research Notes

> Status: Complete | Last Updated: 2026-08-16 | Needed Before: HartsyInference.Audio (wake models), HartsyInference.Engine (wake service), HartsyInference.API (audio ingest)

## Summary

Always-on wake word detection for network satellites (a Pi Pico W streaming 24/7 mic audio). The satellite is too small to run a wake model, so detection is **server-side** — the established "dumb satellite" tier of the Home Assistant ecosystem (M5 ATOM Echo pattern), as opposed to on-device microWakeWord on an ESP32-S3.

The runtime stack is the **openWakeWord pattern**: a shared mel front-end and a frozen Google `speech_embedding` backbone feed N tiny per-phrase classifier heads. Shared front-end cost is paid once regardless of how many wake words are active, which is why 15-20 models run in real time on a single Raspberry Pi 3 core. User-settable phrases come from training a new head on synthetic TTS audio — the engine already has the two expensive ingredients (multi-voice TTS, and the backbone that produces the training features).

**Every constant in this document was extracted from the actual checkpoints** (ONNX graph dumps and the safetensors header), not from upstream READMEs. Several published descriptions are wrong; those are flagged inline. Sources: [openWakeWord](https://github.com/dscripka/openWakeWord), [Google speech_embedding](https://arxiv.org/abs/2002.01322), [Silero VAD](https://github.com/snakers4/silero-vad), [hey-buddy](https://github.com/painebenjamin/hey-buddy), [microWakeWord](https://github.com/kahrendt/microWakeWord).

## Detailed Findings

### 1. Why server-side, and why not the alternatives

| Option | Verdict |
|---|---|
| **openWakeWord pattern** (chosen) | Shared backbone + tiny heads. Weights Apache-2.0 (backbone) and trivially portable — conv/pool/activation, stateless, F32. Custom phrases need a trained head, which we can do in-engine. |
| sherpa-onnx Zipformer open-vocab KWS | The only train-nothing arbitrary-text option, but the engine has **no ONNX graph executor** (a hard architectural rule), so it needs a faithful Zipformer port — BiasNorm, Swoosh activations, nonlin-attention, multi-rate stacks, six cached tensors per layer — for ~84-86% recall untuned vs >95% for a dedicated head. Weeks of work for worse accuracy. Rejected. |
| microWakeWord | Value is int8 microcontroller streaming; on a desktop it buys nothing and its fixed-point PCAN micro-frontend must be reproduced bit-exactly. Rejected. |
| Picovoice Porcupine | Closed engine behind an activation key; the free tier was terminated 2026-06-30. Nothing to port. Rejected. |
| Whisper as the always-on detector | 10-100x the compute, 1-5 s latency, and hallucinated text in silence produces false accepts. Correct as a **verification second stage** only. |

### 2. Mel front-end — verified from `melspectrogram.onnx`

⚠️ **The upstream README does not state these and the commonly cited "25 ms window" is wrong.** Graph dump (opset 13, producer pytorch 1.11.0):

| Constant | Value | Evidence |
|---|---|---|
| n_fft / win_length | **512** (32 ms) | `stft.conv_real.weight` shape `[257, 1, 512]` — 257 = 512/2+1 bins, kernel 512 |
| hop_length | **160** (10 ms) | Conv `strides=[160]` |
| Padding | **none** (`center=False`) | Conv `pads=[0, 0]` |
| Mel bins | **32** | `melW` shape `[257, 32]` |
| Spectrum | **power** (real² + imag²) | `Pow(2)` on both branches then `Add` |
| Mel application | MatMul on power spectrum | `MatMul(power, melW)` |
| Log scale | **10·log10(x)**, floor 1e-10 | `Clip(1e-10, inf)` → `Log` → `*10` → `/2.3025851` |
| Dynamic range | **global_max − 80 dB** | `ReduceMax(keepdims=0)` → `Sub(80)` → `Clip(min=floor)` |
| Post-transform | **x/10 + 2** | Applied in Python (`_get_melspectrogram`), not in the graph |

The STFT is a **fixed-basis Conv1d** (a materialized DFT matrix), not a radix-2 FFT — 257 real + 257 imag filters of length 512. Frames produced for `n` samples: `ceil(n/160 − 3)`. Verified: 1280 → 5, 1760 → 8, 16000 → 97.

> **The `ReduceMax` is global over the input buffer**, so this front-end is *not* a pure streaming mel — the dB floor depends on the whole window passed in. See §3 for how openWakeWord makes this streamable and why we must copy that chunking exactly.

### 3. The streaming contract — verified from `openwakeword/utils.py`

This is the part that a naive "incremental STFT" implementation gets wrong. openWakeWord does **chunked recompute with fixed left context**, not continuous incremental STFT:

```
on 1280 new samples (80 ms):
    raw_buffer.extend(new_samples)                       # deque, maxlen = 10 s
    mel_input = raw_buffer[-(1280 + 480):]               # 480 = 160*3 samples of LEFT CONTEXT
    new_frames = melspec_model(mel_input) / 10 + 2       # exactly 8 frames
    mel_buffer = vstack(mel_buffer, new_frames)[-970:]   # 970 = 10 s at 97 frames/s
    window = mel_buffer[-76:]                            # 76 frames ~ 775 ms
    feature_buffer = vstack(feature_buffer, embedding(window))[-120:]
    score = head(feature_buffer[-16:])                   # 16 frames = 1.28 s context
```

Consequences that must be reproduced for parity:
- The 80 ms cadence is fixed: one mel call, one embedding call, one score per 1280 samples.
- The **480-sample left context** is what makes 1760 samples yield exactly 8 frames.
- The dB floor is recomputed **per 1760-sample window**, so it is per-chunk, not per-stream. Upstream's own docstring concedes this makes streaming "not exactly the same as when the melspectrogram of the entire clip is calculated" — that difference is inherent to the model as shipped, so we match the streaming path, not the whole-clip path.
- Initial state: `mel_buffer = ones((76, 32))`, `feature_buffer` = embeddings of 4 s of random audio. A reset restores exactly this.

### 4. Speech embedding backbone — verified from `embedding_model.onnx`

Input `[batch, 76, 32, 1]` → reshaped to NCHW `[-1, 1, 76, 32]`; output `[batch, 1, 1, 96]`. 41 weight tensors, ~330k params, **stateless** (all state is host-side buffers).

Repeated block: `Conv2d → LeakyReLU(alpha=0.2) → Max(x, −0.4)`, with `MaxPool(2×2, stride 2)` between stages.

⚠️ **The activation is a *clipped* LeakyReLU, not ReLU** as commonly described: `max(leaky_relu(x, 0.2), −0.4)`. Getting this wrong is a silent accuracy loss, not a crash.

Kernels alternate `1×3` (temporal) and `3×1` (frequency) after an initial `3×3`; channel progression **24 → 48 → 72 → 96**. Convolution biases are BatchNorm-folded already (`*_weights_fused_bn` / `*_bias_fused_bn`), so there is no separate norm to implement. Asymmetric padding appears as `pads=[0,1,0,1]` on the `1×3`/`3×3` convs — honor it exactly. Final `conv2d_19` is `[96,96,3,1]` with no fused BN, then a reshape to 96-d.

### 5. Classifier heads — three distinct architectures ship in the wild

Heads are **not** one architecture. Verified by dumping each file:

| Model | Architecture | Notes |
|---|---|---|
| `alexa_v0.1` | Flatten → Linear(1536→128) → ReLU → Linear(128→128) → ReLU → Linear(128→1) → Sigmoid | **No LayerNorm** |
| `hey_mycroft_v0.1` | Same but with **LayerNorm(128) after each hidden Linear** (eps 1e-5) | Weight names `1/2/4/5/7.*` |
| `hey_jarvis_v0.1` | Same as mycroft, weights prefixed `model.*` | Also bundles a **second `verifier_model.*`** (64-wide) in the same file — skip it when loading |
| hey-buddy family | LayerNorm(1536) → gated MLP (hidden/output/**gate**) → residual blocks with LayerNorm(96) | Paine's own design, Apache-2.0; **not** the openWakeWord head shape |

⚠️ The widely repeated "3-layer FC, 102,849 params, Linear→64 + LayerNorm" description matches none of the shipped v0.1 heads: they are **128-wide** (alexa ≈ 213k params) and LayerNorm is present in some and absent in others.

**Implementation decision:** support one parameterized family — `Flatten → [Linear → optional LayerNorm → ReLU] × 2 → Linear → Sigmoid` — with width and LayerNorm-presence detected from the weight names. That covers alexa/mycroft/jarvis (the day-one words) and is the architecture our trainer emits, so trained heads and shipped heads share one code path. hey-buddy's gated/residual design is a separate importer if it is ever wanted.

All heads take `[1, 16, 96]` and emit a single sigmoid probability.

### 6. Silero VAD v6 — verified from the safetensors header

`silero_vad_16k.safetensors`, **309,633 params**, all F32, MIT licensed. Exactly 15 tensors:

| Tensor | Shape | Role |
|---|---|---|
| `stft_conv.weight` | `[258, 1, 256]` | Fixed-basis DFT as Conv1d — 129 real + 129 imag, kernel 256 → magnitude → 129 bins |
| `conv1.{weight,bias}` | `[128, 129, 3]` | Takes the 129-bin magnitude spectrogram |
| `conv2.{weight,bias}` | `[64, 128, 3]` | |
| `conv3.{weight,bias}` | `[64, 64, 3]` | |
| `conv4.{weight,bias}` | `[128, 64, 3]` | |
| `lstm_cell.{weight_ih,weight_hh,bias_ih,bias_hh}` | `[512, 128]` ×2, `[512]` ×2 | One LSTMCell, hidden 128 |
| `final_conv.{weight,bias}` | `[1, 128, 1]` | → sigmoid, one probability per chunk |

I/O contract: fixed **512-sample (32 ms) chunks** at 16 kHz, with **64 samples of audio context** carried from the previous chunk (input is 576 samples). State is `(2, batch, 128)` — stacked LSTM h and c, zero-initialized, reset on stream discontinuity.

Streaming defaults: enter at prob ≥ **0.5**, exit below **0.35** (hysteresis, `threshold − 0.15`), `min_speech_duration_ms=250`, `min_silence_duration_ms=100`, `speech_pad_ms=30`.

Quality (vendor benchmark, 17 h multi-domain ROC-AUC): Silero v6 0.97 vs WebRTC VAD 0.73 — which is why the energy-threshold option in `STREAMING_AUDIO_INFERENCE.md` §Open-Questions Q7 is resolved in favor of porting these weights. The port is small: reflection pad, 6 Conv1d (one of them a fixed DFT), magnitude, one LSTM cell, sigmoid.

⚠️ Source the weights from the silero-vad repo, **not** from openWakeWord's v0.5.1 release — that ships a pinned old version.

### 7. Speaker identification

No new model needed. `CamPlusSpeakerEncoder` (CAM++, 192-d embedding, 80-bin Kaldi fbank input) is already loaded and validated in-engine, and `Engine/Audio/Stt/SpeakerDiarizer.cs` already does embed + cosine + agglomerative clustering.

What is missing is enrollment storage, a calibrated threshold (the diarizer's `MergeDistance = 0.40f` is documented as uncalibrated), and the protocol. Household convention (Odyssey 2022 baselines, arXiv:2205.00288): speaker model = centroid of L2-normalized enrollment embeddings, score = cosine, identify = argmax with an open-set threshold below which the answer is "guest". 3-5 enrollment utterances.

⚠️ **Short utterances degrade text-independent verification badly** — a model at 0.7% EER on full utterances lands in the several-percent range at ~1 s. Two mitigations, both used: enroll **text-dependent** on the wake phrase itself (train/test content match), and score on the wake word **plus the following command audio**, not the wake word alone.

Because speaker ID and Whisper verification both run on *completed captured segments*, the existing whole-buffer `KaldiFbankExtractor` is sufficient — no incremental fbank is needed.

## Key Numbers / Constants

- Audio: **16 kHz, 16-bit mono PCM** everywhere (satellite wire format, mel input, VAD input, CAM++ input)
- Wake cadence: **1280 samples = 80 ms** per score; mel left context **480 samples**; mel call input **1760 samples → 8 frames**
- Mel: n_fft/win **512**, hop **160**, **32** bins, power spectrum, 10·log10, floor `global_max − 80 dB`, transform `x/10 + 2`
- Embedding: input **76×32**, output **96-d**, stride **8 mel frames (80 ms)**, receptive field ~775 ms
- Head: input **16×96 = 1536** flattened, hidden **128**, output 1 sigmoid; context **1.28 s**
- Buffers: raw **10 s**, mel **970 frames**, features **120 frames** (~10 s)
- VAD: **512-sample** chunks, **64-sample** context, hidden **128**, threshold 0.5 / 0.35
- Frames for n samples: `ceil(n/160 − 3)`; 97 mel frames per second of audio
- Detection convention: threshold a **smoothed** score (moving average), not the raw per-step probability, plus a refractory period. microWakeWord ships `probability_cutoff 0.97` over a **5-sample moving average**; openWakeWord heads are trained to work at **0.5**.
- Release criteria openWakeWord targets: **false-reject < 5%** at **< 0.5 false-accepts/hour**, measured on the Dinner Party Corpus.

## Data Layouts / Formats

### Wake pipeline state (per device session)
```
raw ring buffer      float[160000]     10 s @ 16 kHz, drop-oldest
mel buffer           float[970][32]    ring, initialized to 1.0
feature buffer       float[120][96]    ring, initialized from 4 s random audio
per-head score hist  float[N][window]  moving average for smoothing
refractory deadline  long              sample index before which detection is suppressed
```

### VAD state
```
audio context        float[64]         carried from previous chunk
lstm h, c            float[128] each   zero on reset
```

### On-disk model layout
```
{models}/audio/wake/
    vad/silero_vad_16k.safetensors
    backbone/melspectrogram.onnx        (weights only — DFT basis + mel matrix)
    backbone/embedding_model.onnx       (weights only — 41 conv tensors)
    heads/<name>.onnx                   imported shipped heads
    heads/<name>.safetensors            heads trained in-engine
```

ONNX files are read by `ModelAssets/Onnx/OnnxWeightLoader` for **weights only** — the engine has no ONNX graph executor and every forward pass is reimplemented in C# against `IBackend`.

## Reference Implementations

- **openWakeWord** — [`openwakeword/utils.py`](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/utils.py) is the authority for the streaming contract: `_streaming_melspectrogram`, `_streaming_features`, `_get_embeddings`, `get_features`. [`model.py`](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/model.py) has the prediction buffer and debounce.
- **Silero VAD** — [`utils_vad.py`](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py) `OnnxWrapper` (I/O contract, state handling) and `VADIterator` (hysteresis).
- **hey-buddy** — [painebenjamin/hey-buddy](https://github.com/painebenjamin/hey-buddy) for the fully text-driven training pipeline (100k TTS positives + 100k phonetically-similar adversarial negatives, then augment).
- **Google speech_embedding** — [arXiv:2002.01322](https://arxiv.org/abs/2002.01322) for the backbone's design and the synthesized-speech training method.
- **Household speaker recognition** — [arXiv:2205.00288](https://arxiv.org/abs/2205.00288) + [code](https://github.com/underdogliu/household-speaker-recognition) for enrollment/centroid/adaptation protocols.

## Differences Between Implementations

- **Whole-clip vs streaming mel**: the global-max dB floor makes these differ; upstream documents the discrepancy and ships the streaming path anyway. Parity fixtures must be generated through the **streaming** chunking (1280 + 480 left context), or the comparison is against a path production never takes.
- **Head architectures differ per file** (§5) — a loader that assumes one shape silently mismatches weights or, worse, loads a `verifier_model` as the main model (jarvis bundles both).
- **openWakeWord vs microWakeWord front-ends are not interchangeable**: 32 log-mel bins with a global dB floor vs 40 channels from the TFLite-Micro fixed-point micro-frontend with PCAN auto-gain and stateful noise reduction. Heads are bound to their front-end.
- **Threshold conventions**: openWakeWord heads are trained for a raw 0.5; microWakeWord ships 0.97 over a 5-frame average. Shipping a "universal" threshold across head families is wrong.

## Open Questions

- **Q1**: The mel dB floor is per-chunk, so a very loud transient inside one 1760-sample window shifts that window's floor. Does adding a short score-smoothing window fully absorb this, or is a stream-level floor estimate measurably better? Measure FA/hr both ways during the soak test before adding complexity.
- **Q2**: `feature_buffer` is seeded with embeddings of *random audio* upstream. Confirm that seeding with silence (cheaper, deterministic, and better for a parity fixture) does not change early-stream scores once 16 real frames have accumulated (~1.3 s).
- **Q3**: What smoothing window and refractory period minimize FA/hr at fixed recall for our own trained heads? Upstream's 5-frame/0.97 is tuned for microWakeWord's front-end, not this one.
- **Q4**: For speaker ID at wake-phrase length, how much does scoring wake+command jointly beat wake-only in practice? Needs the calibration trial set.
- **Q5**: Does the openWakeWord precomputed negative-feature dataset remain byte-compatible with our backbone port (it must, since it is backbone output), and is its CC-BY-NC-4.0 license a constraint for a personal deployment? Verify shapes on first use.

## Implementation Notes for HartsyInference

### Package boundaries
- `HartsyInference.Audio/Models/Wake/` — `SileroVad`, `WakeMelFrontend`, `SpeechEmbeddingModel`, `WakeHead`. Model code only, no transport.
- `HartsyInference.Audio/Pipelines/WakeDetectionPipeline.cs` — composes the above into the §3 streaming contract.
- `HartsyInference.Engine/Audio/Wake/` — sessions, registry, the always-on worker, wake-word config, training jobs, speaker profiles.
- `HartsyInference.API` — the socket listener wiring. The listener itself must be host-agnostic so the SwarmUI extension can host it too.

### Threading — the rule that matters
The always-on path (VAD + wake scoring) runs on **one dedicated thread with its own private `CpuBackend`**, and must never take `AudioRuntime._genLock` or an `InferenceQueue` slot — those are serialized (concurrency 1) and an 80 ms-cadence listener holding them would starve every TTS/diffusion request on the engine. Post-trigger work (Whisper verification, full ASR, speaker embedding) is burst work and goes through the normal queue.

Inference cost is <5 ms per 80 ms chunk for the whole chain, so a single thread services dozens of streams; this is a correctness/robustness design, not a throughput one.

### Zero-alloc
The 80 ms cadence means ~12.5 iterations/second/stream forever — a per-iteration allocation is a permanent GC treadmill. All buffers are preallocated per session; `Span<float>` in and out; the mel/embedding scratch tensors live for the session's lifetime, not the chunk's.

### Things to NOT do
- Do not implement the mel front-end as a continuous incremental STFT — it does not match the shipped model (§3).
- Do not assume one head architecture (§5), and do not load `verifier_model.*` tensors as the main head.
- Do not use plain ReLU in the backbone — it is a clipped LeakyReLU (§4).
- Do not gate the audio stream on client-side VAD for a server-side wake design: the satellite streams continuously and the server decides. (Client VAD is an option only for bandwidth-constrained links, which 256 kbps on WiFi is not.)
- Do not source Silero from the openWakeWord release (§6).
- Do not ship a single threshold across head families (§Differences).
