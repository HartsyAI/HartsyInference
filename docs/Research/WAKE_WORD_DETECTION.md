# Wake Word Detection — Research Notes

> **Stub.** This is built and verified end-to-end (front-end + backbone match onnxruntime to mel max-abs 2e-3 /
> embedding relL2 2e-3; Silero VAD 4.77e-6 max abs vs ONNX; custom wake-word training verified 2026-08-16 firing
> at 0.6961 on an unseen voice — see `docs/Checklists/MODEL_STATUS_AUDIO.md`), so the C# is the source of truth
> for *how it works*. What remains is what the code cannot tell you: upstream provenance, reference constants
> to diff a suspect port against, where implementations disagree, and bring-up traps.

## Summary

Always-on wake word detection for network satellites (a Pi Pico W streaming 24/7 mic audio). The satellite is
too small to run a wake model, so detection is **server-side** — the established "dumb satellite" tier of the
Home Assistant ecosystem (M5 ATOM Echo pattern), as opposed to on-device microWakeWord on an ESP32-S3.

The runtime stack is the **openWakeWord pattern**: a shared mel front-end and a frozen Google `speech_embedding`
backbone feed N tiny per-phrase classifier heads. Shared front-end cost is paid once regardless of how many
wake words are active. User-settable phrases come from training a new head on synthetic TTS audio — the engine
already has the two expensive ingredients (multi-voice TTS, and the backbone that produces the training features).

**Every constant in this document was extracted from the actual checkpoints** (ONNX graph dumps and the
safetensors header), not from upstream READMEs — several published descriptions are wrong; those are flagged
inline. Sources: [openWakeWord](https://github.com/dscripka/openWakeWord), [Google speech_embedding](https://arxiv.org/abs/2002.01322),
[Silero VAD](https://github.com/snakers4/silero-vad), [hey-buddy](https://github.com/painebenjamin/hey-buddy),
[microWakeWord](https://github.com/kahrendt/microWakeWord).

## Key Numbers / Constants

- Audio: **16 kHz, 16-bit mono PCM** everywhere (satellite wire format, mel input, VAD input, CAM++ input)
- Wake cadence: **1280 samples = 80 ms** per score; mel left context **480 samples**; mel call input **1760 samples → 8 frames**
- Mel: n_fft/win **512** (32 ms, `center=False`), hop **160** (10 ms), **32** bins, power spectrum (real²+imag²)
  via fixed-basis Conv1d STFT (not an FFT), log scale `10·log10(x)` floored at 1e-10, dynamic range
  `global_max − 80 dB`, post-transform `x/10 + 2`. ⚠️ The commonly cited "25 ms window" is wrong.
- The `ReduceMax` dB floor is **global over the input buffer passed in**, so the front-end is not a pure
  streaming mel — the streaming contract below recomputes it per-chunk, not per-stream, matching upstream's
  own documented (and accepted) discrepancy from whole-clip mel.
- Embedding: input **76×32**, output **96-d**, stride **8 mel frames (80 ms)**, receptive field ~775 ms.
  Backbone is 41 tensors, ~330k params, stateless. Repeated block `Conv2d → clipped LeakyReLU(0.2, floor
  −0.4) → MaxPool(2×2) between stages`; kernels alternate 1×3 (temporal)/3×1 (frequency) after an initial
  3×3; channels progress 24→48→72→96; conv biases are BatchNorm-folded already; asymmetric padding
  `[0,1,0,1]` on the 1×3/3×3 convs must be honored exactly.
- Head: input **16×96 = 1536** flattened, hidden **128**, output 1 sigmoid; context **1.28 s**.
- Buffers: raw **10 s**, mel **970 frames** (init `1.0`), features **120 frames** (init from 4s random audio).
- VAD: **512-sample** chunks, **64-sample** context, hidden **128**, threshold enter 0.5 / exit 0.35.
- Frames for n samples: `ceil(n/160 − 3)`; 97 mel frames per second of audio.
- Detection convention: threshold a **smoothed** score (moving average) plus a refractory period, not the raw
  per-step probability. microWakeWord ships `probability_cutoff 0.97` over a 5-sample moving average;
  openWakeWord heads are trained to work at 0.5 — these are not interchangeable (see Differences).
- Release criteria openWakeWord targets: false-reject < 5% at < 0.5 false-accepts/hour (Dinner Party Corpus).

**The streaming contract** (`openwakeword/utils.py`) is chunked recompute with fixed left context, not
continuous incremental STFT — the part a naive port gets wrong:

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

## Data Layouts / Formats

### Wake pipeline state (per device session)
```
raw ring buffer      float[160000]     10 s @ 16 kHz, drop-oldest
mel buffer           float[970][32]    ring, initialized to 1.0
feature buffer       float[120][96]    ring, initialized from 4 s random audio
per-head score hist  float[N][window]  moving average for smoothing
refractory deadline  long              sample index before which detection is suppressed
```

### Classifier heads — three distinct architectures ship in the wild, not one

| Model | Architecture | Notes |
|---|---|---|
| `alexa_v0.1` | Flatten → Linear(1536→128) → ReLU → Linear(128→128) → ReLU → Linear(128→1) → Sigmoid | **No LayerNorm** |
| `hey_mycroft_v0.1` | Same but with **LayerNorm(128) after each hidden Linear** (eps 1e-5) | Weight names `1/2/4/5/7.*` |
| `hey_jarvis_v0.1` | Same as mycroft, weights prefixed `model.*` | Also bundles a **second `verifier_model.*`** (64-wide) in the same file — skip it when loading |
| hey-buddy family | LayerNorm(1536) → gated MLP (hidden/output/**gate**) → residual blocks with LayerNorm(96) | Paine's own design, Apache-2.0; **not** the openWakeWord head shape |

⚠️ The widely repeated "3-layer FC, 102,849 params, Linear→64 + LayerNorm" description matches none of the
shipped v0.1 heads: they are **128-wide** (alexa ≈ 213k params) and LayerNorm is present in some, absent in
others. **Implementation decision:** support one parameterized family — `Flatten → [Linear → optional
LayerNorm → ReLU] × 2 → Linear → Sigmoid` — width and LayerNorm-presence detected from weight names, covering
alexa/mycroft/jarvis and matching what our own trainer emits. hey-buddy's gated/residual design would need a
separate importer. All heads take `[1, 16, 96]` and emit a single sigmoid probability.

### Silero VAD v6

15 tensors, 309,633 params, F32, MIT licensed:

| Tensor | Shape | Role |
|---|---|---|
| `stft_conv.weight` | `[258, 1, 256]` | Fixed-basis DFT as Conv1d — 129 real + 129 imag, kernel 256 → magnitude → 129 bins |
| `conv1.{weight,bias}` | `[128, 129, 3]` | Takes the 129-bin magnitude spectrogram |
| `conv2.{weight,bias}` | `[64, 128, 3]` | |
| `conv3.{weight,bias}` | `[64, 64, 3]` | |
| `conv4.{weight,bias}` | `[128, 64, 3]` | |
| `lstm_cell.{weight_ih,weight_hh,bias_ih,bias_hh}` | `[512, 128]` ×2, `[512]` ×2 | One LSTMCell, hidden 128 |
| `final_conv.{weight,bias}` | `[1, 128, 1]` | → sigmoid, one probability per chunk |

Architecture as actually ported (see `MODEL_STATUS_AUDIO.md` for the parity number): **right-only** reflect
pad of 64 (not symmetric) → fixed-DFT Conv1d STFT (kernel 256/stride 128) → magnitude → 4-conv encoder (k3
pad1, strides 1/2/2/1) → **ReLU on the LSTM hidden state** (absent from every written description of this
model) → LSTMCell(128) → final conv → sigmoid. For the fixed 576-sample contract the encoder collapses to
T=1, so the graph's trailing `ReduceMean` is vacuous and unimplemented.

I/O contract: fixed 512-sample (32 ms) chunks at 16 kHz with 64 samples of context carried from the previous
chunk (input is 576 samples). State is `(2, batch, 128)` — stacked LSTM h and c, zero-initialized, reset on
stream discontinuity. Streaming defaults: enter ≥0.5, exit below 0.35 (hysteresis), `min_speech_duration_ms=250`,
`min_silence_duration_ms=100`, `speech_pad_ms=30`. Quality (vendor benchmark, 17h multi-domain ROC-AUC):
Silero v6 0.97 vs WebRTC VAD 0.73.

⚠️ **Upstream ships two different revisions of this model** — `silero_vad_16k.safetensors` and
`silero_vad.onnx` share architecture and DFT basis, but every learned tensor differs (correlations 0.90–0.99,
max abs up to 18). This port derives from the **ONNX**, since that's what silero's own `utils_vad.py` runs
and what everyone benchmarks against. Source weights from the silero-vad repo directly, **not** from
openWakeWord's v0.5.1 release (ships a pinned old version).

### On-disk model layout
```
{models}/audio/wake/
    vad/silero_vad_16k.safetensors
    backbone/melspectrogram.onnx        (weights only — DFT basis + mel matrix)
    backbone/embedding_model.onnx       (weights only — 41 conv tensors)
    heads/<name>.onnx                   imported shipped heads
    heads/<name>.safetensors            heads trained in-engine
```

ONNX files are read by `ModelAssets/Onnx/OnnxWeightLoader` for **weights only** — the engine has no ONNX graph
executor and every forward pass is reimplemented in C# against `IBackend`.

## Reference Implementations

- **openWakeWord** — [`openwakeword/utils.py`](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/utils.py) is the authority for the streaming contract: `_streaming_melspectrogram`, `_streaming_features`, `_get_embeddings`, `get_features`. [`model.py`](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/model.py) has the prediction buffer and debounce.
- **Silero VAD** — [`utils_vad.py`](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py) `OnnxWrapper` (I/O contract, state handling) and `VADIterator` (hysteresis).
- **hey-buddy** — [painebenjamin/hey-buddy](https://github.com/painebenjamin/hey-buddy) for the fully text-driven training pipeline (100k TTS positives + 100k phonetically-similar adversarial negatives, then augment).
- **Google speech_embedding** — [arXiv:2002.01322](https://arxiv.org/abs/2002.01322) for the backbone's design and the synthesized-speech training method.
- **Household speaker recognition** — [arXiv:2205.00288](https://arxiv.org/abs/2205.00288) + [code](https://github.com/underdogliu/household-speaker-recognition) for enrollment/centroid/adaptation protocols. No new model needed here: `CamPlusSpeakerEncoder` (CAM++, 192-d, 80-bin Kaldi fbank) is already loaded and validated in-engine.

## Differences Between Implementations

- **Whole-clip vs streaming mel**: the global-max dB floor makes these differ; upstream documents the
  discrepancy and ships the streaming path anyway. Parity fixtures must be generated through the **streaming**
  chunking (1280 + 480 left context), or the comparison is against a path production never takes.
- **Head architectures differ per file** — a loader that assumes one shape silently mismatches weights or,
  worse, loads a `verifier_model` as the main model (jarvis bundles both).
- **openWakeWord vs microWakeWord front-ends are not interchangeable**: 32 log-mel bins with a global dB
  floor vs 40 channels from the TFLite-Micro fixed-point micro-frontend with PCAN auto-gain and stateful
  noise reduction. Heads are bound to their front-end.
- **Threshold conventions**: openWakeWord heads are trained for a raw 0.5; microWakeWord ships 0.97 over a
  5-frame average. Shipping a "universal" threshold across head families is wrong.
- **sherpa-onnx Zipformer** (open-vocab KWS, train-nothing) was rejected: the engine has no ONNX graph
  executor, so it would need a faithful Zipformer port (BiasNorm, Swoosh, nonlin-attention, multi-rate
  stacks) for ~84-86% recall untuned vs >95% for a dedicated head.
- **Speaker ID at short utterance length degrades badly** (a model at 0.7% EER on full utterances lands in
  the several-percent range at ~1s) — mitigated by enrolling text-dependent on the wake phrase itself and
  scoring wake+command jointly, not the wake word alone.

## Implementation Notes for HartsyInference

### Package boundaries
- `HartsyInference.Audio/Models/Wake/` — `SileroVad`, `WakeMelFrontend`, `SpeechEmbeddingModel`, `WakeHead`. Model code only, no transport.
- `HartsyInference.Audio/Pipelines/WakeDetectionPipeline.cs` — composes the above into the streaming contract above.
- `HartsyInference.Engine/Audio/Wake/` — sessions, registry, the always-on worker, wake-word config, training jobs, speaker profiles.
- `HartsyInference.API` — the socket listener wiring. The listener itself must be host-agnostic so the SwarmUI extension can host it too.

### Threading — the rule that matters
The always-on path (VAD + wake scoring) runs on **one dedicated thread with its own private `CpuBackend`**, and must never take `AudioRuntime._genLock` or an `InferenceQueue` slot — those are serialized (concurrency 1) and an 80 ms-cadence listener holding them would starve every TTS/diffusion request on the engine. Post-trigger work (Whisper verification, full ASR, speaker embedding) is burst work and goes through the normal queue.

Inference cost is <5 ms per 80 ms chunk for the whole chain, so a single thread services dozens of streams; this is a correctness/robustness design, not a throughput one.

### Zero-alloc
The 80 ms cadence means ~12.5 iterations/second/stream forever — a per-iteration allocation is a permanent GC treadmill. All buffers are preallocated per session; `Span<float>` in and out; the mel/embedding scratch tensors live for the session's lifetime, not the chunk's.

### Things to NOT do
- Do not implement the mel front-end as a continuous incremental STFT — it does not match the shipped model.
- Do not assume one head architecture, and do not load `verifier_model.*` tensors as the main head.
- Do not use plain ReLU in the backbone — it is a clipped LeakyReLU.
- Do not gate the audio stream on client-side VAD for a server-side wake design: the satellite streams continuously and the server decides. (Client VAD is an option only for bandwidth-constrained links, which 256 kbps on WiFi is not.)
- Do not source Silero from the openWakeWord release (ships a pinned old version).
- Do not ship a single threshold across head families.
