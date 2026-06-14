# Streaming Audio Inference — Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (streaming pipelines), HartsyInference.Server (Phase 7)

## Summary

"Streaming" in audio inference is overloaded. Some models are **natively streaming** (causal or bounded-lookahead encoders that emit partial output as audio arrives — Parakeet-TDT, Moonshine, CosyVoice 2, Sesame CSM, RNN-T family, CTC family). Others are **non-streaming** but admit **pseudo-streaming wrappers** (overlap-and-dedupe sliding windows — Whisper, distil-Whisper, F5-TTS, Kokoro). The latency that matters for a voice agent is not throughput (RTF) but **first-token latency** and the **algorithmic lookahead** baked into the encoder.

This document fixes vocabulary (RTF/RTFx, latency vs throughput, causal vs lookahead, partial vs final hypotheses), enumerates the streaming architectures we will encounter (chunked-and-overlap, cache-aware encoder + per-layer state, RNN-T/TDT joint, chunk-aware causal flow-matching, autoregressive codec-token streaming), and lists the concrete C# infrastructure HartsyInference.Audio needs to build before any streaming model can ship: `AudioRingBuffer`, `StreamingMelExtractor`, `StreamingKvCache`, `IStreamingPipeline<TIn,TOut>`, and an `IAsyncEnumerable<AudioChunk>` output surface that mirrors dotLLM's existing token-streaming API.

Sources: [Moonshine v2 paper](https://arxiv.org/abs/2602.12241), [CosyVoice 2 paper](https://arxiv.org/abs/2412.10117), [NVIDIA Parakeet-TDT blog](https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt/), [Parakeet-TDT 0.6b-v2 card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [whisper.cpp stream.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp), [whisper_streaming](https://github.com/ufal/whisper_streaming), [CosyVoice repo](https://github.com/FunAudioLLM/CosyVoice), [Sesame CSM](https://github.com/SesameAILabs/csm), [Mimi codec](https://huggingface.co/kyutai/mimi), [Moonshine repo](https://github.com/moonshine-ai/moonshine), [Coqui XTTS docs](https://docs.coqui.ai/en/latest/models/xtts.html), [HiFi-GAN](https://github.com/jik876/hifi-gan), [RNN-T / Transducer overview](https://www.assemblyai.com/blog/an-overview-of-transducer-models-for-asr), [TDT explained (Speechmatics)](https://www.speechmatics.com/company/articles-and-news/token-duration-transducer-tdt-explained), [F5-TTS](https://github.com/SWivid/F5-TTS), [Canary streaming docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/streaming_decoding/canary_chunked_and_streaming_decoding.html).

## Detailed Findings

### 1. Definitions

**Real-Time Factor (RTF).** `RTF = processing_seconds / audio_seconds`. `RTF < 1` means the model can keep up with real time. The inverse, **RTFx**, is the number reported by NVIDIA (`RTFx = audio_seconds / processing_seconds`). Parakeet-TDT 0.6b-v2 reports **RTFx = 3380** at batch 128 on H100 — i.e., it processes 3380 seconds of audio in 1 wall-clock second, *but only when batched offline*. Single-stream RTFx is much lower (~50–100x), and is what determines whether the model can support a live mic.

> **Trap.** A 3000x RTFx headline does not imply low latency. It only says throughput is high. If the model still needs the full 30 s window before emitting anything, first-token latency is 30 s no matter how fast the GPU is.

**Latency vs throughput.** Two independent axes:
- **Throughput** = total audio seconds processed per wall-clock second (RTFx). Optimised by batching, FlashAttention, FP16/INT8.
- **Latency** = wall-clock time from "audio sample at t=N arrived" to "the token covering that sample emitted". Bounded below by `algorithmic_lookahead + compute_time_per_chunk`.

A batched offline pipeline can have huge throughput and infinite latency (it returns the whole transcript at the end). A live agent needs the opposite: small per-chunk compute and small lookahead.

**Causal vs lookahead.**
- **Causal model**: output at frame `N` depends only on input frames `≤ N`. Latency = compute time only.
- **Lookahead model**: output at frame `N` needs frames `N+1..N+k`. Latency ≥ `k * frame_duration + compute`. Moonshine v2 has `w_right=4` × 20 ms = **80 ms algorithmic lookahead**. Cache-aware streaming Conformer typically uses 80–2080 ms right-context windows.

**Streaming STT vs streaming TTS — different problem shapes.**
- **STT streaming**: input is a continuous audio stream, output is a sparse and bursty text stream. Hard problems: when to *commit* tokens vs hold them tentative, how to deduplicate across overlapping windows, how to maintain encoder state across chunks.
- **TTS streaming**: input is text (often fully available up front), output is a continuous audio stream that must be playable as it arrives. Hard problems: first-chunk latency, vocoder boundary clicks, prosody continuity across chunks, codec decoder context.

**Partial vs final hypotheses.** Most live ASR systems emit two kinds of output:
- **Partial** — best current guess, may change. Rendered in italics in transcription UIs.
- **Final** — committed, will not change. Emitted when (a) silence (VAD) ends a segment, (b) the LocalAgreement algorithm sees the same prefix from two consecutive windows, or (c) the encoder has slid past a hypothesis far enough that revision is impossible.

### 2. STT streaming patterns

#### 2.1 Whisper (and faster-whisper, whisper.cpp) — pseudo-streaming via sliding window

Whisper's encoder consumes a **fixed 30 s window** with sinusoidal positional encoding (see `WHISPER_ARCHITECTURE.md`). Truly streaming Whisper is not possible without modifying the encoder. The standard workaround is **sliding-window pseudo-streaming**:

1. Accumulate audio in a buffer.
2. When the buffer reaches `length_ms`, run the full encoder+decoder on it.
3. Slide the buffer forward by `step_ms`, retaining the last `keep_ms` for context.
4. Dedupe tokens that overlap the previous window's output.

**whisper.cpp `examples/stream/stream.cpp` defaults**:
- `step_ms = 3000` (process every 3 s)
- `length_ms = 10000` (10 s window per pass)
- `keep_ms = 200` (200 ms overlap kept)
- VAD mode: `step_ms ≤ 0` → trigger only when energy > `vad_thold = 0.6` for 2 s of audio polled every 100 ms.

**whisper_streaming (UFAL)** — wraps faster-whisper with the **LocalAgreement-2** stabilisation algorithm:
- Keep a `HypothesisBuffer` of committed prefix + new tentative tokens.
- After each transcription pass, compute the longest common prefix between the previous tentative output and the new output (1–5 word n-gram match). That prefix is **committed**; the rest stays tentative.
- Trim the audio buffer when it exceeds 15 s, preferring sentence boundaries (via tokenizer) or Whisper segment boundaries.
- Reported latency: **~500–800 ms** in "online" mode with min-chunk 1 s.

**Verdict for HartsyInference**: Whisper streaming is a wrapper layer, not an encoder change. Build `WhisperStreamingPipeline` on top of the standard `WhisperPipeline`. Implement LocalAgreement in C# as a separate `HypothesisBuffer` class. Do not pretend it is "real" streaming — first-token latency floor is `min_chunk_size + encoder_pass_time` (~1.1–1.5 s for a large-v3 single-stream pass).

#### 2.2 Parakeet-TDT — native streaming RNN-T descendant

**Architecture**: FastConformer encoder (XL variant in 0.6b-v2, 600 M params) + TDT joint network.
- **FastConformer** = Conformer with 8× depthwise-separable convolutional subsampling early. At 16 kHz audio with the standard 10 ms hop mel front-end, 8× subsampling gives **one encoder frame per 80 ms of audio**.
- **TDT (Token-and-Duration Transducer)** = generalisation of RNN-T. The joint network outputs **two distributions** per (encoder frame, predictor state) pair: a token distribution over vocab+blank, and a duration distribution over `D = {0, 1, 2, 3, 4}` frames. After emitting token `y`, the decoder advances `d` encoder frames, *skipping blank frames the model knows are silent*.
- **Effect**: a typical TDT decode advances `d=2` or `d=3` per non-blank token, so the predictor is called roughly half as often as in vanilla RNN-T. **~64% faster than RNN-T at equal accuracy.**
- **Streaming**: the FastConformer encoder is causal in its convolutions when configured for streaming (cache-aware mode); past-frame state is kept in per-layer caches. The joint+predictor naturally streams (each new encoder frame triggers a greedy loop).
- **`max_symbols` guard**: prevents the decoder getting stuck on `d=0` (emit-without-advance). Forces a frame advance after K consecutive zero-duration emissions.

**Numbers** (parakeet-tdt-0.6b-v2):
- Sample rate: 16 kHz
- Encoder frame rate: **12.5 Hz (80 ms/frame)** after 8× subsampling
- Params: 600 M (XL)
- Avg WER: 6.05% on Open ASR Leaderboard
- RTFx: 3380 at batch 128 (H100); single-stream RTFx around 50–100x (estimate, not in card)
- Max audio per pass: 24 min full-attention

**Cache-aware streaming**: NeMo's `nemotron-speech-streaming-en-0.6b` exposes chunk size + right context as inference-time knobs without retraining. Latency = `chunk_size + right_context`, configurable from **80 ms to 2080 ms in 80 ms steps**.

#### 2.3 Moonshine — edge-streaming, raw waveform, no mel

**Architecture** ([arXiv:2602.12241](https://arxiv.org/abs/2602.12241)):
1. **Audio preprocessor**: raw waveform → 50 Hz feature sequence via stride-2 causal convs + CMVN. No STFT, no mel. Each feature = 20 ms of audio.
2. **Streaming encoder**: sliding-window self-attention, **no positional embeddings** ("ergodic"). Window per layer:
   - First 2 and last 2 layers: `(w_left=16, w_right=4)` → **80 ms lookahead** per these layers.
   - Intermediate layers: `(w_left=16, w_right=0)` → strictly causal.
3. **Adapter** → small autoregressive decoder.

**Model sizes**:

| Variant | Params | Enc/Dec layers | Enc dim | Dec dim |
|---|---|---|---|---|
| tiny | 34 M | 6 / 6 | 320 | 320 |
| small | 123 M | 10 / 10 | 620 | 512 |
| medium | 245 M | 14 / 14 | 768 | 640 |

**Lookahead**: 80 ms total (the `w_right=4` × 20 ms in the four boundary layers — the lookaheads do not stack across layers when only some layers have them; the bottleneck is the deepest lookahead layer). **Bounded latency independent of utterance length** — unlike Whisper, latency does not grow with audio length.

**Verdict**: Moonshine is the cleanest natively-streaming open ASR for edge. No mel front-end simplifies our work (skip `StreamingMelExtractor` entirely for this model).

#### 2.4 CTC streaming (wav2vec2-CTC, Conformer-CTC)

CTC is *trivially* streaming if the encoder is causal: each frame's logits depend only on past frames, so we just feed chunks and concatenate outputs. The CTC loss permits blank-collapsing post-hoc. Chunks of 1–2 s with the encoder's natural causal mask are typical. Decoding (best-path or prefix-beam) is independent per chunk; duplicate-removal happens naturally because CTC collapses repeats.

The downside: CTC has no language model and no output dependency, so accuracy lags transducer models by 1–3 WER points.

#### 2.5 RNN-T (Recurrent Neural Transducer) — the classic streaming ASR

Three components:
- **Encoder** (acoustic) — usually LSTM/Conformer, causal. Produces `f_t` per audio frame.
- **Predictor** (label) — LM-like, takes the last emitted non-blank token, produces `g_u`.
- **Joint** — feed-forward over `(f_t, g_u)` → distribution over vocab + blank.

**Decoding loop** (greedy, per encoder frame `t`):
```
g_u = predictor(last_token)
while True:
    y = argmax(joint(f_t, g_u))
    if y == blank: break    # advance to t+1
    emit(y); last_token = y
    g_u = predictor(y)
```

Streams naturally because the encoder is causal and each new encoder frame triggers at most a few predictor calls. TDT (above) replaces the blank-loop with explicit duration prediction.

**Reference implementation** in NeMo: `nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py` — classes `GreedyRNNTInfer`, `GreedyBatchedRNNTInfer`, `GreedyBatchedTDTLabelLoopingComputer`.

#### 2.6 Canary (NeMo encoder-decoder, FastConformer + Transformer)

Multilingual ASR + translation. **Not natively streaming** — Transformer decoder needs cross-attention to the full encoder output. NeMo supports two streaming-ish modes:
- **Chunked**: non-overlapping `chunk_len_in_secs` audio segments, run independently, concatenate outputs.
- **Streaming with cross-attention policies**: `waitk` (decoder waits k encoder frames before emitting) or `alignatt` (use cross-attention scores to detect when a token's acoustic span is complete). Configured via `decoding.streaming_policy` and `decoding.xatt_scores_layer`.

#### 2.7 Sesame CSM — bi-directional streaming for conversational AI

**Architecture**:
- **Backbone**: Llama-family transformer (CSM-1B = ~1 B params total), takes interleaved text and audio tokens.
- **Audio decoder**: smaller autoregressive transformer that produces Mimi RVQ codes.
- **Codec**: Kyutai's **Mimi** — 24 kHz audio at **12.5 Hz frame rate (80 ms/frame)**, 8 RVQ codebooks × 2048 entries, **1.1 kbps bitrate**, 80 ms streaming latency.

**Bi-directional streaming**: simultaneously consumes incoming user-audio tokens and emits agent-audio tokens, conditioning each new agent frame on the full interleaved history (text + both speakers' audio). This is "full duplex" — the model can react mid-utterance.

**Streaming protocol**: emit one Mimi frame (8 RVQ codes) per 80 ms step. The Mimi decoder streams these into 24 kHz PCM with no extra lookahead.

**Why this is a different shape from Whisper streaming**: there is no chunking. The model state advances frame-by-frame the entire conversation. The infrastructure needed is closer to dotLLM's KV-cache management than to whisper_streaming's hypothesis buffer.

#### 2.8 Pseudo-streaming wrappers that are NOT real streaming

For completeness: any encoder-decoder with full attention (Whisper, Canary, F5-TTS, Kokoro) can be wrapped in a sliding window. The wrapper does not change what the model is — first-token latency is still bounded below by `chunk_size`. Don't confuse a streaming *transport* (chunked HTTP, websocket) with streaming *inference*.

### 3. TTS streaming patterns

#### 3.1 Autoregressive codec-token TTS (Bark, XTTS-v2, CosyVoice 2, Sesame CSM)

Three-stage pipeline:
1. **Text → semantic/codec tokens** (autoregressive transformer, sometimes hierarchical: semantic → coarse → fine).
2. **Codec tokens → mel or features** (chunked flow-matching / chunked decoder).
3. **Mel/features → PCM** (vocoder: HiFi-GAN, Vocos, or codec decoder).

Streaming works as: stage 1 emits tokens incrementally; once N tokens are buffered, stage 2 runs on that chunk; stage 3 runs on the resulting mel and writes audio to output. Each stage adds latency.

**Bark** (Suno) — three sequential autoregressive transformers:
- `BarkSemanticModel`: text → semantic tokens.
- `BarkCoarseModel`: semantic → first 2 EnCodec codebooks (causal AR).
- `BarkFineModel`: coarse → remaining 6 codebooks (**non-causal**, iterative). EnCodec at 75 Hz, 8 codebooks.
- Streaming is per-semantic-chunk; the fine model's non-causal nature limits chunk size and forces some lookahead.

**XTTS-v2** (Coqui, 750 M):
- GPT-2 style AR over discrete audio tokens.
- HiFi-GAN vocoder, 24 kHz output.
- `inference_stream(stream_chunk_size=20, overlap_wav_len=1024)` — yields wav chunks every 20 codec tokens, crossfading 1024 PCM samples between chunks to hide vocoder boundary clicks.
- First-chunk latency **< 200 ms** on consumer GPU.

**CosyVoice 2** (FunAudioLLM, [arXiv:2412.10117](https://arxiv.org/abs/2412.10117)):
- Text → speech tokens via LLM (chunk-aware causal masking).
- **FSQ (finite-scalar quantisation)** speech tokens, 100% codebook utilisation (vs ~23% for plain VQ).
- Streaming chunk sizes (from repo):
  - `token_hop_len = 25` speech tokens per LLM yield.
  - `token_max_hop_len = 100` (4× growth, doubles each chunk to amortise startup cost).
  - `pre_lookahead_len` added to first chunk.
  - Speech token rate ≠ mel rate → 2× upsampling before chunk-aware causal Transformer aligns to mel.
- Flow matching (CFM) decoder: speech tokens → mel, **chunk-aware causal masking** so each chunk only attends to past chunks.
- HiFi-GAN: mel → 24 kHz PCM; `inference(finalize=False)` streams chunks, `True` flushes tail.
- Quoted latency: **150 ms** end-to-end, **nearly lossless** vs offline mode.

**Sesame CSM** — see §2.7. One Mimi frame per 80 ms, no chunking.

#### 3.2 Streaming flow-matching / diffusion TTS — generally hard

The Conditional Flow Matching (CFM) ODE solver typically denoises the **entire mel sequence jointly**: the field `v(x, t)` at step `t` depends on every position. F5-TTS, NaturalSpeech 3, and the like are non-streaming by default.

Workarounds:
- **Chunked CFM with causal masking** (CosyVoice 2's approach): retrain so the velocity field uses only causal/chunk-causal attention. Adds quality cost but enables streaming.
- **Per-utterance streaming** (F5-TTS official): split text into sentences, generate each as a complete utterance, play sentences as they finish. Latency = one full sentence (typically 1–3 s of audio + 0.5–1 s compute).
- **Community chunked-CFM forks** of F5-TTS exist but quality varies (boundary discontinuities, prosody breaks).

**Verdict**: do not promise streaming for flow-matching TTS unless the checkpoint was trained chunk-aware. For F5-TTS / Kokoro, sentence-level streaming is the only honest option.

#### 3.3 Vocoder streaming (HiFi-GAN, Vocos)

Both vocoders are convolutional. Their **receptive field** (sum of upsample kernels + MRF dilated kernels) sets the minimum chunk + context. For HiFi-GAN-v1 with default config the receptive field is roughly a few hundred ms of mel, but causal-only HiFi-GAN variants achieve **~15 ms forward-receptive-field latency**.

**Chunking protocol**:
1. Pick chunk size `C` mel frames; pick context `L = receptive_field // 2`.
2. For each new chunk, feed `[L past mel | C new mel | L future mel]`.
3. Discard the first `L * upsample_factor` and last `L * upsample_factor` PCM samples; emit the middle `C * upsample_factor` samples.
4. At stream start, zero-pad the left context. At stream end, defer emission until "finalize" is called and zero-pad the right context.

Practical chunk sizes: **4096 PCM samples** per emit (≈170 ms at 24 kHz) is a commonly cited streaming HiFi-GAN target.

Vocos (Fourier-domain vocoder) has smaller receptive fields and is friendlier to streaming, but the iSTFT step still needs OLA (overlap-add).

### 4. Mel preprocessing for streaming

The mel pipeline is `PCM → pre-emphasis → STFT (window, hop) → power → mel-filterbank → log`. For streaming we need **incremental STFT**.

**Incremental STFT algorithm** (matches `pyroomacoustics`, `librosa.stream` with `center=False`):

```
state: tail = empty buffer of capacity (win_length - hop_length) samples

on chunk(new_samples):
    buffer = concat(tail, new_samples)
    out_frames = []
    i = 0
    while i + win_length <= len(buffer):
        frame = window * buffer[i : i + win_length]
        out_frames.append(rfft(frame))
        i += hop_length
    tail = buffer[i:]   # carry over leftover for next call
    return out_frames

on finalize():
    if pad_end:
        buffer = concat(tail, zeros(win_length - len(tail)))
        # emit one final frame
    tail = empty
```

**Whisper exact constants** (see `MEL_SPECTROGRAM.md`):
- `n_fft = win_length = 400` samples (25 ms at 16 kHz)
- `hop_length = 160` samples (10 ms)
- `tail capacity = 240` samples
- mel bins: 80 (large-v1/v2) or 128 (large-v3)

**Padding decisions**:
- **At stream start**: Whisper reference pads with reflection/zero of `win_length / 2` samples ("center=True" semantics). For streaming we choose `center=False` to avoid synthesising fake audio at t=0; alternatively, zero-pad once at start and never again.
- **At stream end**: do NOT zero-pad until `Finalize()` is called. A premature zero-pad causes the last partial frame to attenuate, producing artifacts the encoder may transcribe as glitches.
- **Pre-emphasis filter**: if used (Whisper does not, classic kaldi does), the IIR state must also be carried between chunks.

### 5. KV-cache management for streaming TTS / bi-directional models

dotLLM already has a KV-cache pattern for LLM inference; HartsyInference inherits the same idea but applies it to autoregressive *audio* models. The cache holds `(K_layer, V_layer)` tensors per layer; each new generated token appends one row.

**Append-and-grow** (the common case for short utterances):
- Pre-allocate `[max_seq_len, n_heads, head_dim]` per layer.
- Track `position` (next write index) as an atomic int.
- Each step: project new `k_new, v_new`; write into row `position`; increment.

**Ring buffer** (for long conversations like Sesame CSM):
- When `position >= context_window`, write wraps to row `position % context_window` — old context is overwritten.
- Attention mask must respect the rotated layout (RoPE position IDs become `position - context_window + i`).

**Context truncation** (for token-limited dialogue):
- When `position` exceeds `target`, evict the oldest N tokens by sliding K/V down by N rows (memcpy in place) and reset `position -= N`.
- Cheap on GPU (one launch per layer), but loses the evicted prefix forever.

**For HartsyInference**:
- Phase 6 (audio LLM-style models) needs `StreamingKvCache` as a first-class component.
- We do NOT need this for diffusion image models (each step processes the full latent).
- Storage: unmanaged buffer (`NativeMemory.AlignedAlloc`) sized for `n_layers * max_seq_len * n_heads * head_dim * 2 (K and V) * dtype_size`. For CSM-1B at 32 layers, 2048 ctx, 16 heads, 128 dim, FP16: `32 * 2048 * 16 * 128 * 2 * 2 = 512 MB`. Pre-allocate.

### 6. Chunked-and-overlap pattern (cross-cutting)

The canonical pattern for adapting non-causal encoders to streaming:

```
chunk_size  = N seconds   (e.g. 30 s for Whisper, 1 s for Conformer-CTC)
overlap     = K seconds   (e.g. 1–3 s for Whisper, 0.2 s for Conformer)
emit_window = N - 2K      (output for the centre N-2K seconds only)

for each window i:
    audio_window = audio[i*(N-K) : i*(N-K) + N]   # advance by N-K samples
    output = model(audio_window)
    if i == 0:
        emit(output[0 : N-K])                     # no left edge to discard
    elif is_last:
        emit(output[K : N])                       # no right edge to discard
    else:
        emit(output[K : N-K])                     # discard both edges
```

**For STT**: edges typically contain truncated words. The dedupe step inspects the *committed token sequence* in the overlap region and aligns against the new window's first tokens, dropping duplicates by longest common subsequence. LocalAgreement-2 is the safe variant: only commit a token when it appears at the same position in two consecutive windows.

**For mel preprocessing**: re-process the overlap region in each window — do *not* try to splice mel frames computed from different windows. Conv layers near the input rely on continuity; small numerical differences at chunk boundaries cause large encoder-output deltas.

### 7. Latency budget breakdown (real-time voice agent)

Target: **<500 ms p50** for "conversational", **<200 ms p50** for "natural turn-taking" (Sesame, Moshi class).

| Stage | Component | Typical | Notes |
|---|---|---|---|
| 1 | Mic → VAD chunk | 20–80 ms | Silero VAD polls at 32 ms; speech-end detection adds 200–400 ms tail. |
| 2 | Audio queue → encoder | 5 ms | AudioRingBuffer drain. |
| 3 | STT encoder (chunk pass) | 30–100 ms | Parakeet-TDT single stream on RTX 4090; Moonshine-tiny <20 ms. |
| 4 | STT decoder (joint+predictor) | 5–20 ms | TDT skips blanks; few iterations per chunk. |
| 5 | LLM first-token | 50–400 ms | Depends on model size, context length, GPU. Largest variable. |
| 6 | TTS first codec token | 30–80 ms | LLM-style AR generation; first-token only. |
| 7 | TTS chunk decode (codec → mel) | 20–60 ms | Chunk-aware CFM or codec decoder. |
| 8 | Vocoder chunk (mel → PCM) | 5–30 ms | HiFi-GAN streaming chunk. |
| 9 | PCM → speaker | 10–30 ms | OS audio buffer (typically 10–20 ms). |

**Sub-200 ms requires**: Moonshine-tiny (≤80 ms lookahead + ~20 ms compute), a 1–3 B LLM with short context, and a Sesame-style integrated TTS that produces audio frames at 12.5 Hz directly (no separate codec→mel→PCM pipeline).

### 8. Concrete C# infrastructure HartsyInference.Audio will need

This is the dependency surface that every streaming audio model will share.

```csharp
namespace HartsyInference.Audio.Streaming;

/// <summary>Thread-safe circular PCM buffer for live capture.</summary>
public sealed class AudioRingBuffer : IDisposable
{
    public AudioRingBuffer(int sampleRate, int capacitySamples);
    public int Push(ReadOnlySpan<float> samples);             // returns bytes written
    public int Drain(Span<float> destination);                 // returns bytes read
    public int Available { get; }
    public int Capacity { get; }
}

/// <summary>Stateful mel extractor that carries STFT tail between chunks.</summary>
public sealed class StreamingMelExtractor : IDisposable
{
    public StreamingMelExtractor(MelConfig config);            // win, hop, n_fft, n_mels, sample_rate
    public int Push(ReadOnlySpan<float> pcm, Span<float> outMelFrames);   // returns frames written
    public int Finalize(Span<float> outMelFrames);             // flush remaining + optional pad
    public void Reset();
}

/// <summary>Per-layer K/V append store with optional ring-buffer or trim behaviour.</summary>
public sealed class StreamingKvCache : IDisposable
{
    public StreamingKvCache(int nLayers, int maxSeqLen, int nKvHeads, int headDim, DType dtype, KvCachePolicy policy);
    public int Position { get; }
    public TensorRef GetK(int layer);                          // shape [position, nKvHeads, headDim]
    public TensorRef GetV(int layer);
    public void Append(int layer, TensorRef kNew, TensorRef vNew);
    public void Trim(int dropFromHead);                        // for context truncation
    public void Reset();
}

public enum KvCachePolicy { Grow, Ring, TrimOldest }

/// <summary>Generic chunked streaming surface. Push audio, get text (or vice versa for TTS).</summary>
public interface IStreamingPipeline<TInput, TOutput> : IDisposable
{
    ValueTask<TOutput?> PushAsync(TInput chunk, CancellationToken ct = default);   // null = no output yet
    ValueTask<TOutput?> FinalizeAsync(CancellationToken ct = default);
    void Reset();
}

/// <summary>STT-shaped output: an emitted partial or final hypothesis.</summary>
public readonly record struct TranscriptionChunk(string Text, bool IsFinal, double StartSec, double EndSec);

/// <summary>TTS-shaped output: a raw PCM chunk.</summary>
public readonly record struct AudioChunk(ReadOnlyMemory<float> Pcm, int SampleRate, bool IsFinal);

/// <summary>TTS pipelines expose IAsyncEnumerable to match dotLLM's IAsyncEnumerable&lt;Token&gt; streaming.</summary>
public interface IStreamingTtsPipeline : IDisposable
{
    IAsyncEnumerable<AudioChunk> SynthesizeAsync(string text, CancellationToken ct = default);
}

/// <summary>Hypothesis stabilisation for Whisper-style pseudo-streaming (LocalAgreement-2).</summary>
public sealed class HypothesisBuffer
{
    public void Update(IReadOnlyList<TimedToken> newHypothesis);
    public IReadOnlyList<TimedToken> Commit();                 // returns longest-stable prefix; advances state
    public IReadOnlyList<TimedToken> Tentative { get; }
}
```

**Allocation/perf constraints** (per CODE_STYLE.md):
- `AudioRingBuffer` backing storage: `NativeMemory.AlignedAlloc` so producer/consumer can be P/Invoked safely.
- `StreamingMelExtractor`: `Span<float>` in/out, FFT work buffer pre-allocated in ctor, zero alloc per Push.
- `StreamingKvCache`: storage in CUDA device memory or pinned host (depending on backend); `Append` is a layout copy + position bump, no managed alloc.
- All `Push/Finalize` calls return `ValueTask` (not `Task`) — see §dotLLM async patterns.

### 9. Per-model streaming status table

| Model | Native streaming? | Pseudo-stream possible? | Min chunk / lookahead | RTF (single stream) | Sample rate | First-token latency target |
|---|---|---|---|---|---|---|
| Whisper large-v3 | No | Yes (sliding + LocalAgreement) | 1–3 s chunk, 0.2 s overlap | ~0.05× (turbo: ~0.02×) | 16 kHz | 500–800 ms |
| Whisper-turbo | No | Yes | same | ~3× faster than v3 | 16 kHz | 300–500 ms |
| Distil-Whisper | No | Yes | same | ~5× faster than v3 | 16 kHz | 200–400 ms |
| Parakeet-TDT 0.6b-v2 | Yes (cache-aware) | — | 80 ms frame; chunk 80–2080 ms | RTFx 3380 (batch 128); ~50–100× single | 16 kHz | 100–200 ms |
| Nemotron-speech-streaming-en-0.6b | Yes | — | 80 ms minimum | — | 16 kHz | 80 ms + compute |
| Canary-1b | Partial (decoder needs waitk/alignatt) | Yes (chunked) | 1 s chunk | ~100× | 16 kHz | 1 s |
| Moonshine-tiny | Yes (ergodic enc, 80 ms lookahead) | — | 20 ms frame, 80 ms LA | ~1000×+ | 16 kHz | 100 ms |
| Moonshine-medium | Yes | — | 20 ms frame, 80 ms LA | ~500× | 16 kHz | 100 ms |
| wav2vec2-CTC | Yes (causal cfg) | Trivially | depends on causal mask | varies | 16 kHz | < frame |
| Sesame CSM-1B | Yes (bi-dir, frame-by-frame) | — | 80 ms Mimi frame | ~1× (real-time dialogue) | 24 kHz (Mimi) | 80 ms + compute |
| Bark | Partial (semantic + coarse causal; fine non-causal) | Yes (per-semantic-chunk) | 1 semantic chunk (~10 EnCodec frames) | ~1× | 24 kHz (EnCodec 75 Hz) | 500 ms+ |
| XTTS-v2 | Yes (`inference_stream`) | — | 20 GPT tokens (`stream_chunk_size=20`), 1024-sample overlap | ~1× | 24 kHz | <200 ms |
| CosyVoice 2 | Yes (chunk-aware CFM) | — | `token_hop_len=25`, doubling to 100 | 1–2× | 22.05/24 kHz | 150 ms |
| Kokoro | No | — | per utterance | ~10× (offline) | 24 kHz | utterance length |
| F5-TTS | No (non-causal CFM) | Sentence-level only | per sentence | RTF 0.15 | 24 kHz | one sentence |
| HiFi-GAN vocoder (streaming variant) | Yes (causal cfg) | — | 4096 PCM (~170 ms) chunk | ~100× | 22.05/24 kHz | receptive-field/2 |
| Vocos vocoder | Yes (smaller RF) | — | smaller than HiFi-GAN | ~100× | 24 kHz | receptive-field/2 |

## Key Numbers / Constants

### Sample rates and frame rates
- ASR audio input: **16 kHz** (Whisper, Parakeet, Moonshine, Canary, wav2vec2)
- TTS audio output: **24 kHz** (XTTS, CosyVoice 2, Mimi, Bark/EnCodec) or **22.05 kHz** (some HiFi-GAN configs)
- Whisper mel: `n_fft=400 (25 ms)`, `hop=160 (10 ms)` → **100 Hz** mel frames
- FastConformer encoder: 8× subsample of 100 Hz mel → **12.5 Hz (80 ms/frame)**
- Moonshine raw-waveform front-end: **50 Hz (20 ms/frame)**
- Mimi codec: **12.5 Hz (80 ms/frame)**, 8 RVQ codebooks × 2048, **1.1 kbps**, 24 kHz audio
- EnCodec (Bark): **75 Hz**, 8 codebooks, 24 kHz audio

### whisper.cpp `stream.cpp` defaults
- `step_ms = 3000`
- `length_ms = 10000`
- `keep_ms = 200`
- VAD: poll 100 ms, capture 2 s, `vad_thold = 0.6`, `freq_thold = 100 Hz`

### whisper_streaming
- `--min-chunk-size = 1.0` s
- Trim buffer at 15 s, prefer sentence boundaries
- LocalAgreement-2 stabilisation, 1–5 word n-gram overlap detection
- Quoted latency: 500–800 ms

### TDT
- Duration set: **D = {0, 1, 2, 3, 4}** frames (configurable)
- `max_symbols` guard against d=0 loops
- ~64% faster than equal-quality RNN-T

### Moonshine v2 sliding window
- Window per layer: `(w_left, w_right)` = (16, 4) for outer 4 layers, (16, 0) for inner layers
- Lookahead = `w_right × 20 ms = 80 ms`

### CosyVoice 2 chunk parameters
- `token_hop_len = 25` (initial)
- `token_max_hop_len = 100`
- Speech-token / mel ratio: **2×**
- End-to-end latency: **150 ms**
- FSQ codebook utilisation: 100% (vs VQ 23%)

### XTTS-v2 streaming
- `stream_chunk_size = 20` codec tokens
- `overlap_wav_len = 1024` PCM samples (crossfade)
- First-chunk latency: **<200 ms**

### Sesame CSM-1B
- Codec: Mimi at **80 ms/frame**, 8 RVQ codes/frame
- Backbone: Llama-family, ~1 B params
- Audio decoder: smaller AR transformer

### Latency targets
- **Natural turn-taking** (Sesame/Moshi class): **<200 ms p50**
- **Conversational**: **<500 ms p50**
- **"Live captioning"**: **<1000 ms p50** (Whisper streaming acceptable)

## Data Layouts / Formats

### AudioRingBuffer (single-producer, single-consumer)

```
struct AudioRingBuffer {
    float*  data;          // capacity * 4 bytes, NativeMemory.AlignedAlloc, 64-byte align
    int     capacity;      // power of two for fast modulo
    long    writeIdx;      // monotonically increasing; reader sees Volatile.Read
    long    readIdx;       // monotonically increasing
    int     sampleRate;
}
```

`Available = (int)(writeIdx - readIdx)`. Wrap-handled by `writeIdx & (capacity - 1)`.

### StreamingMelExtractor state

```
struct StreamingMelState {
    float*  tail;          // win_length - hop_length floats carried between chunks
    int     tailLen;       // current valid samples in tail
    float*  windowFn;      // pre-computed Hann/Hamming window, win_length floats
    float*  fftWork;       // n_fft * 2 floats (real + imag)
    float*  melFilterbank; // n_mels * (n_fft/2 + 1) floats, pre-computed
    bool    centerPaddedAtStart;
}
```

Per Push: copy `tail | new_pcm` into a scratch buffer, slide by `hop_length`, rFFT each window, magnitude², matmul against mel filterbank, log. Write mel frames to caller's output `Span<float>`.

### StreamingKvCache layout

Per layer:
```
K: [maxSeqLen, nKvHeads, headDim]   contiguous (row-major)
V: [maxSeqLen, nKvHeads, headDim]   contiguous (row-major)
```

For ring policy: `effectivePos = position % maxSeqLen`. RoPE/ALiBi must compute position IDs from absolute `position`, not the ring slot.

For autoregressive append: writing token `t` puts new K/V into row `t` of each layer's tensor. Attention reads slice `[0..t+1]`.

### Transducer joint network state

```
struct TransducerState {
    int     lastToken;             // last non-blank emitted
    float*  predictorHidden;       // LSTM/GRU hidden state, [n_layers, hidden_dim]
    float*  predictorCell;         // LSTM cell state (RNN-T LSTM predictor)
    float*  encoderCache;          // per-layer FastConformer cache for cache-aware streaming
}
```

TDT additionally tracks `framesToAdvance` (the predicted duration `d ∈ {0..4}`) per step.

## Algorithm Steps

### Whisper sliding-window pseudo-streaming (whisper_streaming style)

```
state:
    audioBuffer = []
    committed   = []           # list of (token, t_start, t_end)
    tentative   = []

on pcmChunk:
    audioBuffer.append(pcmChunk)
    if audioBuffer.duration_s >= min_chunk_size:
        result = whisper_transcribe(audioBuffer, initial_prompt=joinText(committed[-N:]))
        # LocalAgreement-2: longest common prefix of tentative and result
        newCommit = longest_common_prefix(tentative, result)
        committed.extend(newCommit)
        tentative = result[len(newCommit):]
        emit_partial(committed + tentative)
        if audioBuffer.duration_s > 15:
            cutPoint = last_sentence_boundary_or_segment_end(committed)
            audioBuffer = audioBuffer[cutPoint:]
            # drop committed tokens before cutPoint
        emit_final(newCommit)

on stop:
    final = whisper_transcribe(audioBuffer)
    committed.extend(final[len(tentative):])
    emit_final(committed)
```

### Parakeet-TDT cache-aware streaming greedy decode

```
state:
    encCache  = init_per_layer_cache()    # FastConformer streaming cache
    predState = init_predictor_state()
    lastToken = BOS

on pcmChunk (e.g. 80 ms):
    mel = streamingMel.push(pcmChunk)                 # 8 mel frames per 80 ms
    f_t_chunk, encCache = encoder(mel, encCache)      # encoder emits 1 frame per 80 ms after 8x subsample
    for f_t in f_t_chunk:
        emitsThisFrame = 0
        while emitsThisFrame < max_symbols:
            g_u = predictor(lastToken, predState)
            joint_out = joint(f_t, g_u)
            token = argmax(joint_out.token_logits)
            d     = argmax(joint_out.duration_logits)   # in {0..4}
            if token != BLANK:
                emit(token); lastToken = token; predState = updated
                emitsThisFrame += 1
            if d > 0:
                break       # advance d-1 more frames (skip)
        skipFrames(d - 1)   # next iteration jumps frames
```

### CosyVoice 2 streaming TTS

```
state:
    llmState     = init_llm()
    flowState    = init_chunk_aware_cfm()
    hiftState    = init_hifigan()
    tokenBuf     = []
    chunkIdx     = 0
    tokenOffset  = 0

LLM thread (continuously):
    for tok in llm.generate_stream(text, prompt_tokens):
        tokenBuf.append(tok)

main loop:
    hopLen = min(token_hop_len * (2 ** chunkIdx), token_max_hop_len)
    while len(tokenBuf) - tokenOffset < hopLen:
        wait or break_if_llm_done
    chunkTokens = tokenBuf[tokenOffset : tokenOffset + hopLen + pre_lookahead_len]
    mel, flowState = flow_chunk(chunkTokens, flowState, finalize=isLast)
    melNew = mel[:, tokenOffset * token_mel_ratio :]
    pcm, hiftState = hifigan(melNew, hiftState, finalize=isLast)
    emit(pcm)
    tokenOffset += hopLen
    chunkIdx    += 1
```

### Incremental STFT (mel preprocessing)

See §4 — kept inline since it's the canonical implementation we'll port.

### HiFi-GAN chunked vocoding

```
state:
    L = receptive_field // 2
    melLeftCtx = zeros(L)        # at stream start; later, last L frames of prev chunk

on melChunk(C frames):
    fullInput = concat(melLeftCtx, melChunk, melRightCtx)   # melRightCtx may be zeros at stream end
    pcm = hifigan(fullInput)
    if first_chunk:
        emit(pcm[0 : (L + C) * upsample])                   # no left edge to drop
    elif last_chunk:
        emit(pcm[L * upsample : (L + C) * upsample])
    else:
        emit(pcm[L * upsample : (L + C) * upsample])        # drop both edges
    melLeftCtx = melChunk[-L:]
```

## Reference Implementations

- **whisper.cpp streaming**: [`examples/stream/stream.cpp`](https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp) — read the `step_ms`/`length_ms`/`keep_ms` handling and the VAD energy detector. README: [`examples/stream/README.md`](https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/README.md).
- **whisper_streaming (UFAL)**: [`whisper_online.py`](https://github.com/ufal/whisper_streaming/blob/main/whisper_online.py) — `HypothesisBuffer`, `OnlineASRProcessor`. This is the cleanest LocalAgreement-2 implementation in Python and the model for our `HypothesisBuffer` class.
- **NeMo TDT decoding**: [`nemo/collections/asr/parts/submodules/`](https://github.com/NVIDIA-NeMo/NeMo/tree/main/nemo/collections/asr/parts/submodules) — look for `rnnt_greedy_decoding.py` (classes `GreedyBatchedRNNTInfer`, `GreedyBatchedTDTLabelLoopingComputer`) and `transducer_decoding.py`. Cache-aware streaming tutorial: [`tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb).
- **CosyVoice 2 streaming**: [`cosyvoice/cli/model.py`](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/model.py) — `token_hop_len`, `token_max_hop_len`, `llm_job` thread, `hift.inference(finalize=...)`. Paper: [arXiv:2412.10117](https://arxiv.org/abs/2412.10117).
- **Sesame CSM streaming demo**: [`davidbrowne17/csm-streaming`](https://github.com/davidbrowne17/csm-streaming) — community real-time wrapper around CSM-1B. Mimi codec details: [`kyutai/mimi` model card](https://huggingface.co/kyutai/mimi).
- **Moonshine**: [`moonshine-ai/moonshine`](https://github.com/moonshine-ai/moonshine), [streaming model cards](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny), paper [arXiv:2602.12241](https://arxiv.org/abs/2602.12241).
- **XTTS-v2 streaming**: [`TTS.tts.models.xtts.Xtts.inference_stream`](https://docs.coqui.ai/en/latest/_modules/TTS/tts/models/xtts.html) — `stream_chunk_size`, `overlap_wav_len`, crossfade.
- **F5-TTS sentence streaming**: [`F5-TTS/src/f5_tts/infer/utils_infer.py`](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/utils_infer.py) — sentence-level chunking. Real-time-stream issue tracking: [#700](https://github.com/SWivid/F5-TTS/issues/700).
- **HiFi-GAN streaming variants**: see [HiFi-Stream paper (arXiv 2503.17141)](https://arxiv.org/html/2503.17141v1) for causal-only configuration and ~15 ms forward receptive field.
- **Canary chunked / streaming**: [NeMo docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/streaming_decoding/canary_chunked_and_streaming_decoding.html) — `waitk`/`alignatt` policies, `chunk_len_in_secs`.

## Differences Between Implementations

- **Whisper "streaming" semantics**:
  - whisper.cpp `stream.cpp` re-runs full transcription every `step_ms` over the last `length_ms`. It emits each pass's output wholesale (no LocalAgreement) — text *flickers* mid-stream.
  - whisper_streaming uses LocalAgreement-2 — text only commits when stable, lowering perceived flicker at the cost of slightly higher commit latency.
  - HuggingFace `transformers` `pipeline(..., return_timestamps='word', chunk_length_s=30, stride_length_s=(5,5))` uses non-overlapping 30s chunks with 5s stride for context; tokens in the stride regions are dropped — different deduplication strategy from LocalAgreement and different WER.

- **TDT duration sets**: most NeMo Parakeet-TDT checkpoints use `D = {0, 1, 2, 3, 4}`. A few public configs omit `d=0` to prevent the stuck-loop case; in those, `max_symbols` is unnecessary.

- **FastConformer subsampling**: standard is 8× (80 ms/frame at 16 kHz mel-hop=10 ms). Some "streaming Conformer" configs use 4× (40 ms/frame) for lower latency at compute cost.

- **Moonshine lookahead**: paper specifies the deepest lookahead layer dominates (single 80 ms), but in practice the 4 outer layers all have `w_right=4`. Whether the actual algorithmic lookahead is 80 ms or 80 ms × 4 = 320 ms depends on how chunked inference is implemented (recurrent layer-by-layer vs full-window). Trust the paper's "80 ms total" claim for the recurrent-streaming case but **verify by measuring** when we port.

- **Mimi codec at 12.5 Hz vs EnCodec at 75 Hz**: Mimi is 6× sparser than EnCodec, so Sesame CSM produces 6× fewer codec tokens than Bark for the same audio duration. This directly explains why CSM is real-time on a single GPU while Bark is not.

- **CosyVoice 2 chunk-aware CFM vs F5-TTS full-utterance CFM**: same underlying flow-matching family, but CV2 trained with causal/chunk-causal masks (so streaming works without retraining), F5 trained with full attention (so it cannot stream without quality loss). Lesson: streaming capability is a **training-time decision**, not just an inference trick.

- **Vocoder boundary handling**:
  - XTTS-v2 uses **PCM-domain crossfade** over 1024 samples (~42 ms at 24 kHz) — simple but slight phase distortion.
  - HiFi-GAN streaming forks use **mel-domain context windows** (discard PCM at receptive-field/2) — phase-correct but adds receptive-field/2 latency.
  - Vocos uses **iSTFT OLA** — naturally phase-consistent if window/hop align.

## Open Questions

- **Q1**: Does cache-aware streaming for FastConformer require specific weights, or can any FastConformer checkpoint be loaded with streaming caches? NeMo docs imply the same checkpoint works in both offline and streaming mode — confirm by inspecting `nemotron-speech-streaming-en-0.6b` vs `parakeet-tdt-0.6b-v2` configs.
- **Q2**: For Moonshine, is the algorithmic lookahead truly 80 ms total or does it stack across the 4 lookahead-enabled layers in a non-recurrent streaming impl? Needs measurement on our port.
- **Q3**: What is the exact `pre_lookahead_len` for CosyVoice 2 chunk-aware CFM? The repo references it but a precise default isn't in the paper. Need to read `cosyvoice/flow/decoder.py` after our initial bring-up.
- **Q4**: Sesame CSM bi-directional streaming — does the audio decoder need its own KV cache distinct from the backbone, and can both caches grow independently? Public docs are thin; will need to read the model code on first integration.
- **Q5**: F5-TTS community chunked-CFM forks — is any of them quality-competitive with sentence-level streaming? Worth a quick A/B before committing engineering effort to support a chunked path for F5.
- **Q6**: For our `StreamingKvCache`, when the ring policy wraps, what is the correct RoPE handling? We need to either (a) recompute RoPE with absolute positions and accept the cost, or (b) use a sliding-window cache (k_recent only). Decision deferred to Phase 6.
- **Q7**: VAD selection — Silero-VAD (PyTorch, ~1.8 MB) is the de-facto standard but requires its own runtime. WebRTC-VAD (pure C) is leaner but worse. Should we port Silero-VAD weights to our runtime or use an energy-threshold detector for v1? Affects audio pipeline architecture.
- **Q8**: For `AudioRingBuffer`, is single-producer / single-consumer enough, or do we need multi-consumer (e.g. simultaneous VAD + STT readers)? SPSC is much simpler and 10× faster.

## Implementation Notes for HartsyInference

### Package boundaries (per NUGET_PACKAGE_DESIGN)
- `HartsyInference.Audio` owns `AudioRingBuffer`, `StreamingMelExtractor`, `StreamingKvCache`, the `IStreamingPipeline` interfaces, `HypothesisBuffer`, and `AudioChunk`/`TranscriptionChunk` records. No model-specific code lives here.
- `HartsyInference.Audio.Whisper` (or wherever Whisper lives) owns `WhisperStreamingPipeline` which composes `StreamingMelExtractor` + `WhisperPipeline` + `HypothesisBuffer`.
- `HartsyInference.Audio.Parakeet` owns the TDT joint+predictor loop and the cache-aware encoder front-end. Reuses `StreamingMelExtractor` and `StreamingKvCache`.
- `HartsyInference.Audio.Vocoders` (HiFi-GAN, Vocos) owns the chunked-vocoder context-window logic. Exposes a streaming `Synthesize(mel) → IAsyncEnumerable<AudioChunk>` API.

### Phase ordering
1. **Audio primitives first** (`AudioRingBuffer`, `StreamingMelExtractor`) — required by every audio model anyway.
2. **Then `HypothesisBuffer`** — gives us Whisper streaming with no encoder changes (lowest engineering risk, validates the architecture).
3. **Then `StreamingKvCache`** — gates Sesame CSM, XTTS-v2 streaming, CosyVoice 2 streaming. This is the biggest new component; budget time.
4. **Then cache-aware Conformer encoder support** — enables Parakeet-TDT streaming and Canary chunked.
5. **Last: chunk-aware CFM streaming** for CosyVoice 2. This is the most invasive change to a flow-matching pipeline; defer until we have a working offline CV2.

### Threading model
- Audio capture runs on its own thread (typically the OS audio callback, ~10 ms cadence). It writes to `AudioRingBuffer`. Never block this thread.
- Streaming pipeline (`PushAsync`) runs on a worker thread or `ThreadPool`. Drains the ring buffer at its own pace; if it falls behind, the ring buffer drops oldest samples (log a warning).
- LLM inference (if part of a voice-agent pipeline) runs on a CUDA stream; emits tokens via `IAsyncEnumerable` to the TTS pipeline.
- TTS pipeline writes finished `AudioChunk`s to another `AudioRingBuffer` consumed by the OS audio output callback.

This is a classic three-buffer pipeline: capture → STT → LLM → TTS → playback, each pair separated by a thread-safe buffer.

### Zero-alloc on hot paths
- All `Push` methods take `ReadOnlySpan<float>` / `Span<float>` and return frame counts (no `Memory<T>` allocations).
- `TranscriptionChunk` and `AudioChunk` use `ReadOnlyMemory<float>` only for the *final* boundary (consumer needs a stable reference). Internal pipeline plumbing uses `Span<float>`.
- `HypothesisBuffer`'s longest-common-prefix algorithm pre-allocates a `List<TimedToken>` of size `maxTokensPerWindow` and reuses it; never `new` in `Update`.
- Mel filterbank, window function, FFT twiddle tables: pre-computed once at ctor time, stored in `NativeBuffer`s.

### Validation tolerance
- Whisper streaming: same WER as offline whisper within ±0.1% on LibriSpeech test-clean (LocalAgreement should be quality-preserving).
- Parakeet streaming: within ±0.2% WER of offline Parakeet (cache-aware streaming has known small accuracy delta from chunk boundaries).
- TTS streaming: PESQ ≥ offline PESQ - 0.1 (chunk boundaries should not be audible).
- Vocoder streaming: SNR ≥ 30 dB vs offline-vocoded PCM at chunk boundaries.

### Things to NOT do
- Do not introduce a `Task`-allocating `Push` API; use `ValueTask` everywhere (most pushes return synchronously).
- Do not store K/V cache on the .NET managed heap — it will pin GC and crush throughput. Always `NativeMemory` or device memory.
- Do not pretend Whisper or F5-TTS are streaming models. Document the pseudo-streaming wrapper honestly; expose `IsNativeStreaming` on the pipeline so callers can pick the right model for the latency budget.
- Do not couple streaming pipelines to a specific audio I/O backend (NAudio, PortAudio, etc.). The pipelines accept and produce `Span<float>` PCM; let the application choose its I/O library.
