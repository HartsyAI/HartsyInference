# OpenVoice v2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (OpenVoice pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

OpenVoice (MyShell, [arXiv:2312.01479](https://arxiv.org/abs/2312.01479)) is an open-source voice-cloning TTS system that factors voice cloning into two completely decoupled stages: (1) a **Base TTS** that synthesizes speech in a generic "base speaker" voice from text, and (2) a **Tone Color Converter (TCC)** — a VITS-style normalizing-flow voice converter that replaces the base voice's timbre with that of a reference speaker while preserving linguistic content, prosody, and style. The reference timbre is supplied as a 256-dim speaker embedding extracted from a few seconds of reference audio by a small **Tone Color Extractor** (`ReferenceEncoder`). The two stages share no parameters and are trained independently, which is the key design choice: it lets the system add a new language by training only a new base TTS (the converter and extractor are language-agnostic).

In v1 (Dec 2023) the base TTS was a custom VITS-derived model trained on en/zh. In **v2 (April 2024)** the base TTS was replaced by **[MeloTTS](https://github.com/myshell-ai/MeloTTS)**, giving native multilingual support for English (US/UK/IN/AU accents), Spanish, French, Chinese, Japanese, and Korean. The converter and extractor are unchanged between v1 and v2. License is MIT (commercial-friendly).

This file covers the **Tone Color Converter** and **Tone Color Extractor** — the OpenVoice-specific components. The stage-1 base TTS is documented separately in [MELOTTS_ARCHITECTURE.md](MELOTTS_ARCHITECTURE.md). Mel/STFT preprocessing shared with VITS is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). HiFi-GAN-style ResBlock decoders are in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md).

Sources: [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice), [arXiv:2312.01479](https://arxiv.org/abs/2312.01479), [myshell-ai/OpenVoiceV2 HF](https://huggingface.co/myshell-ai/OpenVoiceV2), [myshell-ai/OpenVoice (v1) HF](https://huggingface.co/myshell-ai/OpenVoice).

## Key Numbers / Constants

From `converter/config.json` ([HF](https://huggingface.co/myshell-ai/OpenVoiceV2/raw/main/converter/config.json)):

```json
{
  "data": {
    "sampling_rate": 22050,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_speakers": 0
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "gin_channels": 256,
    "zero_g": true
  }
}
```

Derived constants:

| Symbol | Value | Source |
|---|---|---|
| `n_fft` | 1024 | = `filter_length` |
| Linear-spec bins | 513 | = `n_fft // 2 + 1` |
| Total upsample factor | 256 | = 8·8·2·2 (matches `hop_length`) |
| Posterior z dim | 192 | = `inter_channels` |
| Flow blocks | 4 | (Coupling + Flip) × 4 |
| WN layers in posterior encoder | 16 | from code |
| WN layers per coupling block | 4 | from code |
| WN kernel size | 5 | from code |
| WN dilation rate | 1 | from code |
| Speaker embedding dim | 256 | = `gin_channels` |
| Output sample rate | 22050 Hz | mono, float32 in [-1, 1] |

## Data Layouts / Formats

### Checkpoint format

`converter/checkpoint.pth` is a PyTorch pickle of a dict:

```python
{
  "iteration": int,
  "model": OrderedDict[str, torch.Tensor],   # state dict
  "optimizer": ...,                          # optional, may be missing
  "learning_rate": float
}
```

Relevant state-dict keys (prefix `enc_p.*` is dead text-encoder — skip):

- `enc_q.pre.weight/bias` — posterior pre-conv `(192, 513, 1)`
- `enc_q.enc.in_layers.{0..15}.weight/bias` — 16 WN dilated convs
- `enc_q.enc.res_skip_layers.{0..15}.weight/bias` — 16 residual+skip projections
- `enc_q.enc.cond_layer.weight/bias` — single fused `Conv1d(256, 2*192*16, 1)` for speaker conditioning (or per-layer `cond_layers.{i}` depending on checkpoint vintage)
- `enc_q.proj.weight/bias` — `(384, 192, 1)` produces `m_q, logs_q`
- `flow.flows.{0,2,4,6}.*` — 4 coupling layers (indices 0/2/4/6 because Flip occupies 1/3/5/7)
  - `.pre.weight/bias` `(192, 96, 1)`
  - `.enc.in_layers.{0..3}.weight/bias`
  - `.enc.res_skip_layers.{0..3}.weight/bias`
  - `.enc.cond_layer.weight/bias`
  - `.post.weight/bias` `(96, 192, 1)` (mean_only)
- `dec.*` — HiFi-GAN generator (conv_pre, ups.{0..3}, resblocks.{0..11}, conv_post, cond)
- `ref_enc.convs.{0..5}.weight/bias` — 6 Conv2d layers (`InstanceNorm2d` has no learnable affine by default)
- `ref_enc.gru.weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`
- `ref_enc.proj.weight/bias` `(256, 128)`

### Speaker embedding files

`base_speakers/ses/{lang}.pth` is a PyTorch pickle of one tensor of shape `(1, 256, 1)`, dtype float32. Convert offline to a raw `float32[256]` little-endian binary at packaging time.

### Audio I/O

- Input reference: any sample rate, librosa resamples to 22.05 kHz mono float32.
- Stage-1 audio: directly produced at 22.05 kHz float32 by MeloTTS.
- Stage-2 spec: linear (not mel) STFT with Hann window, n_fft=1024, hop=256, win=1024, centered, reflective padding.
- Magnitude: `sqrt(real² + imag² + 1e-6)`. No log scaling, no mel projection.
- Output: float32 PCM in [-1, 1] at 22.05 kHz. Save to WAV via standard 16-bit PCM round-trip.

## Implementation Notes for HartsyInference

1. **Stage 1 (MeloTTS)** — implement per [MELOTTS_ARCHITECTURE.md](MELOTTS_ARCHITECTURE.md). The OpenVoice C# wrapper just consumes the stage-1 PCM and the pre-extracted source `g_src` embedding, and does not need to know what's inside MeloTTS.

2. **Stage 2 (Tone Color Converter)** is the unique part of OpenVoice. Modules we need in `HartsyInference.Audio`:

   - **`PosteriorEncoder`** — `Conv1d(513→192, k=1)` → 16× WN(k=5, d=1) with speaker cond → `Conv1d(192→384, k=1)` → split into (mean, logσ). One reparameterized sample with `tau`.
   - **`ResidualCouplingBlock`** — 4× (`ResidualCouplingLayer` + `Flip`).
   - **`ResidualCouplingLayer`** with `mean_only=true` — split-half, `Conv1d(96→192)`, 4× WN(k=5, d=1) cond, `Conv1d(192→96)`, shift only. Must be exactly invertible (the same code path with reverse=True must produce a bit-identical inverse to within numerical precision; we should test this round-trip with the reference once we have weights loaded).
   - **`Flip`** — channel-axis reversal; trivial.
   - **`HiFiGanGenerator`** — already needed for [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) implementations. Reuse with config: `upsample_rates=[8,8,2,2]`, `upsample_kernel_sizes=[16,16,4,4]`, `upsample_initial_channel=512`, MRF kernels `[3,7,11]`, dilations `[[1,3,5],[1,3,5],[1,3,5]]`, gin_channels=256, optional `zero_g` flag.
   - **`ReferenceEncoder`** — 6× Conv2d(stride=2, k=3) with **InstanceNorm2d** (affine=false by default!) and ReLU → GRU(hidden=128) → Linear(128→256) → tanh. The InstanceNorm here has NO learnable affine parameters and uses per-feature-map running stats only at inference (or rather, instance-norm computes per-sample stats so no running stats either). Need a `InstanceNorm2dNoAffine` op.

3. **WN (WaveNet) block** is the central reusable primitive — it appears in posterior encoder, every coupling layer, and (with a different shape) HiFi-GAN. Implement once with parameters `(in_channels, hidden, kernel, dilation_rate, n_layers, gin_channels)`. Hot-path ops: dilated `Conv1d(k=5)`, fused tanh-sigmoid gate, residual+skip projection (single `Conv1d(hidden→2·hidden, k=1)`). The speaker conditioning is a single big `Conv1d(gin → 2·hidden·n_layers, k=1)` whose output is sliced per layer — implement that exact layout to match checkpoint shape.

4. **GRU implementation** — PyTorch's `GRU(input, hidden, batch_first=True)` with sequence input. We only need the final hidden state, not all timesteps. Standard fused GRU cell: `r = σ(Wir·x + bir + Whr·h + bhr)`, `z = σ(Wiz·x + biz + Whz·h + bhz)`, `n = tanh(Win·x + bin + r * (Whn·h + bhn))`, `h' = (1-z)·n + z·h`. Add `Gru` to `HartsyInference.Core/Modules` (Kokoro also needs LSTM — same family, different gates).

5. **InstanceNorm2d** — per-(batch, channel) mean/variance across spatial dims, eps=1e-5, no learnable affine. Trivial elementwise kernel; can share with our existing LayerNorm path. Different from BatchNorm (no running stats).

6. **STFT / iSTFT** — we already need this for the vocoder. Reuse the same forward STFT here. The converter does **not** use iSTFT (HiFi-GAN generates waveforms directly); only the forward magnitude spectrogram is needed. Validate against `torch.stft` with `center=True`, `pad_mode='reflect'`, `return_complex=True`, Hann window, then `sqrt(real²+imag²+1e-6)`.

7. **Weight-norm folding** — VITS uses `torch.nn.utils.weight_norm` on most convs. At checkpoint conversion time (Python script we run once), call `remove_weight_norm()` on the model before exporting to safetensors. Don't carry `weight_g` / `weight_v` into the runtime — fold to a single `weight` tensor.

8. **Checkpoint conversion** — write a one-time Python helper to: (a) load `checkpoint.pth`, (b) drop unused `enc_p` (TextEncoder) and `dp`/`sdp` (duration predictors) keys, (c) call `remove_weight_norm`, (d) export the remainder to safetensors, (e) also export each `base_speakers/ses/*.pth` as a raw float32 binary. Result: one safetensors (~120 MB after dropping dead weights) plus tiny per-language embedding blobs.

9. **Determinism vs sampling** — the posterior reparameterization `z = m + exp(logs) * randn * tau` uses Gaussian noise. For reproducibility expose a `seed` parameter; default to deterministic by passing `tau=0.0` if the caller wants reproducible output (this becomes "take the posterior mean", which loses some prosody nuance but is bit-stable). The reference defaults to `tau=0.3`.

10. **API shape** for `HartsyInference.Audio.OpenVoiceCloner`:
    ```csharp
    // One-shot per target voice (slow only because of I/O; model is tiny)
    SpeakerEmbedding ExtractReference(ReadOnlySpan<float> audio22kMono, int sampleRate = 22050);

    // Per-utterance
    float[] Convert(ReadOnlySpan<float> audio22kBase, SpeakerEmbedding source,
                    SpeakerEmbedding target, float tau = 0.3f, int? seed = null);

    // Convenience end-to-end (composes with MeloTTS engine)
    float[] CloneSpeak(string text, Language lang, SpeakerEmbedding target,
                       float tau = 0.3f, int? seed = null);
    ```

11. **Memory budget** — fp16 weights ≈ 65 MB for the converter; allocate one persistent activation arena sized for the longest expected utterance (e.g. 30 s @ 22.05 kHz → T ≈ 2580 frames after STFT; peak activation `192 * 2580 * 4 ≈ 2 MB` per layer, comfortable). No per-utterance allocations on the hot path; pre-allocate scratch buffers in the cloner instance.

12. **Validation tolerance** — reference outputs are stochastic (tau > 0), so bit-exact comparison won't work. Validate by (a) running the reference with `tau=0` and matching within 1e-3 PCM, and (b) comparing speaker-embedding similarity (cosine over ref_enc embedding) of our output vs reference output ≥ 0.95.

## Reference Implementations

- [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) — official Python/PyTorch reference (MIT). Models in `openvoice/models.py`, modules in `openvoice/modules.py`, attention in `openvoice/attentions.py`, runtime API in `openvoice/api.py`.
- [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) — official stage-1 base TTS for v2.
- [myshell-ai/OpenVoiceV2 HF](https://huggingface.co/myshell-ai/OpenVoiceV2) — v2 converter checkpoint + base-speaker embeddings.
- [myshell-ai/OpenVoice HF](https://huggingface.co/myshell-ai/OpenVoice) — v1 checkpoint with legacy custom base TTS.
- [arXiv:2312.01479](https://arxiv.org/abs/2312.01479) — original paper (Versatile Instant Voice Cloning, Qin et al. 2023).
- [research.myshell.ai/open-voice](https://research.myshell.ai/open-voice) — official demo and v2 announcement.
- [jasonppy/VITS](https://github.com/jaywalnut310/vits) — original VITS reference (RealNVP coupling, posterior encoder, HiFi-GAN decoder all originate here).
- [Kim et al. 2021, arXiv:2106.06103](https://arxiv.org/abs/2106.06103) — VITS paper. Required reading for the flow + posterior encoder math.
- [Kong et al. 2020, arXiv:2010.05646](https://arxiv.org/abs/2010.05646) — HiFi-GAN paper. Required reading for the generator.
