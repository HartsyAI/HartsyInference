# OpenVoice v2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (OpenVoice pipeline)

## Summary

OpenVoice (MyShell, [arXiv:2312.01479](https://arxiv.org/abs/2312.01479)) is an open-source voice-cloning TTS system that factors voice cloning into two completely decoupled stages: (1) a **Base TTS** that synthesizes speech in a generic "base speaker" voice from text, and (2) a **Tone Color Converter (TCC)** — a VITS-style normalizing-flow voice converter that replaces the base voice's timbre with that of a reference speaker while preserving linguistic content, prosody, and style. The reference timbre is supplied as a 256-dim speaker embedding extracted from a few seconds of reference audio by a small **Tone Color Extractor** (`ReferenceEncoder`). The two stages share no parameters and are trained independently, which is the key design choice: it lets the system add a new language by training only a new base TTS (the converter and extractor are language-agnostic).

In v1 (Dec 2023) the base TTS was a custom VITS-derived model trained on en/zh. In **v2 (April 2024)** the base TTS was replaced by **[MeloTTS](https://github.com/myshell-ai/MeloTTS)**, giving native multilingual support for English (US/UK/IN/AU accents), Spanish, French, Chinese, Japanese, and Korean. The converter and extractor are unchanged between v1 and v2. License is MIT (commercial-friendly).

This file covers the **Tone Color Converter** and **Tone Color Extractor** — the OpenVoice-specific components. The stage-1 base TTS is documented separately in [MELOTTS_ARCHITECTURE.md](MELOTTS_ARCHITECTURE.md). Mel/STFT preprocessing shared with VITS is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). HiFi-GAN-style ResBlock decoders are in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md).

Sources: [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice), [arXiv:2312.01479](https://arxiv.org/abs/2312.01479), [myshell-ai/OpenVoiceV2 HF](https://huggingface.co/myshell-ai/OpenVoiceV2), [myshell-ai/OpenVoice (v1) HF](https://huggingface.co/myshell-ai/OpenVoice).

## Detailed Findings

### Variants

| Variant | Year | Base TTS | Languages | Converter Params | HF Path |
|---|---|---|---|---|---|
| **OpenVoice v1** | Dec 2023 | Custom VITS (en, zh) | en, zh | ~33 M | [myshell-ai/OpenVoice](https://huggingface.co/myshell-ai/OpenVoice) |
| **OpenVoice v2** | Apr 2024 | MeloTTS | en (US/UK/IN/AU), es, fr, zh, ja, ko | ~33 M | [myshell-ai/OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2) |

The **converter** (`checkpoint.pth`) is **131 MB** in float32 (≈33 M parameters). The **base_speakers/ses/** directory holds per-language base-speaker tone-color embeddings (each a `(1, 256, 1)` tensor, ~1 KB). Total v2 repo size ≈ 131 MB plus the MeloTTS checkpoints (one per language, ~50–150 MB each, hosted separately under `myshell-ai/MeloTTS-{English,Spanish,French,Chinese,Japanese,Korean}`).

### Two-Stage Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 1 — Base TTS (MeloTTS)                                     │
│ text + base_speaker_id → audio_base @ 22.05 kHz                  │
│ See MELOTTS_ARCHITECTURE.md                                      │
└──────────────────────────────────────────────────────────────────┘
                            │
                            │ audio_base
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 2 — Tone Color Converter (this document)                   │
│ audio_base + ref_embedding(target) − base_embedding(source)      │
│ → audio_cloned @ 22.05 kHz                                       │
└──────────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌──────────────────────────────────────────────────────────────────┐
│ Tone Color Extractor (run once per reference clip)               │
│ reference_audio → 256-dim speaker embedding (avg over clips)     │
└──────────────────────────────────────────────────────────────────┘
```

The base TTS speaker embeddings (`base_speakers/ses/{en-default,en-us,es,fr,zh,jp,kr}.pth`) are pre-extracted: when the converter runs, the source embedding is the cached embedding of the base speaker the stage-1 audio is in, and the target embedding is the freshly-extracted reference embedding. This factoring means **stage 2 sees the base voice as just another speaker** and the converter never needed to be trained on multilingual data.

### Stage 2 — Tone Color Converter (TCC)

The TCC is a **VITS** ([Kim et al. 2021](https://arxiv.org/abs/2106.06103)) configured for voice-conversion: it keeps the posterior encoder, residual coupling flow, and HiFi-GAN-style decoder, **drops the text encoder and duration predictors**, and conditions every module on a 256-dim speaker embedding `g`. The voice-conversion forward pass (from `SynthesizerTrn.voice_conversion`, [openvoice/models.py](https://github.com/myshell-ai/OpenVoice/blob/main/openvoice/models.py)) is:

```python
# inputs:  y         = linear spectrogram of source audio, shape (1, n_fft//2+1, T) = (1, 513, T)
#          y_lengths = scalar length
#          g_src     = source speaker embedding (1, 256, 1)
#          g_tgt     = target speaker embedding (1, 256, 1)
z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src, tau=tau)
z_p   = self.flow(z, y_mask, g=g_src)                  # forward flow → speaker-free latent
z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)  # inverse flow with new speaker
o_hat = self.dec(z_hat * y_mask, g=g_tgt)              # HiFi-GAN decoder → waveform
```

Three neural sub-modules and one fixed log-determinant op make up the converter:

#### 1. PosteriorEncoder (`enc_q`)
- Input: linear spectrogram `(B, 513, T)` (n_fft=1024 → 513 bins).
- Layers: `Conv1d(513 → 192)` pre-net → **WN (WaveNet)** stack of 16 dilated gated 1D-conv layers, kernel=5, dilation=1 throughout, hidden=192, conditioned on `g` via a per-layer `Conv1d(256 → 2·192)` projection → `Conv1d(192 → 2·192)` post-net producing `m_q, logs_q` (mean, log-σ).
- Reparameterized sample: `z = m_q + exp(logs_q) * randn * tau` (tau default 0.3 — lower tau ⇒ more deterministic / closer to mean ⇒ cleaner timbre, less prosody variation).
- Output: `z (1, 192, T)`, `m_q (1, 192, T)`, `logs_q (1, 192, T)`, `y_mask (1, 1, T)`.

#### 2. ResidualCouplingBlock (`flow`)
- **4** `ResidualCouplingLayer` blocks each followed by a `Flip()`. Inversion is exact (RealNVP-style coupling).
- Each `ResidualCouplingLayer`:
  - Split: `x0, x1 = chunk(x, 2, dim=1)` → each `(B, 96, T)`.
  - Pre-conv `Conv1d(96 → 192)` on `x0`.
  - **WN** stack — 4 dilated gated 1D-conv layers, kernel=5, dilation_rate=1, hidden=192, gin_channels=256.
  - Post-conv `Conv1d(192 → 96)` producing only the shift `m` (default `mean_only=True`, no log-scale → log-det = 0, but invertibility holds).
  - Forward: `x1' = x1 + m`. Reverse: `x1' = x1 − m`. Concat → `(B, 192, T)`.
- `Flip()`: reverses channel order along dim 1, log-det = 0. Needed so the next coupling layer sees a different half of the channels.
- **Speaker conditioning enters the flow** in every WN block via the `g` projection. This is what lets the flow "subtract source speaker, add target speaker": calling forward with `g_src` produces a representation that — under the model's training assumption — is conditionally Gaussian (speaker-independent) given the flow output, and calling reverse with `g_tgt` paints the new timbre back on.

#### 3. Generator / Decoder (`dec`)
- HiFi-GAN ([Kong et al. 2020](https://arxiv.org/abs/2010.05646)) decoder, parameterized by `upsample_initial_channel=512`, `upsample_rates=[8,8,2,2]` (product = 256, matching the STFT `hop_length=256`), `upsample_kernel_sizes=[16,16,4,4]`, three parallel MRF ResBlocks per stage with `resblock_kernel_sizes=[3,7,11]` and `resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]`.
- Pre-conv `Conv1d(192 → 512, kernel=7)` ingests `z_hat`. Speaker embedding `g` is added every upsample stage via a `Conv1d(256 → channels_out)` projection.
- Final `Conv1d(channels → 1, kernel=7)` + `tanh` → 22.05 kHz mono waveform.
- For the v2 converter `zero_g=true` is set in config, which zeros the generator's own speaker-embedding projection — speaker identity is carried entirely by the flow's reverse pass. The decoder still consumes `g` but its effect is gated to zero. (This is a quality knob from the v2 retraining.)

#### Disabled components
The base `SynthesizerTrn` class also defines a `TextEncoder` (transformer w/ relative attention, 6 layers, 192 hidden, 768 FFN, 2 heads) and `StochasticDurationPredictor` / `DurationPredictor`. These are **not used** in `voice_conversion()` — text is irrelevant to the converter. They are present in the checkpoint for code-reuse reasons but their weights are essentially dead; we can skip loading them entirely for the pure-C# port.

### Tone Color Extractor (`ReferenceEncoder`)

A small CNN+GRU that turns an arbitrary-length reference spectrogram into a fixed 256-dim speaker embedding. From [openvoice/models.py](https://github.com/myshell-ai/OpenVoice/blob/main/openvoice/models.py):

```
spec (1, 513, T)
   reshape → (1, 1, T, 513)
   ↓ 6× [Conv2d(stride=(2,2), kernel=(3,3), pad=1) → InstanceNorm2d → ReLU]
        filters: 32 → 32 → 64 → 64 → 128 → 128
   ↓ permute → (1, T', features=128 · ceil(513 / 2⁶)) ≈ (1, T', 1024)
   ↓ GRU(input=128 · F', hidden=128, batch_first=True)
   ↓ take last hidden state → (1, 128)
   ↓ Linear(128 → 256) → tanh
   = g  (1, 256)  →  unsqueeze(-1) → (1, 256, 1)
```

To extract from a reference clip the API call is:

```python
gs = []
for audio_path in reference_paths:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    y = torch.FloatTensor(y).unsqueeze(0)
    spec = spectrogram_torch(y, n_fft=1024, hop=256, win=1024)  # (1, 513, T)
    g = model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)       # (1, 256, 1)
    gs.append(g)
return torch.stack(gs).mean(0)
```

Multiple reference clips are simply averaged in embedding space. Embeddings are stored as PyTorch `.pth` files of one tensor `(1, 256, 1)` — ~1 KB each.

### Inference Pipeline

Full end-to-end clone of an utterance:

1. **Once per target voice** — extract target embedding:
   - Load reference audio (recommended: 10–60 s, mono, any language).
   - Resample to 22.05 kHz.
   - Compute linear STFT spec, n_fft=1024, hop=256, win=1024, Hann window.
   - Run `ReferenceEncoder` → `g_tgt (1, 256, 1)`.
   - (Optional) average across multiple clips.
2. **Stage 1** — base TTS:
   - Pick a base speaker matching the desired output language (e.g. `en-us` for English-US output, `es` for Spanish).
   - Run MeloTTS: text → audio_base @ 22.05 kHz (in the base voice). See [MELOTTS_ARCHITECTURE.md](MELOTTS_ARCHITECTURE.md).
   - Load the cached `g_src = base_speakers/ses/{base}.pth`.
3. **Stage 2** — tone color converter:
   - STFT audio_base → spec `(1, 513, T)`.
   - `z = enc_q(spec, g=g_src, tau=0.3)` (tau ∈ [0.0, ~1.0]; default 0.3).
   - `z_p = flow(z, g=g_src)`.
   - `z_hat = flow(z_p, g=g_tgt, reverse=True)`.
   - `audio_out = dec(z_hat, g=g_tgt)` @ 22.05 kHz.
4. (Optional) Apply audio watermarking via `wavmark` model — partitions into 16000-sample chunks and embeds message bits. Disabled by default in v2; we can skip this in SharpInference.

### Language Support

| Language | Stage-1 model | Stage-2 base embedding key |
|---|---|---|
| English (default = US) | `MeloTTS-English` | `en-default`, `en-us`, `en-br`, `en-india`, `en-au` |
| Spanish | `MeloTTS-Spanish` | `es` |
| French | `MeloTTS-French` | `fr` |
| Chinese (Mandarin) | `MeloTTS-Chinese` | `zh` |
| Japanese | `MeloTTS-Japanese` | `jp` |
| Korean | `MeloTTS-Korean` | `kr` |

Adding a new output language requires only a new MeloTTS checkpoint + a fresh `base_speakers/ses/{lang}.pth`. The converter and extractor never need retraining.

### Cross-Lingual Capabilities

OpenVoice's headline trick: **the reference language is independent of the output language.**

- The Tone Color Extractor is trained on a multilingual mix and learns a language-agnostic speaker embedding. A 10-second Mandarin reference yields a usable `g_tgt` even if you then drive stage 1 with Spanish text. The system has been demonstrated cross-lingually for all pairwise combinations of the 6 supported output languages.
- **What works well**:
  - Timbre transfer (voice color, formant structure) is robust across languages.
  - Speaking rate and broad pitch range carry through.
- **What partially works**:
  - Accent and prosody come from the **base TTS speaker**, not the reference. Cloning a British speaker into Spanish gives Spanish prosody with the British speaker's timbre. You cannot get "British accent on Spanish text" — pick the right MeloTTS base for the prosody you want.
  - Speaker-specific micro-expressions (laughter, breathiness, vocal fry) are partly lost.
- **What doesn't work**:
  - **Cloning into a language MeloTTS doesn't support**. Pure-zero-shot to e.g. Hindi requires either a v1-style cross-lingual TTS or a custom MeloTTS variant.
  - Singing voice — model is trained on speech.
  - Phonemes absent from the reference language but present in the target (e.g. clicks, ejectives) are approximated from MeloTTS's generic distribution.
- The `tau` parameter (posterior sampling temperature) trades off prosody fidelity vs. timbre cleanliness; cross-lingual generally wants tau in [0.2, 0.4].

### HuggingFace Files

**[myshell-ai/OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2)** (131 MB total):

| Path | Size | Purpose |
|---|---|---|
| `converter/checkpoint.pth` | 131 MB | TCC weights (PosteriorEncoder + ResidualCouplingBlock + HiFi-GAN Generator + ReferenceEncoder). PyTorch pickle. |
| `converter/config.json` | 838 B | TCC hyperparameters (see "Key Numbers" below). |
| `base_speakers/ses/en-default.pth` | ~1 KB | Pre-extracted `(1,256,1)` embedding for default English base. |
| `base_speakers/ses/en-us.pth` | ~1 KB | US English base. |
| `base_speakers/ses/en-br.pth` | ~1 KB | British English base. |
| `base_speakers/ses/en-india.pth` | ~1 KB | Indian English base. |
| `base_speakers/ses/en-au.pth` | ~1 KB | Australian English base. |
| `base_speakers/ses/es.pth` | ~1 KB | Spanish base. |
| `base_speakers/ses/fr.pth` | ~1 KB | French base. |
| `base_speakers/ses/zh.pth` | ~1 KB | Mandarin base. |
| `base_speakers/ses/jp.pth` | ~1 KB | Japanese base. |
| `base_speakers/ses/kr.pth` | ~1 KB | Korean base. |

Stage-1 MeloTTS checkpoints are in separate repos: **myshell-ai/MeloTTS-English**, **MeloTTS-Spanish**, **MeloTTS-French**, **MeloTTS-Chinese**, **MeloTTS-Japanese**, **MeloTTS-Korean** (50–150 MB each).

**v1** lives at [myshell-ai/OpenVoice](https://huggingface.co/myshell-ai/OpenVoice) with a similar converter checkpoint plus its own `base_speakers/EN/` and `base_speakers/ZH/` folders containing the legacy custom base TTS (not MeloTTS).

### Memory and Performance

| Item | Value |
|---|---|
| Converter VRAM (fp32, inference) | ~250 MB peak for 10 s utterance |
| Converter VRAM (fp16) | ~150 MB |
| Stage-1 MeloTTS VRAM | ~400 MB per language |
| Tone Color Extractor VRAM | <50 MB (only run once per reference) |
| Stage-1 RTF (RTX 3090, fp16, MeloTTS) | ~0.02 |
| Stage-2 RTF (RTX 3090, fp16, TCC) | ~0.03 |
| End-to-end RTF (fp16) | ~0.05 — well under real time |
| Sample rate (input & output) | 22.05 kHz |
| Reference clip length (recommended) | 10–60 s; minimum useful ~3 s |
| Extraction is one-shot | Embedding can be cached and reused for unlimited generations |

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

## Algorithm Steps

### Reference embedding extraction (one-shot per voice)

```
1. Load wav, resample to 22.05 kHz, mono, float32.
2. STFT: n_fft=1024, hop=256, win=1024, center=True, Hann.
3. spec = sqrt(real² + imag² + 1e-6).        # (1, 513, T)
4. x = spec.transpose(1, 2).unsqueeze(1)     # (1, 1, T, 513)
5. For i in 0..5:
       x = Conv2d(stride=2, kernel=3, pad=1)(x)
       x = InstanceNorm2d(x)
       x = ReLU(x)
       channels: 32, 32, 64, 64, 128, 128
6. x = x.transpose(1, 2).flatten(2)          # (1, T'', 128 * F')
7. _, h = GRU(x)                             # last hidden state (1, 1, 128)
8. emb = tanh(Linear(128, 256)(h.squeeze(0))) # (1, 256)
9. g = emb.unsqueeze(-1)                     # (1, 256, 1) — cache to disk.
```

### Voice conversion (per utterance)

```
INPUTS  audio_base, g_src, g_tgt, tau=0.3

1. STFT audio_base → spec (1, 513, T).
2. mask = sequence_mask(T)                   # (1, 1, T) all ones for batch 1
3. Posterior encoder:
     h  = enc_q.pre(spec) * mask
     h  = WN_16(h, mask, g=g_src)            # 16 dilated gated convs
     stats = enc_q.proj(h) * mask            # (1, 384, T)
     m_q, logs_q = split(stats, 192)
     z  = (m_q + exp(logs_q) * randn() * tau) * mask
4. Flow forward (g_src):
     for layer in [Coupling0, Flip, Coupling1, Flip, Coupling2, Flip, Coupling3, Flip]:
         z = layer.forward(z, mask, g=g_src)
     z_p = z
5. Flow reverse (g_tgt):
     for layer in reversed([Coupling0, Flip, Coupling1, Flip, Coupling2, Flip, Coupling3, Flip]):
         z_p = layer.forward(z_p, mask, g=g_tgt, reverse=True)
     z_hat = z_p
6. Decoder (HiFi-GAN):
     h = dec.conv_pre(z_hat * mask)
     if zero_g == False:  h += dec.cond(g_tgt)
     for stage in 0..3:
         h = leaky_relu(h, 0.1)
         h = dec.ups[stage](h)               # ConvTranspose1d
         xs = sum(dec.resblocks[3*stage + k](h) for k in 0..2) / 3
         h = xs
     h = leaky_relu(h)
     audio = tanh(dec.conv_post(h)).squeeze(1)   # (1, samples) @ 22.05 kHz
```

### WN block forward (used in posterior encoder and coupling layers)

```
def WN(x, mask, g):
    # n_layers iterations
    skip_acc = 0
    if cond_layer is not None:
        g_proj = cond_layer(g)               # (B, 2*hidden*n_layers, 1)
    for i in 0..n_layers-1:
        x_in = in_layers[i](x)               # dilated Conv1d, kernel=5
        if g is not None:
            cond = g_proj[:, 2*hidden*i : 2*hidden*(i+1), :]
            x_in = x_in + cond               # broadcasts on T
        acts = tanh(x_in[:, :hidden]) * sigmoid(x_in[:, hidden:])  # gated
        res_skip = res_skip_layers[i](acts)
        if i < n_layers - 1:
            x = (x + res_skip[:, :hidden]) * mask     # residual
            skip_acc = skip_acc + res_skip[:, hidden:]
        else:
            skip_acc = skip_acc + res_skip
    return skip_acc * mask
```

Padding for dilated convs: `pad = (kernel * dilation - dilation) // 2`. With kernel=5, dilation=1 → pad=2 each side.

### Coupling-layer forward / reverse

```
def coupling_forward(x, mask, g, reverse=False):
    x0, x1 = chunk(x, 2, dim=1)              # 96 + 96
    h = pre(x0) * mask
    h = WN_4(h, mask, g)
    m = post(h) * mask                       # mean_only → only shift, no scale
    if not reverse:
        x1 = (m + x1) * mask
    else:
        x1 = (x1 - m) * mask
    return concat([x0, x1], dim=1)
```

### Flip

```
def flip(x): return x.flip(dim=1)            # reverse channel order; log-det = 0
```

## Open Questions

- [ ] Exact parameter count breakdown per sub-module (PosteriorEncoder vs Flow vs Generator vs ReferenceEncoder). Total is ≈33 M but per-component split is not in the paper.
- [ ] Whether the v2 converter checkpoint has any cond-layer weight-norm wrappers we need to fold into raw weights at conversion time. (VITS uses `nn.utils.weight_norm`; need to `remove_weight_norm` before exporting.)
- [ ] Whether `zero_g=True` in v2 means the decoder's `cond` weights are literally zeros in the checkpoint (we can skip loading them) or just gated to zero at runtime (we still need to load). Inspect checkpoint to confirm.
- [ ] The minimum useful reference duration. The README says "a few seconds" but quality clearly improves with more reference; we should benchmark.
- [ ] Whether MeloTTS speaker IDs map 1-to-1 to the `base_speakers/ses/*.pth` files, or whether the pth files were extracted from a specific recording of each base voice (matters for stage-1/stage-2 alignment).
- [ ] Whether tau ∈ [0.0, 1.0] is the recommended range or it can usefully go higher.

## Implementation Notes for SharpInference

1. **Stage 1 (MeloTTS)** — implement per [MELOTTS_ARCHITECTURE.md](MELOTTS_ARCHITECTURE.md). The OpenVoice C# wrapper just consumes the stage-1 PCM and the pre-extracted source `g_src` embedding, and does not need to know what's inside MeloTTS.

2. **Stage 2 (Tone Color Converter)** is the unique part of OpenVoice. Modules we need in `SharpInference.Audio`:

   - **`PosteriorEncoder`** — `Conv1d(513→192, k=1)` → 16× WN(k=5, d=1) with speaker cond → `Conv1d(192→384, k=1)` → split into (mean, logσ). One reparameterized sample with `tau`.
   - **`ResidualCouplingBlock`** — 4× (`ResidualCouplingLayer` + `Flip`).
   - **`ResidualCouplingLayer`** with `mean_only=true` — split-half, `Conv1d(96→192)`, 4× WN(k=5, d=1) cond, `Conv1d(192→96)`, shift only. Must be exactly invertible (the same code path with reverse=True must produce a bit-identical inverse to within numerical precision; we should test this round-trip with the reference once we have weights loaded).
   - **`Flip`** — channel-axis reversal; trivial.
   - **`HiFiGanGenerator`** — already needed for [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) implementations. Reuse with config: `upsample_rates=[8,8,2,2]`, `upsample_kernel_sizes=[16,16,4,4]`, `upsample_initial_channel=512`, MRF kernels `[3,7,11]`, dilations `[[1,3,5],[1,3,5],[1,3,5]]`, gin_channels=256, optional `zero_g` flag.
   - **`ReferenceEncoder`** — 6× Conv2d(stride=2, k=3) with **InstanceNorm2d** (affine=false by default!) and ReLU → GRU(hidden=128) → Linear(128→256) → tanh. The InstanceNorm here has NO learnable affine parameters and uses per-feature-map running stats only at inference (or rather, instance-norm computes per-sample stats so no running stats either). Need a `InstanceNorm2dNoAffine` op.

3. **WN (WaveNet) block** is the central reusable primitive — it appears in posterior encoder, every coupling layer, and (with a different shape) HiFi-GAN. Implement once with parameters `(in_channels, hidden, kernel, dilation_rate, n_layers, gin_channels)`. Hot-path ops: dilated `Conv1d(k=5)`, fused tanh-sigmoid gate, residual+skip projection (single `Conv1d(hidden→2·hidden, k=1)`). The speaker conditioning is a single big `Conv1d(gin → 2·hidden·n_layers, k=1)` whose output is sliced per layer — implement that exact layout to match checkpoint shape.

4. **GRU implementation** — PyTorch's `GRU(input, hidden, batch_first=True)` with sequence input. We only need the final hidden state, not all timesteps. Standard fused GRU cell: `r = σ(Wir·x + bir + Whr·h + bhr)`, `z = σ(Wiz·x + biz + Whz·h + bhz)`, `n = tanh(Win·x + bin + r * (Whn·h + bhn))`, `h' = (1-z)·n + z·h`. Add `Gru` to `SharpInference.Core/Modules` (Kokoro also needs LSTM — same family, different gates).

5. **InstanceNorm2d** — per-(batch, channel) mean/variance across spatial dims, eps=1e-5, no learnable affine. Trivial elementwise kernel; can share with our existing LayerNorm path. Different from BatchNorm (no running stats).

6. **STFT / iSTFT** — we already need this for the vocoder. Reuse the same forward STFT here. The converter does **not** use iSTFT (HiFi-GAN generates waveforms directly); only the forward magnitude spectrogram is needed. Validate against `torch.stft` with `center=True`, `pad_mode='reflect'`, `return_complex=True`, Hann window, then `sqrt(real²+imag²+1e-6)`.

7. **Weight-norm folding** — VITS uses `torch.nn.utils.weight_norm` on most convs. At checkpoint conversion time (Python script we run once), call `remove_weight_norm()` on the model before exporting to safetensors. Don't carry `weight_g` / `weight_v` into the runtime — fold to a single `weight` tensor.

8. **Checkpoint conversion** — write a one-time Python helper to: (a) load `checkpoint.pth`, (b) drop unused `enc_p` (TextEncoder) and `dp`/`sdp` (duration predictors) keys, (c) call `remove_weight_norm`, (d) export the remainder to safetensors, (e) also export each `base_speakers/ses/*.pth` as a raw float32 binary. Result: one safetensors (~120 MB after dropping dead weights) plus tiny per-language embedding blobs.

9. **Determinism vs sampling** — the posterior reparameterization `z = m + exp(logs) * randn * tau` uses Gaussian noise. For reproducibility expose a `seed` parameter; default to deterministic by passing `tau=0.0` if the caller wants reproducible output (this becomes "take the posterior mean", which loses some prosody nuance but is bit-stable). The reference defaults to `tau=0.3`.

10. **API shape** for `SharpInference.Audio.OpenVoiceCloner`:
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
