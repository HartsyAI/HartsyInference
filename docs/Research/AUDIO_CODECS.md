# Neural Audio Codecs — Research Notes
> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (codecs)

## Summary

Modern TTS and music-generation models (Bark, MusicGen, AudioGen, Moshi/Sesame CSM, Orpheus, IndexTTS-2, ChatTTS, Llasa, F5-TTS, etc.) almost never operate on raw audio. Instead they predict (autoregressively or via diffusion) a stream of discrete integer "audio tokens" produced by a small CNN autoencoder called a **neural audio codec**. The codec has three pieces:

1. **Encoder** — a strided 1-D convolutional network (sometimes with a tail LSTM or transformer) that downsamples 16/24/44.1kHz waveform into a low-rate latent (typically 12.5 – 75 Hz).
2. **Quantizer** — a Residual Vector Quantizer (RVQ), Finite Scalar Quantizer (FSQ), or single VQ that converts each latent frame into 1–32 small integer codebook indices.
3. **Decoder** — a mirrored CNN (or ConvNeXt + iSTFT in Vocos-style codecs) that reconstructs the waveform.

For HartsyInference we only need **inference**: the encoder (for voice cloning / reference audio embedding) and the decoder (to turn LM-predicted tokens back into PCM). We do **not** need the GAN discriminators, EMA codebook updates, or distillation teachers used at training time. This document covers EnCodec, DAC, Mimi, SNAC, WavTokenizer, SpeechTokenizer, XCodec/XCodec2, and BigCodec — the eight codecs that cover essentially every modern TTS / music model in our scope.

## Detailed Findings

### 1. EnCodec (Meta / Défossez 2022)

**Used by:** Bark (24 kHz mono), MusicGen (32 kHz mono, 4-codebook), AudioGen (16/32 kHz), AudioCraft / Multi-Band Diffusion, plus dozens of derivative TTS systems.

**Architecture (SEANet encoder/decoder).** Default parameters from `encodec/modules/seanet.py`:

- Channels (audio in): 1 (24 kHz model) or 2 (48 kHz model).
- Base filters `n_filters = 32`, internal dimension `dimension = 128`.
- Encoder downsample ratios: `[8, 5, 4, 2]` → product = **hop length 320** → 24000 / 320 = **75 Hz** frame rate for 24 kHz model.
- Initial Conv1d: kernel 7, channels 1 → 32, padding mode `reflect`.
- Four `EncoderBlock`s, each: one `SEANetResnetBlock` (kernel 3, dilation `2^i`, hidden = dim/2, ELU activation, `weight_norm`) then a strided Conv1d (kernel = 2*stride, stride = ratio).
- After all blocks: 2-layer LSTM at `dimension=128` (recurrent skip-connected to the conv output).
- Final Conv1d: kernel 7, channels 128 → 128.
- Decoder mirrors with `ConvTranspose1d` (kernel = 2*stride, stride = ratio) and ends with a Conv1d → 1 channel.

**24 kHz variant differences:**
- `causal = True` — all convs use left-only padding, LSTM is unidirectional. This permits real-time streaming with ~13 ms (320-sample) lookahead.
- Normalization: `weight_norm` on every conv.

**48 kHz variant differences:**
- `causal = False`, normalization is `time_group_norm` (GroupNorm applied along time), and the model processes 1-second segments with overlap-add. Stereo (2-channel) input.
- Stereo handled by treating channels as independent then re-mixing — see `encodec/model.py::_decode_frame`.

**Quantizer.** `ResidualVectorQuantization` with `EuclideanCodebook` (`quantization/core_vq.py`):

- Codebook shape: `(codebook_size=1024, dim=128)`.
- Encode: rearrange (B,D,T)→(B,T,D), compute `-( |x|² − 2·x·Eᵀ + |E|² )`, take argmin (max of the negated formula).
- Decode: `F.embedding(indices, codebook)`.
- RVQ loop: `residual = x; out = 0; for layer: q, idx = layer(residual); out += q; residual -= q.detach(); codes.append(idx)`.
- Number of *active* codebooks at inference is bandwidth-dependent: `n_q = int(1000 * bw // (fps * 10))`. So at 75 Hz and 6 kbps you get n_q = 8 codebooks (`6000 / (75 * 10) = 8`).
- Target bandwidths supported: `[1.5, 3, 6, 12, 24]` kbps (24 kHz) / `[3, 6, 12, 24]` kbps (48 kHz). The same checkpoint serves all bandwidths — you just keep the first N codebooks.

**API shapes.** For T-sample input:
- `encode(wav) -> codes [B, n_q, T/320], scales [B, num_segments]` (scales = per-segment amplitude normalization).
- `decode(codes, scales) -> wav [B, 1, T]`.

**Checkpoint.** `facebook/encodec_24khz/model.safetensors` (93.1 MB) and `facebook/encodec_48khz/model.safetensors`. Also packaged inside Bark and MusicGen checkpoints.

### 2. DAC — Descript Audio Codec (Kumar 2023)

**Used by:** IndexTTS-2 (24 kHz semantic + acoustic stack), ChatTTS DVAE variant, many post-2024 academic TTS, Parler-TTS (44 kHz), Llasa-style stacks. Drop-in replacement for EnCodec at higher quality.

**Architecture (`dac/model/dac.py`).** Defaults for the canonical 44.1 kHz model:

- `encoder_dim = 64`, `encoder_rates = [2, 4, 8, 8]` → **hop length 512** → 44100 / 512 ≈ **86 Hz**.
- `decoder_dim = 1536`, `decoder_rates = [8, 8, 4, 2]` (mirror).
- `n_codebooks = 9`, `codebook_size = 1024`, `codebook_dim = 8` (factorized VQ — codes live in 8-D after a projection from `latent_dim` 1024).
- `latent_dim = encoder_dim * 2^len(encoder_rates) = 64 * 16 = 1024`.
- `sample_rate = 44100`; same architecture also released at 16 kHz and 24 kHz (different rates/dim).
- Bitrate: 9 codebooks × 10 bits × 86 Hz ≈ **7.75 kbps**.

**Blocks (raw code).**

```python
class ResidualUnit(nn.Module):                 # 7-tap snake + 1×1
    def __init__(self, dim=16, dilation=1):
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

class EncoderBlock(nn.Module):                 # three dilated residuals then down-conv
    def __init__(self, dim=16, stride=1):
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(dim // 2, dim, kernel_size=2*stride, stride=stride,
                     padding=math.ceil(stride / 2)),
        )

class DecoderBlock(nn.Module):                 # up-conv then three dilated residuals
    def __init__(self, input_dim=16, output_dim=8, stride=1):
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(input_dim, output_dim, kernel_size=2*stride,
                              stride=stride, padding=math.ceil(stride / 2)),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )
```

Encoder topology: `Conv1d(1,64,7) → 4 × EncoderBlock(dim=64·2^i, stride=rate_i) → Snake1d → Conv1d(1024, latent_dim, 3)`.
Decoder topology: `Conv1d(latent_dim, 1536, 7) → 4 × DecoderBlock → Snake1d → Conv1d(?, 1, 7) → Tanh`.

**Snake1d activation** (DAC's signature op, not in our IBackend yet):

```python
def snake(x, alpha):                           # learnable per-channel
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
# alpha shape: (1, channels, 1)
```

**Factorized VQ.** Each VQ layer carries `in_proj` (latent_dim→8, WNConv1d k=1) and `out_proj` (8→latent_dim). Distance is computed on L2-normalized 8-D codes — equivalent to cosine similarity (ViT-VQGAN trick). API: `from_codes(codes) -> z, latents, _`.

**Causal?** Default DAC is **not** causal; convs use centered padding. There are streaming forks but the official checkpoints are offline-only.

**Checkpoints.** `descript/descript-audio-codec` HF repo and the `python -m dac download --model_type {16khz,24khz,44khz}` script puts `.pt` files in `~/.cache/descript/dac/`. Approximate sizes 300–400 MB each.

### 3. Mimi (Kyutai, the Moshi codec)

**Used by:** Moshi, Sesame CSM, Hibiki translation, and several 2025 streaming TTS. The combination of 12.5 Hz frame rate + dual semantic/acoustic tokens + true streaming made it the dominant codec for full-duplex dialogue models.

**Architecture (`moshi/models/loaders.py`).** Confirmed values from the `_seanet_kwargs` / `_quantizer_kwargs` / `_transformer_kwargs` dicts:

- Sample rate **24000 Hz**, frame rate **12.5 Hz** → hop length 1920.
- Encoder convolution ratios: `[8, 6, 5, 4]` (decoder reverses to `[4, 5, 6, 8]`). Product = 960 — but the **bottleneck transformer downsamples by an additional factor of 2** giving final 1920-sample hop = 12.5 Hz.
- SEANet `dimension=512`, `kernel_size=7`, `n_filters` similar to EnCodec.
- Bottleneck **Transformer** (one before quantizer, one after decoder input): 8 layers, 8 heads, model dim 512, MLP dim 2048, RoPE, GELU, **causal** with finite context 250 frames (20 s).
- All convs are causal (left-padded). Streaming latency ≈ 80 ms.

**Quantizer.** `SplitResidualVectorQuantizer`:

- Total `n_q = 32` codebooks but at inference Moshi uses only the first **8** (one semantic + seven acoustic).
- Codebook size **2048**, codebook dim **256**.
- The first quantizer is a *standalone* VQ distilled from WavLM (semantic tokens — phonetic content).
- The remaining (up to 31) are an RVQ acting on the residual after the semantic quantizer — these encode timbre/prosody/acoustic detail.
- Bitrate at 8 codebooks × 11 bits × 12.5 Hz = **1.1 kbps**.

**API.**
- `encode(wav) -> codes [B, n_q=8, T_frames]` (T_frames = wav_samples / 1920).
- `decode(codes) -> wav [B, 1, T_frames * 1920]`.
- Continuous-mode (pre-quantization latents) is exposed via `encode_to_latent` for downstream non-RVQ uses.

**Checkpoint.** `kyutai/mimi/model.safetensors` (385 MB). Original Kyutai filename: `tokenizer-e351c8d8-checkpoint125.safetensors`. HuggingFace `transformers` ships a port (`transformers.MimiModel`).

### 4. SNAC (Multi-Scale Neural Audio Codec)

**Used by:** Orpheus TTS (3 B / 1 B Llama-3.2-based), Canopy Labs' speech LMs, several recent Apache-2.0 TTS forks. The hierarchical (multi-rate) codebooks let the LM emit a small number of coarse tokens plus more fine tokens, matching the long-range structure of speech.

**Architecture (`snac/snac.py`).** Same DAC-style blocks (Snake + WNConv + dilated residuals) but with **different downsample factors per VQ level**:

| Variant | sampling_rate | encoder_dim | encoder_rates | decoder_dim | decoder_rates | codebook_size | codebook_dim | vq_strides    | attn_window | depthwise |
| ------- | ------------- | ----------- | ------------- | ----------- | ------------- | ------------- | ------------ | ------------- | ----------- | --------- |
| 24 kHz  | 24000         | 48          | [2,4,8,8]     | 1024        | [8,8,4,2]     | 4096          | 8            | [4,2,1]       | null        | true      |
| 32 kHz  | 32000         | 64          | [2,3,8,8]     | 1536        | [8,8,3,2]     | 4096          | 8            | [8,4,2,1]     | 32          | true      |
| 44 kHz  | 44100         | 64          | [2,3,8,8]     | 1536        | [8,8,3,2]     | 4096          | 8            | [8,4,2,1]     | 32          | true      |

**`vq_strides` is the key idea.** For the 24 kHz model:
- After encoder downsample (2·4·8·8 = 512), bottleneck runs at 24000/512 ≈ **46.875 Hz**.
- Three VQ layers operate at different temporal pooling: stride 4, stride 2, stride 1 → coarse codebook fires at ~11.7 Hz, mid at ~23.4 Hz, fine at ~46.9 Hz.
- Orpheus's "7 tokens per audio frame" pattern means: one coarse + two mid + four fine per super-frame. ~83 tokens/sec real-time generation target.

For 32/44 kHz: four VQ layers with strides [8,4,2,1].

**Decoder noise injection.** When `noise=true`, each `DecoderBlock` injects scaled noise (Gaussian, learnable scale) to model unvoiced frication. Not present in DAC.

**Depthwise convs.** All conv layers in 24/32/44 kHz SNAC use `groups = in_channels` (depthwise) plus a 1×1 point-wise — drastically smaller than DAC. 24 kHz model is **79.5 MB** total.

**Local attention.** 32/44 kHz models add `LocalMHA` with `attn_window_size = 32` (windowed self-attention) at the encoder/decoder bottleneck.

**API.**
- `encode(wav) -> List[Tensor]` of three (or four) integer tensors, each `[B, T_i]` with T_i differing per stride.
- `decode(codes_list) -> wav [B, 1, T]`.

**Checkpoint.** `hubertsiuzdak/snac_24khz/pytorch_model.bin` (79.5 MB), `hubertsiuzdak/snac_32khz`, `hubertsiuzdak/snac_44khz`.

### 5. WavTokenizer (Ji et al., ICLR 2025)

**Used by:** various 2025 single-stream audio LMs that don't want to deal with delay-stack RVQ; emerging Chinese TTS systems.

**Architecture.** Encoder is EnCodec/SEANet-style (`n_filters`, ratios [8,5,4,2] for 75 Hz, ratios [8,5,5,3] for 40 Hz variant). Decoder is **Vocos-style**:
- `VocosBackbone` of 12 × ConvNeXt blocks at dim 768, intermediate 2304, with optional self-attention.
- iSTFT head: predicts magnitude + phase spectrogram, inverse-STFT (`n_fft = 1280`, `hop = 320`) → waveform.

**Quantizer.** Single VQ codebook, size **4096**, dim 512. K-means initialized (200 clusters). Achieves high utilization via the "random awakening" trick used at training time — at inference it's just a normal VQ lookup.

**Config (75 Hz variant).**
- `sample_rate = 24000`, `hop = 320`, `n_fft = 1280`, `frame_rate = 75 Hz`, single codebook → 75 × 12 bits = **0.9 kbps**.
- 40 Hz variant uses larger ratios → 600-sample hop, 0.48 kbps.

**API.** `encode(wav) -> codes [B, 1, T/320]` (just one codebook). `decode(codes) -> wav`.

**Checkpoint.** `novateur/WavTokenizer/WavTokenizer_small_320_24k_4096.ckpt` (1.58 GB — large because the Vocos-style decoder is heavy).

### 6. SpeechTokenizer (FunCodec / Zhang 2023)

**Used by:** SoundStorm, USLM, FunCodec-based Chinese TTS, several semantic-conditioned diffusion TTS. First codec to deliberately **disentangle** semantic content (codebook 0) from acoustic detail (codebooks 1–7).

**Architecture (`speechtokenizer/model.py` + pretrained config).**

- Encoder: SEANet with `n_filters=64`, `dimension=1024`, ratios `[8, 5, 4, 2]` (product 320), ELU activation, `weight_norm`, **2 BiLSTM** layers in encoder, regular LSTM in decoder.
- `sample_rate = 16000` → hop 320 → **50 Hz** frame rate.
- Decoder: mirror SEANet, 1024 → 1.
- Quantizer: RVQ with `n_q = 8`, `codebook_size = 1024`, `codebook_dim = 1024` (no factorization).
- Distillation: codebook 0 is trained to match HuBERT layer-9 features via L2 distance (`semantic_dimension = 768` via linear projection). At inference there's no teacher — codebook 0 is just the first RVQ entry.

**Bitrate.** 8 codebooks × 10 bits × 50 Hz = **4 kbps**.

**API.** `encode(wav, n_q=k) -> codes [k, B, T/320]`. `decode(codes) -> wav`.

**Checkpoint.** `fnlp/SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt` (~250 MB).

### 7. XCodec and XCodec2

**XCodec (v1).** Combines an acoustic codec (DAC-like) with a HuBERT semantic encoder. Both feature streams concatenate before a single shared RVQ. Used by early Llasa experiments.

**XCodec2 (the current production model).** Used by **Llasa** (1B / 3B / 8B Llama-based TTS), F5-TTS-MLX variants, and several Chinese voice cloning systems. Major redesign:

- Sample rate **16000 Hz**, hop 320 → **50 Hz** frame rate.
- **Semantic encoder:** `facebook/w2v-bert-2.0`, the 16th hidden layer (1024-D).
- **Acoustic encoder:** `CodecEncoder` (BigCodec-derived, see §8) → 1024-D features.
- The two streams are concatenated (1024 + 1024 = 2048), passed through `fc_prior` (Linear 2048→2048), then through a **single Finite Scalar Quantizer** (FSQ).
- **FSQ codebook size: 65 536** (16 bits per token, equivalent to 2^16). FSQ encodes each scalar in a learned discrete range independently, then combines them — no codebook embedding table is stored, just per-dimension level counts.
- 99% codebook usage reported (vs ~20-30% for vanilla VQ at this scale).
- `fc_post_a` (Linear 2048→1024) projects back to decoder dim.
- **Decoder:** `CodecDecoderVocos` — ConvNeXt-style with iSTFT head producing 16 kHz waveform.

**Bitrate.** 1 token × 16 bits × 50 Hz = **0.8 kbps**.

**API.** `encode(wav) -> codes [B, T/320]` (single integer stream). `decode(codes) -> wav`.

**Checkpoint.** `HKUSTAudio/xcodec2/model.safetensors` (3.29 GB — large because of w2v-BERT). The w2v-BERT semantic encoder is mandatory at encode time; for **decode-only** workloads (LM → audio) we only need `CodecDecoderVocos` + the FSQ index→latent inverse, dramatically smaller.

### 8. BigCodec

**Used by:** standalone high-bitrate TTS demos, base for XCodec2's acoustic branch, and several emerging single-codebook research systems.

**Architecture.**

- Encoder (`vq/codec_encoder.py`):
  - `ngf = 48` initial channels, doubles each block to final 1024.
  - First Conv1d kernel 7, padding 3.
  - Strides `(2, 2, 2, 5, 5)` → hop **200** samples → at 16 kHz = **80 Hz** frame rate. (The earlier 1000-sample/16 Hz number reported in some sources is the BigCodec **decoder** side; encoder is 200.)
  - Dilations per residual block `(1, 3, 9)`.
  - Tail: 2-layer unidirectional LSTM at 1024-D.
  - Final Conv1d kernel 3.

- Decoder (`vq/codec_decoder.py`):
  - Mirror upsample ratios `(5, 5, 2, 2, 2)`, starting channels 1536.
  - Optional 2-layer `ResLSTM` after the initial conv (`use_rnn=True`).
  - 159 M parameters total (~11× larger than EnCodec).
  - Trained with HiFi-GAN-style multi-period + multi-scale discriminators.

- Quantizer: `ResidualVQ` with `num_quantizers = 1`, `codebook_size = 8192`, `codebook_dim = 8`. Single VQ, not RVQ, at inference.

**Bitrate.** 1 × 13 bits × 80 Hz ≈ **1.04 kbps**.

**API.** `encode(wav) -> codes [B, T/200]`. `decode(codes) -> wav`.

**Notable.** BigCodec is the codec that proved you can match 4-6 kbps codecs with a single 8192-entry codebook at ~1 kbps if you make the network big enough — directly inspired XCodec2's "one giant codebook" design.

## Key Numbers / Constants

### Comparison Table

| Codec           | Sample rate (Hz) | Frame rate (Hz) | # codebooks (default) | Codebook size | Codebook dim | Bitrate (kbps) | Causal? | Used by                                              |
| --------------- | ---------------- | --------------- | --------------------- | ------------- | ------------ | -------------- | ------- | ---------------------------------------------------- |
| EnCodec 24k     | 24000            | 75              | 2–32 (variable)       | 1024          | 128          | 1.5–24         | yes     | Bark, AudioGen-16k, dozens of TTS                    |
| EnCodec 48k     | 48000            | 150             | 4–32                  | 1024          | 128          | 3–24           | no      | MusicGen-stereo, AudioCraft music                    |
| EnCodec 32k     | 32000            | 50              | 4 (typical)           | 2048          | 128          | 2.2            | no      | MusicGen mono                                        |
| DAC 16k         | 16000            | 50              | 12                    | 1024          | 8            | 6              | no      | research TTS                                         |
| DAC 24k         | 24000            | 75              | 32                    | 1024          | 8            | 24             | no      | IndexTTS-2 acoustic, ChatTTS (DVAE swap)             |
| DAC 44k         | 44100            | ~86             | 9                     | 1024          | 8            | ~7.75          | no      | Parler-TTS, music TTS                                |
| Mimi            | 24000            | 12.5            | 8 (1 sem + 7 acou)    | 2048          | 256          | 1.1            | yes     | Moshi, Sesame CSM, Hibiki                            |
| SNAC 24k        | 24000            | ~12/24/47       | 3 (multi-scale)       | 4096          | 8            | ~0.98          | no*     | Orpheus TTS                                          |
| SNAC 32k        | 32000            | ~8/16/31/62     | 4 (multi-scale)       | 4096          | 8            | ~1.9           | no      | music TTS forks                                      |
| SNAC 44k        | 44100            | ~9/17/34/69     | 4 (multi-scale)       | 4096          | 8            | ~2.6           | no      | music TTS forks                                      |
| WavTokenizer 75 | 24000            | 75              | 1                     | 4096          | 512          | 0.9            | no      | single-stream audio LM research                      |
| WavTokenizer 40 | 24000            | 40              | 1                     | 4096          | 512          | 0.48           | no      | low-rate variants                                    |
| SpeechTokenizer | 16000            | 50              | 8 (1 sem + 7 acou)    | 1024          | 1024         | 4              | no      | SoundStorm, USLM, FunCodec systems                   |
| XCodec2 / Llasa | 16000            | 50              | 1 (FSQ)               | 65536         | n/a (FSQ)    | 0.8            | no      | Llasa, F5-TTS-MLX, Steveeeeeeen/llasagna             |
| BigCodec        | 16000            | 80              | 1 (VQ)                | 8192          | 8            | 1.04           | no      | standalone HQ demos; XCodec2 acoustic backbone       |

\* SNAC encoder/decoder use centered padding by default but the architecture is symmetric and could be re-padded for streaming.

### Bandwidth / codebook math

For an RVQ codec at bandwidth `bw` (bps), `n_q = bw / (frame_rate * log2(codebook_size))`. EnCodec 24k @ 6 kbps: `6000 / (75 * 10) = 8` codebooks. Mimi @ 1.1 kbps: `1100 / (12.5 * 11) = 8`.

## Data Layouts / Formats

### Codes tensor

Almost every codec returns codes as `int64` (or `int32`) with one of:
- `[batch, n_codebooks, T_frames]` — EnCodec, DAC, Mimi, SpeechTokenizer.
- `List[Tensor[batch, T_frames_i]]` — SNAC (different T per level).
- `[batch, T_frames]` — WavTokenizer, XCodec2, BigCodec (single stream).

### Codebook tensor (in checkpoint)

- **Standard VQ** (EnCodec, SpeechTokenizer, Mimi): `embed: (codebook_size, dim)` stored as `float32`. Some implementations also store `cluster_size` and `embed_avg` from EMA training — these are unused at inference.
- **Factorized VQ** (DAC, SNAC, BigCodec): `in_proj.weight: (codebook_dim, latent_dim, 1)` (Conv1d k=1), `out_proj.weight: (latent_dim, codebook_dim, 1)`, `_codebook.embed: (codebook_size, codebook_dim)`. At encode time you project latent → `codebook_dim`, L2-normalize both sides, take argmin. At decode time you embed-lookup then project up via `out_proj`.
- **FSQ** (XCodec2): there is **no codebook tensor**. The model stores per-dimension level counts (e.g. `[8, 8, 8, 5, 5, 5]` so the 6 dims combine to 8·8·8·5·5·5 = 64 000). Tokens are converted via base-N decomposition: `dim_i_value = (token // prod(levels[i+1:])) % levels[i]`.

### Weight normalization

EnCodec, DAC, SNAC, BigCodec, WavTokenizer all use PyTorch `weight_norm`. The checkpoint stores **two** tensors per conv: `weight_g` (scale, shape (out_channels,) or (out_channels,1,1)) and `weight_v` (direction, full conv shape). To use at inference: pre-compute fused `weight = weight_g * weight_v / ||weight_v||_per_output_channel` once at load time, then forward is a plain conv.

```
# fuse at load time
w_g = state_dict["conv.weight_g"]                   # (out, 1, 1)
w_v = state_dict["conv.weight_v"]                   # (out, in, k)
norm = w_v.reshape(w_v.shape[0], -1).norm(dim=1)    # (out,)
w   = w_g.squeeze() / norm                           # (out,)
fused = w_v * w.view(-1, 1, 1)                       # (out, in, k)
```

### Mmap-safetensors plan

All HF checkpoints listed are <500 MB except XCodec2 (3.3 GB, only ~1 GB needed for decode-only) and BigCodec/WavTokenizer (~1.5 GB). Standard HartsyInference mmap-safetensors loader handles all of these. PyTorch `.pt` checkpoints (WavTokenizer, official SpeechTokenizer, SNAC 24k) must be converted offline using our existing pt→safetensors tool.

## Algorithm Steps

### Generic RVQ encode (EnCodec / DAC / SpeechTokenizer / Mimi)

```
x  = encoder(wav)                    # (B, D, T_frames)
residual = x.transpose(-1, -2)       # (B, T_frames, D)
codes = []                           # List of (B, T_frames) int
for i in range(n_q):
    z   = quantizers[i].in_proj(residual)         # optional factorization
    z_n = l2norm(z)                                # only for DAC/SNAC/BigCodec
    e_n = l2norm(quantizers[i].embed)              # cached at load time
    dist = z_n @ e_n.T                             # (B, T, K), cosine
    idx  = dist.argmax(dim=-1)                     # (B, T)
    q    = quantizers[i].embed[idx]                # (B, T, D_code)
    q_p  = quantizers[i].out_proj(q)               # back to D
    residual = residual - q_p
    codes.append(idx)
return stack(codes, dim=1)                         # (B, n_q, T)
```

### Generic RVQ decode

```
z = 0
for i, idx in enumerate(codes):                    # codes: (B, n_q, T)
    e = quantizers[i].embed[idx]                   # lookup
    z += quantizers[i].out_proj(e)
wav = decoder(z.transpose(-1, -2))                 # (B, 1, T*hop)
```

### SNAC multi-scale decode

```
# codes[0]: (B, T/4)  coarse
# codes[1]: (B, T/2)  mid
# codes[2]: (B, T)    fine
z = 0
for i, idx in enumerate(codes):
    stride = vq_strides[i]
    e = quantizers[i].embed[idx]                          # (B, T_i, D_code)
    e = quantizers[i].out_proj(e).transpose(-1, -2)       # (B, D, T_i)
    e = upsample_nearest_or_linear(e, factor=stride)      # to (B, D, T_max)
    z = z + e
wav = decoder(z)
```

### FSQ decode (XCodec2)

```
# token: int in [0, prod(levels))
# levels: e.g. (8, 8, 8, 5, 5, 5)
indices = []
for L in reversed(levels):
    indices.append(token % L)
    token //= L
indices.reverse()                                  # (n_dim,)
# project each int back to its quantized scalar value
scalars = [round_to_grid(i, L) for i, L in zip(indices, levels)]
z = stack(scalars)                                 # (n_dim,)
z = fc_post_a(z)                                   # → decoder dim
wav = decoder(z)
```

## Reference Implementations

| Codec           | Repo                                                    | Key file(s)                                                                   |
| --------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| EnCodec         | `facebookresearch/encodec`                              | `encodec/model.py`, `encodec/modules/seanet.py`, `encodec/quantization/core_vq.py`, `encodec/quantization/vq.py` |
| DAC             | `descriptinc/descript-audio-codec`                      | `dac/model/dac.py`, `dac/model/encoder.py`, `dac/model/decoder.py`, `dac/nn/layers.py`, `dac/nn/quantize.py` |
| Mimi            | `kyutai-labs/moshi`                                     | `moshi/moshi/models/loaders.py`, `moshi/moshi/modules/seanet.py`, `moshi/moshi/quantization/vq.py`, `moshi/moshi/quantization/core_vq.py` |
| SNAC            | `hubertsiuzdak/snac`                                    | `snac/snac.py`, `snac/layers.py`, `snac/vq.py`, `snac/attention.py`           |
| WavTokenizer    | `jishengpeng/WavTokenizer`                              | `decoder/models.py`, `decoder/feature_extractors.py`, `decoder/heads.py` (iSTFT), `encoder/modules.py` |
| SpeechTokenizer | `ZhangXInFD/SpeechTokenizer`                            | `speechtokenizer/model.py`, `speechtokenizer/modules/seanet.py`, `speechtokenizer/quantization/core_vq.py` |
| XCodec          | `zhenye234/xcodec`                                      | `vq/codec_encoder.py`, `vq/codec_decoder.py`, `vq/module.py`                  |
| XCodec2         | `zhenye234/X-Codec-2.0`                                 | `inference.py`, `vq/codec_decoder_vocos.py`, `vq/codec_encoder.py`, `modeling_xcodec2.py` (on HF) |
| BigCodec        | `Aria-K-Alethia/BigCodec`                               | `vq/codec_encoder.py`, `vq/codec_decoder.py`, `vq/residual_vq.py`             |

HuggingFace ports worth cross-referencing for cleaner code:

- `transformers.EncodecModel` — clean re-implementation of EnCodec 24/48 kHz, weight names already standardized.
- `transformers.MimiModel` — clean port; includes the transformer + split-RVQ stack.
- `transformers.DacModel` — clean port for 16/24/44 kHz.

These three are the easiest reference targets when porting to C# because the HF code uses straightforward `nn.Module` (no PyTorch script tricks, no custom CUDA ops).

## Differences Between Implementations

- **Distance metric in VQ.** EnCodec/Mimi/SpeechTokenizer use plain Euclidean distance. DAC/SNAC/BigCodec L2-normalize first (cosine). Both yield the same nearest neighbor *for unit-norm inputs* but the projection layers around them differ.
- **Codebook dim.** EnCodec/SpeechTokenizer keep `codebook_dim = latent_dim` (no factorization). DAC/SNAC/BigCodec factorize to 8-D. Mimi factorizes to 256-D.
- **Causality.** Only EnCodec 24k, Mimi, and (custom) streaming forks are causal out of the box. Everything else uses centered padding and is offline-only — for streaming TTS we either accept a one-frame look-ahead or re-pad at load time.
- **Weight normalization.** Almost universal. Mimi is the exception — its modern bottleneck-transformer + RMSNorm path uses no `weight_norm` inside the transformer; only the SEANet convs do.
- **Activation function.** EnCodec/SpeechTokenizer/WavTokenizer: ELU. DAC/SNAC/BigCodec/XCodec2: Snake (learnable per-channel). Mimi: GELU inside transformer, ELU in SEANet convs.
- **Decoder topology.** EnCodec/DAC/SNAC/SpeechTokenizer/BigCodec/Mimi: symmetric ConvTranspose. WavTokenizer/XCodec2: asymmetric Vocos-style ConvNeXt + iSTFT head.
- **Final activation.** DAC ends with `tanh`. EnCodec, SNAC, BigCodec end with a plain Conv1d (no activation); waveform is clipped at inference. Vocos-style (WavTokenizer, XCodec2) emits magnitude + phase, no waveform activation at all.
- **Bandwidth selection.** EnCodec and DAC let you keep a prefix of the RVQ codebooks at inference (same checkpoint for many bitrates). Mimi exposes the same trick — Moshi uses 8/32 codebooks. SNAC, WavTokenizer, XCodec2, BigCodec all use a fixed codebook count.
- **Continuous mode.** Mimi, DAC, and EnCodec all expose pre-quantization continuous features (`encode_to_latent`), useful for diffusion-conditioning models that want soft features instead of integer tokens. WavTokenizer / XCodec2 / BigCodec are integer-only by design.

## Open Questions

- **Exact 16 kHz / 24 kHz DAC configs.** Official `dac/model/dac.py` hardcodes the 44.1 kHz defaults. The 16 kHz and 24 kHz checkpoints override `sample_rate`, `encoder_rates`, `decoder_rates`, and `n_codebooks` at construction time but the official repo doesn't expose those values cleanly — we'd need to inspect the `metadata` block inside each `.pt` file (or use the HuggingFace `transformers.DacModel` `config.json`s for each variant) before implementing.
- **Mimi's `n_q=32` vs runtime `8`.** Confirmed via `loaders.py` that the codec is trained with 32 codebooks but Moshi/Sesame use 8. Need to verify that the remaining 24 codebook weights in the checkpoint can be skipped without breaking decoder reconstruction (they should, since RVQ is additive — but worth a numerical test).
- **XCodec2 FSQ exact level list.** Source code references both "65 536 codebook" and FSQ. Need to read `inference.py`'s FSQ instantiation to pull the exact `levels` list (likely `[8, 8, 8, 5, 5, 5]` or `[8, 8, 5, 5, 5, 5, 5]` — both yield ~65 536). Until confirmed, our FSQ decoder needs to be parameterized.
- **SNAC 24kHz `vq_strides=[4,2,1]` decoder lookup.** Three-level SNAC implies the LM emits a pattern like `[c0, m0, m1, f0, f1, f2, f3]` per super-frame (7 tokens). Need to confirm the exact interleaving Orpheus uses — there may be more than one valid order.
- **Continuous-feature endpoint for SpeechTokenizer.** Whether the `semantic_dimension=768` projection produces directly-usable HuBERT-like features at inference, or whether it's only a training auxiliary, is unclear. Doesn't block discrete-token implementation but matters for downstream models that want soft conditioning.
- **WavTokenizer iSTFT phase prediction.** The Vocos head predicts `log(magnitude)` and `phase` then runs torch's `istft`. Need to confirm whether the phase head outputs raw radians or sin/cos pair (the Vocos paper uses sin/cos).
- **BigCodec encoder vs decoder hop mismatch.** Encoder strides `(2,2,2,5,5)` product 200, decoder strides `(5,5,2,2,2)` product 1000 — that's a 5× asymmetry. Source comments suggest internal up/down rates and bypassing — needs careful re-reading of `forward()` before implementation.

## Implementation Notes for HartsyInference

### Ops we need that aren't in IBackend today

1. **ConvTranspose1d.** Required by every codec decoder (except Vocos-style WavTokenizer / XCodec2 which use Conv1d + iSTFT). Implementation is straightforward — equivalent to a Conv1d on dilated/padded input or, more efficiently, a direct strided write-back. Reuse Conv2d kernel by reshape if needed (treat 1-D as `[B, C, 1, T]`).
2. **Snake1d activation.** `y = x + (1 / (alpha + 1e-9)) * sin(alpha * x)^2`, with `alpha` a learnable per-channel parameter of shape `(1, C, 1)`. Trivial to add (element-wise). Used by DAC, SNAC, BigCodec, XCodec2.
3. **Causal Conv1d with KV-cache.** Required only by Mimi and EnCodec-24k streaming mode. For the offline path we can treat causal convs as regular convs with left-only padding. For real-time streaming (LM generates one frame, codec emits 1920 samples) we need to keep the last `(kernel-1)*dilation` input samples per conv layer as a small ring buffer. Recommend implementing this as a wrapper around the existing Conv1d op rather than a new kernel.
4. **LSTM (unidirectional + bidirectional).** EnCodec, SpeechTokenizer, BigCodec use 2-layer LSTM at the encoder/decoder bottleneck. Already on the HartsyInference roadmap for Kokoro — same op. SpeechTokenizer's encoder LSTM is **bi**directional; that doubles the parameters but is just two unidirectional LSTMs running on the forward and reversed sequence with output concat.
5. **Local windowed self-attention.** Only SNAC 32k/44k need this (`attn_window_size = 32`). Implementable as standard scaled-dot-product attention with a band-diagonal mask. Not needed for SNAC 24k (Orpheus's choice).
6. **Bottleneck Transformer (RoPE + GELU).** Mimi only. Standard transformer block — we already have all the pieces (matmul, RMSNorm/LayerNorm, GELU, attention, RoPE).
7. **iSTFT.** WavTokenizer and XCodec2 decoders. We have FFT (for mel spectrogram). iSTFT is FFT⁻¹ + overlap-add — small op, write it once.
8. **FSQ decode/encode.** XCodec2 only. Pure integer math (base decomposition) + a small per-dim quantization grid. No learned codebook tensor.
9. **L2 normalize + cosine VQ lookup.** DAC/SNAC/BigCodec. Easy: `x / ||x||` per row then a matmul with the also-normalized codebook, take argmax. Cache the normalized codebook at load time (the codebook is fixed at inference, no need to re-normalize each call).
10. **Noise injection (SNAC decoder).** Per-channel scaled Gaussian noise added inside each `DecoderBlock`. Needs a deterministic PRNG (Philox or similar) for reproducibility. Not on the critical path quality-wise — can default to zero noise without breaking output.
11. **Depthwise Conv1d (SNAC).** All SNAC convs use `groups = in_channels`. Existing Conv2d kernel can handle grouped convs if it supports the `groups` parameter; otherwise we need a special path (very cheap — it's basically C parallel 1×k convolutions).

### Weight-norm fusion at load time

Implement once as a load-time transform inside the safetensors→IBackend tensor pipeline. Detect any pair of `*.weight_g` + `*.weight_v` keys, fuse, store the result under `*.weight`. This eliminates all runtime cost of weight_norm. Apply this to EnCodec, DAC, SNAC, BigCodec, WavTokenizer, SpeechTokenizer load paths.

### Codebook caching

Store codebooks contiguously in unmanaged memory as `float32[codebook_size, codebook_dim]`. For factorized codecs, also precompute the L2-normalized version and keep both. For RVQ stacks, store as one big `float32[n_q, codebook_size, codebook_dim]` to keep the inner-loop addressing simple and cache-friendly.

### Numerical tolerance vs reference

EnCodec and DAC are float32 throughout; we should match PyTorch float32 reconstruction within ~1e-4 RMS error on standard audio. Mimi uses bf16 in the bundled checkpoint (`moshiko-pytorch-bf16`) — when porting we should upcast bf16 → fp32 at load time for the encoder/decoder and quantizer (the speed cost is negligible compared to the LM), and only keep bf16 for the optional in-bottleneck transformer if memory matters.

### Priority order for HartsyInference

1. **EnCodec 24k** — needed for Bark. Smallest, simplest, well-documented. First codec to implement.
2. **DAC 24k** — needed for IndexTTS-2 and any modern TTS that's swapped EnCodec for DAC. Adds Snake1d.
3. **SNAC 24k** — needed for Orpheus TTS. Adds multi-scale RVQ + depthwise convs + (optional) noise injection.
4. **Mimi** — needed for Moshi and Sesame CSM. Adds bottleneck transformer, causal streaming, split-RVQ.
5. **EnCodec 32k / 48k** — needed for MusicGen / AudioGen / AudioCraft music. Mostly free once #1 works.
6. **SpeechTokenizer** — needed if we add SoundStorm / USLM. Adds BiLSTM.
7. **WavTokenizer** — useful for single-stream LM systems, but requires iSTFT + Vocos backbone (more work).
8. **XCodec2** — needed for Llasa and F5-TTS-MLX. Largest implementation effort because of FSQ + Wav2Vec2-BERT (encoder side). Decode-only path is small.
9. **BigCodec** — lowest priority; mostly subsumed by XCodec2's acoustic backbone.

### Package boundaries

All codecs should live in **HartsyInference.Audio.Codecs**. The Wav2Vec2-BERT semantic encoder required by XCodec2 *encoding* is a separate concern and belongs in **HartsyInference.Audio.SpeechEncoders** (alongside HuBERT, WavLM, Whisper-encoder). Decoder-only usage of XCodec2 doesn't need that dependency, so split the load paths cleanly to avoid forcing every Llasa user to load a 2 GB w2v-BERT they don't need.

### Reference tests

For every codec, ship a regression test that:
1. Loads a 1-second 1 kHz sine wave at the codec's sample rate.
2. Encodes via Python reference (HF transformers or original repo) → saves `codes.npy`.
3. Encodes via HartsyInference → asserts integer-exact match of codes (allow off-by-one drift on edge frames if causality differs, but assert match on the middle 80% of frames).
4. Decodes the reference codes via both → assert per-sample L1 difference < 1e-3 in float32.

XCodec2 FSQ is the only codec where integer-exact code match is mandatory across the entire stream — its quantization grid is deterministic with no near-miss tolerance.
