# HiFiGAN + iSTFTNet + Vocos Vocoder — Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (vocoder)

## Summary

A vocoder converts mel-spectrogram features (or other compressed audio features) into raw audio waveform. This doc covers three families used by SharpInference's TTS pipelines:

1. **HiFiGAN** (Kong et al., 2020) — GAN-trained generator with transposed convolutions + Multi-Receptive Field Fusion. The reference baseline. Used standalone by some TTS systems and by HiFi-GAN-based codecs.
2. **iSTFTNet** (Kaneko et al., 2022) — modifies HiFiGAN by replacing the final upsampling stages with an inverse STFT, predicting magnitude+phase spectrograms instead of waveform samples directly. Used by **StyleTTS 2** and therefore by **Kokoro** ([KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md)).
3. **Vocos** (Siuzdak, 2023) — fully iSTFT-based, ConvNeXt backbone, no temporal upsampling at all. ~13x faster than HiFiGAN, comparable or better quality. Used by F5-TTS ([F5_TTS_ARCHITECTURE.md](F5_TTS_ARCHITECTURE.md)) and others.

Only the generator is needed at inference time — the discriminators are training-only and aren't shipped in checkpoints.

Sources: [HiFi-GAN paper (arXiv:2010.05646)](https://arxiv.org/abs/2010.05646), [iSTFTNet paper (arXiv:2203.02395)](https://arxiv.org/abs/2203.02395), [Vocos paper (arXiv:2306.00814)](https://arxiv.org/abs/2306.00814), [jik876/hifi-gan](https://github.com/jik876/hifi-gan), [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos), [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2).

## Detailed Findings

### Standard HiFiGAN Generator Architecture (Kong et al., 2020)

HiFiGAN is a GAN-based neural vocoder that converts mel spectrograms into raw audio waveforms. The generator uses transposed convolutions for progressive temporal upsampling, with Multi-Receptive Field Fusion (MRF) modules at each stage. The standard V1 configuration uses upsampling factors 8 × 8 × 2 × 2 = 256 (matching a 256-sample hop size) at 22.05 kHz.

Generator has three main sections:

1. **Pre-convolution** (`conv_pre`): Conv1d projecting input mel spectrogram (80 channels) into the hidden dimension (512 for V1). Kernel size 7, padding 3.

2. **Upsampling blocks**: Transposed 1D convolutions, each increasing temporal resolution by a factor specified in `upsample_rates`. After each, the channel count is halved. Each upsampling layer is followed by an MRF module.

3. **Post-convolution** (`conv_post`): Conv1d projecting to a single output channel (waveform). Kernel size 7, padding 3. Followed by `tanh` to constrain output to [-1, 1].

#### Multi-Receptive Field Fusion (MRF) Module

```
MRF(x) = sum(ResBlock_k(x) for k in resblock_kernel_sizes) / num_kernels
```

For V1 with `resblock_kernel_sizes = [3, 7, 11]`, three residual blocks run in parallel per upsampling stage.

#### ResBlock1 (resblock_type=1, used by V1 and V2)

Contains 3 pairs of dilated convolution layers with residual connections:

```
for each dilation_group in dilation_sizes:
    xt = LeakyReLU(x, slope=0.1)
    xt = Conv1d(xt, kernel_size=kr, dilation=d[0])  # dilated conv
    xt = LeakyReLU(xt)
    xt = Conv1d(xt, kernel_size=kr, dilation=1)      # non-dilated conv
    x = x + xt                                        # residual connection
```

For V1 with dilation `[1, 3, 5]` and kernel size 3, a single ResBlock1 has 6 Conv1d layers total. All convolutions use weight normalization during training (removed at inference).

#### ResBlock2 (resblock_type=2, used by V3)

Contains 2 dilated convolution layers per dilation group (instead of 3 pairs). Same structure but fewer layers.

#### Channel Progression (V1)

| Stage | Operation | Input Ch | Output Ch | Temporal Scale |
|-------|-----------|----------|-----------|----------------|
| pre | Conv1d(k=7) | 80 | 512 | 1x |
| up_0 | ConvTranspose1d(k=16, s=8) | 512 | 256 | 8x |
| MRF_0 | 3 × ResBlock1 | 256 | 256 | 8x |
| up_1 | ConvTranspose1d(k=16, s=8) | 256 | 128 | 64x |
| MRF_1 | 3 × ResBlock1 | 128 | 128 | 64x |
| up_2 | ConvTranspose1d(k=4, s=2) | 128 | 64 | 128x |
| MRF_2 | 3 × ResBlock1 | 64 | 64 | 128x |
| up_3 | ConvTranspose1d(k=4, s=2) | 64 | 32 | 256x |
| MRF_3 | 3 × ResBlock1 | 32 | 32 | 256x |
| post | Conv1d(k=7) | 32 | 1 | 256x |

Total upsampling: 8 × 8 × 2 × 2 = 256 (matches hop_size=256).

### HiFiGAN Configuration Variants

| Parameter | V1 | V2 | V3 |
|-----------|----|----|-----|
| `upsample_initial_channel` (hu) | 512 | 128 | 256 |
| `upsample_rates` | [8, 8, 2, 2] | [8, 8, 2, 2] | [8, 8, 4] |
| `upsample_kernel_sizes` (ku) | [16, 16, 4, 4] | [16, 16, 4, 4] | [16, 16, 8] |
| `resblock_type` | 1 | 1 | 2 |
| `resblock_kernel_sizes` (kr) | [3, 7, 11] | [3, 7, 11] | [3, 5, 7] |
| `resblock_dilation_sizes` (Dr) | [[1,3,5], [1,3,5], [1,3,5]] | [[1,3,5], [1,3,5], [1,3,5]] | [[1,2], [2,6], [3,12]] |
| Sample rate | 22,050 Hz | 22,050 Hz | 22,050 Hz |
| Hop size | 256 | 256 | 256 |
| FFT size | 1,024 | 1,024 | 1,024 |
| Mel bins | 80 | 80 | 80 |

V1 is the highest quality (largest), V2 is the fastest (smallest hidden dim), V3 uses fewer upsampling stages with ResBlock2.

### iSTFTNet Modification (Kaneko et al., 2022)

iSTFTNet modifies HiFiGAN by replacing output-side transposed convolution layers with an inverse Short-Time Fourier Transform. Instead of generating waveform samples directly, the network predicts magnitude and phase spectrograms, which are then converted to audio via iSTFT.

Key insight: after sufficient upsampling reduces the frequency dimension, the remaining time-to-waveform conversion can be handled analytically by iSTFT rather than learned convolutions.

Three variants (applied to HiFiGAN V1 with upsample_rates [8,8,2,2]):
- **C8C8I**: Keep 2 upsampling stages (8×, 8×), replace last 2 with iSTFT. Best quality/speed tradeoff.
- **C8I**: Keep 1 upsampling stage (8×), replace rest with iSTFT. Fastest but lower quality.
- **CI**: No upsampling, all done by iSTFT. Poor quality.

The final layer outputs `(n_fft/2 + 1) * 2` channels: half for magnitude (passed through `exp()`) and half for phase (passed through `sin()`). The iSTFT then reconstructs the waveform.

### Kokoro/StyleTTS2 iSTFTNet Decoder (What We Implement for Kokoro)

Kokoro-82M is fine-tuned from [StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) and uses StyleTTS2's iSTFTNet decoder with additional modifications. This is NOT standard HiFiGAN.

#### Configuration (from StyleTTS2 config.yml)

```yaml
decoder:
  type: 'istftnet'
  resblock_kernel_sizes: [3, 7, 11]
  upsample_rates: [10, 6]
  upsample_initial_channel: 512
  resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
  upsample_kernel_sizes: [20, 12]
  gen_istft_n_fft: 20
  gen_istft_hop_size: 5

preprocess_params:
  sr: 24000
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300
    n_mels: 80
```

#### Key Differences from Standard HiFiGAN

1. **Only 2 upsampling stages** (not 4): rates [10, 6] instead of [8, 8, 2, 2]
2. **iSTFT output head**: Replaces the final conv_post + tanh with magnitude/phase prediction + iSTFT
3. **AdaIN conditioning**: Each ResBlock uses Adaptive Instance Normalization (AdaINResBlock1) that modulates features using a style vector
4. **Snake activation**: Uses `x + (1/alpha) * sin^2(alpha * x)` instead of LeakyReLU in residual blocks
5. **Harmonic-plus-noise source module** (SourceModuleHnNSF): Generates F0-conditioned harmonic source signals (8 harmonics) and noise, injected at each upsampling stage
6. **Style dimension**: 64 (internal decoder style, separate from the 256-dim model-level style embedding)
7. **24 kHz sample rate** (not 22.05 kHz)

#### Decoder Class Detail

Source: [`kokoro/istftnet.py`](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py)

1. **F0 conv**: `Conv1d(1, hidden_dim, kernel_size=3, stride=2)` — downsamples F0 by 2x.
2. **N conv**: `Conv1d(1, hidden_dim, kernel_size=3, stride=2)` — downsamples energy by 2x.
3. **Encode block**: `AdainResBlk1d(dim_in + 2*hidden_dim, 1024, style_dim)` — fuses ASR features, F0, and energy.
4. **Decode blocks**: 4 stages of `AdainResBlk1d` that progressively refine, with residual ASR features, F0, and N concatenated at each stage.
5. **Generator**: The actual iSTFTNet waveform synthesis.

#### Generator Class Detail

1. **Source module (SourceModuleHnNSF)**: Generates harmonic sine waves (8 harmonics) + noise from F0 at the target sample rate. Uses SineGen with `voiced_threshold=10`.
2. **Upsampling**: Two stages of `ConvTranspose1d` with rates `[10, 6]` and kernel sizes `[20, 12]`. Starting from 512 channels, halving at each stage: 512 -> 256 -> 128.
3. **Residual blocks**: At each upsampling level, 3 `AdaINResBlock1` blocks with kernel sizes `[3, 7, 11]` and dilations `[[1,3,5], [1,3,5], [1,3,5]]`. All conditioned on the style vector.
4. **Noise conditioning**: Parallel `Conv1d` + `AdaINResBlock1` path processes the STFT of the source signal at each level.
5. **Final projection**: `Conv1d` -> `(n_fft + 2)` channels, split into magnitude (exponentiated) and phase (sin/cos), then inverse STFT.

#### Channel Progression (Kokoro/StyleTTS2)

| Stage | Operation | Input Ch | Output Ch | Temporal Scale |
|-------|-----------|----------|-----------|----------------|
| pre | Conv1d(k=7) | 512 (style-conditioned input) | 512 | 1x |
| up_0 | ConvTranspose1d(k=20, s=10) | 512 | 256 | 10x |
| noise_0 | Conv1d(har, k=12, s=6) | 22 | 256 | (harmonic injection) |
| MRF_0 | 3 × AdaINResBlock1 | 256 | 256 | 10x |
| up_1 | ConvTranspose1d(k=12, s=6) | 256 | 128 | 60x |
| noise_1 | Conv1d(har, k=1, s=1) | 22 | 128 | (harmonic injection) |
| MRF_1 | 3 × AdaINResBlock1 | 128 | 128 | 60x |
| post | Conv1d -> split | 128 | 22 (11 mag + 11 phase) | 60x |
| iSTFT | iSTFT(n_fft=20, hop=5) | 11 complex | 1 | 300x (60 × 5) |

Total effective upsampling: 10 × 6 × 5 = 300 (matches hop_length=300 at 24 kHz).

#### iSTFT Output Head

The post-convolution outputs `gen_istft_n_fft + 2 = 22` channels:
- Channels 0-10: log-magnitude, passed through `exp()` to get magnitude
- Channels 11-21: raw phase, passed through `sin()` to get phase

Then `iSTFT(magnitude * exp(j * phase), n_fft=20, hop_size=5, window=hann)` reconstructs the waveform.

**Snake activation**: `x + (1/alpha) * sin^2(alpha * x)` where alpha is a learnable parameter. This provides a periodic, smooth non-linearity well-suited for audio.

**AdaIN1d**: `(1 + gamma) * InstanceNorm(x) + beta` where gamma and beta are projected from the style vector via `Linear(style_dim, num_features * 2)`.

### Vocos Architecture (Alternative)

Vocos (Siuzdak, 2023) is a fundamentally different approach:

- **No transposed convolutions**: Maintains constant temporal resolution throughout
- **ConvNeXt backbone**: Stack of 1D depthwise convolution + inverted bottleneck blocks with GELU activation and LayerNorm
- **All upsampling via iSTFT**: The only temporal expansion happens in the final iSTFT layer
- **iSTFT head**: Predicts magnitude via `exp(m)` and phase via unit circle projection `(cos(p), sin(p))`
- **Speed**: Approximately 13x faster than HiFiGAN, approximately 70x faster than BigVGAN on GPU
- **Quality**: Comparable or better than HiFiGAN (VISQOL 4.66 vs 4.57; PESQ 3.70 vs 3.09)
- **Parameters**: Trained on LibriTTS at 24 kHz with n_fft=1024, hop=256, 100 mel bins

#### Vocos Variants (Pretrained)

- `charactr/vocos-mel-24khz` — mel-input (80 bins), 24kHz output. Used as a drop-in HiFiGAN replacement.
- `charactr/vocos-encodec-24khz` — consumes EnCodec discrete codes (4-8 codebooks), 24kHz output. Codec vocoder.

F5-TTS uses `vocos-mel-24khz` as its default vocoder. See [F5_TTS_ARCHITECTURE.md](F5_TTS_ARCHITECTURE.md) and [AUDIO_CODECS.md](AUDIO_CODECS.md).

#### Vocos ConvNeXt Block

```
x_in = x
x = DepthwiseConv1d(x, kernel=7, padding=3)       # spatial mixing
x = LayerNorm(x)                                    # channels-last layer norm
x = Linear(x, dim * 4)                              # inverted bottleneck up
x = GELU(x)
x = Linear(x, dim)                                  # inverted bottleneck down
x = AdaLNGamma(x) if conditioning else x            # optional scale
return x_in + x                                     # residual
```

Stack 8 of these (standard config), with optional adaptive LayerNorm for conditioning on speaker / discrete token / language id.

### Comparison: Standard HiFiGAN vs Kokoro iSTFTNet vs Vocos

| Aspect | Standard HiFiGAN V1 | Kokoro/StyleTTS2 iSTFTNet | Vocos |
|--------|---------------------|--------------------------|-------|
| Sample rate | 22,050 Hz | 24,000 Hz | 24,000 Hz |
| Hop size | 256 | 300 | 256 |
| Upsampling stages | 4 (8,8,2,2) | 2 (10,6) + iSTFT(hop=5) | All iSTFT |
| Total upsampling | 256x | 300x (60x conv + 5x iSTFT) | Via iSTFT |
| Output method | tanh(conv_post) | iSTFT(magnitude, phase) | iSTFT(magnitude, phase) |
| Activation | LeakyReLU(0.1) | Snake activation | GELU |
| Normalization | Weight norm | AdaIN (style-conditioned) | LayerNorm |
| Conditioning | None (unconditional) | Style vector + F0 | None (mel-only) or AdaLN |
| Source model | None | HnNSF (harmonic+noise, F0) | None |
| Backbone | Transposed conv + ResBlocks | Transposed conv + ResBlocks | ConvNeXt blocks |
| Temporal resolution | Changes through network | Changes through network | Constant |
| Speed vs HiFiGAN | 1x | ~1.7x faster | ~13x faster |
| Parameters (generator) | ~14M (V1) | Part of 82M total | ~13.5M |

## Key Numbers / Constants

### Standard HiFiGAN V1 (Reference)

| Constant | Value | Notes |
|----------|-------|-------|
| Input mel bins | 80 | Standard across all variants |
| Sample rate | 22,050 Hz | Standard HiFiGAN |
| Hop size | 256 | Matches total upsampling factor |
| FFT size | 1,024 | For mel spectrogram computation |
| Window size | 1,024 | Hann window |
| Total upsampling | 256x | 8 × 8 × 2 × 2 |
| Initial channels | 512 | V1 hidden dimension |
| LeakyReLU slope | 0.1 | Used throughout generator |
| Output range | [-1.0, 1.0] | tanh activation |

### Kokoro/StyleTTS2 iSTFTNet

| Constant | Value | Notes |
|----------|-------|-------|
| Analysis FFT size | 2,048 | `n_fft` for mel computation |
| Analysis window size | 1,200 | `win_length` for mel computation |
| Synthesis iSTFT n_fft | 20 | `gen_istft_n_fft` |
| Synthesis iSTFT hop | 5 | `gen_istft_hop_size` |
| Style dimension (decoder) | 64 | AdaIN conditioning |
| Harmonic count | 8 | SourceModuleHnNSF |
| Voiced threshold | 10 | SourceModuleHnNSF |
| Post conv output channels | 22 | n_fft + 2 = 22 (11 mag + 11 phase) |

### Vocos (vocos-mel-24khz)

| Constant | Value | Notes |
|----------|-------|-------|
| Sample rate | 24,000 Hz | |
| Mel bins | 100 | (Note: 100, not 80) |
| FFT size (mel computation) | 1,024 | |
| Hop size (mel computation) | 256 | |
| iSTFT n_fft | 1,024 | Synthesis matches analysis |
| iSTFT hop size | 256 | |
| Hidden dim | 512 | |
| ConvNeXt depth | 8 layers | |
| Parameters | ~13.5M | |

## Data Layouts / Formats

### Input: Mel Spectrogram
```
Shape: [batch, n_mels, T_mel]   (n_mels = 80 or 100 depending on model)
Type: float32
Range: log-scale mel spectrogram values (typically -11.0 to 2.0)
```

### Internal: HiFiGAN After Upsampling
```
After pre-conv:   [batch, 512, T_mel]
After up_0:       [batch, 256, T_mel × 8]
After up_1:       [batch, 128, T_mel × 64]
After up_2:       [batch, 64,  T_mel × 128]
After up_3:       [batch, 32,  T_mel × 256]
After post-conv:  [batch, 1,   T_mel × 256]
After tanh:       [batch, 1,   T_mel × 256]  values in [-1, 1]
```

### Internal: Kokoro iSTFTNet
```
After pre-conv:   [batch, 512, T_mel]
After up_0:       [batch, 256, T_mel × 10]
After up_1:       [batch, 128, T_mel × 60]
After post-conv:  [batch, 22,  T_mel × 60]
  - channels 0-10: log-magnitude spectrogram
  - channels 11-21: phase spectrogram
After iSTFT:      [batch, 1, T_mel × 300]
```

### Internal: Vocos
```
After embed:      [batch, 100, T_mel] -> [batch, 512, T_mel]
After 8 ConvNeXt: [batch, 512, T_mel]
After head:       [batch, n_fft + 2, T_mel] = [batch, 1026, T_mel]
After iSTFT:      [batch, T_mel × 256]
```

### Internal: Harmonic Source (Kokoro)
```
F0 input:         [batch, 1, T_mel]  (fundamental frequency per mel frame)
F0 upsampled:     [batch, 1, T_mel × 300]  (upsampled to sample rate)
Harmonic source:  [batch, 1, T_mel × 300]  (sinusoidal signal)
Harmonic STFT:    [batch, 22, T_mel × 60]  (STFT of harmonic source, n_fft=20)
  - channels 0-10: harmonic magnitude
  - channels 11-21: harmonic phase
```

### Output: Audio Waveform
```
Shape: [batch, 1, T_audio] where T_audio = T_mel × hop_size
Type: float32, range [-1.0, 1.0]
For 16-bit PCM: multiply by 32767, clamp to [-32768, 32767], cast to int16
```

## Algorithm Steps

### Standard HiFiGAN V1 Forward Pass

```
1. x = conv_pre(mel)                           # [B, 80, T] -> [B, 512, T]
2. For each upsampling stage i in [0, 1, 2, 3]:
   a. x = LeakyReLU(x, slope=0.1)
   b. x = ups[i](x)                            # transposed convolution
   c. xs = 0
   d. For each kernel j in [0, 1, 2]:           # resblock_kernel_sizes = [3, 7, 11]
      xs += resblocks[i*3 + j](x)              # ResBlock1 with dilations [1,3,5]
   e. x = xs / 3                                # average MRF outputs
3. x = LeakyReLU(x, slope=0.1)
4. x = conv_post(x)                             # [B, 32, T*256] -> [B, 1, T*256]
5. x = tanh(x)
6. return x
```

### ResBlock1 Forward Pass

```
For each dilation d in [1, 3, 5]:
  xt = LeakyReLU(x, slope=0.1)
  xt = Conv1d(xt, kernel=kr, dilation=d, padding=d*(kr-1)//2)
  xt = LeakyReLU(xt, slope=0.1)
  xt = Conv1d(xt, kernel=kr, dilation=1, padding=(kr-1)//2)
  x = x + xt
return x
```

### Kokoro/StyleTTS2 iSTFTNet Forward Pass

```
1. Upsample F0 to audio sample rate:
   f0_up = interpolate(f0, scale_factor=300)    # [B, 1, T] -> [B, 1, T*300]

2. Generate harmonic+noise source from F0:
   har_source, noi_source, uv = SourceModuleHnNSF(f0_up)

3. Compute STFT of harmonic source:
   har_spec, har_phase = STFT(har_source, n_fft=20, hop=5)
   har = concat(har_spec, har_phase, dim=1)     # [B, 22, T*60]

4. Process through upsampling stages:
   For each stage i in [0, 1]:
     a. x = LeakyReLU(x, slope=0.1)
     b. x_source = noise_convs[i](har)           # harmonic feature injection
        x_source = noise_res[i](x_source, style) # style-conditioned
     c. x = ups[i](x)                            # transposed conv upsample
     d. x = x + x_source                         # add harmonic conditioning
     e. xs = 0
     f. For each kernel j in [0, 1, 2]:
        xs += adain_resblocks[i*3+j](x, style)  # AdaIN + Snake activation
     g. x = xs / 3

5. Generate magnitude and phase:
   x = conv_post(x)                              # [B, 128, T*60] -> [B, 22, T*60]
   magnitude = exp(x[:, :11, :])                 # positive magnitude
   phase = sin(x[:, 11:, :])                     # phase wrapped to [-1, 1]

6. Reconstruct waveform via iSTFT:
   audio = iSTFT(magnitude, phase, n_fft=20, hop_size=5, window=hann)

7. return audio
```

### AdaINResBlock1 Forward Pass (Snake + AdaIN)

```
For each dilation d in [1, 3, 5]:
  xt = AdaIN1d(x, style)                       # normalize, then scale/shift by style
  xt = Snake(xt, alpha)                         # x + (1/alpha) * sin^2(alpha * x)
  xt = Conv1d(xt, kernel=kr, dilation=d)
  xt = AdaIN1d(xt, style)
  xt = Snake(xt, alpha)
  xt = Conv1d(xt, kernel=kr, dilation=1)
  x = x + xt
return x
```

### SourceModuleHnNSF (Harmonic Source Generation)

```
Given f0 (fundamental frequency in Hz) at sample rate:
1. For k in 0..harmonic_num (0..8):
   phase[k] = cumsum(f0 * (k+1) / sample_rate) * 2 * pi
   sine[k] = sin(phase[k])
   sine[k] *= voiced_mask(f0 > threshold)       # zero out unvoiced
2. harmonic_source = linear_combination(sines)   # learned weights
3. noise_source = gaussian_noise * learned_scale
4. return harmonic_source, noise_source, uv_flag
```

### Vocos Forward Pass

```
1. x = mel_embed(mel)                            # [B, 100, T] -> [B, 512, T]
2. For block in convnext_blocks:                 # 8 blocks
   a. residual = x
   b. x = depthwise_conv1d(x, kernel=7)
   c. x = layernorm(x.transpose) .transpose      # channels-last LN
   d. x = linear(x, 2048)                        # inverted bottleneck up
   e. x = GELU(x)
   f. x = linear(x, 512)                         # inverted bottleneck down
   g. x = residual + x
3. x = layernorm(x)
4. h = head_linear(x)                            # [B, 512, T] -> [B, 1026, T]
5. magnitude = exp(h[:, :513, :])
6. phase = atan2(h[:, 513:1026, :], shifted)     # via cos+sin head
7. audio = iSTFT(magnitude * exp(j*phase), n_fft=1024, hop=256, window=hann)
8. return audio
```

### Weight Normalization Notes

Standard HiFiGAN uses `weight_norm` on all Conv1d layers during training. At inference, weight normalization should be fused/removed for speed:
```
weight = weight_g * (weight_v / norm(weight_v))
```
This can be pre-computed and stored as a single weight tensor. Kokoro's iSTFTNet uses a similar pattern with `ConvWeighted` separating magnitude (`weight_g`) and direction (`weight_v`). At conversion time (safetensors / .pth -> our format) we fuse these into a single Conv1d weight tensor; no runtime cost.

## Reference Implementations

- [jik876/hifi-gan](https://github.com/jik876/hifi-gan) — Official HiFiGAN V1/V2/V3.
- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) — iSTFTNet decoder with AdaIN, what Kokoro uses.
- [hexgrad/kokoro](https://github.com/hexgrad/kokoro) — Kokoro TTS, fine-tuned from StyleTTS2-LJSpeech.
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch) — Standalone iSTFTNet implementation.
- [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) — Vocos reference implementation.
- [charactr-platform/vocos](https://huggingface.co/charactr) — Vocos pretrained weights (vocos-mel-24khz, vocos-encodec-24khz).
- [Blaizzy/mlx-audio](https://deepwiki.com/Blaizzy/mlx-audio/3.2-api-reference) — MLX port, good architecture reference.
- [torchaudio HiFiGANVocoder](https://docs.pytorch.org/audio/2.8/generated/torchaudio.prototype.models.HiFiGANVocoder.html) — Clean reference with V1/V2/V3 factory functions.
- [SpeechBrain HiFiGAN](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.HifiGAN.html) — Well-documented HiFiGAN with both ResBlock types.

## Open Questions

- [ ] Exact parameter count of the iSTFTNet decoder portion alone (vs. full 82M model)
- [ ] Whether Kokoro's iSTFTNet weights can be loaded independently from the rest of the model
- [ ] Whether the F0 (pitch) input is required or can be set to zero for inference (probably not — the model is trained with F0 as a strong conditioning signal)
- [ ] Whether to ship Vocos as a separate package or bundle it into SharpInference.Audio (probably bundle — Vocos is small and shared by F5-TTS, Kokoro-vocos variants, EnCodec decoding paths)

## Implementation Notes for SharpInference

1. **Implement iSTFTNet first, not standard HiFiGAN.** Kokoro is our first TTS target and it uses iSTFTNet. Standard HiFiGAN is only useful as a reference/test target.

2. **Key operations to implement:**
   - 1D transposed convolution (ConvTranspose1d) with stride > 1
   - 1D dilated convolution (Conv1d with dilation parameter)
   - Adaptive Instance Normalization (AdaIN1d): normalize per-channel, then apply learned affine from style vector
   - Snake activation: `x + (1/alpha) * sin^2(alpha * x)` — requires element-wise sin and multiply
   - iSTFT: inverse Short-Time Fourier Transform with Hann window, n_fft=20, hop=5
   - SourceModuleHnNSF: cumulative sum of phase, sinusoidal generation, learned mixing

3. **iSTFT implementation:** With n_fft=20 and hop_size=5, this is a very small FFT (only 20 points). Can be implemented efficiently with a direct DFT formula rather than a full FFT library. The overlap-add synthesis window is a 20-point Hann window.

4. **Memory layout:** All intermediate tensors are `[batch, channels, time]` (channels-first). The time dimension grows through the network: T_mel -> T_mel*10 -> T_mel*60 -> T_audio (T_mel*300).

5. **Weight normalization fusion:** Pre-compute `weight = weight_g * weight_v / norm(weight_v)` during model conversion (offline) to avoid runtime overhead and simpler safetensors layout.

6. **Transposed convolution padding:** For `ConvTranspose1d(kernel=k, stride=s)`, the output length is `(input_length - 1) * stride + kernel - 2 * padding`. The kernel sizes are chosen as 2x the stride (k=20 for s=10, k=12 for s=6), which with appropriate padding produces clean upsampling.

7. **16-bit PCM output:** After the vocoder produces float32 audio in [-1, 1]:
   ```
   pcm16 = clamp(round(audio * 32767), -32768, 32767)
   ```
   Write as little-endian int16 for WAV file output at 24,000 Hz sample rate.

8. **Vocos for F5-TTS path**: F5-TTS uses `vocos-mel-24khz` directly. Implementing Vocos doubles as the F5-TTS vocoder. ConvNeXt blocks are simpler than HiFiGAN — no transposed convs at all. Plan to implement Vocos alongside iSTFTNet.

9. **iSTFT shared kernel**: All three vocoder families need iSTFT (HiFiGAN's tanh head is the only outlier). Implement once in SharpInference.Audio as `IStftLayer`. Match `torch.istft(window=hann, center=True, normalized=False)` exactly.

10. **GPU kernels needed for the vocoder:**
    - Conv1D (existing CUDA backend has 2D via im2col; need 1D variant — implement via im2col_1d or special-case)
    - ConvTranspose1D (new — implement via the "stride-2 input with zero-stuffing + Conv1D" trick or directly)
    - InstanceNorm1D (new — similar to GroupNorm with num_groups=channels)
    - Snake activation (new — element-wise; trivial PTX)
    - iSTFT (new — composes FFT, complex multiply, overlap-add window-sum)

11. **Cross-references:** [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md) for the model that calls this vocoder, [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) for the input mel computation (note: Kokoro and HiFiGAN use different mel parameters), [F5_TTS_ARCHITECTURE.md](F5_TTS_ARCHITECTURE.md) for the Vocos consumer.
