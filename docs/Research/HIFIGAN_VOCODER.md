# HiFiGAN Vocoder — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Audio

## Summary

HiFiGAN (Kong et al., 2020) is a GAN-based neural vocoder that converts mel spectrograms into raw audio waveforms. The generator uses transposed convolutions for progressive temporal upsampling, with Multi-Receptive Field Fusion (MRF) modules at each stage that combine multiple residual blocks with different kernel sizes and dilation rates. The standard V1 configuration uses upsampling factors 8 x 8 x 2 x 2 = 256 (matching a 256-sample hop size) at 22.05 kHz. Only the generator is needed at inference time.

**Kokoro does NOT use standard HiFiGAN.** Kokoro-82M uses a modified iSTFTNet decoder (based on StyleTTS2's architecture) that replaces the final upsampling stages with an inverse Short-Time Fourier Transform. The Kokoro/StyleTTS2 iSTFTNet uses only 2 transposed convolution stages (10x, 6x = 60x total) followed by an iSTFT with n_fft=20 and hop_size=5 (effective 5x), yielding 60 x 5 = 300 total upsampling (matching the 300-sample hop at 24 kHz). It also uses AdaIN (Adaptive Instance Normalization) conditioning, Snake activation, and harmonic-plus-noise source filtering (F0-conditioned).

Vocos is a faster alternative (approximately 13x faster than HiFiGAN on GPU) that uses ConvNeXt blocks at constant temporal resolution and performs all upsampling via iSTFT, but Kokoro currently ships with iSTFTNet.

Sources: [HiFi-GAN paper (NeurIPS 2020)](https://papers.neurips.cc/paper_files/paper/2020/file/c5d736809766d46260d816d8dbc9eb44-Paper.pdf), [HiFi-GAN GitHub](https://github.com/jik876/hifi-gan), [iSTFTNet paper (ICASSP 2022)](https://arxiv.org/abs/2203.02395), [StyleTTS2 GitHub](https://github.com/yl4579/StyleTTS2), [Kokoro-82M HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M), [Vocos paper (ICLR 2024)](https://arxiv.org/abs/2306.00814), [torchaudio HiFiGAN](https://docs.pytorch.org/audio/2.8/generated/torchaudio.prototype.models.HiFiGANVocoder.html)

## Detailed Findings

### Standard HiFiGAN Generator Architecture (Kong et al., 2020)

The generator is a fully convolutional neural network. The architecture has three main sections:

1. **Pre-convolution** (`conv_pre`): A 1D convolution that projects the input mel spectrogram (80 channels) into the hidden dimension (e.g., 512 channels for V1). Kernel size 7, padding 3.

2. **Upsampling blocks**: A series of transposed 1D convolutions, each increasing temporal resolution by a factor specified in `upsample_rates`. After each transposed convolution, the channel count is halved. Each upsampling layer is followed by an MRF module.

3. **Post-convolution** (`conv_post`): A 1D convolution that projects to a single output channel (the waveform). Kernel size 7, padding 3. Followed by `tanh` activation to constrain output to [-1, 1].

#### Multi-Receptive Field Fusion (MRF) Module

The MRF module sums the outputs of multiple residual blocks, each with a different kernel size from `resblock_kernel_sizes`. This creates diverse receptive field patterns:

```
MRF(x) = sum(ResBlock_k(x) for k in resblock_kernel_sizes) / num_kernels
```

For V1 with `resblock_kernel_sizes = [3, 7, 11]`, three residual blocks run in parallel per upsampling stage.

#### ResBlock1 (resblock_type=1)

Used by V1 and V2. Contains 3 pairs of dilated convolution layers with residual connections. Each pair applies:

```
for each dilation_group in dilation_sizes:
    xt = LeakyReLU(x, slope=0.1)
    xt = Conv1d(xt, kernel_size=kr, dilation=d[0])  # dilated conv
    xt = LeakyReLU(xt)
    xt = Conv1d(xt, kernel_size=kr, dilation=1)      # non-dilated conv
    x = x + xt                                        # residual connection
```

For V1 with dilation `[1, 3, 5]` and kernel size 3, a single ResBlock1 has 6 Conv1d layers total (3 dilated + 3 non-dilated). All convolutions use weight normalization during training (removed at inference).

#### ResBlock2 (resblock_type=2)

Used by V3. Contains 2 dilated convolution layers per dilation group (instead of 3 pairs). Same structure but fewer layers, making it lighter.

#### Channel Progression (V1 Example)

| Stage | Operation | Input Channels | Output Channels | Temporal Scale |
|-------|-----------|---------------|----------------|----------------|
| pre | Conv1d(k=7) | 80 | 512 | 1x |
| up_0 | ConvTranspose1d(k=16, s=8) | 512 | 256 | 8x |
| MRF_0 | 3 x ResBlock1 | 256 | 256 | 8x |
| up_1 | ConvTranspose1d(k=16, s=8) | 256 | 128 | 64x |
| MRF_1 | 3 x ResBlock1 | 128 | 128 | 64x |
| up_2 | ConvTranspose1d(k=4, s=2) | 128 | 64 | 128x |
| MRF_2 | 3 x ResBlock1 | 64 | 64 | 128x |
| up_3 | ConvTranspose1d(k=4, s=2) | 64 | 32 | 256x |
| MRF_3 | 3 x ResBlock1 | 32 | 32 | 256x |
| post | Conv1d(k=7) | 32 | 1 | 256x |

Total upsampling: 8 x 8 x 2 x 2 = 256 (matches hop_size=256).

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
| FFT size | 1024 | 1024 | 1024 |
| Window size | 1024 | 1024 | 1024 |
| Mel bins | 80 | 80 | 80 |

V1 is the highest quality (largest), V2 is the fastest (smallest hidden dim), V3 uses fewer upsampling stages with ResBlock2.

### iSTFTNet Modification (Kaneko et al., 2022)

iSTFTNet modifies HiFiGAN by replacing output-side transposed convolution layers with an inverse Short-Time Fourier Transform. Instead of generating waveform samples directly, the network predicts magnitude and phase spectrograms, which are then converted to audio via iSTFT.

Key insight: after sufficient upsampling reduces the frequency dimension, the remaining time-to-waveform conversion can be handled analytically by iSTFT rather than learned convolutions.

Three variants were proposed (applied to HiFiGAN V1 with upsample_rates [8,8,2,2]):
- **C8C8I**: Keep 2 upsampling stages (8x, 8x), replace last 2 stages (2x, 2x) with iSTFT. Best quality/speed tradeoff.
- **C8I**: Keep 1 upsampling stage (8x), replace rest with iSTFT. Fastest but lower quality.
- **CI**: No upsampling, all done by iSTFT. Poor quality.

The final layer outputs `(n_fft/2 + 1) * 2` channels: half for magnitude (passed through `exp()`) and half for phase (passed through `sin()`). The iSTFT then reconstructs the waveform.

### Kokoro/StyleTTS2 iSTFTNet Decoder (What We Actually Need)

Kokoro-82M is fine-tuned from [StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) and uses StyleTTS2's iSTFTNet decoder with additional modifications. This is NOT standard HiFiGAN.

#### Configuration (from [StyleTTS2 config.yml](https://github.com/yl4579/StyleTTS2/blob/main/Configs/config.yml))

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
3. **AdaIN conditioning**: Each ResBlock uses Adaptive Instance Normalization (AdaINResBlock1) that modulates features using a style vector, enabling voice/prosody control
4. **Snake activation**: Uses `x + (1/alpha) * sin^2(alpha * x)` instead of LeakyReLU in residual blocks
5. **Harmonic-plus-noise source module** (SourceModuleHnNSF): Generates F0-conditioned harmonic source signals (8 harmonics) and noise, injected at each upsampling stage via noise convolutions
6. **Style dimension**: 64 (internal decoder style, separate from the 256-dim model-level style embedding)
7. **24 kHz sample rate** (not 22.05 kHz)

#### Channel Progression (Kokoro/StyleTTS2)

| Stage | Operation | Input Ch | Output Ch | Temporal Scale |
|-------|-----------|----------|-----------|----------------|
| pre | Conv1d(k=7) | 512 (style-conditioned input) | 512 | 1x |
| up_0 | ConvTranspose1d(k=20, s=10) | 512 | 256 | 10x |
| noise_0 | Conv1d(har, k=12, s=6) | 22 | 256 | (harmonic injection) |
| MRF_0 | 3 x AdaINResBlock1 | 256 | 256 | 10x |
| up_1 | ConvTranspose1d(k=12, s=6) | 256 | 128 | 60x |
| noise_1 | Conv1d(har, k=1, s=1) | 22 | 128 | (harmonic injection) |
| MRF_1 | 3 x AdaINResBlock1 | 128 | 128 | 60x |
| post | Conv1d -> split | 128 | 22 (11 mag + 11 phase) | 60x |
| iSTFT | iSTFT(n_fft=20, hop=5) | 11 complex | 1 | 300x (60 x 5) |

Total effective upsampling: 10 x 6 x 5 = 300 (matches hop_length=300 at 24 kHz).

#### iSTFT Output Head

The post-convolution outputs `gen_istft_n_fft + 2 = 22` channels:
- Channels 0-10: log-magnitude, passed through `exp()` to get magnitude
- Channels 11-21: raw phase, passed through `sin()` to get phase

Then `iSTFT(magnitude * exp(j * phase), n_fft=20, hop_size=5, window=hann)` reconstructs the waveform.

### Vocos Architecture (Alternative)

Vocos (Siuzdak, 2023) is a fundamentally different approach:

- **No transposed convolutions**: Maintains constant temporal resolution throughout
- **ConvNeXt backbone**: Stack of 1D depthwise convolution + inverted bottleneck blocks with GELU activation and LayerNorm
- **All upsampling via iSTFT**: The only temporal expansion happens in the final iSTFT layer
- **ISTFT head**: Predicts magnitude via `exp(m)` and phase via unit circle projection `(cos(p), sin(p))`
- **Speed**: Approximately 13x faster than HiFiGAN, approximately 70x faster than BigVGAN on GPU
- **Quality**: Comparable or better than HiFiGAN (VISQOL 4.66 vs 4.57; PESQ 3.70 vs 3.09)
- **Parameters**: Trained on LibriTTS at 24 kHz with n_fft=1024, hop=256, 100 mel bins

Vocos is a compelling alternative for future consideration, but Kokoro currently uses iSTFTNet, so implementing iSTFTNet is required first.

## Key Numbers/Constants

### Standard HiFiGAN V1 (Reference)

| Constant | Value | Notes |
|----------|-------|-------|
| Input mel bins | 80 | Standard across all variants |
| Sample rate | 22,050 Hz | Standard HiFiGAN |
| Hop size | 256 | Matches total upsampling factor |
| FFT size | 1,024 | For mel spectrogram computation |
| Window size | 1,024 | Hann window |
| Total upsampling | 256x | 8 x 8 x 2 x 2 |
| Initial channels | 512 | V1 hidden dimension |
| LeakyReLU slope | 0.1 | Used throughout generator |
| Output range | [-1.0, 1.0] | tanh activation |

### Kokoro/StyleTTS2 iSTFTNet (What We Implement)

| Constant | Value | Notes |
|----------|-------|-------|
| Input mel bins | 80 | `n_mels` in config |
| Sample rate | 24,000 Hz | Kokoro native rate |
| Hop size | 300 | `hop_length` in config |
| Analysis FFT size | 2,048 | `n_fft` for mel computation |
| Analysis window size | 1,200 | `win_length` for mel computation |
| Synthesis iSTFT n_fft | 20 | `gen_istft_n_fft` |
| Synthesis iSTFT hop | 5 | `gen_istft_hop_size` |
| Upsample rates | [10, 6] | 2 stages |
| Upsample kernel sizes | [20, 12] | kernel = 2x rate |
| Upsample initial channel | 512 | Hidden dimension |
| Resblock kernel sizes | [3, 7, 11] | 3 parallel residual blocks per MRF |
| Resblock dilation sizes | [[1,3,5], [1,3,5], [1,3,5]] | Same pattern for each kernel |
| Style dimension (decoder) | 64 | AdaIN conditioning |
| Harmonic count | 8 | SourceModuleHnNSF |
| Voiced threshold | 10 | SourceModuleHnNSF |
| Total effective upsampling | 300x | 10 x 6 x 5 = 300 |
| Post conv output channels | 22 | n_fft + 2 = 22 (11 mag + 11 phase) |

## Data Layouts/Formats

### Input: Mel Spectrogram

```
Shape: [batch, 80, T_mel]
  - batch: batch dimension (1 for single utterance)
  - 80: mel frequency bins
  - T_mel: number of mel frames = ceil(audio_samples / hop_length)
Type: float32
Range: log-scale mel spectrogram values (typically -11.0 to 2.0)
```

### Internal: After Upsampling

```
After pre-conv:   [batch, 512, T_mel]
After up_0:       [batch, 256, T_mel * 10]
After up_1:       [batch, 128, T_mel * 60]
After post-conv:  [batch, 22,  T_mel * 60]
  - channels 0-10: log-magnitude spectrogram
  - channels 11-21: phase spectrogram
```

### Internal: Harmonic Source

```
F0 input:         [batch, 1, T_mel]  (fundamental frequency per mel frame)
F0 upsampled:     [batch, 1, T_mel * 300]  (upsampled to sample rate)
Harmonic source:  [batch, 1, T_mel * 300]  (sinusoidal signal)
Harmonic STFT:    [batch, 22, T_mel * 60]  (STFT of harmonic source, n_fft=20)
  - channels 0-10: harmonic magnitude
  - channels 11-21: harmonic phase
```

### Output: Audio Waveform

```
Shape: [batch, 1, T_audio]
  - T_audio = T_mel * 300 (approximately)
Type: float32
Range: [-1.0, 1.0] (normalized)
```

For 16-bit PCM output: multiply by 32767 and clamp to [-32768, 32767], cast to int16.

## Algorithm Steps

### Standard HiFiGAN V1 Forward Pass (Inference)

```
1. x = conv_pre(mel)                           # [B, 80, T] -> [B, 512, T]
2. For each upsampling stage i in [0, 1, 2, 3]:
   a. x = LeakyReLU(x, slope=0.1)
   b. x = ups[i](x)                            # transposed convolution, doubles/octuples time
   c. xs = 0
   d. For each kernel j in [0, 1, 2]:           # resblock_kernel_sizes = [3, 7, 11]
      xs += resblocks[i*3 + j](x)              # ResBlock1 with dilations [1,3,5]
   e. x = xs / 3                                # average MRF outputs
3. x = LeakyReLU(x, slope=0.1)
4. x = conv_post(x)                             # [B, 32, T*256] -> [B, 1, T*256]
5. x = tanh(x)                                  # constrain to [-1, 1]
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

### Kokoro/StyleTTS2 iSTFTNet Forward Pass (Inference)

```
1. Upsample F0 to audio sample rate:
   f0_up = F.interpolate(f0, scale_factor=300)  # [B, 1, T] -> [B, 1, T*300]

2. Generate harmonic+noise source from F0:
   har_source, noi_source, uv = SourceModuleHnNSF(f0_up)
   # har_source: sinusoidal at f0 with 8 harmonics

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
   # [B, 22, T*60] -> [B, 1, T*300]

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

## Reference Implementations

| Implementation | Language | Notes |
|---------------|----------|-------|
| [jik876/hifi-gan](https://github.com/jik876/hifi-gan) | Python/PyTorch | Official HiFiGAN, V1/V2/V3 configs |
| [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) | Python/PyTorch | iSTFTNet decoder with AdaIN, what Kokoro uses |
| [hexgrad/kokoro](https://github.com/hexgrad/kokoro) | Python/PyTorch | Kokoro TTS, fine-tuned from StyleTTS2-LJSpeech |
| [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch) | Python/PyTorch | Standalone iSTFTNet implementation |
| [Blaizzy/mlx-audio](https://deepwiki.com/Blaizzy/mlx-audio/3.2-api-reference) | Python/MLX | MLX port of Kokoro, good architecture reference |
| [characterai/vocos](https://github.com/gemelo-ai/vocos) | Python/PyTorch | Vocos reference implementation |
| [torchaudio HiFiGANVocoder](https://docs.pytorch.org/audio/2.8/generated/torchaudio.prototype.models.HiFiGANVocoder.html) | Python/PyTorch | Clean reference with V1/V2/V3 factory functions |
| [SpeechBrain HiFiGAN](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.HifiGAN.html) | Python/PyTorch | Well-documented HiFiGAN with both ResBlock types |

## Differences Between Implementations

### Standard HiFiGAN vs. Kokoro's iSTFTNet

| Aspect | Standard HiFiGAN V1 | Kokoro/StyleTTS2 iSTFTNet |
|--------|---------------------|--------------------------|
| Sample rate | 22,050 Hz | 24,000 Hz |
| Hop size | 256 | 300 |
| Upsampling stages | 4 (8,8,2,2) | 2 (10,6) + iSTFT(hop=5) |
| Total upsampling | 256x | 300x (60x conv + 5x iSTFT) |
| Output method | tanh(conv_post) | iSTFT(magnitude, phase) |
| Activation | LeakyReLU(0.1) | Snake activation |
| Normalization | Weight norm | AdaIN (style-conditioned) |
| Conditioning | None (unconditional) | Style vector + F0 |
| Source model | None | HnNSF (harmonic+noise, F0) |
| Kernel sizes (upsample) | [16, 16, 4, 4] | [20, 12] |
| ResBlock kernels | [3, 7, 11] | [3, 7, 11] |
| ResBlock dilations | [[1,3,5]] x 3 | [[1,3,5]] x 3 |
| Initial channels | 512 | 512 |
| Parameters (generator) | ~14M (V1) | Part of 82M total model |

### iSTFTNet vs. Vocos

| Aspect | iSTFTNet (Kokoro) | Vocos |
|--------|-------------------|-------|
| Backbone | Transposed conv + ResBlocks | ConvNeXt blocks |
| Upsampling strategy | Partial conv + iSTFT | All iSTFT |
| Temporal resolution | Changes through network | Constant throughout |
| Speed (vs HiFiGAN) | ~1.7x faster (C8C8I) | ~13x faster |
| Quality | Comparable to HiFiGAN | Comparable or better |
| Phase estimation | sin() on raw output | Unit circle (cos, sin) |
| F0 conditioning | Yes (HnNSF source) | No (mel-only input) |
| Mel bins | 80 | 100 (in paper) |

### Weight Normalization Notes

Standard HiFiGAN uses `weight_norm` on all Conv1d layers during training. At inference, weight normalization should be fused/removed for speed:
```
weight = weight_g * (weight_v / norm(weight_v))
```
This can be pre-computed and stored as a single weight tensor.

Kokoro's iSTFTNet uses a similar pattern with `ConvWeighted` separating magnitude (`weight_g`) and direction (`weight_v`).

## Open Questions

- [x] Whether Kokoro uses standard HiFiGAN or a modified variant — **Answer: Modified iSTFTNet from StyleTTS2, NOT standard HiFiGAN**
- [x] Exact upsampling kernel sizes and rates for the Kokoro vocoder — **Answer: rates=[10,6], kernels=[20,12], iSTFT n_fft=20 hop=5**
- [x] Whether Vocos is a better alternative — **Answer: Vocos is 13x faster with comparable quality, but Kokoro ships with iSTFTNet. Could be a future optimization.**
- [ ] Exact parameter count of the iSTFTNet decoder portion alone (vs. full 82M model)
- [ ] Whether Kokoro's iSTFTNet weights can be loaded independently from the rest of the model
- [ ] Whether the F0 (pitch) input is required or can be set to zero for inference
- [ ] Exact weight tensor names in the Kokoro safetensors file for the decoder portion

## Implementation Notes

### For SharpInference

1. **Implement iSTFTNet first, not standard HiFiGAN.** Kokoro uses iSTFTNet. Standard HiFiGAN is only useful as a reference/test target.

2. **Key operations to implement:**
   - 1D transposed convolution (ConvTranspose1d) with stride > 1
   - 1D dilated convolution (Conv1d with dilation parameter)
   - Adaptive Instance Normalization (AdaIN1d): normalize per-channel, then apply learned affine from style vector
   - Snake activation: `x + (1/alpha) * sin^2(alpha * x)` — requires element-wise sin and multiply
   - iSTFT: inverse Short-Time Fourier Transform with Hann window, n_fft=20, hop=5
   - SourceModuleHnNSF: cumulative sum of phase, sinusoidal generation, learned mixing

3. **iSTFT implementation:** With n_fft=20 and hop_size=5, this is a very small FFT (only 20 points). Can be implemented efficiently with a direct DFT formula rather than a full FFT library. The overlap-add synthesis window is a 20-point Hann window.

4. **Memory layout:** All intermediate tensors are `[batch, channels, time]` (channels-first). The time dimension grows through the network: T_mel -> T_mel*10 -> T_mel*60 -> T_audio (T_mel*300).

5. **Weight normalization fusion:** Pre-compute `weight = weight_g * weight_v / norm(weight_v)` during model loading to avoid runtime overhead.

6. **Transposed convolution padding:** For `ConvTranspose1d(kernel=k, stride=s)`, the output length is `(input_length - 1) * stride + kernel - 2 * padding`. The kernel sizes are chosen as 2x the stride (k=20 for s=10, k=12 for s=6), which with appropriate padding produces clean upsampling.

7. **16-bit PCM output:** After the vocoder produces float32 audio in [-1, 1]:
   ```
   pcm16 = clamp(round(audio * 32767), -32768, 32767)
   ```
   Write as little-endian int16 for WAV file output at 24,000 Hz sample rate.

8. **Cross-reference:** See [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) for mel spectrogram computation details (note: that doc covers Whisper's parameters at 16kHz; Kokoro uses different STFT parameters at 24kHz with n_fft=2048, hop=300, win=1200, 80 mel bins).
