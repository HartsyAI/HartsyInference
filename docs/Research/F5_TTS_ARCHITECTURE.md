# F5-TTS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (F5-TTS pipeline)

## Summary

F5-TTS (Shanghai Jiao Tong University X-LANCE Lab / SWivid, 2024) is a **fully non-autoregressive, zero-shot voice-cloning TTS** built on **Conditional Flow Matching (CFM)** over mel spectrograms. It takes a short reference audio clip (3-15 s, 12 s hard cap) plus its transcript and a target text, and produces target speech in the reference speaker's voice. The whole utterance is generated **jointly** in a single forward integration pass — not autoregressively — which is what distinguishes it from XTTS/Bark/Sesame-style decoder TTS. At ~336 M params (DiT) + ~14 M (Vocos) it is currently the leading open-weight zero-shot voice cloning model for English and Mandarin Chinese; community fine-tunes cover ~10 additional languages.

The pipeline is `(ref_audio, ref_text, target_text) → mel-prep + char-tokenize → Flow-Matching DiT (32 NFE, CFG=2.0, Sway Sampling s=-1.0) → Vocos vocoder → 24 kHz waveform`. There is **no G2P, no phonemizer, no learned duration predictor** — characters go straight into a 256-token byte-level embedding and duration is a closed-form ratio of reference and target character counts. The DiT uses standard SD3/Flux-style AdaLN-Zero blocks (`dim=1024, depth=22, heads=16, head_dim=64, ff_mult=2`) preceded by a **ConvNeXt V2** text stem (4 blocks, depthwise-Conv1D kernel=7 + GRN). The vocoder is `charactr/vocos-mel-24khz`, **but fine-tuned by the F5-TTS team to 100 mel bins** (the public charactr checkpoint is 100-bin in F5's mel parameterization — `n_fft=1024, hop=256, win=1024`).

This file covers the model architecture and inference pipeline. The Sway-Sampling scheduler math is in [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) F5-TTS section. Mel preprocessing is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). Vocos vocoder implementation details are in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) Vocos section.

**Sources**:
- Paper: ["F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"](https://arxiv.org/abs/2410.06885) (Chen et al., ACL 2025)
- Repo: [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) (`src/f5_tts/model/cfm.py`, `dit.py`, `modules.py`, `infer/utils_infer.py`)
- Weights: [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS), [SWivid/E2-TTS](https://huggingface.co/SWivid/E2-TTS)
- Vocoder: [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz)
- Community variants index: [`src/f5_tts/infer/SHARED.md`](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/SHARED.md)

## Detailed Findings

### 1. Model Variants

All official models share the same DiT shape (`dim=1024, depth=22, heads=16, ff_mult=2, conv_layers=4`) — they differ only in training data, training steps, vocab, and zero-init policy.

| Variant | Params | Languages | HF path | Checkpoint | File size | License |
|---|---|---|---|---|---|---|
| **F5TTS_v1_Base** *(current default, 2024-12)* | ~335.8 M | EN + ZH (code-switch) | `SWivid/F5-TTS` | `F5TTS_v1_Base/model_1250000.safetensors` | ~1.34 GB FP32 / ~672 MB FP16 | CC-BY-NC-4.0 |
| **F5TTS_v1_Base_no_zero_init** | ~335.8 M | EN + ZH | `SWivid/F5-TTS` | `F5TTS_v1_Base_no_zero_init/model_1250000.safetensors` | ~1.34 GB FP32 | CC-BY-NC-4.0 |
| **F5TTS_Base** *(legacy v0)* | ~335.8 M | EN + ZH | `SWivid/F5-TTS` | `F5TTS_Base/model_1200000.safetensors` | ~1.34 GB FP32 | CC-BY-NC-4.0 |
| **F5TTS_Base_bigvgan** | ~335.8 M | EN + ZH | `SWivid/F5-TTS` | `F5TTS_Base_bigvgan/...` | ~1.34 GB FP32 | CC-BY-NC-4.0 |
| **E2TTS_Base** *(paper predecessor)* | ~333 M | EN + ZH | `SWivid/E2-TTS` | `E2TTS_Base/model_1200000.safetensors` | ~1.33 GB FP32 | CC-BY-NC-4.0 |

Training data for the base models: **Emilia 95k h ZH + EN** ([amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)). Repo total (all variants) is **6.74 GB**.

**v1 vs v0 differences**: v1 fixes the AdaLN-Zero init (the `_no_zero_init` debug variant was published to demonstrate the difference), uses a slightly cleaner Emilia split, and trained 50 k more steps. Both have identical architecture — checkpoint swap only.

**Community language variants** (from [`SHARED.md`](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/SHARED.md), 2025):

| Language | Base | Training data | Notable changes |
|---|---|---|---|
| Arabic | F5-TTS-**Small** | EN + AR mixed (tens of thousands h) | smaller DiT |
| Finnish | F5-TTS-Base | Common Voice + VoxPopuli | vocab.txt extended |
| French | F5-TTS-Base | LibriVox FR | |
| German | F5-TTS-Base | Mozilla CV 19.0 + 800 h crowdsourced | |
| Hindi | F5-TTS-**Small** | IndicTTS + IndicVoices-R | smaller DiT, Devanagari chars |
| Italian | F5-TTS-Base | `ylacombe/cml-tts` | |
| Japanese | F5-TTS-Base | Emilia JA 1.7 k + Galgame 5.4 k | |
| Latvian | F5-TTS-Base | Common Voice LV | |
| Russian | F5-TTS-Base | Common Voice RU | Cyrillic chars |
| Spanish | F5-TTS-Base | VoxPopuli ES + TEDx (218 h) | |

There is also **Cross-Lingual F5-TTS** ([arXiv:2509.14579](https://arxiv.org/abs/2509.14579)) — a 2025 follow-up adding language-agnostic voice cloning, and **Fast F5-TTS** ([fast-f5-tts.github.io](https://fast-f5-tts.github.io/)) — a 7-NFE distilled variant.

The **"F5-TTS-Small"** topology (used by Arabic, Hindi) is approximately `dim=768, depth=18, heads=12, ff_mult=2, conv_layers=4` (~155 M params). The official repo carries the YAML; planned variants in the SharpInference loader must read `dim/depth/heads/ff_mult/conv_layers/text_dim/text_num_embeds` from the YAML next to the safetensors and dispatch accordingly.

### 2. Architecture

#### 2.1 Text encoder — character-level, NO G2P

F5-TTS replaces phoneme conditioning with **raw character conditioning**. There are three supported tokenization modes (`get_tokenizer()` in `src/f5_tts/model/utils.py`):

1. **`byte`** — UTF-8 byte values, vocab size = 256, zero G2P needed. Index 0 is reserved for "unknown / pad". This is the simplest mode and is the **recommended one for SharpInference's first cut**.
2. **`char`** — load `vocab.txt` next to the checkpoint, one symbol per line, look each character up. The shipped `F5TTS_v1_Base/vocab.txt` (EN+ZH model) contains ASCII printables + pinyin syllables + ZH punctuation; vocab size ≈ 2545.
3. **`pinyin`** *(used by all official EN+ZH checkpoints)* — Chinese characters are first segmented with `rjieba` and converted to **TONE3-style pinyin** ("ni3 hao3 ma5"), then looked up in `vocab.txt`. ASCII text is passed through character-by-character. **Spaces separate syllables.**

For the official EN+ZH model the inference flow is:
```
"你好, world" → segment+pinyin → "ni3 hao3 , w o r l d" → idx via vocab.txt
```

The tokenizer must satisfy: **"space character is at index 0 in vocab.txt because 0 is also the unknown-char index"** (assert in `get_tokenizer`).

The **embedding layer** has `text_num_embeds + 1` rows (the +1 is a "filler" token at the last index used to pad the text to the audio length). For v1 base with pinyin vocab that is `(2545 + 1, 512)`.

After embedding, the character sequence is **padded with the filler token to the mel frame count** (so text and mel have the same length T_mel before they enter the DiT). Then it goes through **rotary positional embedding + 4 ConvNeXt V2 blocks** (the "text stem") to produce a `(B, T_mel, 512)` text condition. `precompute_max_pos = 8192` mel frames ≈ 87 s of audio at 24 kHz / hop 256.

#### 2.2 Audio encoder — mel spectrogram

The reference audio is converted to mel features with the following **exact** parameters (from `F5TTS_v1_Base.yaml` and reproduced in `MEL_SPECTROGRAM.md`):

| Parameter | Value |
|---|---|
| `target_sample_rate` | **24 000 Hz** |
| `n_mel_channels` | **100** |
| `n_fft` | **1024** |
| `win_length` | **1024** |
| `hop_length` | **256** |
| Windowing | Hann |
| Mel filter range | 0 Hz – 12 000 Hz (full Nyquist; no f_min/f_max clip) |
| Mel scale | HTK slaney mel (matches torchaudio default `MelSpectrogram(power=1)`) |
| Log compression | `log(clamp(mel, min=1e-5))` (log, **not** log10, **not** dB) |
| RMS normalization | `target_rms = 0.1` applied to ref audio before mel |

Frame rate is 24 000 / 256 = **93.75 mel frames per second**. Mel input to the DiT is shape `(B, T_mel, 100)`.

Reference audio is **clipped to ≤ 12 s** by progressive silence detection (1000 ms threshold first, then 100 ms) inside `preprocess_ref_audio_text`. Longer text input is **chunked**: each chunk targets ≤ 135 UTF-8 bytes by default (dynamically rescaled by ref-audio duration), and chunks are stitched with a **0.15 s linear crossfade**.

#### 2.3 DiT (Diffusion Transformer)

| Hyperparameter | F5TTS_v1_Base |
|---|---|
| `dim` (model hidden) | **1024** |
| `depth` (transformer blocks) | **22** |
| `heads` | **16** |
| `dim_head` | **64** (= 1024 / 16) |
| `ff_mult` | **2** (FFN intermediate = 2048) |
| `text_dim` | **512** |
| `text_num_embeds` | 256 (byte) or vocab size (char/pinyin) |
| `conv_layers` (ConvNeXt text stem depth) | **4** |
| `mel_dim` | **100** (input/output channel count) |
| Positional encoding | **Rotary (RoPE)** on Q/K, computed by `RotaryEmbedding(dim_head=64)` |
| `qk_norm` | Optional RMSNorm on Q/K (config flag, **off** in base) |
| Attention backend | `torch` (eager) or `flash_attn`; SharpInference target = own kernel |
| Long skip connection | Optional symmetric U-net-style skip (off in base) |
| FFN | GeGLU? **No** — plain Linear → GELU → Linear (`FeedForward` in `modules.py`) |
| Final norm | `AdaLayerNorm_Final` (zero-init) → `Linear(dim → mel_dim)` |

**Block layout** (one of 22 identical blocks):
```
x, c = block_input
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp =
    AdaLNZero(time_emb).chunk(6, dim=-1)
y = LayerNorm(x) * (1 + scale_msa) + shift_msa
y = Attention_RoPE(y)                # MHA with rotary on Q,K
x = x + gate_msa * y
y = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
y = FeedForward(y)                   # Linear(d,2d) -> GELU -> Linear(2d,d)
x = x + gate_mlp * y
```

The **input to the DiT** is built by `CFM.forward` like this:
```
noisy_mel    : (B, T_mel, 100)   # x_t = (1-t)*x0 + t*epsilon
cond_mel     : (B, T_mel, 100)   # ref-audio mel zero-padded over target region
text_emb     : (B, T_mel, 512)   # ConvNeXt-stem output, filler-padded
time_emb     : (B, dim)          # sinusoidal+MLP, see 2.5

input        = concat([noisy_mel, cond_mel], dim=-1)   # (B, T_mel, 200)
input        = Linear(200 → dim)(input)                # (B, T_mel, 1024)
input        = input + Linear(text_dim → dim)(text_emb) # add text condition
output       = DiT_blocks(input, time_emb)              # (B, T_mel, 1024)
velocity     = Linear(dim → 100)(AdaLNFinal(output, time_emb))  # (B, T_mel, 100)
```

That output `velocity` is the **predicted velocity field** `v_theta(x_t, t, cond, text)` consumed by the flow-matching ODE solver.

#### 2.4 ConvNeXt V2 text stem (the "ConvNeXt" in the paper title)

Four blocks, each:
```
y = depthwise_Conv1d(dim=512, kernel=7, padding=3, groups=512)(x)   # depthwise temporal
y = LayerNorm(y)
y = Linear(512 → 1024)(y)        # pointwise expand (intermediate_dim=1024)
y = GELU(y)                      # paper uses GELU; modules.py uses Mish/GELU variant
y = GRN(y)                       # Global Response Normalization (the V2 in ConvNeXt V2)
y = Linear(1024 → 512)(y)        # pointwise contract
x = x + y                        # residual
```

**GRN** (Global Response Normalization, from ConvNeXt V2):
```
Gx = ||x||_2 along token dim         # (B, 1, C)
Nx = Gx / (mean(Gx, dim=C) + 1e-6)   # (B, 1, C)
out = gamma * (x * Nx) + beta + x    # gamma, beta learnable (init 0)
```

This stem is **only** applied to the embedded character sequence (text path), **not** to the audio. RoPE is also applied inside the text-stem blocks. After 4 blocks, output goes into the DiT as the text condition.

#### 2.5 Time embedding

```
freqs = SinusoidalPositionEmbedding(dim=256)(t)   # t in [0,1]
       # i.e.  emb_i = log(10000)/(half_dim-1);  e = exp(-i*emb) ; t*e -> sin,cos concat
time  = Linear(256 → 1024) → SiLU → Linear(1024 → 1024)(freqs)
```

The final `(B, 1024)` time embedding feeds the AdaLN-Zero modulation in every DiT block and in `AdaLayerNorm_Final`. No condition dropout on the time embedding; CFG is realized by zeroing both `cond_mel` and `text_emb` on the unconditional branch.

#### 2.6 Vocoder — Vocos (charactr/vocos-mel-24khz)

F5-TTS uses **Vocos** to invert the predicted mel back to a 24 kHz waveform. The published F5-TTS pipeline expects **100-bin mel** input (matching the DiT output), which corresponds to the `charactr/vocos-mel-24khz` checkpoint as configured for F5-TTS (`n_fft=1024, hop=256, win=1024, 100 mel bins, sr=24 kHz`).

Vocos architecture (full design in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) → "Vocos Architecture (Alternative)"):
- Mel input → 1D Conv embed → **8 ConvNeXt blocks** (no temporal upsampling)
- Final Linear projects to `(n_fft/2 + 1) * 2 = 1026` channels = predicted **magnitude + phase** of the STFT
- A single **inverse STFT** synthesizes the waveform — no transposed convs, no upsampling tower
- ~13.5 M params, 13× faster than HiFiGAN-V1 at higher quality

For BigVGAN-trained checkpoints (e.g. `F5TTS_Base_bigvgan`), substitute the `nvidia/bigvgan_v2_24khz_100band_256x` vocoder. SharpInference will target Vocos first (smaller, simpler, matches the default v1 checkpoint).

### 3. In-context inference pattern (the key idea)

F5-TTS treats voice cloning as **infilling**, not generation. The model never "starts from a speaker embedding"; it sees the entire utterance and predicts only the masked tail.

```
Reference audio:   [-----ref mel (N_ref frames)-----]
Target audio:                                       [-----noise/mask (N_tgt frames)-----]

Conditioning mel:  [-----ref mel (N_ref frames)-----][-----zeros (N_tgt frames)-----]
Noisy mel x_t:     [-----random noise (whole T)----- ----- whole T -----]
Text:              [ref_chars + target_chars padded with filler to T frames]
```

At each Euler step the DiT predicts velocity over the full `T = N_ref + N_tgt` sequence. The reference-region predictions are **discarded** — the loop overwrites that region with the original `cond_mel` at every step (this is how the model is "anchored" to the reference voice). Only the target-region samples accumulate. After NFE steps the target region is the final mel and is sent to Vocos.

Why this works: training masks a random suffix of every utterance and asks the model to reconstruct it from the prefix + full text. At inference, "prefix + full text" is exactly the reference clip + concatenated transcript, so zero-shot voice cloning falls out for free. **No speaker embedding, no learned style encoder, no fine-tuning required.**

This is fundamentally different from autoregressive TTS (Bark, XTTS, Tortoise, Sesame): F5-TTS processes the whole utterance jointly, in parallel, in 32 forward passes total — not one decoder step per token.

### 4. Flow matching — Sway Sampling

F5-TTS uses **rectified flow / Conditional Flow Matching** with **Euler integration** in time `t: 0 → 1` (data at `t=0`, noise at `t=1` — opposite sign convention from SD3 sigmas, but the per-step update is identical).

**Defaults** (from `src/f5_tts/infer/utils_infer.py`):

| Knob | Value | Notes |
|---|---|---|
| `nfe_step` | **32** | ablation flat above ~16 NFE |
| `cfg_strength` | **2.0** | both `cond_mel` and `text_emb` zeroed on uncond branch |
| `sway_sampling_coef` | **−1.0** | enables Sway; `None`/`0` = uniform |
| ODE solver | **Euler** | `midpoint` available as fallback |
| `target_rms` | **0.1** | RMS of ref audio normalized to this before mel |
| `cross_fade_duration` | **0.15 s** | between text chunks |
| `speed` | **1.0** | multiplies target duration (smaller = faster speech) |

**Sway Sampling** is a one-shot remap of the uniform timestep grid applied **before** the ODE loop:

```python
# from cfm.py — exact code
t = torch.linspace(0, 1, steps)            # uniform NFE grid
if sway_sampling_coef is not None:
    t = t + sway_sampling_coef * (torch.cos(torch.pi/2 * t) - 1 + t)
# t still starts at 0 and ends at 1; only interior density shifts
```

With `s = -1.0` more samples cluster near `t = 0` (data end), so the solver spends more NFE polishing fine detail and fewer NFE on coarse structure near noise. Paper ablation: `s = -1.0` reduces WER ~10 % and raises SIM-O ~0.02 vs. uniform at NFE=32. **See [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) F5-TTS section** for the full derivation, the sigma-space relationship, and a per-step value table — that doc is canonical for the scheduler implementation.

**CFG combination** (per step):
```
v_cond  = DiT(x_t, t, cond_mel, text_emb)
v_uncond = DiT(x_t, t, zeros_like(cond_mel), zeros_like(text_emb))
v = v_uncond + cfg_strength * (v_cond - v_uncond)
x_{t-dt} = x_t - v * dt          # Euler step (negative because we integrate 1→0)
# at every step, overwrite the ref region of x with ref_mel
```

E2-TTS (predecessor) is mathematically the same model with no ConvNeXt text stem and `sway_sampling_coef = 0`. For SharpInference: **E2-TTS = F5-TTS scheduler with sway off, plus drop the ConvNeXt blocks from the text path.**

### 5. Duration prediction — closed-form heuristic, no learned predictor

F5-TTS has **no neural duration model**. The target mel length is computed by simple ratio (from `_infer_basic` in `utils_infer.py`):

```python
ref_audio_len = audio.shape[-1] // hop_length          # ref mel frames
duration = ref_audio_len + int(
    ref_audio_len / ref_text_len * gen_text_len / local_speed
)
```

Or, written algebraically:

```
N_tgt = round( N_ref * (len(target_text) / len(ref_text)) / speed )
T     = N_ref + N_tgt                                  # total DiT sequence length
```

`ref_text_len` and `gen_text_len` are **character counts after tokenization** (so for the pinyin EN+ZH model they are the post-pinyin lengths — that is the key to keep ZH and EN proportional). The CFM sample loop additionally enforces `duration >= max(text_len, ref_audio_len) + 1` (see `cfm.sample()`):

```python
duration = torch.maximum(
    torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
)
```

`speed > 1.0` → faster (shorter target). The lack of a learned duration predictor is intentional — the paper argues that the joint DiT itself implicitly handles fine-grained alignment, and a global length estimate is all the conditioning needs.

### 6. Multilingual handling

The official models use **pinyin for Chinese, raw characters for everything else** (segmentation by `rjieba`, pinyin by `pypinyin.lazy_pinyin(style=TONE3)`, spaces inserted at language boundaries — `convert_char_to_pinyin` in `src/f5_tts/model/utils.py`).

Community fine-tunes typically just **extend `vocab.txt`** with the script's characters (Cyrillic, Devanagari, Arabic, kana/kanji, etc.) and re-init the embedding rows for new tokens. The DiT itself is script-agnostic.

For SharpInference, the **`byte` tokenization mode is the easiest first cut** — works for any UTF-8 input, no vocab file needed, vocab size = 257. Quality will be slightly lower than the pinyin path on Chinese (no syllable structure) but is fully general. Pinyin support can be added later as a separate Tokenizer implementation behind an interface.

### 7. Voice cloning quality

Widely regarded as the **SOTA open-weight zero-shot voice cloning model for EN/ZH at the < 500 M-param scale** as of late 2024 / early 2025. Reported on LibriSpeech-PC (paper Table 3):

| Model | WER ↓ | SIM-O ↑ |
|---|---|---|
| VALL-E 2 | 2.6 | 0.643 |
| NaturalSpeech 3 | 1.94 | 0.67 |
| **E2-TTS** (NFE=32) | 2.19 | **0.71** |
| **F5-TTS** (NFE=32, Sway=-1.0) | **1.83** | 0.66 |

Subjective community consensus: F5-TTS clones English/Mandarin voices convincingly from **5–10 s** of clean reference. Common failure modes: poor performance on heavy accents not in Emilia, very emotive/whispered/shouted speech, and prosody is occasionally flat because there is no explicit prosody model. Out-of-distribution languages need a community fine-tune.

### 8. Inference pipeline pseudocode (exact tensor shapes)

For `ref_audio_path` (wav), `ref_text` ("Hello world."), `target_text` ("This is the cloned voice.") on the `F5TTS_v1_Base` checkpoint:

```
# ---------- 0. Preprocess ref audio ----------
wav, sr = load(ref_audio_path)
wav = resample(wav, sr → 24000)
wav = clip_to_max_silence(wav, max_dur=12.0)
wav = wav * (target_rms / rms(wav))          # normalize to RMS 0.1
ref_mel = mel_spec(wav,
                   n_fft=1024, hop=256, win=1024,
                   n_mels=100, sr=24000,
                   log=True, clamp_min=1e-5)
ref_mel : (T_ref, 100)         # T_ref = len(wav) // 256

# ---------- 1. Tokenize text ----------
# byte mode (simplest):
ref_ids    = list(ref_text.encode('utf-8'))            # ints in [0,255]
target_ids = list(target_text.encode('utf-8'))
text_ids   = ref_ids + target_ids                      # concatenated
# pinyin mode (official EN+ZH model): apply convert_char_to_pinyin first.

# ---------- 2. Compute target duration ----------
N_ref = T_ref
N_tgt = round( N_ref * (len(target_ids) / len(ref_ids)) / speed )
T     = N_ref + N_tgt
T     = max(T, len(text_ids) + 1)                      # safety lower bound

# ---------- 3. Build inputs ----------
text_padded = pad_with_filler(text_ids, length=T)      # filler = text_num_embeds
text_padded : (1, T) int
text_emb    = Embedding(text_padded)                   # (1, T, 512)
text_emb    = ConvNeXt_stem_4_blocks(text_emb)         # (1, T, 512)

cond_mel = zeros(1, T, 100)
cond_mel[:, :T_ref, :] = ref_mel                       # zero-pad target region

x_t = randn(1, T, 100)                                 # init at t=1 (pure noise)

# ---------- 4. Sway-sampled Euler integration ----------
t_grid = linspace(0, 1, steps=33)                      # 32 NFE = 33 nodes
t_grid = t_grid + (-1.0) * (cos(pi/2 * t_grid) - 1 + t_grid)   # Sway s=-1.0

for i in range(32, 0, -1):                             # integrate noise→data
    t  = t_grid[i]
    dt = t_grid[i] - t_grid[i-1]
    time_emb = TimeMLP(SinusoidalEmbed(t))             # (1, 1024)
    v_cond   = DiT(x_t, cond_mel,        text_emb,   time_emb)
    v_uncond = DiT(x_t, zeros_like(cond_mel), zeros_like(text_emb), time_emb)
    v = v_uncond + 2.0 * (v_cond - v_uncond)           # CFG 2.0
    x_t = x_t - v * dt                                 # Euler
    x_t[:, :T_ref, :] = ref_mel                        # anchor ref region

generated_mel = x_t[:, T_ref:, :]                      # (1, N_tgt, 100)

# ---------- 5. Vocoder ----------
generated_mel = generated_mel.transpose(1, 2)          # (1, 100, N_tgt)
waveform = Vocos.decode(generated_mel)                 # (1, N_tgt * 256) at 24 kHz
waveform = waveform * (rms(orig_wav) / target_rms)     # restore original loudness
return waveform
```

For multi-chunk generation (target text > ~135 UTF-8 bytes): split target into chunks at sentence boundaries, run the loop above per chunk, **0.15 s linear crossfade** the resulting waveforms together.

### 9. Memory and performance

| Resource | F5TTS_v1_Base |
|---|---|
| Disk (FP32 safetensors) | ~1.34 GB |
| Disk (FP16) | ~672 MB |
| VRAM @ FP16 inference | ~2.5 GB peak (model 0.7 GB + activations + Vocos 0.05 GB) |
| VRAM @ FP32 inference | ~5 GB peak |
| Min recommended GPU | 8 GB VRAM (FP16); 16 GB comfortable; 24 GB for training |
| **RTF @ 32 NFE, FP16, RTX 4090** | **~0.15** (≈ 6.5× realtime) |
| RTF @ 16 NFE, FP16, RTX 4090 | ~0.075 (≈ 13× realtime) |
| RTF @ 7 NFE distilled (Fast F5-TTS, 3090) | ~0.030 |
| Quantized (community Q8) | ~400 MB VRAM |
| Latency per 1 s of audio @ 32 NFE | ~150 ms on 4090 |

The bottleneck is the **DiT** (22 layers × 32 NFE × 2 CFG branches × ~T forward passes), specifically attention over `T` tokens where `T ≈ 94 * audio_seconds`. For a 10 s output `T ≈ 940`, so the attention is well-suited to flash-attention / our own fused-attention kernel; this is where the bulk of optimization payoff lives. Vocos is negligible cost (one feed-forward conv stack + one iSTFT per call).

### 10. C# implementation notes (SharpInference)

| Component | Reuse / new | Source of truth |
|---|---|---|
| **DiT block (AdaLN-Zero, RoPE, GELU FFN)** | **Reuse** Flux/SD3 block; F5 differs only in `dim_head=64`, `ff_mult=2`, no GeGLU | [FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md), [SD3_ARCHITECTURE.md](SD3_ARCHITECTURE.md) |
| **Sinusoidal + MLP time embedding** | **Reuse** from image diffusion stack | [DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md) |
| **Mel preprocessing (24 kHz / 100 bin / 1024 FFT / 256 hop / Hann / log-clamp)** | **Reuse** existing mel module; add the 100-bin variant | [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) |
| **ConvNeXt V2 text stem (4 blocks, dwconv k=7, GRN, intermediate=1024)** | **New** — implement `ConvNeXtV2Block` + `GRN` for 1D temporal data. Depthwise Conv1D groups=channels. | this doc § 2.4 |
| **Char/byte/pinyin tokenizer** | **New trivial component**. Start with byte mode (a `byte[] → int[]` map plus a filler token). Defer pinyin (it would pull in a Jieba-style segmenter + pinyin tables). | this doc § 2.1 |
| **Sway-sampling scheduler** | **New** — small scheduler class extending the existing `FlowMatchEulerDiscreteScheduler`: just apply the cosine remap to the timestep grid before the loop. | [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) F5-TTS section |
| **CFG combiner for velocity** | **Reuse** existing CFG helper from image flow matching (identical formula) | [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md) |
| **In-context infilling loop (concat ref+target, overwrite ref region each step)** | **New** wrapper around the Euler loop | this doc § 8 |
| **Vocos vocoder (8 ConvNeXt blocks → magnitude+phase → iSTFT)** | **New** — implement once, shared with Kokoro-vocos variant + EnCodec decode | [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) "Vocos Architecture (Alternative)" |
| **Safetensors loader** | **Reuse** | [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md) |
| **GGUF quantized variants (community Q8)** | **Reuse** GGUF backend; verify tensor naming matches `f5_tts.*` convention | [GGUF_BACKEND.md](GGUF_BACKEND.md) |

**Package placement**: `SharpInference.Audio` owns the F5-TTS pipeline, Vocos vocoder, mel preprocessor, and Sway scheduler. The DiT block kernels live in `SharpInference.Core` (already shared with image diffusion). No new native code required — pure C# + PTX.

**Recommended first-milestone scope**:
1. Load `F5TTS_v1_Base/model_1250000.safetensors` (FP32 → FP16) into a typed weight dictionary.
2. Implement `ByteTokenizer` (vocab size 257, filler at 256).
3. Implement `ConvNeXtV2Block1D` + `GRN1D`.
4. Wire 22-layer DiT using existing AdaLN-Zero blocks, with the 200-channel input projection (concat noisy + cond).
5. Implement Sway-sampled Euler loop with the ref-region overwrite anchor.
6. Implement Vocos (mel-24kHz config) and validate against `vocos.decode(mel)` from Python reference within 1e-3 L2 on a held-out mel.
7. End-to-end test: `(reference 5 s wav, ref text, target text) → 24 kHz wav`, compare against Python F5-TTS output on the same inputs at same seed/NFE/CFG/Sway; SIM-O should match within 0.005, WER within 1 %.

**Reference fidelity targets**: validate the DiT block-by-block (mel → first block output, then mid-network, then final velocity) against the Python reference with all stochasticity removed (fixed noise tensor, NFE=32, deterministic). Tolerances: 1e-3 atol FP16, 1e-5 atol FP32, both relative to torch reference.

## Open Questions

- [ ] Should we support `pinyin` tokenization in v1, or ship `byte` only and require fine-tuned weights for ZH quality? (Pinyin pulls in a 200 KB+ Chinese segmentation table.)
- [ ] Which size class should "F5-TTS-Small" community fine-tunes use as their default — confirm by inspecting their YAML configs in each HF repo.
- [ ] BigVGAN vs Vocos for the BigVGAN-trained variant — defer BigVGAN until Vocos works end-to-end; BigVGAN is ~3× larger and slower.
- [ ] Streaming output: can the in-context infilling be chunk-streamed (i.e., emit waveform as the DiT integrates)? Probably not natively — the whole T_mel is needed for each step. Streaming would require chunked text + crossfade like the Python infer loop already does.

## Cross-References

- [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) — F5-TTS scheduler, Sway Sampling derivation, exact CFM math.
- [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) — 24 kHz / 100-bin mel parameters.
- [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) — Vocos architecture and implementation.
- [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md) — alternative TTS pipeline (autoregressive-style with G2P) for comparison.
- [FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md), [SD3_ARCHITECTURE.md](SD3_ARCHITECTURE.md) — reusable AdaLN-Zero DiT blocks.
- [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md) — CFG combination for velocity fields.
- [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md), [GGUF_BACKEND.md](GGUF_BACKEND.md) — weight loading.
