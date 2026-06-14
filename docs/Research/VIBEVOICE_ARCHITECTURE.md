# VibeVoice — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-20 | Needed Before: HartsyInference.Audio (VibeVoice pipeline)

## Summary

VibeVoice (Microsoft Research, 2025; community-maintained fork since the official repo was taken down) is a **long-form, multi-speaker, expressive conversational TTS** based on a **next-token diffusion** framework. The text+voice prompt is fed into a Qwen2.5 LM that autoregresses over a **tiny constrained vocabulary** (`speech_start`, `speech_end`, `speech_diffusion`, `eos`) at the token level. Whenever the LM emits a `speech_diffusion` token, a small **4-layer DDPM head** runs a 20-step denoise (cosine schedule, v-prediction, CFG) over a single **64-d continuous latent**, which is decoded by a **causal 1D-ConvNeXt acoustic VAE** at **7.5 Hz** (3200× downsample over 24 kHz audio = one latent every 133 ms ≈ one waveform chunk of 3200 samples). The decoded audio is then **re-encoded** by a separate **semantic VAE** (128-d) and both embeddings are fed back into the LM as the next-step embedding. The semantic↔acoustic dual feedback loop is what gives VibeVoice its prosody and turn-taking quality.

The pipeline is `(multi-speaker script, ref audio per speaker) → Qwen2.5 LM prefill → AR loop {emit token → if diffusion: DDPM-20 + acoustic decode → semantic re-encode → feedback embed} → concat 7.5 Hz chunks → 24 kHz waveform`. There is **no mel front-end, no phonemizer, no learned duration predictor, no separate vocoder** — the acoustic VAE decoder *is* the vocoder. The model handles **up to 4 speakers and 90-minute outputs** (1.5B) or 45-minute outputs (7B), and has a **single-speaker streaming-0.5B variant** with a split-LM architecture and binary EOS classifier for low-latency real-time TTS.

This file covers the model architecture, scheduler, and inference pipeline for the three official checkpoints. The DDPM math and v-prediction noise-prediction loss are covered by the existing diffusion-pipeline scheduler infra in `HartsyInference.Diffusion`. Voice-cloning works zero-shot from any 5-30 s reference clip per speaker (24 kHz mono). License is **MIT** (community fork) — fully commercially usable.

**Sources**:
- Paper: ["VibeVoice Technical Report"](https://arxiv.org/pdf/2508.19205) (Microsoft Research, 2025)
- Repo: [vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice) (`vibevoice/modular/`, `vibevoice/configs/`, `vibevoice/schedule/`, `vibevoice/processor/`)
- Weights: [vibevoice (HF org)](https://huggingface.co/vibevoice) — `VibeVoice-1.5B`, `VibeVoice-7B` (alias "Large"), `VibeVoice-Streaming-0.5B`
- Companion paper for next-token diffusion: [arXiv:2412.08635](https://arxiv.org/abs/2412.08635)

## Detailed Findings

### 1. Model Variants

All three official checkpoints share the same VAE topology (encoder ratios `[8,5,5,4,2,2]` → 3200× downsample, 7.5 Hz on 24 kHz audio) and the same 4-layer diffusion head — they differ in the Qwen2.5 LM size, context length, and (for streaming) presence of the semantic VAE.

| Variant | LM Backbone | LM hidden / layers / heads / KV-heads | Context | Speakers | Max output | Tokenizer vocab | License | HF path |
|---|---|---|---|---|---|---|---|---|
| **VibeVoice-1.5B** | Qwen2.5-1.5B | 1536 / 28 / 12 / 2 | 65 536 | up to 4 | ~90 min | 151 936 (Qwen2.5 base) | MIT | `vibevoice/VibeVoice-1.5B` |
| **VibeVoice-7B** ("Large") | Qwen2.5-7B | 3584 / 28 / 28 / 4 | 32 768 | up to 4 | ~45 min | 152 064 (Qwen2.5 base) | MIT | `vibevoice/VibeVoice-Large` |
| **VibeVoice-Streaming-0.5B** | Qwen2.5-0.5B (split) | 896 / 24 (4 lower + 20 upper TTS) / 14 / 2 | ~32 768 | 1 (single-speaker only) | streaming, real-time | 151 936 | MIT | `vibevoice/VibeVoice-Streaming-0.5B` |

Common to all variants:
- `acoustic_vae_dim` = 64, `semantic_vae_dim` = 128 (streaming has no semantic VAE)
- Tokenizer encoder/decoder: `encoder_n_filters` = `decoder_n_filters` = 32, `encoder_ratios` = `decoder_ratios` = `[8,5,5,4,2,2]`, `encoder_depths` = `"3-3-3-3-3-3-8"` (parsed string), causal, RMSNorm, depthwise-Conv mixer, no weight-norm
- Diffusion head: `hidden_size` = LM hidden_size, `head_layers` = 4, `head_ffn_ratio` = 3.0, `latent_size` = 64, `prediction_type` = "v_prediction", `diffusion_type` = "ddpm", `ddpm_num_steps` = 1000, `ddpm_num_inference_steps` = 20, `ddpm_beta_schedule` = "cosine", RMSNorm eps 1e-5
- Acoustic VAE `fix_std` = 0.5, `std_dist_type` = "gaussian"; semantic VAE `fix_std` = 0, `std_dist_type` = "none" (deterministic mean output only)
- All checkpoints distributed in **bfloat16**

Param-count breakdown (1.5B variant, approximate):
- Qwen2.5-1.5B LM: ~1.55 B
- Acoustic VAE encoder + decoder: ~37 M total (depths sum to 26 ConvNeXt-style blocks)
- Semantic VAE encoder only: ~18 M
- Diffusion head (4 layers @ hidden=1536, ffn=4608): ~50 M
- SpeechConnector × 2 (acoustic 64→1536, semantic 128→1536): ~5 M total
- **Total ~1.66 B params**

### 2. Architecture — Top-Level Composition

The Python class hierarchy (from `vibevoice/modular/modeling_vibevoice.py`):

```
VibeVoiceForConditionalGeneration            # outer wrapper
├── model: VibeVoiceModel
│   ├── language_model: Qwen2Model           # AutoModel.from_config(decoder_config)
│   ├── acoustic_tokenizer: VibeVoiceAcousticTokenizerModel
│   │   ├── encoder: TokenizerEncoder        # 1D ConvNeXt, 6 downsample stages
│   │   └── decoder: TokenizerDecoder        # 1D ConvNeXt-transpose, 6 upsample stages
│   ├── semantic_tokenizer: VibeVoiceSemanticTokenizerModel
│   │   └── encoder: TokenizerEncoder        # same arch as acoustic encoder, vae_dim=128
│   ├── acoustic_connector: SpeechConnector  # 64 → lm_hidden (Linear + RMSNorm + Linear)
│   ├── semantic_connector: SpeechConnector  # 128 → lm_hidden
│   ├── prediction_head: VibeVoiceDiffusionHead  # 4-layer AdaLN-FFN denoiser
│   ├── noise_scheduler: DPMSolverMultistepScheduler  # used at inference (20 steps)
│   ├── speech_scaling_factor: nn.Buffer(scalar)   # latent normalization, learned during training
│   └── speech_bias_factor: nn.Buffer(scalar)
└── lm_head: nn.Linear(lm_hidden → vocab_size)     # tied to embed_tokens for 1.5B (not 7B)
```

The streaming-0.5B variant replaces this with:

```
VibeVoiceStreamingForConditionalGenerationInference
├── model: VibeVoiceStreamingModel
│   ├── language_model: Qwen2Model      # LOWER 4 layers; final norm replaced with nn.Identity()
│   ├── tts_language_model: Qwen2Model  # UPPER 20 layers (embed_tokens unused, only transformer layers used)
│   ├── tts_input_types: nn.Embedding(2, lm_hidden)  # marks position as text(1) or speech(0)
│   ├── acoustic_tokenizer + acoustic_connector + prediction_head + noise_scheduler
│   └── (NO semantic_tokenizer / semantic_connector)
└── tts_eos_classifier: BinaryClassifier(lm_hidden → 1)   # 2-layer MLP for end-of-speech detection
```

The streaming variant has **no `lm_head`** and **no LM-vocab token sampling at all** — speech generation is driven entirely by the diffusion head, and EOS is decided by the binary classifier on the TTS-LM's last hidden state.

### 3. Acoustic Tokenizer (Causal 1D-ConvNeXt VAE)

Implemented in `modular_vibevoice_tokenizer.py`. The full encoder takes a `(B, 1, T)` waveform at 24 kHz and emits a `(B, T/3200, 64)` latent sequence (after permute). The decoder is the mirror.

#### 3.1 Encoder shape

`encoder_ratios = [8, 5, 5, 4, 2, 2]` produces 6 downsample stages with stride products `[8, 5, 5, 4, 2, 2]`. Inside the encoder, the **ratios are reversed** before use (`list(reversed(...))` → `[2, 2, 4, 5, 5, 8]`), and the channel growth follows `n_filters * 2^i`:

| Stage | Stride | Channels in→out | ConvNeXt depth |
|---|---|---|---|
| 0 (stem) | 1 | 1 → 32 | 3 |
| 1 | 2 | 32 → 64 | 3 |
| 2 | 2 | 64 → 128 | 3 |
| 3 | 4 | 128 → 256 | 3 |
| 4 | 5 | 256 → 512 | 3 |
| 5 | 5 | 512 → 1024 | 3 |
| 6 | 8 | 1024 → 2048 | 8 |
| head | 1 | 2048 → 64 | (proj only) |

Each downsampling step is an `SConv1d(in, out, kernel=stride*2, stride=stride, causal=True, pad_mode='constant')`. Each stage then runs `depth` `Block1D` blocks (described below). The final `head` is `SConv1d(2048, 64, kernel=7, causal=True)`, so the latent has 64 channels. `disable_last_norm = True` in config → the optional pre-head RMSNorm is replaced with `nn.Identity()`.

**Total downsample = 8 × 5 × 5 × 4 × 2 × 2 = 3200**, so on 24 kHz input the latent frame rate is `24000 / 3200 = 7.5 Hz`. Each frame = 1/7.5 s = 133.33 ms of audio.

#### 3.2 Decoder shape

Mirror topology: stem is `SConv1d(64 → 2048, kernel=7)`, then 6 upsample stages using `SConvTranspose1d(in, out, kernel=stride*2, stride=stride, causal=True, trim_right_ratio=1.0)` and the **same channel widths in reverse**. Decoder depths default to the **reverse of encoder depths** (`[8, 3, 3, 3, 3, 3, 3]`). Final head is `SConv1d(32, 1, kernel=7, causal=True)`.

The `trim_right_ratio=1.0` setting is critical for causal transpose conv — it removes all "future" padding on the right, which together with cached-left context (`SConvTranspose1d.context_size = kernel - 1`) enables streaming decode one frame at a time.

#### 3.3 Block1D (ConvNeXt-V1-style, not V2)

Per-block path:

```
residual_a = x
x = ConvRMSNorm(x)                              # RMSNorm on channels-last
x = DepthwiseConv1d(dim=d, groups=d, kernel=7)  # causal, no normalization
x = x * gamma                                   # learned per-channel scale, init=1e-6
x = residual_a + drop_path(x)

residual_b = x
x = ConvRMSNorm(x)
x = x.permute(0,2,1)                            # (B, T, C) for FFN
x = Linear(d → 4d)                              # FFN linear1
x = GELU(x)
x = Linear(4d → d)                              # FFN linear2
x = x.permute(0,2,1)                            # back to (B, C, T)
x = x * ffn_gamma
x = residual_b + drop_path(x)
```

`mixer_layer = "depthwise_conv"` (config-default for all official ckpts; no other choice used). FFN expansion is **4×** (different from the diffusion head's 3×). Both `gamma` and `ffn_gamma` are layer-scale parameters initialized to **1e-6**.

#### 3.4 SConv1d / SConvTranspose1d — causal streaming-aware conv

Both wrappers expose a `cache` parameter (a `VibeVoiceTokenizerStreamingCache`) and a `use_cache` flag. In streaming mode, each conv layer keeps its **last `(kernel-1)*dilation - (stride-1)` input samples** as cache and prepends them to the next chunk's input — this is exactly how Mimi / EnCodec causal streaming works.

Cache key is `(f"sconv1d_{id(self)}", sample_index)` — a `Dict[(str,int), Tensor]`. Per-sample cache lets us batch multiple independent streams (e.g. up to 4 speakers in a podcast each generated in their own buffer).

Pad mode is **`"constant"`** (zero-pad) per config — NOT reflect-pad. Bias on every conv. **No weight-norm anywhere** (`conv_norm = "none"`).

#### 3.5 Encoder VAE sampling

`VibeVoiceAcousticTokenizerModel.encode()` returns `VibeVoiceTokenizerEncoderOutput(mean=encoder(audio).permute(0,2,1), std=fix_std=0.5)`. Sampling has three modes:

- `'fix'`: `z = mean + std * N(0,I)` where `std=0.5` is the fixed scalar.
- `'gaussian'` (training and inference default — `std_dist_type = "gaussian"`): a **per-batch random std multiplier** is drawn: `std = N(0, std/0.8) = N(0, 0.625)`, then `z = mean + std * N(0,I)`. The 1/0.8 scale isn't documented in the paper but the code is explicit.
- `'none'`: deterministic — returns mean directly. Used by the semantic VAE.

In the inference pipeline, the acoustic VAE is sampled per-frame during voice-prompt encoding, and the diffusion head outputs are normalized via the learned `speech_scaling_factor` / `speech_bias_factor` (registered as nan-initialized buffers, computed on the first training batch as `1/std` and `-mean` of the VAE latents). At inference time these buffers are loaded from the checkpoint and applied:

```
acoustic_features = (raw_vae_latent + speech_bias_factor) * speech_scaling_factor
... LM consumes acoustic_connector(acoustic_features) ...
... diffusion head emits new normalized latents ...
scaled_latent = (latent / speech_scaling_factor) - speech_bias_factor   # un-normalize
audio_chunk = acoustic_tokenizer.decode(scaled_latent)
```

### 4. Semantic Tokenizer (Encoder-Only, 128-d)

Identical arch to the acoustic encoder but with `vae_dim = 128`, `fix_std = 0`, and `std_dist_type = "none"`. There is **no decoder** — the semantic VAE only produces conditioning embeddings for the LM.

It is used in **two places** at inference:
1. **Voice prompt prefill**: NOT used here — the semantic embedding for the reference audio is computed and added to `speech_input_mask` positions only if `speech_semantic_tensors` is passed to `forward`. In the current inference path, voice prompts go through the acoustic encoder only.
2. **Per-frame feedback in the AR loop**: After the acoustic VAE decodes one new frame's worth of audio, that same audio chunk is re-encoded through the **semantic** VAE (with streaming cache) to produce a 128-d semantic vector. This is then summed with the acoustic embedding (after passing through their respective `SpeechConnector`s) and used as the input embedding for the next LM step.

The semantic-feedback loop is the key insight: it gives the LM access to *what the audio actually sounds like phonetically* after the diffusion head fills in acoustic detail. Without it, prosody and turn-taking would drift.

### 5. SpeechConnector

Tiny MLP at `modular_vibevoice_*.py`:

```
SpeechConnector(input_dim, output_dim):
    fc1: Linear(input_dim, output_dim, bias=True)
    norm: LlamaRMSNorm(output_dim, eps=1e-6)
    fc2: Linear(output_dim, output_dim, bias=True)
```

Two instances exist on the non-streaming model:
- `acoustic_connector`: 64 → lm_hidden
- `semantic_connector`: 128 → lm_hidden

Streaming-0.5B has only `acoustic_connector` (64 → 896).

### 6. Diffusion Head

Implemented in `modular_vibevoice_diffusion_head.py`. Lightweight 4-layer FFN-only transformer (no self-attention) that denoises a single 64-d latent vector per step using the LM's hidden state as cross-step conditioning.

#### 6.1 Topology (1.5B variant: hidden_size=1536, ffn=4608)

```
VibeVoiceDiffusionHead:
  noisy_images_proj: Linear(64 → 1536, bias=False)
  cond_proj:         Linear(1536 → 1536, bias=False)        # projects LM hidden_state
  t_embedder: TimestepEmbedder
    timestep_embedding(t, dim=256)        # sinusoidal (max_period=10000)
    Linear(256 → 1536, bias=False) → SiLU → Linear(1536 → 1536, bias=False)
  layers: 4 × HeadLayer:
    HeadLayer(embed_dim=1536, ffn_dim=4608, cond_dim=1536, eps=1e-5):
      norm: RMSNorm(1536, eps=1e-5)
      adaLN_modulation: Sequential(SiLU, Linear(1536 → 3*1536, bias=False))  # → shift_ffn, scale_ffn, gate_ffn
      ffn: SwiGLU(1536, 4608)        # gate_proj, up_proj, down_proj, all bias=False
        # SwiGLU = down_proj(silu(gate_proj(x)) * up_proj(x))
  final_layer: FinalLayer:
    norm_final: RMSNorm(1536, eps=1e-5, elementwise_affine=False)
    adaLN_modulation: Sequential(SiLU, Linear(1536 → 2*1536, bias=False))   # → shift, scale
    linear: Linear(1536 → 64, bias=False)
```

Per-layer forward (DiT-style AdaLN-Zero applied **only to FFN**, no self-attention):

```
shift_ffn, scale_ffn, gate_ffn = adaLN_modulation(c).chunk(3)
x = x + gate_ffn * ffn(modulate(norm(x), shift_ffn, scale_ffn))
```

`modulate(x, shift, scale) = x * (1 + scale) + shift` (standard DiT modulation).

Final layer:

```
shift, scale = adaLN_modulation(c).chunk(2)
x = linear(modulate(norm_final(x), shift, scale))
```

#### 6.2 Conditioning

`c = cond_proj(lm_hidden_state) + t_embedder(timestep)` — a single 1536-d vector per diffusion step, used by **all 4 layers and the final layer** (no per-layer cond fan-out, unlike Flux/SD3). For CFG, two parallel streams are batched: one with positive condition, one with the negative-prompt condition.

#### 6.3 Init policy (zero-init AdaLN-Zero)

- All `adaLN_modulation[-1].weight` initialized to **zero** (zero-init AdaLN-Zero)
- `final_layer.linear.weight` initialized to **zero**
- `t_embedder.mlp[0].weight` and `t_embedder.mlp[2].weight` initialized to `N(0, 0.02)`

This means the diffusion head starts as an identity-like function and learns to add corrections during training. The HartsyInference port doesn't need to reproduce the init — just load the trained weights — but the math invariant matters for unit tests (random-init output must equal the identity branch).

### 7. Inference Loop (Multi-Speaker, 1.5B / 7B)

From `vibevoice/modular/modeling_vibevoice_inference.py`. The loop is autoregressive at the LM-token level, with a diffusion sub-loop firing on `speech_diffusion` tokens.

#### 7.1 Prefill phase

1. **Build prompt** (via `VibeVoiceProcessor`): `system_prompt + voice_input_tokens + text_input_tokens + speech_output_tokens` (see §8 below for the exact format).
2. **Forward through LM** with `inputs_embeds` after running the voice-prompt audio through `acoustic_tokenizer.encode()` → `acoustic_features` → `acoustic_connector` → **insert at `speech_input_mask=True` positions**. The semantic VAE is NOT applied to voice prompts in the current inference code path.
3. Populates `past_key_values` for the **positive stream** and a separate **negative stream** initialized with a single `speech_start` token.

#### 7.2 Generation loop

Per step (capped at `max_steps = min(max_length - prompt_len, max_length_times * prompt_len)`, default `max_length_times = 2`):

1. **LM forward** on the last token, get logits for the next token.
2. **Logit constraint**: only `{speech_start_id, speech_end_id, speech_diffusion_id, eos_token_id, bos_token_id}` are allowed (other logits set to -inf via `VibeVoiceTokenConstraintProcessor`).
3. **Sample next token** (multinomial or argmax).
4. Branch on the token:
   - **`eos`**: mark sample finished.
   - **`speech_end`**: zero out per-sample acoustic + semantic streaming caches.
   - **`speech_start`**: rotate negative-stream KV cache (move first key/value to last position to keep negative cond aligned).
   - **`speech_diffusion`**: this is where audio is generated. (Sub-loop below.)
5. **Build next-step embedding**:
   - For non-diffusion tokens: `next_inputs_embed = embed_tokens(next_token)`.
   - For diffusion tokens: `next_inputs_embed = acoustic_connector(speech_latent) + semantic_connector(semantic_features)`.

#### 7.3 Diffusion sub-loop (per `speech_diffusion` token)

Triggered by `sample_speech_tokens(condition, neg_condition, cfg_scale)`:

```
scheduler.set_timesteps(20)                              # 20 DDPM steps with cosine beta
condition_pair = cat([positive_cond, negative_cond])    # (2, 1536)
speech = N(0, I) ∈ R^(2, 64)                            # noise init
for t in scheduler.timesteps:                            # 20 iterations
    half = speech[:1]
    combined = cat([half, half])                         # (2, 64), same latent both sides
    eps = prediction_head(combined, t.expand(2), condition_pair)   # (2, 64)
    cond_eps, uncond_eps = split(eps)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)    # CFG
    eps = cat([half_eps, half_eps])
    speech = scheduler.step(eps, t, speech).prev_sample
return speech[:1]                                        # (1, 64)
```

Then:

```
scaled_latent = speech_latent / speech_scaling_factor - speech_bias_factor  # un-normalize
audio_chunk = acoustic_tokenizer.decode(scaled_latent, cache=acoustic_cache, use_cache=True)
# audio_chunk has shape (B, 1, 3200) — one 7.5 Hz frame = 3200 samples at 24 kHz
semantic_features = semantic_tokenizer.encode(audio_chunk, cache=semantic_cache, use_cache=True).mean
acoustic_embed = acoustic_connector(speech_latent)
semantic_embed = semantic_connector(semantic_features)
next_embed = acoustic_embed + semantic_embed
audio_chunks[sample].append(audio_chunk)
```

#### 7.4 Negative-stream maintenance

The negative branch (uncond CFG side) runs **in parallel with the positive branch on every step**, with its own `past_key_values`. Two subtle bits:

- On `speech_start`, the negative attention mask is zeroed except the last position, and its KV-cache last-position is overwritten with its first-position content. This re-anchors the negative branch on the diffusion event.
- On non-diffusion tokens following a diffusion event, the negative KV cache is "shifted" to keep the negative branch tracking the positive branch's logical position (via `correct_cnt` bookkeeping).

This bookkeeping is fiddly but mechanical — it's what makes per-token CFG work efficiently on a single negative prompt.

#### 7.5 Default sampler parameters

- `cfg_scale = 1.3` (most common in demo scripts; the `generate()` default is 1.0)
- `do_sample = True`, `temperature = 0.95`, `top_p = 0.95`
- `ddpm_inference_steps = 20` (config default — overrideable via `set_ddpm_inference_steps`)
- `max_length_times = 2` (per-sample length cap is `min(max_length - prompt, 2*prompt)`)

### 8. Multi-Speaker Prompt Format

From `vibevoice/processor/vibevoice_processor.py`. The processor builds a Qwen-tokenized prompt that looks like:

```
 Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.
 Voice input:
 Speaker 0:<|vision_start|><|vision_pad|>*N_0<|vision_end|>
 Speaker 1:<|vision_start|><|vision_pad|>*N_1<|vision_end|>
 ...
 Text input:
 Speaker 0: Hello, welcome to the podcast.
 Speaker 1: Thanks, glad to be here.
 ...
 Speech output:
<|vision_start|>                                # <-- LM starts generating from here
```

Then during AR generation the LM emits `<|vision_pad|>` (`speech_diffusion_id`) tokens — each one triggers one 7.5 Hz audio frame. The LM eventually emits `<|vision_end|>` (`speech_end_id`) to end a speaker turn, then `<|vision_start|>` again to begin the next turn, until `<|endoftext|>` (eos) terminates the whole utterance.

#### 8.1 Special tokens reuse Qwen2 vision tokens

VibeVoice **does not add new token IDs to the Qwen vocab**. It re-purposes:

| Logical token | Qwen2 token | Default ID (1.5B) |
|---|---|---|
| `speech_start` | `<|vision_start|>` | 151 652 |
| `speech_end` | `<|vision_end|>` | 151 653 |
| `speech_diffusion` | `<|vision_pad|>` | 151 654 |
| `eos` / `pad` / `unk` | `<|endoftext|>` | 151 643 |

That's a **huge** porting win: we don't need any tokenizer changes — just remap these three IDs to our own logical roles in the inference loop.

#### 8.2 N_i computation (voice-prompt latent count)

```
N_i = ceil(len(speaker_i_audio_samples) / 3200)
```

So a 6-second reference at 24 kHz produces `ceil(144000 / 3200) = 45` `<|vision_pad|>` tokens, each replaced (in `speech_input_mask`) with a 64-d acoustic latent from the encoder.

#### 8.3 Audio normalization (input only)

Per `vibevoice_tokenizer_processor.py::AudioNormalizer`:

```
target_dB_FS = -25
eps = 1e-6
rms = sqrt(mean(audio^2))
target_rms = 10^(target_dB_FS / 20) = 0.05623
audio = audio * (target_rms / (rms + eps))
```

This applies to reference audio before encoding. Output audio is **not** normalized by the model — the user can apply their own loudness curve.

### 9. Streaming-0.5B Variant — Differences

The streaming model is a meaningfully different architecture, not just a smaller LM. From `vibevoice/modular/configuration_vibevoice_streaming.py` and `modeling_vibevoice_streaming_inference.py`:

#### 9.1 Split-LM architecture

- **Lower layers** (`config.num_hidden_layers - tts_backbone_num_hidden_layers = 4` by default for the 24-layer Qwen2.5-0.5B): pure text encoding. The **final norm is replaced with `nn.Identity()`** so its output stays in pre-norm space.
- **Upper layers** (`tts_backbone_num_hidden_layers = 20`): TTS generation. The lower-layer hidden states are spliced into the TTS-LM's `inputs_embeds` tail at each step, then a learned **`tts_input_types: nn.Embedding(2, 896)`** is added (index 1 for text positions, index 0 for speech-frame positions).

This split lets the text encoder run "ahead" of the speech generator in a windowed fashion (5 text tokens per window → 6 speech frames per window — see constants `TTS_TEXT_WINDOW_SIZE = 5`, `TTS_SPEECH_WINDOW_SIZE = 6` at the top of the inference file).

#### 9.2 No semantic VAE

The streaming model has **only the acoustic VAE** — no semantic encoder, no `semantic_connector`. Per-frame feedback uses just the acoustic embedding.

#### 9.3 Binary EOS classifier

`tts_eos_classifier = BinaryClassifier(hidden_size)` is a 2-layer MLP (`Linear → ReLU → Linear → 1`) that takes the TTS-LM's last hidden state and emits a single logit. EOS is decided by **thresholding this logit** (default threshold > 0 → stop), NOT by the LM emitting an `eos` token (there is no `lm_head` at all).

#### 9.4 Constrained to batch=1 and single speaker

`assert batch_size == 1` in `generate()`. The processor for streaming (`vibevoice_streaming_processor.py`) is single-speaker only.

#### 9.5 Negative prompt token

For CFG, the streaming variant uses `<|image_pad|>` (Qwen2 vision token, not used by VibeVoice for anything else) as the negative-prompt sentinel — distinct from the multi-speaker variant which uses `<|vision_start|>`.

### 10. DPM-Solver Scheduler

`vibevoice/schedule/dpm_solver.py` is a 1064-line port of HuggingFace `diffusers`' `DPMSolverMultistepScheduler` with the following config at inference:

- `num_train_timesteps = 1000`
- `beta_schedule = "cosine"` (Nichol-Dhariwal cosine)
- `prediction_type = "v_prediction"`
- `algorithm_type = "dpmsolver++"` (default; the file also supports "sde-dpmsolver++" and "deis")
- `solver_order = 2` (multistep)
- `num_inference_steps = 20`

`HartsyInference.Diffusion` already has `DpmppMultiStepScheduler` with v-prediction support — we should verify the cosine beta schedule and exact alpha bar formula match, then **reuse the existing scheduler** rather than porting the file.

Cosine schedule reference (Nichol-Dhariwal):
```
alpha_bar(t) = cos^2((t/T + s) / (1 + s) * pi/2),  s = 0.008
betas[t] = clamp(1 - alpha_bar(t) / alpha_bar(t-1), 0, 0.999)
```

This must match exactly — a small drift in `s` or the clip bound changes the inference trajectory enough to be audible.

### 11. License & Commercial Use

**MIT** — fully commercial-friendly. The community fork preserves the original license terms after Microsoft removed the official repo. Voice samples in `demo/` are CC-BY-4.0 from Common Voice + LibriSpeech (cleared for redistribution). Model weights on HF are MIT.

### 12. Mapping to HartsyInference

Tasks split between "reuse" (existing HartsyInference / dotLLM code) and "new":

| Component | Reuse | New |
|---|---|---|
| Qwen2.5-0.5B / 1.5B / 7B LM | **dotLLM** (full Qwen2 forward + KV cache + RoPE + tied embeddings) | None — dotLLM dependency check only |
| Qwen2 tokenizer (BPE) | **dotLLM** tokenizer | Subclass to expose `speech_start_id`, `speech_end_id`, `speech_diffusion_id` from `<|vision_start/end/pad|>` |
| `<|vision_pad|>` constraint logits processor | — | New: small constrained-vocab logit mask (~30 lines) |
| Cosine beta schedule, v-prediction, DPM++ multistep | `HartsyInference.Diffusion/Schedulers/DpmppMultiStepScheduler` (verify cosine match) | Likely no new code; one validation test |
| Acoustic VAE encoder + decoder | — | **New** `VibeVoiceAcousticTokenizer.cs` (1D causal ConvNeXt + streaming cache). Patterns from F5-TTS Vocos but causal and depthwise-conv. |
| Semantic VAE encoder | — | **New** `VibeVoiceSemanticTokenizer.cs` (encoder only; shares `Block1D` with acoustic) |
| `SpeechConnector` | — | **New** trivial 3-line MLP |
| Diffusion head (4-layer AdaLN+FFN) | AdaLN-Zero patterns from `Flux/SD3` DiT blocks | **New** `VibeVoiceDiffusionHead.cs` — strip out attention, keep only FFN + AdaLN modulation |
| Multi-speaker prompt format | — | **New** `VibeVoiceProcessor.cs` — script parser + voice-prompt builder + audio dB-FS normalizer |
| Streaming cache for SConv1d / SConvTranspose1d | — | **New** `VibeVoiceTokenizerStreamingCache` — per-sample, per-layer ring of last `(k-1)*d - (s-1)` samples |
| Streaming-0.5B split-LM forward | dotLLM Qwen2 (per-layer hooks needed — see deviation below) | **New** wrapper that runs lower-N layers, then upper-(N-K) layers as a separate forward with `tts_input_types` embedding injected |
| Binary EOS classifier | — | **New** trivial 2-layer MLP |
| Audio I/O (24 kHz PCM) | `HartsyInference.Audio/Io` | None |

#### 12.1 Deviation risk: dotLLM Qwen2 must expose per-layer outputs

The streaming-0.5B variant needs to run the **lower 4 Qwen2 layers**, then independently run the **upper 20 layers** with modified inputs. dotLLM's Qwen2Model normally returns only the final hidden state. Two options:

1. **Add a "stop at layer N" parameter** to dotLLM's Qwen2 forward (preferred — minimal API surface).
2. **Construct two independent Qwen2Model instances** with overlapping but disjoint layer slices, sharing the same embedding table.

The first is cleaner and matches what `VibeVoiceStreamingModel` does in Python (it instantiates two `Qwen2Model`s but unused embed_tokens on the second). We should coordinate with dotLLM maintainers (or self, since same author) before settling.

#### 12.2 GPU kernels required

All ops needed already exist in `HartsyInference.Core`/`HartsyInference.Diffusion`:
- `Conv1d` (causal padding + dilation) — already exists for F5-TTS Vocos
- `ConvTranspose1d` (causal + trim_right) — already exists for HiFiGAN/Vocos vocoders
- `RMSNorm`, `LayerNorm`, `SiLU`, `GELU` — all in `Core`
- `Linear` (no bias) — in `Core`
- `nn.functional.pad` (constant zero-pad mode) — in `Core`
- Permute / reshape — in `Core`

No new PTX kernels needed.

#### 12.3 Package boundaries

- `HartsyInference.Audio.Models.VibeVoice/` — all new model code
- `HartsyInference.Audio.Pipelines.VibeVoicePipeline.cs` — orchestration
- **dotLLM** dependency: Qwen2.5 LM only (existing). VibeVoice should not depend on `dotLLM.Cli` or any other dotLLM package beyond what F5-TTS already uses (if any) — see [`docs/Design/NUGET_PACKAGE_DESIGN.md`](../Design/NUGET_PACKAGE_DESIGN.md).

### 13. Open Questions / Pre-Implementation Verification

These need to be resolved during the planning phase, not during research:

1. **Cosine beta exact form**: does our existing `DpmppMultiStepScheduler` use the same `s=0.008` cosine? Run a 20-step diff against `vibevoice/schedule/dpm_solver.py::DPMSolverMultistepScheduler(beta_schedule="cosine")` on identical seeds → match to 1e-6.
2. **`speech_scaling_factor` / `speech_bias_factor` provenance**: confirm these are present in the safetensors file (registered as buffers — should serialize). If not, we need a fallback.
3. **`std_dist_type = 'gaussian'`**: in inference, do we actually need the random std multiplier, or is the deterministic `mean` good enough? Demo scripts always use `gaussian` — match the upstream behavior to avoid quality regression.
4. **Streaming cache key**: Python uses `id(self)` as part of the cache key, which doesn't transfer to a C# port. We need a deterministic per-layer ID assigned at construction time. Use `layer_index` + `block_index` + role (`"enc"|"dec"`).
5. **dotLLM Qwen2 per-layer interrupt**: confirm we can stop forward at layer N or do we need to refactor. (See §12.1.)
6. **bf16 on GPU**: all official ckpts are bf16. Our existing diffusion DiT pipelines already run bf16 — confirm `Tensor.CastTo` covers this path and bf16 matmul kernels are in place. (Should be — Flux/SD3/Z-Image all use bf16.)
7. **CFG-Zero* / APG**: the upstream code uses **vanilla CFG only** (`x = uncond + scale*(cond - uncond)`). Do not enable APG / CFG-Zero* until separately validated — they're not used by VibeVoice.
8. **Long-form stability**: 90-min outputs need a 65k-token context (1.5B variant has `max_position_embeddings = 65536`). Confirm dotLLM's KV cache scales to this without precision loss. RoPE theta is 1e6 (long-context preset).

### 14. Validation Checklist (for the implementation phase)

- [ ] Acoustic VAE encoder forward on a 24 kHz 1s clip matches Python within 1e-3 (bf16) / 1e-5 (f32)
- [ ] Acoustic VAE decoder forward on a known 64-d latent matches Python within 1e-3
- [ ] Acoustic VAE round-trip (encode→decode) STOI > 0.95 on a 5s LibriSpeech clip
- [ ] Semantic VAE encoder forward matches Python within 1e-4
- [ ] Diffusion head single-step forward (random latent, random timestep, random condition) matches Python within 1e-3 (bf16)
- [ ] DPM-Solver 20-step output matches Python's `vibevoice/schedule/dpm_solver.py` on identical seed within 1e-3
- [ ] Voice-prompt prefill: input_ids, speech_input_mask, speech_tensors, speech_masks match Python byte-for-byte
- [ ] One-step end-to-end: same prompt + same RNG seed → same output token + same 64-d latent + same 3200-sample audio chunk within 1e-2
- [ ] Multi-speaker 4-way demo script → audible end-to-end output matches Python demo (subjective)
- [ ] Streaming-0.5B: split-LM forward equivalent to single-pass forward when text-only (no `tts_input_types` activation)
- [ ] Streaming-0.5B: first-frame latency < 250 ms on RTX 3060 12 GB
- [ ] 90-min generation (1.5B): VRAM stable (no leak), no OOM
- [ ] License audit confirms MIT in published artifact metadata
