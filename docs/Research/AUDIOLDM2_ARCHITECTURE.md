# AudioLDM 2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (AudioLDM2 pipeline)

## Summary

AudioLDM 2 (Liu et al., 2023) is a text-to-audio latent diffusion model that generates music, sound effects, and (in the speech variants) intelligible speech from natural-language prompts. It produces 16 kHz mono waveforms (typically ~10.24 s per generation) by denoising in a compact mel-spectrogram latent space and vocoding the decoded mel with HiFi-GAN. The model is architecturally unusual: instead of feeding text embeddings directly into the UNet, it uses a **two-stage conditioning pipeline** — (1) CLAP and FLAN-T5-Large jointly encode the prompt, (2) a small **GPT-2** autoregressively produces a fixed-length "AudioMAE-style" continuous feature sequence from those embeddings. The UNet then cross-attends to both the GPT-2 output *and* the original CLAP/T5 text features via two parallel cross-attention streams. Diffusion is classic Gaussian (eps-prediction, DDIM) at 200 steps with CFG 3.5. The full pipeline is: text → (CLAP + T5) → projection → GPT-2 (8 tokens) → UNet denoise (200 steps) → VAE decode → mel → HiFi-GAN → 16 kHz waveform.

This file covers the model architecture and inference pipeline. The HiFi-GAN vocoder (a SpeechT5HifiGan variant with non-standard upsampling) is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Mel preprocessing reference parameters are in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). Classifier-free guidance details are in [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md). DDIM is in [DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md). T5 encoder implementation notes are in [TEXT_ENCODERS.md](TEXT_ENCODERS.md).

Sources:
- Paper: [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining (arXiv:2308.05734)](https://arxiv.org/abs/2308.05734)
- Reference repo: [haoheliu/AudioLDM2](https://github.com/haoheliu/AudioLDM2)
- Diffusers pipeline: [`pipeline_audioldm2.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py)
- Diffusers modeling: [`modeling_audioldm2.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py)
- Weights: [`cvssp/audioldm2`](https://huggingface.co/cvssp/audioldm2), [`cvssp/audioldm2-large`](https://huggingface.co/cvssp/audioldm2-large), [`cvssp/audioldm2-music`](https://huggingface.co/cvssp/audioldm2-music)

License: cc-by-nc-sa-4.0 (non-commercial — note this is incompatible with paid HartsyInference use; flag in the pipeline docs).

## Detailed Findings

### 1. Variants

All public checkpoints share the same encoders (CLAP, T5, GPT-2, projection model), the same VAE, and the same SpeechT5HifiGan vocoder. They differ only in the UNet weights and (where applicable) `transformer_layers_per_block`.

| Checkpoint | Task | UNet params | Total params | UNet `transformer_layers_per_block` | Training audio | Sample rate | Default length |
|---|---|---|---|---|---|---|---|
| `cvssp/audioldm2` | general (sfx + music + some speech) | 350 M | ~1.1 B | 1 | 1,150 k hours | 16 kHz | 10.24 s |
| `cvssp/audioldm2-large` | general | 750 M | ~1.5 B | 2 | 1,150 k hours | 16 kHz | 10.24 s |
| `cvssp/audioldm2-music` | music only | 350 M | ~1.1 B | 1 | 665 k hours (music) | 16 kHz | 10.24 s |
| `cvssp/audioldm2-speech-gigaspeech` | TTS (GigaSpeech) | 350 M | ~1.1 B | 1 | speech corpus | 16 kHz | 10.24 s |
| `cvssp/audioldm2-speech-ljspeech` | TTS (LJSpeech) | 350 M | ~1.1 B | 1 | LJSpeech | 16 kHz | 10.24 s |

There is also a `audioldm_48k` upstream variant in the reference repo (not on diffusers): 48 kHz output, 256 mel bins, hop 480, `n_fft` 2048, latent embed dim 16, latent time 128, latent freq 32. **Not** wired into the diffusers pipeline — we'll target the diffusers configs for HartsyInference v1 and revisit 48k later.

The base / large / music checkpoints are the focus of this doc; the speech checkpoints additionally use a VITS-based phoneme encoder in place of T5, which the diffusers `AudioLDM2Pipeline` supports but is a separate code path. We will only implement the **CLAP + T5** general/music path in v1.

### 2. Pipeline Components (diffusers `model_index.json`)

```
{
  "feature_extractor":  ClapFeatureExtractor   (transformers)
  "tokenizer":          RobertaTokenizerFast   (transformers)       # for CLAP
  "text_encoder":       ClapModel              (transformers)
  "tokenizer_2":        T5TokenizerFast        (transformers)
  "text_encoder_2":     T5EncoderModel         (transformers)       # FLAN-T5-Large
  "projection_model":   AudioLDM2ProjectionModel (audioldm2)
  "language_model":     GPT2Model              (transformers)
  "unet":               AudioLDM2UNet2DConditionModel (audioldm2)
  "vae":                AutoencoderKL          (diffusers)
  "vocoder":            SpeechT5HifiGan        (transformers)
  "scheduler":          DDIMScheduler          (diffusers)
}
```

The feature_extractor is **inference-only used for ranking** generated waveforms via CLAP audio↔text similarity when `num_waveforms_per_prompt > 1`. The text-only inference path does not require it.

### 3. CLAP Text Encoder

CLAP is a contrastive audio-text encoder (the audio analogue of CLIP). For AudioLDM 2 inference we only use the **text tower**.

- Tokenizer: RoBERTa BPE (vocab 50,265).
- Text tower: 12-layer RoBERTa-base style transformer, hidden 768.
- After the transformer, a projection head reduces hidden states to a **single 512-dim embedding** per prompt: `text_features = clap.get_text_features(input_ids, attention_mask)` → shape `(B, 512)`.
- The pipeline unsqueezes to `(B, 1, 512)` to expose a sequence dimension. This 1-token CLAP embedding is the "global" text condition.

CLAP config fields surface only the fusion/projection sizes (`fusion_hidden_size=768`, `projection_hidden_size=768`); the rest of the architecture matches the standard HuggingFace `LAION/clap-htsat-unfused` text model. HartsyInference must implement:
- RoBERTa BPE tokenization (we have BPE infrastructure; add RoBERTa-specific byte-level pretok + special tokens `<s>` `</s>` `<pad>` `<unk>`).
- RoBERTa transformer (post-LN, learned positional embeddings, gelu, max_position=514 with offset 2).
- A projection MLP (linear → gelu → linear → L2-normalize) producing the 512-dim feature.

We do **not** need the CLAP audio tower for the text-to-audio path. We *might* want it later for the optional CLAP-rerank quality control feature.

### 4. T5 Text Encoder (FLAN-T5-Large)

The second text encoder is FLAN-T5-Large (encoder only).

| Field | Value |
|---|---|
| `d_model` | 1024 |
| `d_ff` | 2816 |
| `num_layers` | 24 |
| `num_heads` | 16 |
| `d_kv` | 64 |
| `feed_forward_proj` | `gated-gelu` (T5 v1.1 style) |
| `is_gated_act` | true |
| `relative_attention_num_buckets` | 32 |
| `relative_attention_max_distance` | 128 |
| `layer_norm_epsilon` | 1e-6 |
| `vocab_size` | 32128 |
| `n_positions` | 512 (no positional embedding — relative bias only) |
| `tie_word_embeddings` | false |

Pad token id 0, EOS token id 1. Input is padded to `max_length` (the pipeline uses the tokenizer's model max length unless overridden). Output is `(B, T5_seq_len, 1024)`.

This is the standard FLAN-T5-Large encoder — HartsyInference already has the T5 encoder kernel from prior pipelines. The only T5-XL/Pile-T5/UMT5 idiosyncrasies (per-layer relative bias) **do not** apply here; this is plain T5 v1.1.

### 5. Projection Model (`AudioLDM2ProjectionModel`)

A tiny module (essentially two linears and four learned vectors). Its job: project the CLAP token and the T5 sequence into the **GPT-2 embedding space** (768) and frame both with learned SOS/EOS markers.

Config:
```
text_encoder_dim     = 512   (CLAP)
text_encoder_1_dim   = 1024  (T5)
langauge_model_dim   = 768   (GPT-2)   # note the typo in upstream config field name
```

Components:
- `projection`: `Linear(512, 768)` — CLAP → GPT-2 space.
- `projection_1`: `Linear(1024, 768)` — T5 → GPT-2 space.
- Four learned parameters: `sos_embed`, `eos_embed`, `sos_embed_1`, `eos_embed_1`, each shape `(768,)`.
- (For VITS variants: an optional positional embedding table — not used in our CLAP+T5 path.)

Forward:
1. Project: `h0 = projection(clap_features)` → `(B, 1, 768)`. Mask `m0 = ones(B,1)`.
2. Project: `h1 = projection_1(t5_features)` → `(B, T5_len, 768)`. Mask `m1 = t5_attention_mask`.
3. Insert SOS/EOS into each stream: prepend `sos_embed` and append `eos_embed` at positions where the mask transitions (so unpadded sequences get `[SOS, ...real..., EOS, pad, pad]`). Implementation in upstream: `add_special_tokens(hidden, attn, sos_token, eos_token)` walks each row.
4. Concatenate along sequence: `hidden = cat([h0_with_special, h1_with_special], dim=1)` → `(B, 1+2+T5_len+2, 768) = (B, T5_len+5, 768)`. Attention mask concatenated likewise.

The result is what GPT-2 consumes as `inputs_embeds`.

### 6. GPT-2 Language Model — the AudioMAE Feature Generator

This is the architecturally novel piece. AudioLDM 2 trains a tiny GPT-2 to predict **continuous AudioMAE-style features** in its hidden-state space, conditioned on the projected CLAP+T5 prefix. At inference, we autoregress 8 hidden states and feed them to the UNet as a second cross-attention stream. The token output head is **never used at inference** — we read the last hidden state directly.

GPT-2 config (= GPT-2 small, but used as a continuous-feature predictor):

| Field | Value |
|---|---|
| `n_layer` | 12 |
| `n_embd` | 768 |
| `n_head` | 12 |
| `n_inner` | null (defaults to 4 × `n_embd` = 3072) |
| `n_positions` / `n_ctx` | 1024 |
| `activation_function` | `gelu_new` |
| `layer_norm_epsilon` | 1e-5 |
| `vocab_size` | 50257 (unused at inference) |
| `max_new_tokens` | **8** |

The default `max_new_tokens=8` is set in the GPT-2 config and consumed by the pipeline. This is the AudioMAE feature length used during training, and *must not change at inference* — the UNet has only ever seen sequences of 8 in this stream.

#### Generation loop (continuous-feature mode)

Standard GPT-2 generation reads `logits = lm_head(hidden)` then samples a token. AudioLDM 2 throws away the logits entirely and treats the **last hidden state** as the next "token embedding":

```python
inputs_embeds = projected_prompt_embeds                # (B, T5_len+5, 768)
for _ in range(max_new_tokens):                        # 8 iterations
    model_inputs = prepare_inputs_for_generation(inputs_embeds, **kwargs)
    out = self.language_model(**model_inputs, output_hidden_states=True, return_dict=True)
    next_hidden = out.hidden_states[-1][:, -1:, :]     # (B, 1, 768) - last layer, last position
    inputs_embeds = torch.cat([inputs_embeds, next_hidden], dim=1)
    kwargs = self.language_model._update_model_kwargs_for_generation(out, kwargs)
return inputs_embeds[:, -max_new_tokens:, :]           # (B, 8, 768) — the AudioMAE feature sequence
```

Notes:
- This is **deterministic** given the prompt embeds — there is no sampling, no temperature, no top-p.
- `use_cache=true` in the GPT-2 config means KV-cache should be used; the prefix is encoded once on the first step.
- The prefix length is `T5_len + 5` (T5 sequence + SOS_clap + EOS_clap + content_clap + SOS_t5 + EOS_t5). Plus 8 generated steps. So GPT-2 sees at most `T5_max_len + 13` positions, well under `n_ctx=1024`.
- For **classifier-free guidance**, GPT-2 is run twice (once on the negative-prompt projection, once on the positive-prompt projection) producing two `(B, 8, 768)` tensors that are concatenated along the batch dim before the UNet loop.

#### What is an "AudioMAE feature"?

Per the paper, AudioMAE is an audio masked autoencoder pretrained on AudioSet that produces patch-level features over a mel spectrogram. During AudioLDM 2 training, AudioMAE features extracted from the *target* audio (8 averaged patch tokens) act as the regression target for GPT-2. At inference, GPT-2 hallucinates those features purely from text. This gives the UNet a much richer, audio-aware conditioning than text alone.

For HartsyInference: we do **not** need to ship AudioMAE — it's a training-time component only. We only need the trained GPT-2.

### 7. UNet (`AudioLDM2UNet2DConditionModel`)

Standard 2D conditional UNet (Stable-Diffusion-style block layout) but applied to a **mel-shaped** latent and with **two parallel cross-attention streams** per transformer block.

Canonical config (base & music; large differs only in `transformer_layers_per_block`):

| Field | Value |
|---|---|
| `sample_size` | 256 (time axis of latent at default 10.24 s) |
| `in_channels` / `out_channels` | 8 / 8 |
| `block_out_channels` | (128, 256, 384, 640) |
| `down_block_types` | `DownBlock2D, CrossAttnDownBlock2D, CrossAttnDownBlock2D, CrossAttnDownBlock2D` |
| `mid_block_type` | `UNetMidBlock2DCrossAttn` |
| `up_block_types` | `CrossAttnUpBlock2D, CrossAttnUpBlock2D, CrossAttnUpBlock2D, UpBlock2D` |
| `layers_per_block` | 2 |
| `transformer_layers_per_block` | 1 (base/music) / 2 (large) |
| `attention_head_dim` | 8 |
| `cross_attention_dim` | nested list — see below |
| `norm_num_groups` / `norm_eps` | 32 / 1e-5 |
| `act_fn` | `silu` |
| `use_linear_projection` | false (use convolutional projection at transformer boundaries) |
| `time_embedding_type` | positional (sinusoidal) |
| `flip_sin_to_cos` / `freq_shift` | true / 0 |
| `resnet_time_scale_shift` | default (additive timestep bias) |
| `mid_block_scale_factor` | 1 |
| `conv_in_kernel` / `conv_out_kernel` | 3 / 3 |
| `downsample_padding` | 1 |

The latent tensor is `(B, 8, T_latent, F_latent)` where the **height axis is time** (`T_latent` = 256 at 10.24 s) and **width axis is mel frequency** (`F_latent` = 16). 8 channels matches the VAE's `latent_channels=8`.

#### `cross_attention_dim` shape (the unusual bit)

`cross_attention_dim` is a **list of lists**: `cross_attention_dim[block_idx]` is itself a list with one entry per cross-attention sub-layer at that block (i.e., per `transformer_layers_per_block` *times* the number of stacked cross-attns within a transformer layer).

Base/music (1 transformer layer per block, 3 cross-attn sub-layers exposed):
```
cross_attention_dim = [
  [null, 768, 1024],   # block 0 (no cross-attn at sub-layer 0 since DownBlock2D has none — placeholder; relevant for mid/up)
  [null, 768, 1024],   # block 1
  [null, 768, 1024],   # block 2
  [null, 768, 1024],   # block 3
]
```

Large (2 transformer layers per block, 4 cross-attn sub-layers):
```
cross_attention_dim = [
  [null, 768, 1024, null],
  [null, 768, 1024, null],
  [null, 768, 1024, null],
  [null, 768, 1024, null],
]
```

Each entry tells the block which conditioning stream that cross-attention layer attends to:
- **null** → skip cross-attention at this sub-layer (self-attn + FFN only).
- **768** → cross-attend to GPT-2 output (`encoder_hidden_states`, shape `(B, 8, 768)`).
- **1024** → cross-attend to CLAP+T5 features (`encoder_hidden_states_1`, shape `(B, ?, 1024)`).

The diffusers routing rule (per `modeling_audioldm2.py`):
```
idx <= 1  → forward_encoder_hidden_states = encoder_hidden_states    (GPT-2)
idx >  1  → forward_encoder_hidden_states = encoder_hidden_states_1  (CLAP+T5)
```
i.e. within a transformer layer the cross-attns alternate **GPT-2 first, then text features**. Only **up to 4** cross-attention layers per block are supported by the diffusers implementation — keep this constraint in C# parity tests.

Wait — the dim 1024 doesn't match either CLAP (512) or T5 (1024). It matches **T5 only**. So `encoder_hidden_states_1` is the *T5-only* tensor (not CLAP+T5 concat). Re-reading the pipeline call:

```python
noise_pred = self.unet(
    latent_model_input, t,
    encoder_hidden_states=generated_prompt_embeds,        # GPT-2 output (B, 8, 768)
    encoder_hidden_states_1=prompt_embeds,                # T5 output     (B, T5_len, 1024)
    encoder_attention_mask_1=attention_mask,              # T5 mask
)
```

Confirmed: stream 0 = GPT-2 (768), stream 1 = T5 (1024). **CLAP is consumed only through the projection model → GPT-2 prefix; it is not directly cross-attended to by the UNet.** This is a key correction to the original task brief.

#### Timestep embedding

Standard sinusoidal at `block_out_channels[0]=128` dims, projected through `Linear → SiLU → Linear` to `4 × 128 = 512` dims, added inside each ResBlock. No class embeddings, no add embeddings (`class_embed_type=null`, `projection_class_embeddings_input_dim=null`).

#### What HartsyInference can reuse

The block layout is **identical** to SD1.5's UNet except for: (a) 8 latent channels instead of 4, (b) `block_out_channels=(128,256,384,640)` instead of `(320,640,1280,1280)`, (c) two cross-attention streams routed per the rule above, (d) the input is shaped `(time, mel_freq)` rather than `(H, W)` but the math is unchanged. The existing SD1.5/SDXL UNet kernels (GroupNorm, ResnetBlock2D, Transformer2D, CrossAttn) port over directly. The only new piece is the dual-stream cross-attention routing, which is a forward-time control-flow change inside the transformer block.

### 8. VAE (`AutoencoderKL`, mel-spectrogram variant)

Standard diffusers `AutoencoderKL` with a 1-channel image (mel) instead of 3-channel RGB.

| Field | Value |
|---|---|
| `in_channels` / `out_channels` | 1 / 1 |
| `latent_channels` | 8 |
| `block_out_channels` | (128, 256, 512) |
| `down_block_types` | `DownEncoderBlock2D × 3` |
| `up_block_types` | `UpDecoderBlock2D × 3` |
| `layers_per_block` | 2 |
| `norm_num_groups` | 32 |
| `act_fn` | `silu` |
| `sample_size` | 1024 |
| `scaling_factor` | **0.4110932946205139** (≠ SD's 0.18215) |
| `force_upcast` | true |

`vae_scale_factor = 2 ^ (len(block_out_channels) - 1) = 2^2 = 4`. (Note: **4, not 8** — there are only 3 down-blocks. This differs from SD's VAE.)

Input mel: `(B, 1, mel_time, mel_freq) = (B, 1, 1024, 64)` at 10.24 s. Encoded latent: `(B, 8, 256, 16)`. The decoder is the inverse: latent `(B, 8, 256, 16)` → mel `(B, 1, 1024, 64)`.

Latent normalization at encoding time: `latent = encoder(mel).sample() * scaling_factor`. At decode: `mel = decoder(latent / scaling_factor)`. (We only need decode at inference — encoding is for audio-to-audio remixing, not in the v1 pipeline.)

The VAE kernels are identical to SD's `AutoencoderKL` (we have these); only the channel counts and block depths change.

### 9. Vocoder (`SpeechT5HifiGan`)

A non-standard HiFi-GAN variant from the SpeechT5 codebase, **5 upsample stages** (vs the canonical V1's 4), totaling 5×4×2×2×2 = **320**. This matches the mel parameters: 16,000 Hz output / 50 Hz mel-frame rate = 320 samples per frame, and `hop_length=160` × **2** = 320 — wait. The vocoder's effective hop is 320 (the product of upsample rates), but the mel preprocessor used during training has hop 160. The mel **frame rate going into the vocoder is half of the mel rate** because the VAE expects a mel spectrogram at `1024 frames / 10.24 s = 100 Hz` not 50 Hz.

Resolving: vocoder takes 512 mel frames and outputs `512 × 320 = 163,840` samples = 10.24 s @ 16 kHz. So the mel sequence given to the vocoder is **half** the length of the mel used for the VAE input. Concretely: the VAE decoder outputs a `(B, 1, 1024, 64)` mel; this is then **reshaped/permuted** to `(B, mel_frames=512, mel_bins=64)` for the vocoder by treating two consecutive time steps as belonging to one vocoder frame (interleaved). See `mel_spectrogram_to_waveform` in `pipeline_audioldm2.py`:

```python
mel_spectrogram = mel_spectrogram.squeeze(1)              # (B, 1024, 64)
waveform = self.vocoder(mel_spectrogram)                  # (B, 1024*hop) ?
```

Actually looking at the SpeechT5HifiGan implementation: it accepts `(B, T_mel, n_mels)` and outputs `(B, T_mel * prod(upsample_rates))` = `(B, T_mel * 320)`. So with `T_mel=1024`, the output is `1024 × 320 = 327,680` samples = **20.48 s @ 16 kHz**. That contradicts the documented 10.24 s default.

The reconciliation (per the paper and reference repo): AudioLDM 2's effective mel rate going into the vocoder is **100 Hz** (hop 160 @ 16 kHz). The vocoder is trained with `prod(upsample_rates) = 160`, *not* 320. Re-reading the config: `upsample_rates = [5, 4, 2, 2, 2]` → product = **160**. (5×4=20, ×2=40, ×2=80, ×2=160.) Wait: 5×4×2×2×2 = 160, not 320. Earlier I mis-multiplied. So:

- vocoder hop (effective) = `prod([5,4,2,2,2]) = 160` ✓ matches mel hop.
- `1024 mel frames × 160 = 163,840 samples = 10.24 s @ 16 kHz` ✓.

**Corrected vocoder spec:**

| Field | Value |
|---|---|
| `model_in_dim` (mel bins) | 64 |
| `sampling_rate` | 16,000 Hz |
| `upsample_rates` | [5, 4, 2, 2, 2] (product = 160 = mel hop) |
| `upsample_kernel_sizes` | [16, 16, 8, 4, 4] |
| `upsample_initial_channel` | 1024 |
| `resblock_kernel_sizes` | [3, 7, 11] |
| `resblock_dilation_sizes` | [[1,3,5], [1,3,5], [1,3,5]] |
| `leaky_relu_slope` | 0.1 |
| `normalize_before` | false |

The generator structure (pre-conv → 5 × {ConvTranspose1d + MRF} → tanh post-conv) is identical to the HiFi-GAN V1 family — see [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). The only differences from canonical V1: (a) 5 upsample stages instead of 4, (b) mel bins 64 instead of 80, (c) 16 kHz instead of 22.05 kHz, (d) `model_in_dim` for the input projection is 64, (e) initial channel 1024 instead of 512. The HiFi-GAN kernel implementation we already have can absorb these as config parameters.

**Per-feature mel normalization**: `SpeechT5HifiGan` supports an optional `normalize_before` flag using a `mean`/`scale` tensor; AudioLDM2 sets this to `false`, so no normalization is applied between VAE decoder output and vocoder input.

### 10. Scheduler (DDIM)

```json
{
  "_class_name": "DDIMScheduler",
  "num_train_timesteps": 1000,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.0015,
  "beta_end": 0.0195,
  "prediction_type": "epsilon",
  "clip_sample": false,
  "set_alpha_to_one": false,
  "steps_offset": 1,
  "timestep_spacing": "leading",
  "rescale_betas_zero_snr": false,
  "thresholding": false
}
```

Notes for parity:
- `beta_schedule="scaled_linear"`: `betas = linspace(sqrt(0.0015), sqrt(0.0195), 1000) ** 2`.
- `set_alpha_to_one=false`: the "final alpha" used for the last denoising step uses `alphas_cumprod[0]` rather than 1.0 — affects the very last DDIM step.
- `steps_offset=1`: shift the inference timestep schedule by +1 (matches original DDIM impl in SD).
- `prediction_type="epsilon"`: standard noise prediction (not v-pred, not x0). CFG applies to the noise prediction.
- DPM-Solver++ is a drop-in replacement at inference (same betas / prediction_type) and reaches comparable quality in ~30–50 steps. HartsyInference should ship both DDIM (default) and DPMSolver++ for users who want speed. Both schedulers are in [DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md).
- **No flow matching.** This is classic Gaussian diffusion.

### 11. Mel Spectrogram Parameters

Used to convert training audio → mel for VAE encoding, and as the implicit "format" the VAE decoder and vocoder agree on. The vocoder expects mel produced with these exact parameters.

| Param | Value |
|---|---|
| Sample rate | 16,000 Hz |
| `n_fft` (filter_length) | 1024 |
| `win_length` | 1024 |
| `hop_length` | 160 |
| Window | Hann (periodic) |
| `n_mels` | 64 |
| `f_min` | 0 |
| `f_max` | 8,000 (Nyquist) |
| Mel scale | Slaney |
| Mel norm | Slaney area |
| Power | 1 (magnitude, not power) |
| Log | natural log (`log(max(x, 1e-5))`) — log-magnitude, not log-power |

At 10.24 s: `163,840 samples / 160 hop = 1024 mel frames`. Mel shape: `(1024, 64)`.

We **don't need mel computation** at inference (text-to-audio path consumes only text), but we do need the reverse: the VAE produces a mel-shaped latent that we decode to mel, then vocode. The mel format must be log-magnitude (not log-power) at the above STFT parameters, otherwise the vocoder produces garbage. Cross-reference [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) for the canonical implementation; AudioLDM2 column needs to be added there.

### 12. End-to-End Inference Walkthrough

For `pipeline("a cat meowing", num_inference_steps=200, guidance_scale=3.5, audio_length_in_s=10.24)`:

1. **Tokenize twice.**
   - RoBERTa BPE → `clap_input_ids` (B, L_clap).
   - T5 sentencepiece → `t5_input_ids` (B, L_t5), padded to model max length, attention mask captured.
2. **CLAP encode.** `clap_feat = clap.get_text_features(...)` → `(B, 512)` → unsqueeze → `(B, 1, 512)`.
3. **T5 encode.** `t5_feat = t5_encoder(t5_input_ids, attn_mask).last_hidden_state` → `(B, L_t5, 1024)`.
4. **Negative prompt path.** Repeat steps 1–3 with the empty string (or user-supplied negative prompt) to produce `neg_clap_feat`, `neg_t5_feat`. Both are needed for CFG.
5. **Project.** For each (positive, negative):
   `proj_seq = projection_model(clap_feat, t5_feat, mask_clap, mask_t5)` → `(B, L_t5 + 5, 768)` plus attention mask.
6. **GPT-2 generate.** For each (positive, negative):
   `gpt_seq = generate_language_model(proj_seq, mask, max_new_tokens=8)` → `(B, 8, 768)`. **No sampling, deterministic, KV-cache enabled.**
7. **Concat for CFG.** Along batch dim:
   - `encoder_hidden_states = cat([neg_gpt, pos_gpt])` → `(2B, 8, 768)`.
   - `encoder_hidden_states_1 = cat([neg_t5, pos_t5])` → `(2B, L_t5, 1024)`.
   - `encoder_attention_mask_1 = cat([neg_t5_mask, pos_t5_mask])` → `(2B, L_t5)`.
8. **Initialize latents.** Shape `(B, 8, height=256, width=16)` where:
   - `height = audio_length_in_s × 16000 / 160 / vae_scale_factor = 10.24 × 100 / 4 = 256`.
   - `width  = vocoder.model_in_dim / vae_scale_factor = 64 / 4 = 16`.
   - `latents = randn(...) * scheduler.init_noise_sigma`.
9. **DDIM loop (200 steps).** For each timestep `t`:
   - `latent_input = cat([latents, latents])` → `(2B, 8, 256, 16)`.
   - `latent_input = scheduler.scale_model_input(latent_input, t)` (no-op for DDIM with epsilon).
   - `noise_pred = unet(latent_input, t, encoder_hidden_states, encoder_hidden_states_1, encoder_attention_mask_1)` → `(2B, 8, 256, 16)`.
   - Split: `neg_pred, pos_pred = noise_pred.chunk(2)`.
   - CFG: `pred = neg_pred + guidance_scale * (pos_pred - neg_pred)`.
   - `latents = scheduler.step(pred, t, latents, eta=0.0).prev_sample`.
10. **VAE decode.** `mel = vae.decode(latents / 0.4110932946205139).sample` → `(B, 1, 1024, 64)`.
11. **Reshape for vocoder.** `mel = mel.squeeze(1)` → `(B, 1024, 64)` (T, n_mels).
12. **Vocode.** `wav = vocoder(mel)` → `(B, 1024 × 160) = (B, 163840)` float32, range ≈ [-1, 1].
13. **(Optional) CLAP rerank.** If `num_waveforms_per_prompt > 1`: extract CLAP audio features from each candidate, score against `clap.get_text_features(prompt)`, keep top-k.
14. **Return.** `(B, 163840)` float32 mono waveform at 16 kHz.

### 13. Conditioning Inputs Summary

| Input | Supported? | Notes |
|---|---|---|
| Text prompt | Yes, required | English; the model is only trained on English captions. |
| Negative prompt | Yes | Defaults to empty string. CFG enabled when `guidance_scale > 1`. |
| Audio duration | Yes, configurable | `audio_length_in_s`; will be rounded so the latent height is a multiple of `vae_scale_factor=4`. Effective resolution = 10.24 ms (one mel frame at 100 Hz, hop 160 @ 16 kHz). Practical range 1–30 s; quality degrades past ~15 s because the UNet was mostly trained at 10.24 s. |
| Melody / audio conditioning | **No** | Diffusers `AudioLDM2Pipeline` does not expose audio-to-audio or melody conditioning. The VAE can encode audio, but no `audio2audio` pipeline ships. (Skip for v1.) |
| Seed | Yes | Standard `torch.Generator`; HartsyInference uses our `cuRand`-equivalent kernel for reproducibility. |
| `num_waveforms_per_prompt` | Yes | If >1, generate multiple candidates and rerank via CLAP audio↔text similarity (requires the CLAP audio tower — defer this to a later milestone). |

### 14. Memory & Performance

Approximate numbers; we'll measure for real on RTX 3090/4090 once the C# pipeline runs.

| Variant | FP16 VRAM (all components resident) | RTF @ 200 DDIM steps on RTX 3090 |
|---|---|---|
| base (350 M UNet) | ~3.5 GB | ~0.3 (3.3 s wall for 10.24 s audio) |
| large (750 M UNet) | ~5.5 GB | ~0.5 (5.0 s wall for 10.24 s audio) |
| music (350 M UNet) | ~3.5 GB | ~0.3 |

Source: HuggingFace blog and community benchmarks. Diffusers' Python reference is ~3× slower than the AudioLDM2 reference repo originally claimed, but the diffusers numbers above already reflect that optimization.

The bottleneck is the UNet × 200 steps × 2 (CFG). Speedups:
- DPM-Solver++ at 50 steps recovers near-identical quality, → ~4× wall-clock.
- FP16 throughout; the VAE has `force_upcast=true` and will need an FP32 path for stability.
- CLAP, T5, projection, GPT-2 each run once per generation → negligible vs UNet × 400.

T5-Large alone is ~750 M params (~1.4 GB FP16). It can be eagerly unloaded after step 5 — only `encoder_hidden_states_1` and `encoder_attention_mask_1` are needed thereafter. Same for CLAP after step 2 (only its 512-dim output is consumed). The GPT-2 is tiny (124 M).

### 15. Music vs Speech Variants

Same architecture, different training data and (for some variants) tokenizer:
- **General (`audioldm2`, `audioldm2-large`)**: trained on 1,150 k hours of mixed sound effects, music, and some speech from AudioSet and other sources. Best all-rounder; mediocre at intelligible speech.
- **Music (`audioldm2-music`)**: trained on 665 k hours of music only. Higher fidelity instrumental output, but cannot generate sound effects or speech.
- **Speech (`audioldm2-speech-*`)**: trained on the named speech corpus. **Uses a VITS phoneme encoder** in place of T5 — the diffusers `AudioLDM2Pipeline` handles this by branching on `text_encoder_2.config.model_type`. Quality is below dedicated TTS models (Kokoro, F5-TTS) and the model has no speaker control. **Recommendation: do NOT prioritize the speech variant** — HartsyInference already has Kokoro / F5-TTS for TTS.

Inference parameters are identical across variants.

### 16. HartsyInference Implementation Notes

This pipeline is roughly 70% reuse from existing HartsyInference components and 30% new.

**Reuse:**
- T5EncoderModel — already implemented (F-Lite / SD3 / Flux pipelines).
- AutoencoderKL — kernels reused; new config (1 channel in/out, 8 latent channels, 3 down-blocks instead of 4).
- UNet 2D — reuse SD1.5 UNet kernels (ResBlock2D, Transformer2D, CrossAttn, GroupNorm). Modifications: (a) accept 8 in/out channels, (b) different `block_out_channels`, (c) route two `encoder_hidden_states` streams per the `cross_attention_dim[block][sublayer]` table, (d) the input "image" is shaped `(time, mel_freq)`; no math changes.
- HiFiGAN generator — reuse `HifiGanGenerator` from the vocoder layer; new config with 5 upsample stages, mel bins 64, sample rate 16 kHz. See [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md).
- DDIM scheduler — already implemented. Verify `set_alpha_to_one=false` + `steps_offset=1` parity.
- DPM-Solver++ scheduler — already implemented (Z-Image / SD3); plug-and-play.

**New:**
- **CLAP text encoder** (RoBERTa-base + projection MLP). Tokenizer: RoBERTa BPE (need to add to `HartsyInference.ModelAssets.Tokenizers`). Model: ~12-layer post-LN transformer; 768 → 512 projection head. **No audio tower needed for v1.**
- **GPT-2 small** (~124 M). Need to implement:
  - Learned token + positional embeddings (`wte` + `wpe`).
  - 12 × decoder-only transformer blocks (pre-LN, GELU-new, causal self-attn).
  - Final LayerNorm.
  - **No `lm_head` evaluation needed** — we read the last layer's hidden state directly.
  - **Continuous-prefix mode**: feed `inputs_embeds` (skip `wte`). The pipeline never tokenizes anything for GPT-2.
  - **KV-cache**: required for performance. Prefix encoded once on iteration 0, then 8 single-step extensions.
  - **No sampling**: deterministic; no temperature / top-p / top-k logic needed.
  - This is the first decoder-only Transformer in HartsyInference. Architecture is similar to dotLLM's Llama (RoPE → swap for learned positional, RMSNorm → LayerNorm, SwiGLU → GELU MLP). Treat it as the foundation kernel for any future small autoregressive transformer; consider a `HartsyInference.Transformer.Gpt2` package shared with future work.
- **AudioLDM2ProjectionModel** (~80 K params). Two linears + four learned vectors + the SOS/EOS insertion logic. Trivial — implement inline in the AudioLDM2 pipeline class.
- **Dual-stream cross-attention routing** inside the UNet transformer block. Per-sublayer table maps to one of two `encoder_hidden_states` tensors. Add a `int[][] CrossAttnStreamIndex` config to the UNet block and switch the K/V tensor accordingly. Implement once in `Transformer2DBlock`; existing single-stream pipelines pass a length-1 table.

**Validation targets** (vs HuggingFace diffusers Python at FP32 on CPU; tolerance per [BENCHMARKING.md](BENCHMARKING.md)):
- CLAP text features: max |Δ| < 1e-4 on a 32-prompt set.
- T5 encoder: max |Δ| < 1e-4 (existing target).
- Projection model output (post SOS/EOS insertion): exact (no nonlinearity beyond linears).
- GPT-2 forward (single step): max |Δ| < 1e-4 per hidden state.
- GPT-2 8-step continuous generation: max |Δ| < 5e-4 (accumulated).
- UNet single forward at fixed timestep / latent / conditioning: max |Δ| < 1e-3 (FP32) on each output channel.
- VAE decode: max |Δ| < 1e-3 on the mel.
- Vocoder: max |Δ| < 1e-3 on waveform sample values (with the same input mel).
- End-to-end: not bit-exact (DDIM noise sampling differs); compare on **CLAP audio score** of the generated waveform vs the prompt — should match Python within ±0.01 CLAP score over a 32-prompt benchmark.

**Package placement** (one folder per package under `src/`, GPU behind `IBackend`):
- `HartsyInference.Audio.AudioLDM2` — pipeline class, projection model, UNet 2D dual-stream override.
- `HartsyInference.TextEncoders.Clap` — new package (text-only for v1; can grow to add audio tower later).
- `HartsyInference.TextEncoders.T5` — existing.
- `HartsyInference.Transformer.Gpt2` — new (consider naming to allow future reuse; keep tiny).
- `HartsyInference.Vocoder.HifiGan` — existing; add the 5-stage 16 kHz config preset.
- `HartsyInference.Vae` — existing; add the 1-channel mel-VAE config preset.

**Build order** (suggested for [Checklists/](../Checklists/)):
1. CLAP text encoder + RoBERTa tokenizer (CPU + CUDA).
2. GPT-2 small (CPU + CUDA, with KV-cache, continuous `inputs_embeds` mode).
3. AudioLDM2ProjectionModel.
4. UNet 2D dual-stream cross-attention extension.
5. Mel-VAE config preset and validation.
6. SpeechT5HifiGan 5-stage preset and validation.
7. Pipeline glue + DDIM/DPMSolver++ wiring.
8. End-to-end validation against HF diffusers reference outputs.

### 17. Known Gotchas

- `langauge_model_dim` (sic) — typo in the upstream projection model config field name. Preserve it for safetensors compatibility; expose with the corrected spelling at the C# API layer.
- VAE `scaling_factor` is `0.4110932946205139` — **not** SD's 0.18215. Bake into the AudioLDM2 VAE config preset.
- VAE has only 3 down-blocks → `vae_scale_factor = 4` (not 8). All downstream shape arithmetic depends on this.
- The vocoder is shape-sensitive: input mel must be `(B, T, 64)` (T = `audio_length_in_s × 100`), in log-magnitude (natural log), with **no per-feature normalization** (`normalize_before=false`).
- GPT-2 `max_new_tokens=8` is **baked into UNet training**. Don't expose it as a user parameter.
- Per the diffusers source, the UNet supports at most 4 cross-attention sublayers per block. HartsyInference's `Transformer2DBlock` should enforce this and fail loudly otherwise.
- License is cc-by-nc-sa-4.0 (non-commercial). The HartsyInference loader should expose the license string at load time so end-user apps can warn / gate accordingly. Music-only and speech-only variants share this license.
- The reference repo's separate 48 kHz variant uses a different VAE (16 latent channels, different mel bins) and is **not** API-compatible — defer to a separate config preset.

## Open Questions

- Confirm whether `audioldm2-large` benefits from FP16 or needs FP32 in the UNet middle block. (Empirical; flag for the QA phase.)
- Decide whether to implement the CLAP **audio** tower for rerank in v1 or defer to v2. (Recommend defer.)
- Decide whether to implement the VITS-encoder speech-variant path. (Recommend skip — Kokoro / F5-TTS supersede.)
- Decide whether to expose audio-to-audio (encode mel → noise → re-denoise) since the VAE supports it. (Recommend yes; small addition once the t2a path works.)
