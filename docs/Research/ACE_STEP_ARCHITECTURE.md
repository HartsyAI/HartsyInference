# ACE-Step — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (ACE-Step pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## 1. Summary

**ACE-Step** is an open-source music generation foundation model from ACE Studio + StepFun. As of mid-2026 two distinct generations have been released:

- **ACE-Step v1 (3.5B, May 2025)** — the original Diffusion Transformer that put the project on the map. Pure flow-matching DiT operating on a 2-D mel-latent produced by a custom "Music-DCAE" autoencoder, conditioned by UMT5-base text embeddings, a Conformer lyric encoder, and a 512-d global speaker/timbre vector. Trained with auxiliary REPA cosine-alignment losses to MERT and m-HuBERT embeddings. Apache 2.0. Generates 4 min of stereo 48 kHz music in ~20 s on an A100. This is the variant the public ecosystem (ComfyUI, Replicate, Diffusers) standardized on.
- **ACE-Step v1.5 (Jan 2026 → April 2026 XL)** — a major architectural rewrite. Replaces the diffusers DiT with a **Qwen3-style decoder LM** (alternating sliding-window / full attention, RMSNorm, SwiGLU, GQA) whose **output is FSQ-quantized audio tokens** rather than continuous latents. Three size tiers: **2B base/sft/turbo** (hidden 2048, 24 layers, 16 heads), **5B XL base/sft/turbo** (hidden 2560, 32 layers, 32 heads), each tier paired with a separate **Music DCAE / 1D Stable-Audio-format VAE** for waveform reconstruction. Adds optional **Qwen3-based "planner" LMs** (0.6B / 1.7B / 4B) that act as omni-capable lyric/style writers. MIT license. <4 GB VRAM for the 2B turbo, ~9 GB for the XL.

Both generations share: UMT5 text encoder, 32k-context RoPE (θ=1e6), in/out 8-channel latents (v1) or 192-channel pre-quant features (v1.5), structured-lyric tags (`[verse]`, `[chorus]`, `[bridge]`, `[instrumental]`, …), 50+ language support, and a flow-matching scheduler with shift=3.0 (v1) or learned `(μ=-0.4, σ=1.0)` timestep distribution (v1.5).

For HartsyInference, the **v1 3.5B model is the implementation target first**: its component boundaries are clean (UMT5 text encoder + Conformer lyric encoder + DiT + DCAE + HiFiGAN vocoder), it reuses UMT5 (already in HartsyInference for AuraFlow) and a flow-match Euler scheduler (already in HartsyInference for Flux/SD3). v1.5 should follow once Qwen3-style decoder LM + FSQ vocoder are scoped — those are HartsyInference.LLM territory plus a new audio codec component.

This file covers ACE-Step architecture, weights, conditioning, and inference. The shared flow-matching mathematics live in [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md). The HiFiGAN vocoder family used by both generations is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). UMT5 details live with the AuraFlow text-encoder notes ([TEXT_ENCODERS.md](TEXT_ENCODERS.md)).

---

## 2. Key Numbers / Constants

### v1 DiT (`ACEStepTransformer2DModel`)

```
num_layers            24
num_attention_heads   20
attention_head_dim    128
inner_dim             2560                   # = 20 * 128
mlp_ratio             2.5                    # FFN hidden = 2560 * 2.5 = 6400
in_channels           8
out_channels          8
patch_size            [16, 1]                # height-collapse only
max_position          32768
rope_theta            1_000_000.0
text_embedding_dim    768                    # UMT5 hidden
speaker_embedding_dim 512
lyric_hidden_size     1024                   # Conformer hidden
lyric_encoder_vocab_size 6693 (config) / 6681 (code)
ssl_names             ["mert", "m-hubert"]
ssl_encoder_depths    [8, 8]                 # DiT layer to tap for SSL alignment
ssl_latent_dims       [1024, 768]            # MERT 1024-d, m-HuBERT 768-d
```

### v1 Music DCAE

```
in_channels                  2              # stereo (left, right)
latent_channels              8
encoder_block_out_channels   [128, 256, 512, 1024]
encoder_block_types          [ResBlock×3, EfficientViTBlock]
encoder_layers_per_block     [2, 2, 3, 3]
decoder_layers_per_block     [3, 3, 3, 3]
attention_head_dim           32             # in EfficientViT blocks
encoder_qkv_multiscales      [[],[],[5],[5]]
scaling_factor               0.41407        # diffusers-style
pipeline scale_factor        0.1786         # actually applied around DCAE in ACE-Step
pipeline shift_factor        -1.9091
mel_chunk_size               1024 frames
latent_chunk_size            128 frames     # = 1024 / 8
time_downsample_factor       8
mel range clip               [-11.0, +3.0]
mel post-clip standardise    mean=0.5, std=0.5
```

### v1 Vocoder (ADaMoSHiFiGANV1)

```
sampling_rate         44100
n_mels                128
num_mels              512                    # internal? see Open Questions
n_fft                 2048
win_length            2048
hop_length            512                    # → 44100/512 ≈ 86.13 mel Hz
f_min                 40 Hz
f_max                 16000 Hz
depths                [3, 3, 9, 3]           # ConvNeXt-style stages
dims                  [128, 256, 384, 512]
upsample_initial_channel 1024
upsample_rates        [4, 4, 2, 2, 2, 2, 2]  # product = 512 (matches hop_length)
upsample_kernel_sizes [8, 8, 4, 4, 4, 4, 4]  # 2× upsample_rates
resblock_kernel_sizes [3, 7, 11, 13]
resblock_dilations    [[1,3,5],[1,3,5],[1,3,5],[1,3,5]]
pre_conv_kernel_size  13
post_conv_kernel_size 13
input_channels        128
```

### Sampler (v1, default scheduler)

```
num_train_timesteps   1000
shift                 3.0
default infer_steps   27 (fast) / 60 (quality)
default guidance      7.0
APG momentum          -0.75
APG norm threshold    2.5
APG eta               0.0
```

### Latent / time math

```
F_lat = int(duration_seconds × 44100 / 512 / 8)
      = duration_seconds × ~10.77 tokens/s
      → 4 min (240 s) ≈ 2585 latent frames
mel_frames        = duration_seconds × 44100 / 512  ≈ 86.13 / s
samples           = duration_seconds × 44100
latent tensor     (B, 8, 16, F_lat)                    # H=16 = 128 mel / 8
mel tensor        (B, 2, 128, mel_frames)              # 2 = stereo
wav tensor        (B, 2, samples)                      # stereo 44.1 kHz
```

### v1.5 base (2B)

```
hidden_size               2048
intermediate_size         6144         # FFN
num_hidden_layers         24
num_attention_heads       16
num_key_value_heads       8            # GQA 2:1
head_dim                  128
sliding_window            128
layer_types               alternating sliding / full
vocab_size                64003
in_channels (features)    192
fsq_dim                   2048
fsq_input_levels          [8,8,8,5,5,5]   # 64000 codes
pool_window_size          5             # ≈ 5 Hz token rate
timestep_mu / sigma       (-0.4, 1.0)   # logit-normal timestep dist
text_hidden_dim           1024
timbre_hidden_dim         64
timbre_fix_frame          750
patch_size                2
rope_theta                1_000_000
max_position_embeddings   32768
dtype                     bfloat16
```

### v1.5 XL (5B)

Same as 2B except:

```
hidden_size               2560
intermediate_size         9728
num_hidden_layers         32
num_attention_heads       32
num_key_value_heads       8            # GQA 4:1
encoder_hidden_size       2048         # asymmetric: 2B-class encoder
encoder_intermediate_size 6144
encoder_num_attention_heads     16
encoder_num_key_value_heads     8
num_audio_decoder_hidden_layers 24
num_lyric_encoder_hidden_layers 8
num_timbre_encoder_hidden_layers 4
```

---

## 3. Data Layouts / Formats

### 4.1 Safetensors keys (v1 3.5B DiT)

Indicative prefix tree (extracted from class definitions, not from a key dump):

```
proj_in.weight, proj_in.bias                  # patch embed conv stack
caption_projection.linear.weight,bias         # 768 → 2560 (text)
lyric_embs.weight                             # (6693, 1024)
lyric_encoder.*                               # Conformer 8 layers
lyric_proj.linear.weight,bias                 # 1024 → 2560
speaker_embedder.linear.weight,bias           # 512 → 2560
time_embed.timestep_embedder.linear_1.weight,bias
time_embed.timestep_embedder.linear_2.weight,bias
transformer_blocks.{0..23}.scale_shift_table  # (6, 2560)
transformer_blocks.{0..23}.norm1, norm2       # RMSNorm (no params if elementwise_affine=False)
transformer_blocks.{0..23}.attn1.{q,k,v,out}_proj.weight,bias    # self-attn
transformer_blocks.{0..23}.attn2.{q,k,v,out}_proj.weight,bias    # cross-attn
transformer_blocks.{0..23}.ff.glumb.*         # GLUMBConv FFN
projectors.{0,1}.*                            # SSL alignment heads (trainable only)
proj_out.norm_final.scale_shift_table
proj_out.linear.weight,bias                   # 2560 → 16*1*8 = 128
```

### 4.2 Tensor shapes through the v1 forward pass

```
text_ids                 (B, T_text)
text_embeds = UMT5(text_ids)             (B, T_text, 768)
text_attn_mask                           (B, T_text)

lyric_ids                                (B, T_lyric)         # VoiceBpe
lyric_mask                               (B, T_lyric)
lyric_hidden = Conformer(lyric_ids)      (B, T_lyric, 1024)

speaker_vec                              (B, 512)             # optional, else None

t (timestep)                             (B,)                 # scalar per batch
temb = TimestepEmbed(t)                  (B, 256) → MLP → (B, 2560)

# noise / latent
z_t                                      (B, 8, 16, F_lat)

# patch embed
h = PatchEmbed(z_t)                      (B, F_lat, 2560)

# build cross-attn K/V
spk_tok = speaker_embedder(speaker_vec)  (B, 1, 2560)         # zero-padded if speaker_vec is None
txt_tok = caption_projection(text_embeds)(B, T_text, 2560)
lyr_tok = lyric_proj(lyric_hidden)       (B, T_lyric, 2560)
ctx = concat([spk_tok, txt_tok, lyr_tok], dim=1)              # (B, 1+T_text+T_lyric, 2560)
ctx_mask = concat([ones(1), text_mask, lyric_mask], dim=1)

# 24 blocks
for blk in transformer_blocks:
    h = blk(h, ctx, ctx_mask, temb)      # self-attn + cross-attn + GLUMB

# unpatch
v_pred = ProjOut(h, temb)                (B, 8, 16, F_lat)

# Euler step
sigma, sigma_next = scheduler.sigmas[i], scheduler.sigmas[i+1]
z_{t-1} = z_t + (sigma_next - sigma) * v_pred       # straight-line interpolation
```

### 4.3 Vocoder data flow

```
latent z_0                  (B, 8, 16, F_lat)
mel_pred = DCAE.decode(z_0) (B, 2, 128, mel_frames)
mel_pred = mel_pred * 14 - 11   # de-normalise back to log-mel range [-11, +3]
for ch in 0,1:
    wav_ch = vocoder(mel_pred[:, ch])    (B, mel_frames * 512)
wav = stack([wav_0, wav_1], dim=1)        (B, 2, T_samples)
```

### 4.4 Lyric token stream layout

```
[ 261 | line1_tok... | 2 | line2_tok... | 2 | ... | 2 ]
       ^ start tag              ^ EOL    ^ blank line → bare 2
```

With language-id-prefix from `SUPPORT_LANGUAGES` injected by the BPE tokenizer per line. Structure tags like `[verse]` are *byte-encoded* into the same stream — no dedicated special-token IDs.

---

## 4. Reference Implementations

### Primary

- **`ace-step/ACE-Step`** (v1) — https://github.com/ace-step/ACE-Step
  - `acestep/models/ace_step_transformer.py` — DiT class, patch embed, final layer
  - `acestep/models/attention.py` — `LinearTransformerBlock`, `Attention`, `GLUMBConv`
  - `acestep/models/customer_attention_processor.py` — attention processors (RoPE application)
  - `acestep/models/lyrics_utils/lyric_tokenizer.py` — `VoiceBpeTokenizer`
  - `acestep/models/lyrics_utils/conformer.py` — Conformer encoder
  - `acestep/music_dcae/music_dcae_pipeline.py` — DCAE encode/decode + mel transform
  - `acestep/music_dcae/music_vocoder.py` — `ADaMoSHiFiGANV1`
  - `acestep/pipeline_ace_step.py` — top-level orchestration, `tokenize_lyrics`, `calc_v`, schedulers
  - `acestep/apg_guidance.py` — APG + CFG-Zero★ + MomentumBuffer
  - `acestep/schedulers/scheduling_flow_match_*.py` — Euler / Heun / PingPong

- **`ace-step/ACE-Step-1.5`** — https://github.com/ace-step/ACE-Step-1.5 — v1.5 inference and training code; ships HF `auto_map` with `modeling_acestep_v15_*` files inside each checkpoint repo.

### HuggingFace

- `ACE-Step/ACE-Step-v1-3.5B` — primary v1 weights
- `ACE-Step/Ace-Step1.5` — turbo 2B
- `ACE-Step/acestep-v15-base`, `-sft`, `-turbo`, `-turbo-continuous`
- `ACE-Step/acestep-v15-xl-{base,sft,turbo,turbo-diffusers}` — 5B XL family
- `ACE-Step/acestep-5Hz-lm-{0.6B,1.7B,4B}` — Qwen3-based planner LMs
- `ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format` — alternative Stable-Audio-format codec
- `Serveurperso/ACE-Step-1.5-GGUF` — community GGUF quantizations

### Paper

- **arXiv 2506.00045** — *"ACE-Step: A Step Towards Music Generation Foundation Model"* (June 2025). v1 architecture, training recipe, REPA loss formulation.
- **arXiv 2602.00744** — *"ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation"* (Feb 2026). v1.5 architecture, FSQ codec, planner LM, 50-language extension.

### Third-party / ecosystem

- **ComfyUI** native ACE-Step nodes (since May 2025) — https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1
- **acestep.cpp** + **acestep.vst3** — C++17 / GGML port + VST3 plugin for v1.5 (CPU/CUDA/Metal/Vulkan)
- **Replicate** `lucataco/ace-step` — hosted inference
- **WaveSpeedAI** Ace-Step 1.5 — hosted, ~70 % speedup over reference
- **DeepWiki** — code-explanation index at https://deepwiki.com/ace-step/ACE-Step (useful but not authoritative)

### Underlying components

- **DCAE** (Sana) — NVIDIA's Deep Compression Autoencoder; HF `mit-han-lab/dc-ae-f32c32-sana-1.0` etc.
- **UMT5** — `google/umt5-base` (already used by AuraFlow in HartsyInference)
- **MERT** — `m-a-p/MERT-v1-330M` (training-only target)
- **m-HuBERT** — `utter-project/mHuBERT-147` (training-only target)
- **VoiceBpeTokenizer** — vendored from XTTS-v2 (`coqui-ai/TTS`)
- **HiFi-GAN** family — base reference https://github.com/jik876/hifi-gan, see [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md)
- **APG** — Sadat et al. 2024 "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models" (arXiv 2410.02416)

---

## 5. Differences Between Implementations

### Inner-dim discrepancy (incorrect doc circulating)

Some third-party docs (including auto-generated wikis) describe the DiT as `num_heads=24, head_dim=64, inner_dim=1536`. The **actual published `config.json`** is `num_heads=20, head_dim=128, inner_dim=2560`. Implementers must use the config-file numbers.

### Lyric vocab size 6681 vs 6693

The Embedding in `ACEStepTransformer2DModel.__init__` uses `vocab_size=6681` (literal in code), but `config.json` declares `lyric_encoder_vocab_size=6693`. The most likely explanation is that the on-disk safetensors weight has shape `(6693, 1024)` (allowing for extra special tokens) and the constructor is overridden by the from-pretrained loader. **Use 6693 when allocating the embedding to match safetensors.**

### `scaling_factor` 0.41407 vs `scale_factor` 0.1786

`AutoencoderDC.config.scaling_factor = 0.41407` is the diffusers-style intrinsic scaling. The ACE-Step pipeline ignores it and applies `scale_factor=0.1786`, `shift_factor=-1.9091` itself in `music_dcae_pipeline.py`. **Implementers must apply the pipeline-level numbers, not the config-level one.**

### Vocoder `num_mels=512` mystery

The vocoder config has both `n_mels=128` (real number of mel bins, matches the DCAE input) and `num_mels=512` (unused at inference; possibly an artifact of the upstream `nemo`/`fish-speech` config schema). **Use 128 as the spectrogram bin count.**

### Three schedulers, no consensus

Different community workflows default to different schedulers. The official pipeline uses Euler by default; ComfyUI exposes all three; PingPong is preferred for "variations" mode because it injects fresh noise per step.

### v1 vs v1.5 — fundamentally different inference

v1 is a continuous-latent flow-matching DiT (diffusers-style). v1.5 is a Qwen3-style decoder over FSQ tokens. They share UMT5, structured lyrics, and the general "music generation" UX, but the inference code is not interchangeable. HartsyInference will need two distinct pipelines.

### Stereo handling

The DCAE config says `in_channels=2`, suggesting joint stereo at the latent level. But the vocoder is mono — at decode time the pipeline runs the vocoder **twice** (once per channel), then stacks. So stereo is preserved through the DCAE but not through the vocoder.

---

## 6. Implementation Notes for HartsyInference

### What we already have

- **UMT5-base encoder** — implemented for AuraFlow. ACE-Step uses the same `google/umt5-base` weights, same tokenizer (SentencePiece, 256k vocab), same per-layer relative-attention-bias path. **Reuse directly.** The only ACE-Step-specific addition is the `SUPPORT_LANGUAGES` ID list, which only the lyric tokenizer cares about — UMT5 is fed plain text.
- **Flow-match Euler scheduler** — implemented for Flux/SD3. Add a `shift=3.0` config knob if not already there, plus PingPong (trivial extension: inject noise per step) and Heun (already present for some samplers).
- **AdaLN-6 modulation, RoPE, RMSNorm, GroupNorm** — all in the kernel library already from Flux/SD3/Lumina work. ACE-Step's `Qwen2RotaryEmbedding(dim=128, base=1e6, max_pos=32768)` is the same RoPE we use for Lumina-Next.
- **GGUF backend** — for loading the community Q4/Q5/Q8 quantizations of the DiT and UMT5.
- **HiFi-GAN vocoder skeleton** — partially built for Kokoro (iSTFTNet, which is a HiFi-GAN cousin). The trunk needs extension to the ConvNeXt-style `depths=[3,3,9,3]` stages used by ADaMoSHiFiGANV1, but ResBlocks + transposed-conv upsamplers + multi-receptive-field fusion are all already implemented.

### What we need to build

1. **GLUMBConv FFN** — gated MBConv-style FFN with depthwise-separable conv + SiLU gate. New op for HartsyInference. Sana / EfficientViT use the same block, so it's worth landing as a reusable `GLUMBConv` layer rather than ACE-Step-specific code.
2. **EfficientViTBlock** — multi-scale linear-attention block used in the deepest DCAE stages. Standard for Sana-family. Implementable on top of our existing scaled-dot-product attention with a lightweight multi-scale projection wrapper.
3. **Conformer encoder** — 8-layer Conformer block (macaron FFN + multi-head self-attn + depthwise conv + macaron FFN). Doesn't exist yet (Whisper uses a plain transformer). Reusable elsewhere (FunASR, SenseVoice, future ASR work).
4. **VoiceBpeTokenizer** — port the XTTS-v2 BPE merge table + multilingual support. Should live in a shared text-tokenization module; can be reused by SongGen, XTTS, or any other vocoder-conditioning model that adopts it.
5. **ADaMoSHiFiGANV1 vocoder** — new vocoder variant; biggest new piece in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Stereo-via-two-passes pattern is a 6-line wrapper around the mono vocoder.
6. **Music-DCAE** — Sana-style 2-D AutoencoderDC over mel spectrograms. New code but shares ResBlock + EfficientViTBlock with the DCAE in Sana-1.0/2.0 (future image work). Land as a generic `AutoencoderDC` module configurable via the diffusers-format `config.json`.
7. **STFT + mel filter bank** — for the vocoder's `mel_transform` step (and the inverse if we ever need to mel-encode reference audio). cuFFT-via-PTX path; matches Whisper / Kokoro mel preprocessing. We have STFT already for Whisper.
8. **APG + CFG-Zero★ guidance helpers** — small kernels: parallel/orthogonal decomposition, momentum buffer, norm thresholding. Useful beyond ACE-Step (Flux, SD3 community workflows are adopting APG).
9. **FlowMatchPingPongScheduler** — Euler with injected noise; trivial extension of the existing flow-match scheduler.
10. **Lyric / structure-tag preprocessor** — utility that runs G2P for CJK lines, language detection per line, line splitting, and the `[261]…[2]…[2]` token-list construction. Pure C# string work; no GPU.

### Suggested implementation order

1. **Scheduler + APG kernels** — small, hot-path-relevant, validates against Python reference quickly with no model weights needed.
2. **GLUMBConv + EfficientViTBlock** — kernel-level building blocks. Validate against AutoencoderDC reference outputs.
3. **Music-DCAE encode/decode** — wire up `AutoencoderDC` + mel transform + scaling. Validate against a known waveform → latent → waveform roundtrip (tolerance: PSNR > 30 dB).
4. **ADaMoSHiFiGANV1 vocoder** — slot into the existing HiFi-GAN code path. Validate per-channel decode against Python reference (numerical match to 1e-3 in waveform L1).
5. **Conformer + VoiceBpe + lyric preprocessor** — text-side pipeline. Validate token-ID stream byte-equal to Python output.
6. **ACEStepTransformer2DModel forward** — the DiT itself. Reuse RMSNorm/RoPE/AdaLN-6/cross-attention from existing DiT code. Validate one forward pass against Python reference at FP32 (tolerance: relative error < 1e-3 for hidden states, < 1e-2 for final velocity).
7. **End-to-end pipeline** — chain it all together. Validate against a published seed/prompt → known waveform (tolerance: PESQ > 3.5 vs reference).
8. **Quantization** — wire up GGUF Q4/Q8 loading paths through the existing GGUF backend. Verify perceptual quality.
9. **Edit / repaint / cover modes** — flow-edit mask-aware loop on top of the basic generation path.
10. **v1.5 2B turbo** — second pipeline. Largely orthogonal; touches HartsyInference.LLM for the decoder and HartsyInference.Audio for the FSQ → mel → wav path.

### Validation strategy

For v1 3.5B, the gold reference is the official PyTorch pipeline with `seed=0`, `prompt="lo-fi hip hop, mellow piano"`, `lyrics=""` (instrumental), `duration=30`, `infer_steps=27`, `guidance=7.0`, `scheduler="euler"`, `guidance_mode="cfg"` (avoid APG for first validation — APG has momentum state that drifts numerically). Verify:

- UMT5 last_hidden_state matches our existing AuraFlow UMT5 implementation byte-for-byte (already a regression test).
- DCAE latent after encoding the reference output matches Python reference (relative error < 1e-4).
- DiT velocity at step 0 matches (relative error < 5e-3 in BF16).
- Final waveform matches Python output in PESQ (>4.0) and short-term spectral L1 (<0.02).

### Memory layout note

The DiT's hidden activation at peak is roughly `(1, F_lat, 2560)` for the audio stream plus `(1, 1 + T_text + T_lyric, 2560)` for the cross-attention context. For a 4-minute song, `F_lat ≈ 2585`, so the audio token sequence dominates. Single-stream allocation in BF16 = `2585 × 2560 × 2 = 13 MB` per layer-residual. Activation checkpointing across 24 layers is unnecessary even on 12 GB cards. Standard `NativeMemory.AlignedAlloc` arenas (one per layer-pair) suffice.

### Package routing

- `HartsyInference.Audio.AceStep` — new sub-namespace for the v1 pipeline (DiT, DCAE, ADaMoS vocoder, lyric preprocessor, edit modes).
- `HartsyInference.Diffusion.Schedulers` — extend with `FlowMatchPingPongScheduler` (already has Euler / Heun for v1).
- `HartsyInference.TextEncoders.UMT5` — reuse (already present for AuraFlow).
- `HartsyInference.Audio.Vocoders` — extend with ADaMoSHiFiGANV1; shared with future Stable-Audio-Open / Suno-style vocoders.
- `HartsyInference.Audio.Codecs.DCAE` — new module; shared with future Sana image work.
- `HartsyInference.Text.VoiceBpe` — new tokenizer module; shared with XTTS / SongGen if/when added.

v1.5 should land in a separate `HartsyInference.Audio.AceStepV15` sub-namespace once HartsyInference.LLM Qwen3 decoder work catches up, because the inference loop is causal-LM-shaped rather than diffusion-shaped.
