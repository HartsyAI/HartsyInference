# Spark-TTS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Spark-TTS pipeline)

## Summary

Spark-TTS ([arXiv:2503.01710](https://arxiv.org/abs/2503.01710), [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS), Mar 2025, CC-BY-NC-SA 4.0) is an efficient LLM-based zero-shot voice-cloning TTS. It pairs a fine-tuned **Qwen2.5-0.5B causal LM** (hidden_size=896, 24 layers, GQA 14:2, vocab=166000 — extended from Qwen2.5's 151,936 with ~14k Spark-specific control + audio tokens) with **BiCodec**, a custom *single-stream* speech codec that decomposes 16 kHz audio into two complementary token sets:

- **Semantic tokens** — a time-varying stream at **50 Hz**, single VQ codebook of **8192 entries (8-D factorized)**, encoding linguistic content from frozen wav2vec2-XLSR-53 features (layers 11, 14, 16 averaged).
- **Global tokens** — a fixed-length set of **32 tokens** per utterance encoding time-invariant speaker timbre, produced by an ECAPA-TDNN over mel-spectrograms → PerceiverResampler → FSQ (6 dims × 4 levels = 4096 codebook).

Both token streams are predicted *autoregressively in a single sequence* by the Qwen LM. The same LM checkpoint handles (a) zero-shot voice cloning from a reference clip and (b) controllable generation from coarse attribute prompts (gender/pitch_label/speed_label) or fine-grained numeric controls (pitch_value 0-1000, speed_value 0-10), unified by a chain-of-thought style prompt format. Cross-lingual cloning (zh ↔ en, including code-switching mid-sentence) works because the LM is bilingual and the global tokens decouple speaker identity from language. Reconstruction is done by a **DAC-style HiFi-GAN-like wave generator** (Snake1d activations, transposed convs at rates [8, 5, 4, 2] for hop=320 → 16 kHz output) fed by Vocos/ConvNeXt backbones that act as prenet/postnet conditioned on the global speaker vector.

Training data is **VoxBox** (102.5k hours, 4.7M utterances across 29 corpora; 47.6k h zh + 54.9k h en) annotated with gender/age/emotion/pitch/speed. The full model ships at ~3.95 GB (LLM 2.03 GB BF16 + BiCodec 626 MB + wav2vec2 1.27 GB FP32). Performance on an L20 GPU via Triton/TensorRT-LLM: RTF 0.14 @ concurrency-1, dropping to RTF 0.07 @ concurrency-4. UTMOS=4.35 (beats CosyVoice2 and even ground truth at 4.08).

This document covers the model + pipeline. The BiCodec codec module is also cross-referenced from [AUDIO_CODECS.md](AUDIO_CODECS.md) (needs a new section). Mel preprocessing in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). HiFi-GAN-style wave generator background in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). The LM reuses the same Qwen2.5-0.5B kernels as dotLLM — see [DOTLLM_ARCHITECTURE.md](DOTLLM_ARCHITECTURE.md).

Sources: [arXiv:2503.01710 (paper)](https://arxiv.org/abs/2503.01710), [HTML version](https://arxiv.org/html/2503.01710v1), [SparkAudio/Spark-TTS-0.5B HF](https://huggingface.co/SparkAudio/Spark-TTS-0.5B), [SparkAudio/Spark-TTS GitHub](https://github.com/SparkAudio/Spark-TTS), [VoxBox](https://github.com/SparkAudio/VoxBox), [demo page](https://sparkaudio.github.io/spark-tts/).

> **Verified-implementation corrections (2026-06-27, bit-exact against real weights).** A few details below
> differ from the actual checkpoint and were corrected during the C# port:
> - **Token IDs** (the placeholders in §2.2/§2.5 were wrong). From the real `added_tokens.json`, **global tokens
>   precede semantic**: `<|bicodec_global_0|>`=151665 (..155760), `<|bicodec_semantic_0|>`=155761 (..163952);
>   start/end_global=165150/165156, start/end_semantic=165151/165157, end_content=165152, end_style_label=165153.
>   Generation stops on `<|end_semantic_token|>`(165157) or `<|im_end|>`(151645).
> - **BiCodec decode path** (`detokenize`): `quantizer.detokenize` → `speaker_encoder.detokenize` → `prenet(z_q, d_vector)`
>   → `x + d_vector[...,None]` → `decoder`. **PostNet is NOT used on the decode path** (training only).
> - **Semantic VQ is factorized**: the codebook is 8-D (`[8192, 8]`); decode = embedding lookup then `out_project`
>   WNConv1d(8→1024). The d-vector flattens the 32 global codes **channel-major** (transpose to `[128, 32]` first),
>   NOT mean-pooling.
> - **Wave generator** uses descript's default `weight_norm` (dim 0, `weight_g` `[C_in,1,1]`) and explicit
>   `kernel_sizes=[16,11,8,4]` with `padding=(k-stride)/2`, no output_padding (the rate-5 stage's kernel is 11, not 10).
> - The LLM `model.safetensors` ships **F32** on disk (not bf16, despite `torch_dtype`).

## Detailed Findings

### 1. Variants

| Variant | LM Params | Total Pkg Size | Languages | HF Path | License |
|---|---|---|---|---|---|
| **Spark-TTS-0.5B** (only official release) | 507M (Qwen2.5-0.5B family, modified vocab) | 3.95 GB | zh + en (zero-shot cross-lingual, code-switching) | [`SparkAudio/Spark-TTS-0.5B`](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | CC-BY-NC-SA 4.0 (non-commercial) |
| Community quantizations (4 listed on HF) | varies (GGUF Q4/Q5/Q8) | ~300-500 MB LM | same | various (e.g. mradermacher) | inherit CC-BY-NC-SA |
| Community fine-tunes (25 listed on HF) | 0.5B | 3.95 GB | varies (some add ja/ko/multilingual) | community | — |

There is **no official larger or smaller variant** as of the May-2026 cutoff. SparkAudio explicitly markets the 0.5B size as the "single, edge-deployable" model. Mirror at [HKUSTAudio/Spark-TTS-0.5B](https://huggingface.co/HKUSTAudio/Spark-TTS-0.5B) is byte-identical.

### 2. Architecture

#### 2.1 Overall Block Diagram

```
                                  Reference Audio (16 kHz mono)
                                           │
                       ┌───────────────────┴───────────────────┐
                       ▼                                       ▼
        wav2vec2-XLSR-53 (frozen)                Mel-spec (128 mels, n_fft=1024,
        layers {11,14,16} avg → (T, 1024)         win=640, hop=320, fmin=10)
                       │                                       │
                       ▼                                       ▼
        ConvNeXt encoder (12 blocks)              ECAPA-TDNN (c=512, emb=192)
        + 2 down blocks → VQ                       + PerceiverResampler (32 queries,
        codebook 8192 × 8-D                          latent=128) + Residual FSQ
                       │                              levels=[4,4,4,4,4,4]
                       ▼                                       │
              SEMANTIC TOKENS                                  ▼
                  50 Hz                                GLOBAL TOKENS
              (one stream)                            32 tokens (fixed)
                       │                                       │
                       └───────────────┬───────────────────────┘
                                       ▼
                       Qwen2.5-0.5B Causal LM (hidden=896, 24 L)
                       Predicts: <|global|>·32 then <|semantic|>·N
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
               sem_tokens (50 Hz)              global_tokens (32)
                       │                               │
                       ▼                               ▼
              VQ → 1024-D                  FSQ→128-D → linear → 1024-D
                       │                          (d-vector)
                       └──────────────┬────────────────┘
                                      ▼
                   PreNet (Vocos backbone, 12 ConvNeXt blocks,
                          AdaLayerNorm conditioned on d-vector)
                                      │
                                      ▼
                   Wave generator (DAC-style HiFi-GAN, Snake1d,
                          transposed convs rates [8,5,4,2],
                          kernels [16,11,8,4], hop = 320)
                                      │
                                      ▼
                          16 kHz mono PCM waveform
```

#### 2.2 Text Tokenizer

- Shared **Qwen2.5 BPE tokenizer** (151,643 base merges + 14,357 Spark-specific tokens → total **vocab = 166,000**).
- Files (in `LLM/`): `vocab.json` (2.78 MB, ~151k base tokens), `merges.txt` (1.67 MB), `tokenizer.json` (14.1 MB merged HF Fast-Tokenizer), `added_tokens.json` (509 KB — these are the new special tokens), `tokenizer_config.json` (2.58 MB), `special_tokens_map.json` (613 B).
- BOS = 151643, EOS = 151645 (standard Qwen2.5 IDs).
- All "added" tokens are pure ASCII textual markers like `<|task_tts|>`, `<|bicodec_global_0|>`, `<|gender_0|>` — they are real string tokens that the tokenizer rewrites to their IDs. **The LM never sees raw audio**; everything is a token.

**Added-token inventory** (from `sparktts/utils/token_parser.py`):

| Family | Range | Count | Purpose |
|---|---|---|---|
| Task | `<\|task_{vc,tts,asr,s2s,t2s,understand,cap,controllable_tts,prompt_tts,edit}\|>` | 10 | Selects mode |
| Structure | `<\|start_content\|>`, `<\|end_content\|>`, `<\|start_global_token\|>`, `<\|end_global_token\|>`, `<\|start_semantic_token\|>`, `<\|end_semantic_token\|>`, `<\|start_style_label\|>`, `<\|end_style_label\|>` | 8 | Section delimiters |
| BiCodec semantic | `<\|bicodec_semantic_0\|>` … `<\|bicodec_semantic_8191\|>` | 8192 | One per VQ codebook entry |
| BiCodec global | `<\|bicodec_global_0\|>` … `<\|bicodec_global_4095\|>` | 4096 | One per FSQ code |
| Gender | `<\|gender_0\|>`, `<\|gender_1\|>` | 2 | female / male |
| Age | `<\|age_0..4\|>` | 5 | Child, Teenager, Youth-Adult, Middle-aged, Elderly |
| Pitch label | `<\|pitch_label_0..4\|>` | 5 | very_low … very_high |
| Pitch value | `<\|pitch_value_0..1000\|>` | 1001 | Quantized Hz bucket |
| Pitch var | `<\|pitch_var_value_0..10\|>`, `<\|pitch_var_label_0..4\|>` | 16 | F0 variance |
| Loudness | `<\|loudness_value_0..30\|>`, `<\|loudness_label_0..4\|>` | 36 | RMS dB |
| Speed label | `<\|speed_label_0..4\|>` | 5 | very_low … very_high |
| Speed value | `<\|speed_value_0..10\|>` | 11 | Phonemes/sec bucket |
| Emotion | `<\|emotion_0..24\|>` | 25 | 25 categories incl. WHISPER, SINGING, LAUGHING |

Total added ≈ 14,322; reserved padding fills to vocab 166,000.

#### 2.3 LM — Qwen2.5-0.5B (modified)

Exact `LLM/config.json`:

| Field | Value |
|---|---|
| architectures | `Qwen2ForCausalLM` |
| model_type | `qwen2` |
| hidden_size | 896 |
| num_hidden_layers | 24 |
| num_attention_heads | 14 |
| num_key_value_heads | 2 (GQA 7:1) |
| intermediate_size | 4864 (SwiGLU FFN) |
| vocab_size | 166000 |
| tie_word_embeddings | true (lm_head shares with token_embed) |
| max_position_embeddings | 32768 |
| sliding_window | 32768 |
| use_sliding_window | false |
| max_window_layers | 21 |
| rope_theta | 1,000,000.0 |
| rms_norm_eps | 1e-6 |
| hidden_act | silu |
| attention_dropout | 0.0 |
| torch_dtype | bfloat16 |
| transformers version (saved with) | 4.43.1 |

This is **structurally identical to Qwen2.5-0.5B** (same layer counts, hidden, GQA, RoPE θ=1M, tied embeddings, RMSNorm, SwiGLU FFN). The only delta is `vocab_size` 151,936 → 166,000, which means `embed_tokens.weight` and `lm_head.weight` (tied) are reshaped to 166,000 × 896 ≈ 149M params (out of 507M total — vocab dominates this small model). FFN/attention weights are identical in shape to Qwen2.5-0.5B and can be loaded with the same shape-handling code.

LM weight file: `LLM/model.safetensors` (2.03 GB BF16). At FP16/BF16 inference this is ~1 GB activations + ~2 GB weights memory.

#### 2.4 BiCodec — Codec Architecture

BiCodec total weight file: `BiCodec/model.safetensors` (626 MB) holding **all of** semantic encoder + decoder + speaker encoder + prenet + postnet + wave generator + both quantizers. Config in `BiCodec/config.yaml` is reproduced verbatim below for unambiguous re-implementation:

```yaml
# Mel front-end (used for speaker encoder + decoder reconstruction loss at train time)
mel_params:
  sample_rate: 16000
  n_fft: 1024
  win_length: 640
  hop_length: 320      # → 16000/320 = 50 Hz mel & token frame rate
  mel_fmin: 10
  mel_fmax: null
  num_mels: 128

# Semantic encoder (consumes wav2vec2 features, NOT mels)
encoder:
  input_channels: 1024            # wav2vec2-XLSR-53 hidden dim
  vocos_dim: 384                  # ConvNeXt hidden
  vocos_intermediate_dim: 2048    # ConvNeXt MLP
  vocos_num_layers: 12            # 12 ConvNeXt blocks
  out_channels: 1024              # before VQ projection
  sample_ratios: [1, 1]           # 2 down-blocks @ stride-1 (no extra downsampling — wav2vec2 already at 50 Hz)

# Semantic quantizer — factorized VQ
quantizer:
  input_dim: 1024
  codebook_size: 8192
  codebook_dim: 8                 # factorized: project 1024 → 8 → VQ → 8 → 1024
  commitment: 0.25
  codebook_loss_weight: 2.0
  use_l2_normlize: true
  threshold_ema_dead_code: 0.2

# Speaker encoder (consumes 128-mel spectrogram)
speaker_encoder:
  input_dim: 128                  # mel bins
  out_dim: 1024                   # d-vector & x-vector output dim
  latent_dim: 128
  token_num: 32                   # 32 global tokens per utterance
  fsq_levels: [4, 4, 4, 4, 4, 4]  # 4^6 = 4096 codes per token
  fsq_num_quantizers: 1

# Prenet — conditions decoder on speaker embedding via AdaLN
prenet:
  input_channels: 1024            # de-quantized semantic codes
  vocos_dim: 384
  vocos_intermediate_dim: 2048
  vocos_num_layers: 12
  out_channels: 1024
  condition_dim: 1024             # d-vector dim
  sample_ratios: [1, 1]
  use_tanh_at_final: false

# Postnet — refines features pre-vocoder
postnet:
  input_channels: 1024
  vocos_dim: 384
  vocos_intermediate_dim: 2048
  vocos_num_layers: 6
  out_channels: 1024
  use_tanh_at_final: false

# Wave generator (DAC HiFi-GAN style, NOT shown in config — hard-coded in code)
decoder:
  input_channel: 1024
  channels: 1536
  rates: [8, 5, 4, 2]             # product = 320 = hop_length ✓
  kernel_sizes: [16, 11, 8, 4]    # 2× corresponding rate
```

##### 2.4.1 Semantic Encoder — `feat_encoder.py`

Input is **wav2vec2-XLSR-53 averaged features**, NOT raw audio:

- Source model: `facebook/wav2vec2-large-xlsr-53` (300M, frozen, 1.27 GB FP32 weights bundled at `wav2vec2-large-xlsr-53/pytorch_model.bin`).
- Input audio is normalized + resampled to 16 kHz.
- All 24 transformer layers of wav2vec2 run; **only `hidden_states[11] + hidden_states[14] + hidden_states[16]` are averaged** (mean over the three) to produce a (T, 1024) sequence at 50 Hz.

The encoder is then a **Vocos-style ConvNeXt backbone**:
- `nn.Conv1d(1024, 384, kernel=7, padding=3)` — input projection.
- 12 × `ConvNeXtBlock(dim=384, intermediate=2048)`:
  - Depthwise `Conv1d(384, 384, kernel=7, padding=3, groups=384)`
  - LayerNorm (over channel dim, transposed before/after)
  - Pointwise `Linear(384, 2048)` → GELU → `Linear(2048, 384)`
  - Layer-scale γ + residual
- 2 × `SamplingBlock(ratio=1)` (no actual downsampling — just structural).
- Final `Linear(384, 1024)` → output (T, 1024) for VQ.

##### 2.4.2 Semantic Quantizer — `FactorizedVectorQuantize`

- Project 1024 → 8 (factorized down-projection, `nn.Linear`).
- L2-normalize, lookup nearest of 8192 codes in 8-D space.
- Look-up index → 8-D code → 1024-D via inverse projection.
- Output: one integer per frame (range 0-8191).
- Single codebook (NOT residual): rate = 50 Hz × log2(8192) = **650 bps** of linguistic info.
- EMA codebook updates (dead-code reset threshold 0.2) — irrelevant for inference; codebook is just a learned `(8192, 8)` matrix.

##### 2.4.3 Speaker Encoder / Global Tokenizer — `speaker_encoder.py`

- **Input**: log-mel spectrogram (128 mels, computed with the same n_fft=1024, win=640, hop=320, fmin=10, fmax=null params).
- **ECAPA-TDNN backbone**: `ECAPA_TDNN_GLOB_c512` (channels=512, embedding=192). Produces both:
  - **x-vector** — global pooled (B, 1024). Used directly as the d-vector at inference; the FSQ branch produces the discrete tokens.
  - Frame-level feature map (B, T, 512) fed to the PerceiverResampler.
- **PerceiverResampler**: 32 learnable query vectors, cross-attention to ECAPA frame features, latent_dim=128. Output: (B, 32, 128) — one latent per global token.
- **Residual FSQ**: levels=[4,4,4,4,4,4] → 6-D code per token, each dim quantized to 4 levels (codebook size 4^6 = 4096), num_quantizers=1 (no residual stack). Each global token is a single integer 0-4095.
- **d-vector path** (decoder conditioning): the 32 quantized 6-D codes → flatten (32 × 6 = 192) → `nn.Linear(192, 1024)` → (B, 1024) d-vector.
- Outputs from `forward()`: `x-vector (B, 1024)`, `d-vector (B, 1024)`, `global_indices (B, 32)`.

##### 2.4.4 Decoder Stack — PreNet → PostNet → WaveGenerator

Input is the de-quantized semantic codes (B, T, 1024) at 50 Hz, conditioning is the d-vector (B, 1024):

1. **PreNet** — `VocosBackbone(dim=384, layers=12, intermediate=2048)` with **AdaLayerNorm** instead of plain LayerNorm. AdaLN takes the d-vector through a small `Linear(1024, 2×384)` to produce scale+shift modulating each ConvNeXt block. Two `SamplingBlock(ratio=1)` (structural only). Output (B, T, 1024).

2. **PostNet** — `VocosBackbone(dim=384, layers=6, intermediate=2048)`, no AdaLN conditioning, refines pre-vocoder features. Output (B, T, 1024).

3. **WaveGenerator** — DAC-style HiFi-GAN (adapted from `descript-audio-codec`):
   - Initial `WNConv1d(1024, 1536, kernel=7)`.
   - 4 × `DecoderBlock`, each with `rates[i]` ∈ {8, 5, 4, 2} and `kernel_sizes[i]` ∈ {16, 11, 8, 4}:
     - `Snake1d(channels)` activation (β-parameterized: `x + (1/β)·sin²(βx)`).
     - `WNConvTranspose1d(in, in//2, kernel, stride=rate, padding=(kernel - rate)//2)` — halves channels each block: 1536 → 768 → 384 → 192 → 96.
     - 3 × `ResidualUnit(dim=out_ch, dilation=∈{1, 3, 9})` — each unit is Snake1d → WNConv1d(k=7, dilation) → Snake1d → WNConv1d(k=1).
   - Final `Snake1d(96)` → `WNConv1d(96, 1, kernel=7)` → `torch.tanh` → (B, 1, T_audio).

Product of upsampling rates: 8 × 5 × 4 × 2 = 320 = `hop_length`. So T_audio = T_semantic × 320 → at 50 Hz tokens, 1 s of audio is 50 tokens generating 16,000 samples. ✓

##### 2.4.5 Summary of Token Streams Predicted by LM

| Stream | Vocab | Rate | Length for 5 s clip |
|---|---|---|---|
| Global | 4096 (FSQ) | fixed 32 per utt | 32 |
| Semantic | 8192 (VQ) | 50 Hz | 250 |

Total tokens for 5 s of audio: ~282 (+ ~40 control/text tokens). The LM autoregresses through all of them in one continuous sequence. This is why the model is so fast: only one stream of tokens per timestep.

#### 2.5 Speaker Encoder (recap)

Already covered in 2.4.3 — ECAPA-TDNN-GLOB-c512 + PerceiverResampler(32 queries) + Residual FSQ. Bundled inside the 626 MB BiCodec checkpoint, not a separate file. Outputs both a continuous 1024-D d-vector (used to condition the decoder) and 32 discrete global tokens (predicted by the LM).

### 3. Inference Pipeline

#### 3.1 Zero-Shot Voice Cloning Mode (most common usage)

Reference: `cli/SparkTTS.py`, `sparktts/models/audio_tokenizer.py`.

**Step A — Encode reference audio (one-shot, ~30 ms on GPU):**

1. Load reference WAV, resample to 16 kHz mono, normalize.
2. Extract a fixed-length **reference clip** for the speaker encoder: `ref_segment_length = int(sr × ref_segment_duration) // hop × hop` (config-defined, typically 6 s clip = 96,000 samples). If audio is shorter, pad-and-tile.
3. Compute the mel-spectrogram (128 mels, n_fft=1024, win=640, hop=320, fmin=10, fmax=null) of the reference clip.
4. Run **wav2vec2-XLSR-53** on the *full* audio (not just the ref clip) — internally upsamples/downsamples to 16 kHz, then takes hidden_states[11], [14], [16] and averages → (T, 1024).
5. Run BiCodec encoder over the wav2vec2 features → semantic codes (T integers 0-8191).
6. Run BiCodec speaker_encoder over the mel of the ref clip → global codes (32 integers 0-4095) + d-vector (1024-D).

**Step B — Build LM prompt (string concatenation; tokenizer handles all):**

```
<|task_tts|>
<|start_content|>{reference_transcript}{target_text}<|end_content|>
<|start_global_token|><|bicodec_global_{g0}|>...<|bicodec_global_{g31}|><|end_global_token|>
<|start_semantic_token|>
```

The LM is then asked to continue from `<|start_semantic_token|>`. Note that for zero-shot cloning, the prompt does **not** include any reference *semantic* tokens — only the global tokens (which encode timbre). The transcript of the reference is concatenated with the target text in the content section so the LM "knows what was said" but it generates only the semantic tokens for the *full* concatenated content.

Some inference recipes also prefix the reference *semantic* tokens after `<|start_semantic_token|>` and let the LM continue — this gives even tighter prosodic mimicry. Both modes are supported.

**Step C — Autoregressive LM decode:**

- `model.generate(prompt_ids, max_new_tokens=3000, temperature=0.8, top_k=50, top_p=0.95)`.
- KV-cache enabled (Qwen2.5 has full kv-cache support; reuse dotLLM's kv-cache infrastructure).
- Stop on `<|end_semantic_token|>` or `<|im_end|>`.
- Output is parsed with regex `<\|bicodec_semantic_(\d+)\|>` to extract integer codes.

**Step D — Decode tokens → waveform:**

1. Convert generated integer list → tensor of semantic indices (B, T).
2. VQ codebook lookup → (B, T, 8) → up-project → (B, T, 1024).
3. Combine with the global codes (which give the d-vector via FSQ decode + Linear(192,1024)).
4. Run PreNet (AdaLN-conditioned ConvNeXt × 12) → PostNet (ConvNeXt × 6) → WaveGenerator.
5. Output: (B, 1, T × 320) → 16 kHz mono float32 PCM in [-1, 1].

#### 3.2 Controllable Generation Mode (no reference audio)

```
<|task_controllable_tts|>
<|start_content|>{target_text}<|end_content|>
<|start_style_label|><|gender_{0|1}|><|pitch_label_{0..4}|><|speed_label_{0..4}|><|end_style_label|>
<|start_semantic_token|>
```

In this mode the LM **predicts the 32 global tokens itself** (after the semantic stream — actually in some variants global tokens are predicted first; check `token_parser.py` for exact slot order). It then generates semantic tokens.

For fine-grained control, replace `<|pitch_label_X|>` with `<|pitch_value_X|>` (X ∈ 0..1000, a Hz quantization) and similarly for speed_value/loudness_value/pitch_var_value. The CoT chain ordering observed in the test code is: `task → age → gender → pitch_value → pitch_label → loudness_value → loudness_label → emotion`.

#### 3.3 Pseudocode for HartsyInference

```csharp
// Pseudocode for HartsyInference.Audio.SparkTtsPipeline
public async Task<float[]> SynthesizeAsync(string text, string referenceWavPath, string referenceTranscript)
{
    // A. Reference encoding
    var refAudio = AudioLoader.Load(referenceWavPath, targetSr: 16_000);
    var refClip  = AudioOps.PadOrTrim(refAudio, length: 96_000);
    var refMel   = MelSpectrogram.Compute(refClip, nFft: 1024, winLength: 640, hopLength: 320, nMels: 128, fmin: 10);
    var w2vFeats = wav2vec2.ExtractAvgHidden(refAudio, layers: [11, 14, 16]);     // (T, 1024)
    var semCodes = biCodec.EncodeSemantic(w2vFeats);                              // int[T]
    var (globCodes, dVector) = biCodec.EncodeSpeaker(refMel);                     // int[32], float[1024]

    // B. Prompt build
    var prompt = $"<|task_tts|><|start_content|>{referenceTranscript}{text}<|end_content|>"
               + "<|start_global_token|>" + string.Join("", globCodes.Select(g => $"<|bicodec_global_{g}|>"))
               + "<|end_global_token|><|start_semantic_token|>";
    var promptIds = qwenTokenizer.Encode(prompt);

    // C. LM generate
    var genIds = await qwenLm.GenerateAsync(promptIds, maxNew: 3000, temperature: 0.8f,
                                             stopTokens: [endSemTokenId, imEndTokenId]);
    var genSem = ParseSemanticTokens(qwenTokenizer.Decode(genIds));               // int[T_out]

    // D. Decode
    var wave = biCodec.Decode(genSem, globCodes);                                 // float[T_out * 320]
    return wave;
}
```

### 4. Cross-Lingual Cloning

Spark-TTS handles cross-lingual cloning naturally because:

1. **The LM is bilingual.** Qwen2.5 is pretrained zh+en, and the Spark fine-tune retains this. The model can read English target text after Chinese reference transcript (or vice versa) and produce sensible phonetic predictions.
2. **Global tokens decouple timbre from language.** The 32 global tokens describe *who is speaking* (vocal tract / timbre / register), not *what language they speak*. They are extracted from a 6-second clip and are constant across the synthesis — so a Chinese reference speaker can fluently "speak" English in the generated semantic stream.
3. **No phonemizer.** Spark does NOT use G2P. Text → BPE → LM directly. This means no language-specific phoneme inventory mismatch.
4. **Code-switching** works mid-sentence because the LM tokenizer accepts mixed scripts (CJK + Latin) and outputs semantic tokens for each segment seamlessly.

**Limitations**: Performance for non-zh/non-en is unsupported on the official checkpoint (community fine-tunes add other languages). Accent transfer is incomplete — a Chinese speaker reading English will sometimes carry over Mandarin prosody/accent because both timbre and some prosodic priors are entangled in the LM, not just the global tokens.

### 5. Style / Instruction Control

Two layers of control, both expressed as ASCII control tokens prepended to the prompt:

**Coarse (5-point labels):**

| Attribute | Token | Values (id → name) |
|---|---|---|
| Gender | `<\|gender_X\|>` | 0=female, 1=male |
| Age | `<\|age_X\|>` | 0=Child, 1=Teenager, 2=Youth-Adult, 3=Middle-aged, 4=Elderly |
| Pitch | `<\|pitch_label_X\|>` | 0=very_low, 1=low, 2=moderate, 3=high, 4=very_high |
| Pitch variance | `<\|pitch_var_label_X\|>` | same 5-point scale |
| Loudness | `<\|loudness_label_X\|>` | same |
| Speed | `<\|speed_label_X\|>` | same |
| Emotion | `<\|emotion_X\|>` | 0..24 — UNKNOWN, NEUTRAL, ANGRY, HAPPY, SAD, FEARFUL, DISGUSTED, SURPRISED, SARCASTIC, EXCITED, SLEEPY, CONFUSED, EMPHASIS, LAUGHING, SINGING, WORRIED, WHISPER, ANXIOUS, NO-AGREEMENT, APOLOGETIC, CONCERNED, ENUNCIATED, ASSERTIVE, ENCOURAGING, CONTEMPT |

**Fine-grained (numeric buckets):**

| Attribute | Token | Range | Mapping |
|---|---|---|---|
| Pitch | `<\|pitch_value_X\|>` | 0..1000 | F0 in Hz, log-spaced bins |
| Pitch variance | `<\|pitch_var_value_X\|>` | 0..10 | Std-dev of F0 |
| Loudness | `<\|loudness_value_X\|>` | 0..30 | RMS dB |
| Speed | `<\|speed_value_X\|>` | 0..10 | Phonemes/sec |

**Prompt CoT order** (from `token_parser.py` test code): `task → age → gender → pitch_value → pitch_label → loudness_value → loudness_label → emotion`. Not all slots have to be filled — the most common subset is `(task_controllable_tts, gender, pitch_label, speed_label)`.

**Tasks** (from `TASK_TOKEN_MAP`): `vc` (voice conversion), `tts` (zero-shot cloning), `asr` (ASR), `s2s` (speech-to-speech), `t2s` (text-to-speech, alt), `understand`, `cap` (audio caption), `controllable_tts`, `prompt_tts` (natural-language style prompt), `edit` (speech edit). The official checkpoint trained primarily on `tts`, `vc`, `controllable_tts`; the others are placeholders.

### 6. Memory and Performance

| Metric | Value (16-bit) | Notes |
|---|---|---|
| **LM weights** | 1.01 GB BF16 (on-disk 2.03 GB because saved as FP32-padded BF16 safetensors with overhead) | Tied embeddings save 149M of duplication. |
| **BiCodec weights** | 626 MB FP32 → ~313 MB FP16 | All of encoder + decoder + speaker + wavgen. |
| **wav2vec2-XLSR-53** | 1.27 GB FP32 → ~635 MB FP16 | Frozen, used only at ref-encode time. Can be unloaded after step A. |
| **Total VRAM steady-state** | ~2 GB (LM only, ref encoders dropped) | Comfortable on 4 GB consumer GPU. |
| **Total VRAM peak** | ~3 GB | When ref encoder is in memory. |
| **KV cache** | (24 layers × 2 KV-heads × 64 head_dim × 2 × seq_len × 2 bytes BF16) = 12 KB/token | A 5 s generation (~280 tokens) needs ~3.3 MB. Negligible. |
| **RTF @ L20 GPU (Triton+TRT-LLM, official)** | concurrency-1: 0.1362 (876 ms latency) / concurrency-4: 0.0704 | RTF = inference_time / audio_duration. Sub-RT means faster than real time. |
| **First-audio latency** | ~870 ms for first 5 s clip | LM has to fully generate before BiCodec decodes. No streaming in official pipeline. |
| **Streaming variant** | Not officially shipped | The single-stream design means LM output can be chunked-decoded by BiCodec every N tokens (≥32 to amortize global), but this is community work, not official. |

**Edge deployability**: At 0.5B LM + 313 MB codec, the model runs on a single 6 GB GPU at FP16 with RTF < 1.0. CPU-only inference is feasible at RTF ~3-5 (per community reports).

### 7. HuggingFace Files

Repo: [`SparkAudio/Spark-TTS-0.5B`](https://huggingface.co/SparkAudio/Spark-TTS-0.5B), total **3.95 GB** across 12 commits.

| Path | Size | Purpose |
|---|---|---|
| `.gitattributes` | 1.77 kB | LFS pointers for large files |
| `README.md` | 6.47 kB | Model card |
| `config.yaml` | 169 B | Top-level pipeline config (paths to LLM/, BiCodec/, wav2vec2/) |
| **`LLM/`** | **2.05 GB** | Qwen2.5-0.5B fine-tune |
| `LLM/model.safetensors` | 2.03 GB | 507M params, BF16 weights |
| `LLM/config.json` | 658 B | Qwen2 config (see §2.3) |
| `LLM/tokenizer.json` | 14.1 MB | HF fast-tokenizer (BPE merges + 14k added tokens) |
| `LLM/vocab.json` | 2.78 MB | Base Qwen2.5 vocab |
| `LLM/merges.txt` | 1.67 MB | BPE merges |
| `LLM/tokenizer_config.json` | 2.58 MB | Lists all added_tokens for re-instantiation |
| `LLM/added_tokens.json` | 509 kB | Maps `<\|task_tts\|>` etc. → token IDs 151643+ |
| `LLM/special_tokens_map.json` | 613 B | BOS/EOS/PAD aliases |
| `LLM/generation_config.json` | (small) | Default sampling params |
| **`BiCodec/`** | **626 MB** | Custom dual-token codec |
| `BiCodec/model.safetensors` | 626 MB | All codec weights (encoder+decoder+speaker+vocoder+quantizers) |
| `BiCodec/config.yaml` | 1.16 kB | Full codec config (see §2.4 — reproduced verbatim) |
| **`wav2vec2-large-xlsr-53/`** | **1.27 GB** | Frozen feature extractor (3rd party: facebook/wav2vec2-large-xlsr-53 mirrored verbatim) |
| `wav2vec2-large-xlsr-53/pytorch_model.bin` | 1.27 GB | 300M params, FP32 |
| `wav2vec2-large-xlsr-53/config.json` | 1.77 kB | wav2vec2 config |
| `wav2vec2-large-xlsr-53/preprocessor_config.json` | 212 B | sample rate + normalization config |
| `wav2vec2-large-xlsr-53/README.md` | 2.29 kB | upstream model card |
| `src/` | (small) | Logo/banner images only — safe to skip in C# port |

### 8. C# Implementation Notes

#### 8.1 Reuse from dotLLM / existing HartsyInference

- **Qwen2.5-0.5B LM = standard Llama-style decoder transformer.** GQA (14:2), SwiGLU FFN, RMSNorm, RoPE θ=1M, tied embeddings. **Reuse dotLLM's `Qwen2ForCausalLM` implementation as-is** — only the embedding/lm_head dimensions change (vocab 151,936 → 166,000). Make sure the safetensors loader accepts the wider embedding matrix.
  - dotLLM patterns for KV-cache, sampling (temp/top-k/top-p), RoPE precomputation, and the GGUF-quantized variants all apply.
  - Sampling stop conditions: stop on `<|end_semantic_token|>` or `<|im_end|>`.
- **Qwen tokenizer**: HartsyInference already has BPE / Qwen tokenizer infra (see [TOKENIZERS.md](TOKENIZERS.md)). The `added_tokens.json` adds ~14k textual tokens — these need to be registered as atomic units that bypass BPE merging.
- **Mel-spectrogram**: HartsyInference already has a mel-spec kernel ([MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md)). Use n_fft=1024, win=640, hop=320, n_mels=128, fmin=10, fmax=sr/2. **Note the unusual win_length=640 (40 ms) with hop=320 (20 ms) — 50% overlap, not the more common 25%.** Use Hann window.
- **HiFi-GAN wave generator** = same DAC-style decoder family covered in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) and [AUDIO_CODECS.md](AUDIO_CODECS.md) §DAC. **The exact code is "Adapted from descript-audio-codec under Apache 2.0"** — the same WNConv1d / WNConvTranspose1d / Snake1d / ResidualUnit primitives we need for DAC are exactly reused here. Implement once, use for both. Snake activation: `x + (1/β)·sin²(βx)` with per-channel learnable β.

#### 8.2 New components needed (Spark-specific, add to `HartsyInference.Audio.SparkTts`)

| Component | Purpose | Notes |
|---|---|---|
| `Wav2Vec2Xlsr53Encoder` | Extract avg(layers 11,14,16) | Frozen, inference-only. Standard Wav2Vec2 transformer (24 layers, 1024 hidden, 16 heads, conv feature extractor with 7 conv layers downsampling to 50 Hz). Loaded from upstream `pytorch_model.bin` (FP32) — convert to safetensors at packaging time. Single forward, no KV cache needed. Can be unloaded after ref encoding. **This is the largest helper module (1.27 GB FP32 / 635 MB FP16).** |
| `VocosBackbone` | 12 (or 6) ConvNeXt blocks at dim=384 | Reusable for encoder + prenet + postnet. AdaLayerNorm variant needed for prenet (conditioning on d-vector). |
| `EcapaTdnnGlobC512` | Speaker backbone | Channels=512, embedding=192. Standard ECAPA-TDNN (Res2Conv1d + SE blocks + attentive stat pooling). Frozen at inference. |
| `PerceiverResampler` | Cross-attn from 32 queries to ECAPA frames | Standard Perceiver — learnable queries, 1 cross-attn + 1 FFN block typical. |
| `FactorizedVectorQuantize` | Semantic quantizer | 1024 → 8 → VQ(8192,8) → 8 → 1024. L2-normalized codebook. **Inference: just argmin in 8-D + embedding lookup.** No EMA updates needed. |
| `ResidualFsq` | Global quantizer | levels=[4,4,4,4,4,4], 1 layer. FSQ is *parameter-free* at inference: just `round(tanh(x) * levels/2) + levels/2`. Single embedding from 6-D code → no learned codebook for FSQ itself; the codebook is just the discretized output of FSQ. |
| `WaveGeneratorDac` | Final vocoder | See [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Rates [8,5,4,2], kernels [16,11,8,4], dilations {1,3,9}, snake activations, weight-norm. |
| `SparkTtsTokenParser` | Build prompts, parse generated semantic tokens | Pure string/regex; tiny. Match `<\|bicodec_semantic_(\d+)\|>` and `<\|bicodec_global_(\d+)\|>`. |
| `SparkTtsPipeline` | Orchestrator | Holds Qwen LM + Wav2Vec2 + BiCodec; exposes `Synthesize(text, refWav, refTranscript)` and `SynthesizeControllable(text, gender, pitch, speed, ...)`. |

#### 8.3 Validation tolerances against Python reference

Per [CODE_STYLE.md](../CODE_STYLE.md) §validate-against-references, each component must match:

- **wav2vec2 features**: pairwise cosine to PyTorch output ≥ 0.9995 per layer at FP16.
- **VQ semantic indices**: 100% identical to PyTorch (deterministic argmin in 8-D — any divergence = numerical bug in the projection).
- **FSQ global indices**: 100% identical to PyTorch.
- **d-vector**: cosine ≥ 0.999 to PyTorch reference.
- **LM logits at temperature=0**: top-1 token match ≥ 99% over 10k tokens.
- **Final waveform**: MEL spectral distance (LS-MSD) ≤ 0.5 dB vs PyTorch reference at FP16; PESQ ≥ 4.5 vs reference; not bit-identical (FP precision + RNG in sampling).

#### 8.4 Cross-references to add

- **[AUDIO_CODECS.md](AUDIO_CODECS.md)** — add a new section "**9. BiCodec (Spark-TTS)**" summarizing: dual-stream (50 Hz semantic VQ-8192 + 32-token global FSQ-4096), wav2vec2-XLSR feature input (not raw audio), ECAPA + PerceiverResampler speaker path, DAC-style HiFi-GAN wave generator at 16 kHz. Cross-link back here for full details.
- **[HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md)** — confirm the DAC-style decoder section covers the exact same rates [8,5,4,2] + kernels [16,11,8,4] used here; if not, add a "Spark-TTS WaveGenerator" subsection.
- **[DOTLLM_ARCHITECTURE.md](DOTLLM_ARCHITECTURE.md)** — note that Spark's LM reuses the Qwen2 implementation; the only delta is vocab size = 166000.
- **[TEXT_ENCODERS.md](TEXT_ENCODERS.md)** — no Spark-specific text encoder; the Qwen tokenizer covers it.
- **[MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md)** — confirm support for n_fft=1024, win=640, hop=320, n_mels=128, fmin=10. Note the 50% overlap (win/hop=2.0).

#### 8.5 Build order

Suggested implementation phases (depends on what's already done):

1. **Phase A (prerequisites)**: dotLLM Qwen2.5-0.5B BF16 inference + Qwen BPE tokenizer with added-token registration. Mel-spec kernel. (Likely already shipped.)
2. **Phase B (reusable codec primitives)**: WNConv1d, WNConvTranspose1d, Snake1d, ResidualUnit, ConvNeXtBlock, AdaLayerNorm, VocosBackbone. These are shared with DAC and other codecs.
3. **Phase C (BiCodec specifics)**: FactorizedVectorQuantize, ResidualFsq, EcapaTdnnGlobC512, PerceiverResampler, WaveGeneratorDac. Wire into BiCodec class.
4. **Phase D (wav2vec2 helper)**: standalone Wav2Vec2Xlsr53Encoder — only the inference path, no training-time CTC head. Loadable from PyTorch `pytorch_model.bin` (use HartsyInference's existing PyTorch-pickle loader or convert to safetensors at packaging).
5. **Phase E (pipeline)**: SparkTtsTokenParser + SparkTtsPipeline + voice-cloning sample app.
6. **Phase F (controllable mode)**: add the attribute control prompt builder + sample.

#### 8.6 Gotchas

- **Vocab size mismatch**: a stock Qwen2.5-0.5B loader will reject Spark's checkpoint because `embed_tokens` is (166000, 896) instead of (151936, 896). Either accept any vocab size in the loader or special-case Spark.
- **Tied embeddings**: `lm_head` is NOT a separate tensor in the safetensors file — it shares storage with `model.embed_tokens.weight`. The loader must handle the tie at materialization time.
- **wav2vec2 hidden state indexing**: layer 0 is the post-feature-extractor input embedding, so "layer 11" means `hidden_states[11]` which is the **output of the 11th transformer block** (1-indexed: blocks 11, 14, 16). Verify against the HF `Wav2Vec2Model.forward(output_hidden_states=True)` semantics.
- **Reference clip duration**: not hard-coded; read `ref_segment_duration` from `config.yaml`. If missing, the inference code uses 6 seconds. Pad short audio by tile-repeat, NOT zero-pad (zero-padding affects ECAPA stats).
- **AdaLayerNorm conditioning**: the d-vector must be projected to `2 × vocos_dim = 768` (scale + shift) per AdaLN block. Verify against `prenet` weights — there should be 12 small Linear(1024, 768) modules.
- **Non-commercial license**: CC-BY-NC-SA 4.0 — flag this in HartsyInference's model registry so commercial users see the restriction.
- **No streaming**: official inference is one-shot. If we want streaming TTS in HartsyInference, we'd need to chunk the BiCodec decoder over windows of N≥32 semantic tokens (~640 ms) while the LM generates. This is a future enhancement, not in the official pipeline.

## Open Questions / TODO During Implementation

- Confirm the exact CoT slot order for `<|task_controllable_tts|>` against the Python `process_prompt_control()` in `cli/SparkTTS.py` — the test code shows `task → age → gender → pitch_value → pitch_label → loudness_value → loudness_label → emotion` but inference may use a subset.
- Confirm whether global tokens are predicted *before* or *after* semantic tokens by the LM in controllable mode. Voice cloning provides global tokens as conditioning (predicted=false); controllable mode requires LM to generate them.
- Verify the `ref_segment_duration` value in the shipping `config.yaml` (paper hints 6 s; code reads from config).
- The 14k "added" tokens — confirm the exact integer ID layout (continuous block starting at 151,643? Or some other start?). Will affect the embedding-layer initialization sanity check.
- The wav2vec2 feature extractor has its own 7-layer 1-D conv front-end (the "feature_extractor") which downsamples raw 16 kHz audio to 50 Hz at 512-channel before the 24-layer transformer. This is part of XLSR-53 and must be reproduced; standard Wav2Vec2 impl handles it.
