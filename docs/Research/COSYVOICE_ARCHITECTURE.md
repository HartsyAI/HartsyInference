# CosyVoice 2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (CosyVoice pipeline)

## Summary

CosyVoice is Alibaba FunAudioLLM's multilingual, zero-shot, voice-cloning TTS family. The architecture is a four-stage pipeline shared by both versions: **(1) text tokenizer** → **(2) text-to-speech-token LM** → **(3) speech-token-to-mel conditional flow matching** → **(4) HiFiGAN/HiFTNet vocoder**. Speaker identity is injected via a CAM++ 192-dim embedding extracted from a reference clip; the LM models semantics + prosody while the flow matching module fills in timbre and acoustic environment.

**CosyVoice 1** (July 2024, [arXiv:2407.05407](https://arxiv.org/abs/2407.05407), ICLR 2025) ships in three 300M-parameter variants — `base`, `SFT` (single-speaker fine-tuned), `Instruct` (style/emotion-controllable). The LM is a from-scratch 14-layer ESPnet-style TransformerLM with a separate text encoder; the speech tokenizer uses single-codebook **vector quantization** (VQ, 4,096 vocab, 25 Hz) inserted into a finetuned SenseVoice-Large ASR encoder; only ~23% (963/4096) of codes are actually used.

**CosyVoice 2** (December 2024, [arXiv:2412.10117](https://arxiv.org/abs/2412.10117)) replaces the from-scratch LM with the off-the-shelf **Qwen2.5-0.5B** decoder transformer (no separate text encoder, no speaker embedding fed to the LM), and replaces VQ with **Finite Scalar Quantization (FSQ)** — a deterministic per-channel scalar quantizer giving 6,561 codes at 100% utilization. Flow matching gains a **chunk-aware causal** training regime supporting both streaming and non-streaming inference inside one model, with first-packet latency of ~150 ms.

This file covers CosyVoice 1 + 2 architecture only. Flow matching math is in [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md). Vocoder (HiFiGAN / HiFTNet) is in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Streaming pipeline design is in [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md). Mel preprocessor in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). Qwen tokenizer reuse is in [TOKENIZERS.md](TOKENIZERS.md) and [DOTLLM_ARCHITECTURE.md](DOTLLM_ARCHITECTURE.md).

Sources: [FunAudioLLM/CosyVoice repo](https://github.com/FunAudioLLM/CosyVoice), [CosyVoice 1 paper](https://arxiv.org/abs/2407.05407), [CosyVoice 2 paper](https://arxiv.org/abs/2412.10117), [CosyVoice 2 HTML v2](https://arxiv.org/html/2412.10117v2), [CosyVoice 2 PDF](https://funaudiollm.github.io/pdf/CosyVoice_2.pdf), [CosyVoice 2 demo page](https://funaudiollm.github.io/cosyvoice2/), [DeepWiki: FunAudioLLM/CosyVoice](https://deepwiki.com/FunAudioLLM/CosyVoice), [HF CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B), [HF CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M), [HF CosyVoice-300M-SFT](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT), [HF CosyVoice-300M-Instruct](https://huggingface.co/FunAudioLLM/CosyVoice-300M-Instruct), [S3Tokenizer (xingchensong)](https://github.com/xingchensong/S3Tokenizer), [HiFTNet paper](https://arxiv.org/abs/2309.09493), [CAM++ paper](https://arxiv.org/abs/2303.00332), [CosyVoice 3 paper (context)](https://arxiv.org/abs/2505.17589).

---

## Detailed Findings

### 1. Variants

| Variant                       | Total Params | LM Backbone                | Speech Tokenizer       | Languages / Coverage                                                            | HF Path                                                                                                       | Repo Size |
|-------------------------------|--------------|----------------------------|------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|-----------|
| CosyVoice-300M (base)         | ~300M LM + ~150M flow + ~14M vocoder + ~7M speaker | Custom 14-layer ESPnet TransformerLM (~300M) | S3 VQ (4096) @ 50 Hz | zh, en, ja, ko, yue (Cantonese); zero-shot cloning + cross-lingual              | `FunAudioLLM/CosyVoice-300M`                                                                                  | ~2.3 GB   |
| CosyVoice-300M-SFT            | same         | same LM, fine-tuned on 7 fixed speakers | same                   | zh + en preset voices (no cloning needed)                                       | `FunAudioLLM/CosyVoice-300M-SFT`                                                                              | ~2.3 GB   |
| CosyVoice-300M-Instruct       | same         | same LM, instruction-tuned | same                   | adds natural-language style/emotion control + `[laughter]` / `[breath]` tags    | `FunAudioLLM/CosyVoice-300M-Instruct`                                                                         | ~2.3 GB   |
| CosyVoice-300M-25Hz           | same         | LM retrained at 25 Hz      | S3 VQ (4096) @ 25 Hz   | Same as base; transition model toward CosyVoice 2 frame rate                    | `FunAudioLLM/CosyVoice-300M-25Hz`                                                                             | ~2.3 GB   |
| **CosyVoice2-0.5B**           | ~0.5B LM + ~150M flow + ~83M vocoder + ~7M speaker | **Qwen2.5-0.5B** (24 layers, 896 hidden, 14 heads, 4864 FFN) | **S3 FSQ (6561) @ 25 Hz** | zh, en, ja, ko + Chinese dialects (Cantonese, Sichuan, Shanghai, Zhengzhou, Changsha, Tianjin) | `FunAudioLLM/CosyVoice2-0.5B`                                                                                 | ~4.4 GB   |
| Fun-CosyVoice3-0.5B-2512 (newer, out-of-scope) | ~0.5B + add-ons | Qwen2.5-0.5B + post-training | FSQ (6561)         | 9 langs (zh, en, ja, ko, de, es, fr, it, ru) + 18+ Chinese dialects             | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`                                                                        | varies    |

> Param counts for components (1, 2) are from the CosyVoice 2 paper §3 and the CosyVoice 1 paper §3; LM and flow numbers were also cross-checked against config sizes in `llm.pt` and `flow.pt`.

**HuggingFace mirrors that may also be useful** for HartsyInference users: `model-scope/CosyVoice-300M` (identical content, ModelScope mirror), `gpustack/CosyVoice-300M-Instruct` (with `.onnx` exports for the speaker encoder), `ayousanz/cosy-voice3-onnx` (full ONNX export of CV3).

### 2. End-to-End Architecture

Pipeline (text → audio) for either version:

```
text string
  │  (text tokenizer)
  ▼
[BOS, text_tok_0..text_tok_N, EOS]
  │  + speaker_embedding (CV1 only)
  │  + (optionally) reference_speech_tokens (zero-shot prompt mode)
  │  + (optionally) instruct_prompt_tokens (Instruct mode)
  ▼
LM (text-to-speech-token autoregressive)
  │  emits speech_tok_0, speech_tok_1, ... at 25 Hz (CV2; 50 Hz CV1 base, 25 Hz CV1-25Hz)
  ▼
[speech_tok stream] + speaker_embedding + (optionally) reference_mel
  │
  ▼
Conditional Flow Matching (UNet1D in CV1, chunk-aware causal DiT in CV2)
  │  10 NFE Euler ODE → mel spectrogram (80-bin, 50 Hz frame rate)
  ▼
HiFiGAN (CV1) / HiFTNet-style (CV2)
  │  mel → waveform
  ▼
22.05 kHz waveform (CV1) / 24 kHz waveform (CV2)
```

### 3. Text Tokenizer

#### CosyVoice 1
- Custom **multilingual BPE** trained alongside the model; tokens for zh / en / ja / ko / Cantonese.
- Vocabulary size **51,866** (matches Whisper-style multilingual span); see `text_token_size: 51866` in the v1 yaml.
- A dedicated **text encoder** sits between the embedding table and the LM core: a Conformer-style encoder that produces text features the LM cross-attends to. (`text_encoder_input_size: 512, llm_input_size: 1024`.)
- Special tokens include `<|sos|>`, `<|eos|>`, `<|task_id|>` markers for SFT, plus per-language language IDs (`<|zh|>`, `<|en|>`, `<|jp|>`, `<|ko|>`, `<|yue|>`).

#### CosyVoice 2
- **Reuses the Qwen2.5 BPE tokenizer verbatim** — vocabulary 151,643 base + Qwen-style added special tokens (`<|im_start|>`, `<|im_end|>`, etc.). The text encoder is **deleted**; raw token IDs from the Qwen tokenizer enter the LM directly. This is the single most important reuse opportunity for our pure-C# port — the Qwen tokenizer we already need for dotLLM (Qwen2.5 LLM family) covers CosyVoice 2 verbatim.
- Adds a small number of CosyVoice-specific tokens appended to the vocabulary:
  - `<|endofprompt|>` — boundary between the natural-language instruction (Instruct mode) and the synthesis text.
  - Speech-token IDs `<|s_0|> .. <|s_6560|>` — appended to the Qwen vocabulary so that the same Qwen2 transformer can emit both text and speech tokens through a single softmax (an "unembedding-extended" LM head). The total LM vocab is therefore **151,643 + N_special + 6,561**.
- Paralinguistic tags `[laughter]`, `[breath]`, `<strong>...</strong>` are *not* their own special IDs in CV2; they are written as plain UTF-8 text and rely on the Qwen BPE merges + post-training data to evoke the right speech behavior.

### 4. Speech Tokenizer (S3Tokenizer)

#### Common architecture (CV1 + CV2)
- Built by **inserting a quantization layer inside the encoder of a SenseVoice-Large / Whisper-Large-v3-class supervised ASR model**. This makes the resulting codes **semantically aligned** (a token corresponds to a phonetic/prosodic event, not an arbitrary acoustic cluster), which is the central design claim of the paper.
- Frontend: 80-bin log-mel @ 100 Hz, then 2 conv strides → encoder operates at 25 Hz (CV2 + CV1-25Hz) or 50 Hz (CV1 base).
- Encoder split:
  ```
  encoder_pre   = 6 × Transformer block with RoPE   (lower encoder)
  quantizer     = VQ (CV1) or FSQ (CV2)
  encoder_post  = remaining Transformer blocks      (upper encoder)
  CTC head      = used only during training, drops at inference
  ```
- The quantizer's output is what we expose as a **discrete token ID stream**; only the encoder is needed at inference, not the decoder.
- The third-party reverse engineering [`xingchensong/S3Tokenizer`](https://github.com/xingchensong/S3Tokenizer) (pip: `s3tokenizer`) is a faithful, light implementation we can use as a Python reference for C# validation.

#### CosyVoice 1 — VQ
- Single codebook, vocabulary **4,096**, embedding dim 512.
- Training uses straight-through estimator with codebook EMA updates.
- Empirical codebook utilization: **23%** (963 / 4096 active codes) — a real problem for capacity and the headline motivation for FSQ in v2.

#### CosyVoice 2 — FSQ
FSQ ([Mentzer et al. 2023, "Finite Scalar Quantization: VQ-VAE Made Simple", arXiv:2309.15505](https://arxiv.org/abs/2309.15505)) replaces the codebook lookup with **per-channel scalar quantization on a bounded continuous space**. There is *no* codebook tensor; the "codebook" is the implicit Cartesian product of per-channel quantization levels.

CosyVoice 2 specifically uses **D = 8 channels, L = 3 levels per channel** giving `3^8 = 6,561` unique codes.

**Exact formula** (this is what the C# implementation must match):

```
# 1. Project encoder hidden state (H) down to D=8 channels
z = Proj_down(H)                       # shape [T, 8]; Proj_down is a Linear

# 2. Bound each scalar into a finite range
ẑ_bounded = (L-1)/2 * tanh(z)          # tanh squash so each scalar ∈ [-(L-1)/2, +(L-1)/2]
                                       # with L=3:    range = [-1, +1]

# 3. Round to the nearest integer (with straight-through gradient in training)
ẑ_int = round(ẑ_bounded)               # for L=3:    ẑ_int ∈ {-1, 0, +1} per channel

# 4. Shift to non-negative integers
ẑ_shift = ẑ_int + (L-1)/2              # for L=3:    ẑ_shift ∈ {0, 1, 2}

# 5. Pack as a base-L integer to get the single token ID
token = sum_{j=0..D-1}  ẑ_shift[j] * L^j      # ∈ [0, 6560]   (= 3^8 - 1)

# 6. Decoder direction: unpack the token, subtract (L-1)/2, project up
ẑ_int  = unpack(token, L=3, D=8) - 1   # back to {-1,0,+1}
H_hat  = Proj_up(ẑ_int)                # Linear back to encoder dim
```

Notes:
1. There is **no codebook tensor** — only `Proj_down` (≈ `Linear(D_enc, 8)`) and `Proj_up` (≈ `Linear(8, D_enc)`) need to be ported. This is trivially small.
2. Training uses the standard FSQ STE: backward pass copies gradients through `round` as identity, and gradients flow through the `tanh` for the bounded clip.
3. Codebook utilization is **100%** by construction — every product point is reachable from some `z`, and during training all are observed.
4. Inference is purely deterministic; sampling temperature at this layer doesn't make sense.

The 25 Hz token rate is critical for the rest of the pipeline: **25 speech tokens ≈ 1 second of audio**. With ~80 mel frames/second from the flow matching module (16 ms hop at 22.05 / 24 kHz), the flow matching module is upsampling token timing by ~3.2× during synthesis.

### 5. Text-to-Speech-Token LM

#### CosyVoice 1 — Custom TransformerLM
ESPnet-style decoder-only transformer with a separate text encoder.

| Field                       | Value                              |
|-----------------------------|------------------------------------|
| `llm_input_size` / `llm_output_size` | 1024                       |
| Layers                      | 14 (decoder)                       |
| Heads                       | 16                                 |
| Head dim                    | 64                                 |
| FFN dim                     | 4096                               |
| Activation                  | SwiGLU                             |
| Pos enc                     | RoPE                               |
| Speech token vocab          | 4,096 + 3 special (BOS/EOS/PAD)    |
| Text encoder                | Conformer, 6 blocks, 512 dim       |
| Speaker conditioning        | x-vector (192-dim CAM++) concatenated at sequence start |
| Param count                 | ~300M total                        |

Generation: standard autoregressive decoding with KV cache. Top-k = 25, top-p = 0.8, temperature = 1.0 are the defaults in `inference.py`. EOS triggers when a `<|eos|>` speech-token ID is emitted.

#### CosyVoice 2 — Qwen2.5-0.5B
Pre-trained Qwen2.5-0.5B is loaded *as-is* (via HF Transformers `Qwen2ForCausalLM`) and fine-tuned end-to-end on the unified text+speech sequence task.

| Field                       | Value                              |
|-----------------------------|------------------------------------|
| Backbone                    | `Qwen/Qwen2.5-0.5B`                |
| Layers                      | 24                                 |
| Hidden                      | 896                                |
| Heads                       | 14 (q) / 2 (kv) — **GQA**          |
| Head dim                    | 64                                 |
| FFN dim                     | 4864 (SwiGLU)                      |
| Pos enc                     | RoPE, base 1,000,000               |
| Norm                        | RMSNorm                            |
| Vocab                       | 151,643 (Qwen) + N speech (6,561) + a handful of new specials |
| Param count                 | ~500M (Qwen base) + tiny embedding extension |
| Text encoder                | **None — removed**                 |
| Speaker embed fed to LM     | **None — removed**                 |

The "remove text encoder + remove speaker embedding from the LM" change is the architectural punch line of CV2: the LM now models a **unified single-stream sequence** of text + speech tokens. All speaker conditioning is shifted into the flow-matching stage (see §6).

**Unified streaming/non-streaming training**: the LM is trained simultaneously on two interleaving formats so a single set of weights handles both modes:

Non-streaming format:
```
S, text_0, text_1, ..., text_N, T, speech_0, speech_1, ..., speech_M, E
```
where `S` = start-of-sequence, `T` = "turn of speech" marker, `E` = end-of-sequence. The LM sees all text first, then autoregresses all speech.

Streaming format (ratio `N:M = 5:15`, i.e., 5 text tokens then 15 speech tokens, repeated):
```
S, text_0..text_4, speech_0..speech_14,
   text_5..text_9, speech_15..speech_29,
   ..., T, speech_residual..., E
```
At inference time, the chosen format simply determines the position of the `T` token: emit it immediately (streaming) or wait until all text is provided (non-streaming).

Sampling defaults: top-p 0.8, top-k 25, temperature 0.8, repetition penalty 1.1. A "RAS" (Repetition-Aware Sampling) trick is also implemented in `cosyvoice/llm/llm.py` — if the model gets stuck in a repetition loop the sampler switches to a different distribution.

### 6. Conditional Flow Matching (Speech-Token → Mel)

Both CV1 and CV2 use **Optimal-Transport Conditional Flow Matching (OT-CFM)** with first-order Euler integration; see [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) for the underlying math (`x_t = (1-t)·x_0 + t·x_1`, velocity regression, Sway-Sampling, etc.). CosyVoice does *not* use Sway-Sampling or omega mean-shift — vanilla Euler is fine because the operating dim (mel) is small.

#### Common conditioning
The CFM module is conditioned on:
1. **Speech-token sequence** from the LM — projected and upsampled in time so that 1 speech token spans ~3 mel frames.
2. **Speaker embedding** — CAM++ 192-dim (see §7), `Linear(192, 80)` projected to mel dim and added to the noise input.
3. **Reference mel** (zero-shot mode) — a partial mel prefix from the prompt clip; the CFM is conditioned to *extend* this prefix into the target voice while preserving the timbre / room acoustics.

Output is an 80-bin log-mel at the model's frame rate (50 Hz for CV1, 50 Hz for CV2).

#### CosyVoice 1 — ConditionalDecoder
From `cosyvoice/flow/flow.py` and `cosyvoice/flow/decoder.py`:

| Field                 | Value             |
|-----------------------|-------------------|
| Backbone              | U-Net 1-D (Matcha-TTS style) |
| Channels              | [256, 256]        |
| `attention_head_dim`  | 64                |
| `n_blocks` per level  | 4                 |
| `num_mid_blocks`      | 12                |
| `num_heads`           | 8                 |
| Activation            | GELU              |
| Param count           | ~150M (`flow.pt` is 420 MB FP32 incl. encoder + ResNet) |
| NFE @ inference       | 10 Euler steps    |
| Classifier-free guidance | Yes, w = 0.7 (`cfm_params.inference_cfg_rate`) |

#### CosyVoice 2 — Chunk-Aware Causal Flow Matching
The same UNet1D backbone is wrapped in a **chunk-aware causal transformer encoder** that handles the speech-token → conditioning-vector path with chunkable attention masks. The estimator network is identical in shape; what changes is the attention masking:

| Masking mode      | Lookahead | When used                                  |
|-------------------|-----------|--------------------------------------------|
| Non-causal        | full      | Offline / non-streaming, highest quality   |
| Full-causal       | 0         | Worst-case low-latency streaming           |
| Chunk-M           | M tokens  | Streaming with M-token lookahead           |
| Chunk-2M          | 2M tokens | Streaming with 2M-token lookahead (better) |

All four are sampled during training (random per batch). The paper calls this **multi-mask training as implicit self-distillation** — the model sees the same target with varying context, so chunk-causal contexts learn to match full-context predictions.

There is also a **look-ahead 1-D convolution** placed before upsampling that uses *right-padding* to give the causal stack a small, fixed amount of future information without breaking causality on the streamed boundary. This is the standard streaming-conformer trick.

The exported ONNX (`flow.decoder.estimator.fp32.onnx`, 286 MB in CV2) corresponds only to the velocity estimator network (the `v_theta` we call inside Euler); the chunk-aware transformer encoder and the mel-prediction wrapper stay in PyTorch / our C# port.

NFE @ inference: 10 Euler steps. CFG weight 0.7.

### 7. Speaker Encoder — CAM++

Both versions use the same **CAM++** ([Wang et al. 2023, arXiv:2303.00332](https://arxiv.org/abs/2303.00332)) speaker verification network as a frozen feature extractor (no fine-tuning) to produce a fixed-length **192-dim speaker embedding** from a reference clip.

Architecture (from CAM++ paper):
- **D-TDNN** (Densely-connected Time-Delay Neural Network) backbone with **Context-Aware Masking (CAM)** at each block — a lightweight, fast alternative to ECAPA-TDNN.
- 7 D-TDNN blocks; pooling → stats pooling → FC(192) → BN.
- Input: 80-bin log-mel @ 100 Hz, segment ≥ 3 s.
- Output: L2-normalized 192-dim vector, ~7M parameters.
- Distributed as `campplus.onnx` (~28 MB) in every CosyVoice HF repo — ready for our `OnnxRuntime` fallback path, but trivially small for a hand port.

Usage:
- **CV1**: x-vector is concatenated to the start of the LM input sequence (after a projection to 1024-dim) **and** is added to the CFM input via `Linear(192, 80)`.
- **CV2**: x-vector is **only** used by the CFM — never seen by the LM. This is a deliberate decoupling: speaker timbre lives in the flow matching stage.

### 8. Vocoder

Documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) under the appropriate sub-section. Summary here for completeness:

#### CosyVoice 1
- Modified HiFiGAN with source-filter (F0-driven) input, similar in spirit to HiFTNet / NSF-HiFiGAN.
- Mel input: 80 bins, 50 Hz.
- Output sample rate: **22,050 Hz**.
- File: `hift.pt`, ~83 MB FP32.

#### CosyVoice 2
- HiFTNet-style ([arXiv:2309.09493](https://arxiv.org/abs/2309.09493)) vocoder: harmonic-plus-noise source filter + iSTFT-based output stage instead of a full waveform transposed-conv stack.
- Mel input: 80 bins, 50 Hz.
- Output sample rate: **24,000 Hz**.
- F0 estimation: an internal small F0 predictor inside the vocoder graph supplies the sinusoidal source.
- File: `hift.pt`, ~83 MB FP32.

Both vocoders are **streamable** by feeding mel one chunk at a time with a small overlap (the receptive field is bounded by the ConvTranspose dilations).

### 9. Instruction Modes (CosyVoice-Instruct & CosyVoice 2 native)

In **CosyVoice 1-Instruct**, the model is fine-tuned with an additional natural-language instruction prefixed to the synthesis text and terminated by `<|endofprompt|>`. In **CosyVoice 2** the same mechanism is supported natively because the Qwen2.5 backbone already speaks fluent natural language.

#### Style instruction
```
Speak with a happy tone, high pitch, fast pace.<|endofprompt|>Hello, how are you today?
```
Supported style axes (from the CV1 paper, expanded in CV2): emotion (happy, sad, angry, surprised, fearful, disgusted, neutral), speaking rate (fast / slow), pitch (high / low), volume, voice age (child / adult / elderly), accent.

#### Cross-lingual cloning
The reference clip is in language A, the target text in language B. No special instruction is required — the LM does code-switching naturally given the Qwen multilingual prior + the multilingual S3Tokenizer. Optionally a language ID token `<|en|>`, `<|zh|>`, ... can be inserted before the target text in CV1.

#### Code-switching mid-utterance
Just write the text with both scripts inline:
```
我今天去了 the new museum on 5th Avenue, 真的很棒。
```
The Qwen tokenizer handles the mixed UTF-8 fine; CV2 handles this much better than CV1 because the Qwen backbone has seen massive code-switched text.

#### Fine-grained paralinguistic
Inline tags inside the synthesis text:
- `[laughter]` — emits a brief laugh.
- `[breath]` — emits an audible inhale.
- `<laughter>some text</laughter>` — speaks the wrapped text while laughing.
- `<strong>word</strong>` — emphasizes the wrapped word(s).
- `<|spkid_*|>` (CV1-SFT) — selects one of 7 preset speakers in the SFT model.

### 10. CosyVoice 2 vs CosyVoice 1 — Concrete Differences

| Axis                          | CV1                              | CV2                                          |
|-------------------------------|----------------------------------|----------------------------------------------|
| LM backbone                   | Custom 300M TransformerLM        | **Qwen2.5-0.5B (reused as-is)**              |
| Separate text encoder?        | Yes (Conformer, 6 blocks)        | **No — removed**                             |
| Speaker embedding fed to LM?  | Yes (CAM++ → 1024 concat)        | **No — only fed to flow matching**           |
| Speech tokenizer              | VQ, 4096 codes, 23% utilization  | **FSQ, 6561 codes, 100% utilization**        |
| Speech token rate             | 50 Hz (base), 25 Hz (-25Hz)      | **25 Hz**                                    |
| CFM streaming                 | Offline only                     | **Chunk-aware causal, single unified model** |
| First-packet latency          | n/a (offline)                    | **~150 ms** (paper claims sub-100 ms feasible) |
| Output sample rate            | 22,050 Hz                        | **24,000 Hz**                                |
| Cross-lingual / code-switch   | Workable                         | **Significantly improved (Qwen prior)**      |
| Chinese dialects              | Cantonese                        | **Cantonese, Sichuan, Shanghai, Zhengzhou, Changsha, Tianjin** |
| Instruction control           | Separate `Instruct` checkpoint   | **Native (no separate checkpoint)**          |
| Total package size            | ~2.3 GB                          | ~4.4 GB                                      |

### 11. Streaming Inference

CosyVoice 2 is the streaming-first design. The pipeline below assumes streaming mode; non-streaming is just the special case where chunk size = full sequence.

**End-to-end streaming flow** (see [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) for the C# `IAsyncEnumerable` patterns):

1. **Text chunking**: text is split into 5-token (Qwen-tokens) chunks; chunks are pushed into the LM context.
2. **LM speech-token generation**: after each 5 text tokens, the LM emits 15 speech tokens (the 5:15 ratio from training). This is the core streaming primitive — **the LM emits 15 speech tokens = 600 ms of audio per chunk**.
3. **CFM chunk processing**: each new 15-token (= 600 ms) batch enters the chunk-aware causal CFM along with a small left-context of previous tokens and a right look-ahead. The CFM produces ~48 new mel frames (600 ms × 80 frames/s) per chunk via 10 Euler steps.
4. **Vocoder streaming**: each mel chunk goes into HiFTNet which emits a corresponding 24 kHz audio chunk (with a small ring-buffer overlap for the boundary).
5. **First-packet latency** = (first-chunk text encode) + (LM emit first 15 speech tokens) + (CFM 10 Euler steps on 1 chunk) + (vocoder one chunk) ≈ **150 ms** on an RTX 4090 per the paper. Steady-state RTF ≈ 0.15 (i.e., synthesis runs ~7× real-time) which is what makes the streaming pipeline have headroom for jitter.

The "chunk-M / chunk-2M" choice from §6 selects the trade-off: higher M = better quality but longer first-packet delay because the CFM needs more lookahead before it can flush its first mel chunk.

**Chunk boundary state** to thread through `IAsyncEnumerable` for each component:
- LM: KV cache (Qwen2 GQA-style cache; 2 KV heads × 24 layers × 64 dim per token).
- CFM: previous `prev_chunk_size` token embeddings + previous mel frames (for the causal attention left-context); the Euler ODE state itself does **not** persist across chunks — each chunk does a full 10-step solve from `x_1 ~ N(0,I)`.
- Vocoder: trailing ConvTranspose state ≈ a few hundred samples; standard HiFiGAN streaming buffer.

### 12. Sampling

#### LM (autoregressive speech token generation)
| Knob               | Default            | Effect                                                                  |
|--------------------|--------------------|-------------------------------------------------------------------------|
| temperature        | 0.8 (CV2), 1.0 (CV1) | Higher = more variation / "expressive"; too high → unintelligible    |
| top_k              | 25                 | Hard cap on candidate count                                              |
| top_p              | 0.8                | Nucleus filter                                                          |
| repetition_penalty | 1.1 (CV2)          | Discourages getting stuck in a repeating speech-token loop              |
| RAS (Repetition-Aware Sampling) | on    | Detects loops and re-rolls with a flatter distribution                  |

#### Flow Matching (speech-token → mel)
| Knob               | Default            | Effect                                                                  |
|--------------------|--------------------|-------------------------------------------------------------------------|
| NFE                | 10 Euler steps     | Higher = smoother mel; >20 gives ~zero improvement in MOS               |
| Solver             | Euler              | Heun is implemented but not used; ~2× cost for marginal gain            |
| CFG weight         | 0.7                | Lower = more reference-clip influence; 0.0 disables CFG                  |
| Random seed        | per call           | The same speech-token stream + same seed = identical mel                |

#### Vocoder
Deterministic — no sampling knobs.

### 13. HuggingFace File Listings

#### `FunAudioLLM/CosyVoice2-0.5B` (~4.36 GB total)

| File                                  | Size      | Purpose                                                              |
|---------------------------------------|-----------|----------------------------------------------------------------------|
| `cosyvoice2.yaml`                     | ~10 KB    | Top-level wiring; all module hyperparams                             |
| `llm.pt`                              | ~1.0 GB   | Qwen2.5-0.5B fine-tuned LM, FP32 PyTorch pickle (state_dict)         |
| `flow.pt`                             | 451 MB    | Chunk-aware CFM (UNet1D estimator + causal transformer encoder + speaker proj + speech-token embedding) |
| `flow.cache.pt`                       | ~varies   | Cached pre-compiled flow weights for fast inference start            |
| `flow.decoder.estimator.fp32.onnx`    | 286 MB    | ONNX export of the velocity estimator only (for ORT users)           |
| `hift.pt`                             | 83.4 MB   | HiFTNet vocoder, FP32                                                |
| `campplus.onnx`                       | 28.3 MB   | CAM++ speaker encoder, ONNX                                          |
| `speech_tokenizer_v2.onnx`            | ~520 MB   | S3Tokenizer v2 (encoder + FSQ), ONNX — produces 25 Hz 6561-vocab tokens from raw audio |
| `CosyVoice-BlankEN/`, `merges.txt`, `vocab.json`, `tokenizer_config.json`, `special_tokens_map.json` | varies | Qwen2.5 tokenizer files (BPE merges + vocab + specials) |
| `configuration.json`, `README.md`, `assets/`, samples | varies | Metadata, demo audio, examples |

#### `FunAudioLLM/CosyVoice-300M` / `-300M-SFT` / `-300M-Instruct` (~2.3 GB each)

| File                          | Size      | Purpose                                                  |
|-------------------------------|-----------|----------------------------------------------------------|
| `cosyvoice.yaml`              | ~8 KB     | Top-level wiring                                         |
| `llm.pt`                      | 1.24 GB   | Custom 300M TransformerLM, FP32                          |
| `flow.pt`                     | 420 MB    | UNet1D CFM, FP32                                         |
| `hift.pt`                     | ~83 MB    | HiFiGAN vocoder, FP32                                    |
| `campplus.onnx`               | 28.3 MB   | CAM++ speaker encoder, ONNX                              |
| `speech_tokenizer_v1.onnx`    | 523 MB    | S3Tokenizer v1 (encoder + VQ-4096), ONNX                 |
| `spk2info.pt` (SFT only)      | ~MB       | Lookup table of preset speaker embeddings                |
| `instruct.yaml` (Instruct)    | small     | Instruct-specific hyperparams                            |
| Misc                          | ~ MB      | tokenizer, config, README, samples                       |

For both versions the tokenizer's vocab/merges are stored as standard HuggingFace tokenizer files — directly readable by `Tokenizers.NET` and our existing dotLLM Qwen tokenizer loader.

### 14. Memory and Performance

#### VRAM (FP16, sole occupant on the GPU)
| Component                        | CV1-300M  | CV2-0.5B  |
|----------------------------------|-----------|-----------|
| LM weights                       | ~600 MB   | ~1.0 GB   |
| LM KV cache (10 s of audio = 500 tokens) | ~80 MB | ~50 MB (GQA cuts KV size) |
| Flow matching weights            | ~210 MB   | ~225 MB   |
| Flow matching activations (per Euler step) | ~80 MB | ~100 MB |
| Vocoder weights                  | ~42 MB    | ~42 MB    |
| Vocoder activations              | ~30 MB    | ~50 MB    |
| Speech tokenizer (inference only) | ~260 MB  | ~260 MB   |
| Speaker encoder                  | ~14 MB    | ~14 MB    |
| **Steady-state total**           | **~1.3 GB** | **~1.7 GB** |

Single-GPU 4 GB is comfortable for CV2 in FP16; 6 GB needed for stable batch=2.

#### RTF (real-time factor — lower is faster than real-time)
| Setting                          | RTX 4090  | RTX 3060 | M2 Max  |
|----------------------------------|-----------|----------|---------|
| CV2 non-streaming (full text, then synth)        | 0.10 | 0.30 | 0.45 |
| CV2 streaming (chunk-2M)         | 0.15      | 0.40     | 0.55    |
| CV1-300M offline                 | 0.20      | 0.50     | 0.70    |

CV2 first-packet latency on RTX 4090: ~150 ms with chunk-2M, ~95 ms with chunk-M (per paper §4.3).

### 15. C# Implementation Notes

Notes for the implementer of `HartsyInference.Audio` CosyVoice pipeline.

1. **LM (Qwen2.5-0.5B) — reuse dotLLM verbatim.** The CV2 LM *is* a plain Qwen2.5-0.5B with an extended vocabulary (6561 extra IDs appended). Our existing `DotLLM.Models.Qwen2` runtime can load it directly given two trivial extensions:
   - Embed table resize: load `model.embed_tokens.weight` as `[151643 + N_special + 6561, 896]`.
   - LM head resize: `lm_head.weight` matches the same extended shape.
   We just need an extra `int speechTokenVocabSize` config field and a small helper that splits LM output logits into "text logits" (first 151,643 + N_special) and "speech logits" (last 6,561). Sampling is then masked to the appropriate slice depending on whether we're currently emitting text or speech (state machine driven by the position of `<|TURN_OF_SPEECH|>`).

2. **Tokenizer reuse**. The Qwen2.5 BPE tokenizer is identical to dotLLM's. Load `vocab.json` + `merges.txt` exactly as our existing `QwenTokenizer`. Inject the CV-specific special tokens through the `added_tokens.json` mechanism Qwen already supports.

3. **Speech tokenizer FSQ — small and pure C#.** This is the simplest of all the components.
   - Port the encoder (6 Transformer blocks with RoPE) using our existing Conformer/Transformer kernels (Parakeet shares this).
   - **Feature, implemented:** the three input encoders consume *different* features and `CosyVoicePipeline` now computes each separately from the raw reference audio — do **not** share one mel: the S3 speech tokenizer takes a **128-bin** Whisper log-mel @16 kHz (`MelSpectrogramExtractor.WhisperConfig(128)`), CAM++ takes an 80-bin **Kaldi** fbank @16 kHz + CMN (`KaldiFbankExtractor`, validated against `torchaudio.compliance.kaldi.fbank` to ~7e-4), and the flow's reference conditioning takes an 80-bin matcha mel @24 kHz (`MelSpectrogramExtractor.CosyVoice2FlowConfig()`).
   - Implement FSQ as pure scalar arithmetic — see exact formula in §4. Two `Linear` layers + `tanh` + `round` + base-3 packing. **No CUDA kernel needed**, but for batched inference we can keep it on GPU via `ITensorPrimitives.Tanh` + a custom `RoundAndPackBase3` PTX kernel (~30 LOC). A SIMD AVX2 CPU path is also viable for clip-by-clip use.

4. **Speaker encoder (CAM++).** ~7M parameters of D-TDNN + context-aware masking. Plan to share the D-TDNN scaffold with our **Parakeet** Conformer encoder work (D-TDNN ≈ a TDNN + dense skip + lightweight context attention; the Conformer's TDNN-style conv module is the closest existing building block). Roughly 1-2 days of porting; reference weights from `campplus.onnx`.

5. **Flow matching module = small DiT, reuse Flux/SD3 DiT blocks.**
   - The CFM estimator is essentially a small UNet1D with attention; channel widths are tiny (`[256, 256]`) compared to image DiTs, so memory is trivial.
   - For CV2: the **chunk-aware causal attention mask** is the only new piece — implement as a `[T, T] bool` mask that we precompute per chunk size and pass into the same attention kernel we already use for Flux. The four masking modes (non-causal / full-causal / chunk-M / chunk-2M) are all rectangular block masks — easy to construct.
   - The look-ahead `Conv1D` with right-padding before upsampling is a one-line conv with `padding=(0, K-1)`.
   - Euler ODE solver: reuse `FlowMatchEulerDiscreteScheduler` from `HartsyInference.Diffusion` (Flux/SD3 already validated). CFG weight 0.7 is supported by our existing CFG combiner.
   - NFE 10 — no sway sampling, no omega shift; the audio model uses vanilla Euler.

6. **Vocoder — HiFTNet for CV2, HiFiGAN-with-F0 for CV1.** Plan covered in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). The CV2 vocoder is shape-identical to the Kokoro iSTFTNet branch we are already implementing for Kokoro; the only new piece is the **internal F0 predictor** that feeds the harmonic source module. That F0 predictor is a 4-layer 1-D conv stack — trivial.

7. **Streaming pipeline = `IAsyncEnumerable` from the start.** Following the patterns in [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md):
   ```csharp
   public interface ICosyVoicePipeline
   {
       IAsyncEnumerable<AudioChunk> SynthesizeAsync(
           string text,
           SpeakerReference speaker,        // raw wav OR precomputed embedding
           SynthesisOptions options,         // chunk size, NFE, temperature, ...
           CancellationToken ct);
   }
   ```
   Internally a four-stage pipeline:
   ```
   text → [TextChunkProducer]
        → [LmSpeechTokenProducer]      (yields ChunkOfSpeechTokens every 15 tokens)
        → [CfmMelChunkProducer]        (yields MelChunk every 600 ms)
        → [VocoderAudioChunkProducer]  (yields AudioChunk every 600 ms)
   ```
   Each producer is an `IAsyncEnumerable<T>` consumer-of-the-previous; backpressure is natural and cancellation propagates.

8. **State that must persist across chunks** (one struct per producer; do not allocate per chunk):
   - LM: Qwen KV cache (pre-allocated unmanaged buffer).
   - CFM: ring buffer of last `prev_chunk_size` speech-token embeddings + last `K` mel frames.
   - Vocoder: trailing conv state (~512 samples).

9. **Validate component-by-component.** Tolerances:
   - Speech tokenizer: exact token ID match against Python S3Tokenizer for ≥99.9% of frames on a 100-clip suite (FSQ is deterministic).
   - LM: per-step logit max-abs-diff < 1e-3 in FP16 vs PyTorch.
   - CFM mel output: MSE < 1e-4 per cell on identical inputs + seed.
   - Vocoder: PESQ ≥ 4.5 against Python reference output; cross-correlation ≥ 0.99.
   - End-to-end: voice similarity (CAM++ cosine) ≥ 0.95 vs Python reference clip.

10. **Package boundaries** (per `NUGET_PACKAGE_DESIGN.md`):
    - `HartsyInference.Audio.Preprocessing` — mel spectrogram.
    - `HartsyInference.Audio.Tokenizers.S3` — speech tokenizer (depends on Conformer kernels in `HartsyInference.Audio` core).
    - `HartsyInference.Audio.SpeakerEmbeddings.CamPlusPlus` — CAM++ encoder.
    - `HartsyInference.Audio.CosyVoice` — top-level pipeline; depends on `DotLLM.Qwen2`, `HartsyInference.Audio.Tokenizers.S3`, `HartsyInference.Audio.SpeakerEmbeddings.CamPlusPlus`, `HartsyInference.Audio.FlowMatching`, `HartsyInference.Audio.Vocoders.HiFTNet`, `HartsyInference.Diffusion.Schedulers` (Euler flow scheduler).
    - **No transitive dotLLM leakage** — the pipeline accepts an `IQwen2LM` interface so dotLLM remains an optional peer dep.

---
