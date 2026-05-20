# SenseVoice + FireRedASR — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (SenseVoice + FireRedASR pipelines)

## Summary

This document covers two strong Chinese / multilingual STT model families that we plan to wrap as pure-C# pipelines under `SharpInference.Audio`:

- **SenseVoice** (Alibaba FunAudioLLM, July 2024) — non-autoregressive, CTC-based, encoder-only "speech understanding" model. The released **SenseVoice-Small** variant matches Whisper-Large-v3 quality on Chinese / Cantonese while running 15× faster (≈70 ms for 10 s of audio). One forward pass emits a 4-part output: `<emotion><language><event><text>`. Encoder uses **SAN-M** (Self-Attention with Memory) — vanilla multi-head attention augmented with an FSMN depthwise-conv "memory" block.
- **FireRedASR** (Xiaohongshu / FireRedTeam, Jan 2025) — industrial-grade Mandarin + dialect + English ASR. Two variants: **AED-L** (1.1 B params, Conformer encoder + Transformer decoder, Whisper-style autoregressive decode) and **LLM-L** (8.3 B params, same Conformer encoder + Linear-ReLU-Linear adapter + Qwen2-7B-Instruct decoder). Both share the same audio frontend (80-bin log-mel, kaldi fbank, CMVN, no LFR) and the same `train_bpe1000` SentencePiece tokenizer (7 832 entries: 1 000 English BPE + 6 827 Chinese chars + 5 special tokens). FireRedASR-LLM-L holds SOTA on Mandarin benchmarks (avg CER 3.05 % across AISHELL-1/2, WenetSpeech-Net/Meeting).

Mel preprocessing for both families is the standard 16 kHz / 80-bin / 25 ms-window / 10 ms-hop kaldi-native-fbank with CMVN. SenseVoice additionally applies LFR (Low Frame Rate) frame stacking with `lfr_m=7, lfr_n=6` — i.e. stack 7 consecutive frames and step by 6 — yielding a 560-dim input vector at a 60 ms frame rate before linear projection to `d_model`. FireRedASR does **not** apply LFR; downsampling happens inside the encoder via a Conv2dSubsampling stack (stride-2 ×2 → 4× downsample → 40 ms frames; FireRedASR-LLM further frame-splices ×2 inside the adapter → 80 ms frames).

Sources:
- SenseVoice: [GitHub FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice), [model.py](https://github.com/FunAudioLLM/SenseVoice/blob/main/model.py), [HuggingFace FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall), [FunAudioLLM paper arXiv:2407.04051](https://arxiv.org/abs/2407.04051), [SenseVoice.cpp port](https://github.com/lovemefan/SenseVoice.cpp), [DeepWiki FunASR SenseVoice page](https://deepwiki.com/modelscope/FunASR/5.2-sensevoice), [sherpa-onnx SenseVoice docs](https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html)
- FireRedASR: [GitHub FireRedTeam/FireRedASR](https://github.com/FireRedTeam/FireRedASR), [FireRedASR paper arXiv:2501.14350](https://arxiv.org/abs/2501.14350), [HuggingFace FireRedASR-AED-L](https://huggingface.co/FireRedTeam/FireRedASR-AED-L), [HuggingFace FireRedASR-LLM-L](https://huggingface.co/FireRedTeam/FireRedASR-LLM-L), [literature review (themoonlight.io)](https://www.themoonlight.io/en/review/fireredasr-open-source-industrial-grade-mandarin-speech-recognition-models-from-encoder-decoder-to-llm-integration), [FireRedASR2 (follow-up) arXiv:2603.10420](https://arxiv.org/abs/2603.10420)

## Detailed Findings

### 1. Variants Tables

#### 1.1 SenseVoice variants

Only **SenseVoice-Small** is publicly released. A larger AR (autoregressive) variant ("SenseVoice-Large") is described in the FunAudioLLM paper but has not been open-sourced as of May 2026.

| Variant | Released | Arch | Params | d_model | Heads | Enc blocks | tp_blocks | FFN | Vocab | Notes |
|---------|----------|------|--------|---------|-------|-----------|-----------|-----|-------|-------|
| SenseVoice-Small | Jul 2024 | Encoder-only (CTC) | ~234 M (Whisper-Small class) | 512 | 4 | 50 (= 1 `encoders0` + 49 `encoders`) | 0 | 2048 | 25 055 | 15× faster than Whisper-Large-v3; matches/exceeds it on zh/yue |
| SenseVoice-Large | unreleased | Encoder-Decoder (AR) | unknown | n/a | n/a | n/a | n/a | n/a | n/a | Cited in arXiv:2407.04051; weights not public |

Notes on SenseVoice-Small:
- `output_size = 512`, `attention_heads = 4`, `linear_units = 2048`, `num_blocks = 50`, `tp_blocks = 0` (the "temporal processor" stack is empty in the released config), `kernel_size = 11` (SANM depthwise conv).
- Despite "Small" in the name, the **encoder is 50 layers deep** (the depth, not the width, drives parameter count). This is unusual compared to Whisper.
- Inference latency: 70 ms / 10 s of audio (Whisper-Small ≈ 350 ms; Whisper-Large-v3 ≈ 1050 ms).
- Trained on **400 000+ hours** of multilingual audio.

#### 1.2 FireRedASR variants

The arXiv:2501.14350 paper trains and ablates AED at four sizes (XS / S / M / L), but **only AED-L and LLM-L are released**. The smaller AED sizes are paper-only.

| Variant | Released | Arch | Params | Enc layers | Dec / LLM layers | d_model (enc) | d_model (dec) | Heads (enc) | Heads (dec) | Max audio | Vocab |
|---------|----------|------|--------|-----------|------------------|---------------|---------------|-------------|-------------|-----------|-------|
| AED-XS | paper only | Conformer + Transformer | 140 M | small | small | — | — | — | — | 60 s | 7 832 |
| AED-S | paper only | Conformer + Transformer | 413 M | medium | medium | — | — | — | — | 60 s | 7 832 |
| AED-M | paper only | Conformer + Transformer | 732 M | large | large | — | — | — | — | 60 s | 7 832 |
| **AED-L** | Jan 2025 | Conformer + Transformer | **1.1 B** | deep (large) | deep | encoder hidden (~1280) | matches enc | ~20 | ~20 | 60 s | 7 832 |
| **LLM-L** | Jan 2025 | Conformer + Adapter + Qwen2-7B | **8.3 B** | same enc as AED-L | 28 (Qwen2-7B-Instruct) | encoder hidden, projected to **768** for adapter input | 3 584 (Qwen2 hidden) | enc heads | 28 (Qwen2) | 30 s | LLM tokenizer (Qwen2) |

Exact per-layer dims for AED-L and the LLM-L encoder are not enumerated in the paper text we could parse (Table 1 cells did not survive the HTML extraction); they have to be read off the released `config.yaml` checkpoints when building the loader. The values above ("~1280", "~20") are what the architecture is designed around per the paper's scaling discussion and are the conventional Conformer-Large dims. **Implementers must confirm the exact `d_model`, `n_head`, `n_layers_enc`, `n_layers_dec` from `config.yaml` (it is stored inside the released HF repos and inside `package["args"]` in the checkpoint `.pth.tar`).** Known confirmed values regardless of size:
- Conformer convolution kernel: **33** (1-D depthwise).
- Subsampling: **4×** via 2× stride-2 Conv2d (kernel 3) → 40 ms frame rate at encoder output.
- FFN expansion: **4× d_model** (`PositionwiseFeedForward`).
- Positional encoding max length: **5000**.
- Residual / general dropout: **0.1**.
- Adapter (LLM only): Linear → ReLU → Linear, plus a `frame_splice` of 2 at adapter input (→ 80 ms frame rate fed to Qwen2).
- LLM adapter output dim must equal Qwen2-7B hidden = **3 584**.
- LoRA on Qwen2 (when enabled during fine-tune): r=64, α=16.

### 2. Architecture

#### 2.1 SenseVoice-Small (encoder-only, CTC)

```
[waveform 16 kHz mono]
    │
    ▼ kaldi-native-fbank: n_mels=80, frame_length=25 ms, frame_shift=10 ms, Hamming, dither=1.0
[T × 80 fbank]
    │
    ▼ LFR stacking (lfr_m=7, lfr_n=6) → frames concatenated, then strided
[T' × 560]                                where T' ≈ T/6
    │
    ▼ optional global CMVN (loaded from am.mvn) — per-feature mean / scale
[T' × 560]
    │
    ▼ prepend 4 learned query embeddings:  [LID] [SER/Emotion] [AED/Event] [ITN]
[(T' + 4) × 560]   ──►  encoders0 (Linear 560→512 + SANM block)  ──►  [(T'+4) × 512]
    │
    ▼ SinusoidalPositionEncoder (scales input by √d_model then adds sinusoidal PE)
    │
    ▼ 49 × EncoderLayerSANM            (the "encoders" stack)
    │
    ▼ 0  × EncoderLayerSANM            (the "tp_encoders" stack — empty in release)
[(T' + 4) × 512]
    │
    ▼ final LayerNorm
    │
    ▼ Linear ctc_lo: 512 → 25 055      (CTC head)
[(T' + 4) × 25 055]   ◄── softmax + argmax over time
    │
    ▼ greedy CTC decode: collapse-consecutive + remove blank(0)
[token ids ...]
    │
    ▼ split prefix: first 4 tokens = <lid> <emo> <event> <itn>;  rest = text BPE
[(lang, emotion, event, itn-flag, text)]
```

`EncoderLayerSANM` (pre-norm, GLU-free — this is **not** a Conformer block, contrary to some third-party blog posts):

```
x → LayerNorm → MultiHeadedAttentionSANM(n_head=4, n_feat=512, kernel_size=11) → +x (residual)
  → LayerNorm → PositionwiseFeedForward(512 → 2048 → 512, ReLU)                → +x (residual)
```

`MultiHeadedAttentionSANM` is the SAN-M block. It is **standard scaled-dot-product MHA augmented with an FSMN memory branch** that runs in parallel:

```
SAN-M(x):
    # FSMN memory branch (depthwise 1-D conv along time)
    mem = DepthwiseConv1D(channels=d_model, kernel=11, padding=center, groups=d_model)(x)
    mem = mem  # the "memory" — captures local temporal context cheaply

    # Standard multi-head self-attention
    q, k, v = Linear(x), Linear(x), Linear(x)   # one fused Linear in the impl
    a = softmax((q @ kᵀ) / √d_k + mask) @ v
    a = Linear(a)

    return a + mem        # additive fusion of attention output and FSMN memory
```

The FSMN branch lets SenseVoice trade some softmax depth for cheap depthwise convs, which is part of why the 50-layer encoder is fast despite the depth. The encoder is **non-streaming** in the released config (the `forward_chunk` / KV-cache path exists in code but is not used by default).

**Special-token prepending.** Four learned query embeddings — language ID hint, emotion query, audio-event query, ITN (inverse text normalization) flag — are concatenated **before** the encoder. After the forward pass the first 4 frame logits are interpreted as classifications over `{lang ids}`, `{emotion ids}`, `{event ids}`, `{itn ids}` respectively. The remaining frames go through standard CTC decoding for the transcript. During training: cross-entropy on the first 4 positions, CTC loss on the rest.

#### 2.2 FireRedASR-AED-L (Conformer encoder + Transformer decoder)

```
[waveform 16 kHz mono, up to 60 s]
    │
    ▼ kaldi-native-fbank: n_mels=80, frame=25 ms, hop=10 ms, Hamming + global CMVN (cmvn.ark)
[T × 80]
    │
    ▼ Conv2dSubsampling: 2× (Conv2d k=3,s=2 + ReLU)   → 4× downsample, 80 → d_model
[T/4 × d_model]                                       (frame rate: 40 ms)
    │
    ▼ Relative-positional encoding (max_len=5000)
    │
    ▼ N_enc × ConformerBlock
[T/4 × d_model]
    │   ┌── encoder output ────────────────────────────────────────────┐
    │                                                                  │
    ▼   start AR decode from [<sos>]                                   │
[B × t × d_model]                                                      │
    │                                                                  │
    ▼ N_dec × TransformerDecoderBlock                                  │
    │       (self-attn pre-norm, cross-attn into encoder output, FFN 4×d_model) ◄┘
    │
    ▼ Linear d_model → vocab(7832)
    │
    ▼ beam search (default beam=3, with length penalty + repetition penalty + softmax temperature)
[token ids ...]
    │
    ▼ ChineseCharEnglishSpmTokenizer.decode  (joins Chinese chars directly, BPE-merges English)
[transcript]
```

`ConformerBlock` (Macaron-FF + MHSA + ConvModule + Macaron-FF, all residual + pre-norm):

```
x → ½·FFN1(x)                                              + x
  → MultiHeadSelfAttention(rel-pos)                         + x
  → ConvModule { PointwiseConv → GLU → DepthwiseConv1D(k=33) → BatchNorm → Swish → PointwiseConv } + x
  → ½·FFN2(x)                                               + x
  → LayerNorm
```

#### 2.3 FireRedASR-LLM-L (Conformer encoder + Adapter + Qwen2-7B decoder)

```
[waveform 16 kHz mono, up to 30 s]
    │
    ▼ same fbank + CMVN as AED-L
[T × 80]
    │
    ▼ same Conv2dSubsampling + N_enc × ConformerBlock as AED-L (encoder weights start from AED-L)
[T/4 × d_enc]                            (40 ms frames)
    │
    ▼ Adapter:
    │     frame_splice(×2)            → [T/8 × 2·d_enc]   (80 ms frames; cuts seq len in half)
    │     Linear(2·d_enc → 768) → ReLU → Linear(768 → 3584)
[T/8 × 3584]   = "speech embeddings"
    │
    ▼ merge with text token embeddings produced by Qwen2 tokenizer
    │     prompt template: "<|im_start|>user\n<SPEECH PLACEHOLDERS>\n请转写音频内容<|im_end|>\n<|im_start|>assistant\n"
    │     _merge_input_ids_with_speech_features() splices speech embeddings in place of placeholder tokens
[(prompt_text_emb || speech_emb || suffix_emb) × 3584]
    │
    ▼ Qwen2-7B-Instruct (28 layers, hidden=3584, heads=28, kv_heads=4, FFN=18944, RoPE, SwiGLU)
    │   — autoregressive AR decode with KV cache, beam or greedy
[generated token ids ...]
    │
    ▼ Qwen2 tokenizer decode (BPE)
[transcript]
```

Key facts for the LLM variant:
- Encoder hidden dim **must be projected** through `2·d_enc → 768 → 3584` to land in Qwen2's embedding space (`hidden_size = 3584`).
- The "frame splicing" inside the adapter halves the speech token count from ~750 (30 s @ 40 ms) to ~375 (30 s @ 80 ms) — important to keep prompt length manageable.
- Special tokens used from Qwen2: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`. No new special tokens are added; speech is injected as embeddings, not token IDs.

### 3. Multilingual / Output Format / Supported Tags

#### 3.1 SenseVoice output format and tag inventory

Output is rendered (when text-formatted) as `<|lang|><|emotion|><|event|><|itn-flag|>text`. The four prefix slots come from four distinct learned queries injected at the encoder input.

**Language tokens** (5 explicit + `auto`/`nospeech`):
- `<|auto|>` (id 0), `<|zh|>` (3), `<|en|>` (4), `<|yue|>` Cantonese (7), `<|ja|>` (11), `<|ko|>` (12), `<|nospeech|>` (13).
- The model is *trained* on 50+ languages but the LID head is constrained to the 5 production tags above (plus `auto`/`nospeech`). Other languages are still transcribed (the CTC head sees full multilingual vocab) but get labeled `auto`.

**Emotion tokens** (7):
- `<|HAPPY|>` (25001), `<|SAD|>` (25002), `<|ANGRY|>` (25003), `<|NEUTRAL|>` (25004), `<|FEARFUL|>`, `<|DISGUSTED|>`, `<|SURPRISED|>`, plus `<|EMO_UNKNOWN|>` (25009).

**Event tokens** (8):
- `<|BGM|>` (background music), `<|Speech|>`, `<|Applause|>`, `<|Laughter|>`, `<|Cry|>`, `<|Sneeze|>`, `<|Breath|>`, `<|Cough|>`.

**Text-formatting tokens** (2):
- `<|withitn|>` (14) — inverse text normalization applied (digits as digits, punctuation present).
- `<|woitn|>` (15) — no ITN (digits spelled out, no punctuation).

**Other vocab constants**:
- `vocab_size = 25055`, `blank_id = 0`, `sos = 1`, `eos = 2`, `ignore_id = -1`.
- The text portion is BPE/SentencePiece over CJK characters + Latin subwords + Japanese / Korean.

#### 3.2 FireRedASR languages

- **Mandarin** (primary).
- **English** (well-supported; trained jointly).
- **Chinese dialects** (paper claims "Chinese dialects" but the v1 release does not have explicit dialect tags — recognition relies on the encoder generalizing). The follow-up **FireRedASR2** expands explicit dialect coverage to 20+ accents (Cantonese HK/GD, Sichuan, Shanghai, Wu, Minnan, Anhui, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan, Jiangxi, Liaoning, Ningxia, Shaanxi, Shanxi, Shandong, Tianjin, Yunnan, etc.) and grows training data from 70 k to ~200 k hours.
- **Singing lyrics** — notable strength; 50–67 % CER reduction vs industrial baselines on lyrics-from-singing audio (FireRedASR2-LLM reaches 1.12 % CER on opencpop).
- No emotion / event side outputs — pure ASR.

### 4. Mel Preprocessing

#### 4.1 SenseVoice frontend (FunASR `WavFrontend`)

| Param | Value |
|-------|-------|
| sample_rate | 16 000 Hz |
| n_mels | 80 |
| frame_length | 25 ms (= 400 samples) |
| frame_shift | 10 ms (= 160 samples) |
| window | Hamming |
| dither | 1.0 (Kaldi-style additive uniform noise) |
| energy_floor | 0 |
| snip_edges | True |
| pre-emphasis | none (kaldi default) |
| **LFR (Low Frame Rate)** | `lfr_m = 7`, `lfr_n = 6` — concatenate 7 consecutive frames, step 6 → input dim 560, time-stride 60 ms |
| CMVN | loaded from `am.mvn` — global per-feature mean + 1/std rescale; applied **after** LFR |
| Final input shape | `[T/6, 560]` |

Implementation note: SenseVoice computes log-mel via the **kaldi-native-fbank** layout (log-power-spectrum → mel filterbank → log, with Kaldi-specific dither + snip-edges + window function). It is **not** byte-compatible with Whisper's STFT/torch-audio mel implementation — features differ by ≈ 1 e-2 typical magnitude. Don't try to reuse the Whisper mel path; build a Kaldi-fbank module.

#### 4.2 FireRedASR frontend (`ASRFeatExtractor`)

| Param | Value |
|-------|-------|
| sample_rate | 16 000 Hz |
| n_mels | 80 |
| frame_length | 25 ms |
| frame_shift | 10 ms |
| window | Hamming (kaldi-native-fbank) |
| dither / snip_edges | kaldi defaults |
| LFR | **not used** (lfr_m=1, lfr_n=1 effectively) |
| CMVN | loaded from `cmvn.ark` — global per-feature mean+var |
| Final input shape | `[T, 80]` |

Downsampling happens inside the encoder via a Conv2dSubsampling (2× stride-2 conv layers → 4×), not in the frontend. FireRedASR-LLM adds another 2× via `frame_splice` in the adapter → 8× total downsampling end-to-end.

### 5. Tokenizer

#### 5.1 SenseVoice tokenizer

- **Type**: SentencePiece (BPE) over multilingual text — Chinese chars + Latin subwords + JA/KO.
- **Vocab size**: 25 055.
- **Special-token reservations** (low IDs): 0=blank/`<|auto|>`-ambiguous, 1=`<sos>`, 2=`<eos>`, 3=`<|zh|>`, 4=`<|en|>`, 7=`<|yue|>`, 11=`<|ja|>`, 12=`<|ko|>`, 13=`<|nospeech|>`, 14=`<|withitn|>`, 15=`<|woitn|>`. High IDs (25001–25009) reserved for emotion labels.
- **Tokenizer file**: shipped as `chn_jpn_yue_eng_ko_spectok.bpe.model` (SentencePiece protobuf) in the HF repo.

#### 5.2 FireRedASR tokenizer (AED variant)

- **Type**: `ChineseCharEnglishSpmTokenizer` — hybrid scheme. English text is BPE-tokenized via SentencePiece (`train_bpe1000.model`, 1 000 merges). Chinese text is split character-by-character (6 827 unique Hanzi/symbol entries). Special tokens: 5 (sos, eos, pad, unk, blank).
- **Vocab size**: **7 832** = 1 000 (en BPE) + 6 827 (zh chars) + 5 (special).
- **Tokenizer file**: `train_bpe1000.model` (SentencePiece protobuf) + a separate `dict.txt` listing Hanzi.
- **Decode**: emit Chinese chars directly with no separator, merge English BPE pieces.

#### 5.3 FireRedASR-LLM tokenizer

- Uses Qwen2's own tokenizer (Qwen2 BPE, ~151 k vocab). Speech is **not** tokenized — it enters as embeddings.

### 6. Inference Loop

#### 6.1 SenseVoice-Small inference

```
1. Load waveform, resample to 16 kHz mono.
2. Compute kaldi-fbank 80-dim → apply LFR (m=7, n=6) → stack to 560-dim.
3. Apply CMVN: feat = (feat - mean) * scale.    # vectors of length 560
4. Build input: prepend 4 learned query embeddings → shape [T'+4, 512] after encoder input projection.
5. Forward through SenseVoiceEncoderSmall (50 SANM blocks).
6. ctc_logits = ctc_lo(encoder_out)              # [T'+4, 25055]
7. Take first 4 frames → argmax → (lang_id, emo_id, event_id, itn_id).  Decode via reserved-ID tables.
8. CTC greedy decode on frames [4:]:
       ids = argmax(ctc_logits[4:], dim=-1)
       ids = unique_consecutive(ids)             # collapse runs
       ids = [i for i in ids if i != 0]          # remove blanks
9. text = sentencepiece.decode(ids)
10. Render output: f"<|{lang}|><|{emo}|><|{event}|><|{itn}|>{text}"
```

No KV cache, no beam search, no autoregressive loop — **single forward pass per utterance**. This is the source of the 15× speedup over Whisper.

#### 6.2 FireRedASR-AED-L inference

Whisper-style encoder-decoder AR:

```
1. Load + resample.  Compute fbank 80-dim.  Apply CMVN.
2. encoder_out = ConformerEncoder(feat)             # [T/4, d_model]
3. tokens = [<sos>]
4. kv_cache = empty
5. while tokens[-1] != <eos> and len(tokens) < max_len:
       y = decoder(tokens, encoder_out, kv_cache)    # cross-attn into encoder_out each layer
       next_token = beam_search_step(y, beam=3,
                                     length_penalty=0.6,
                                     repetition_penalty=3.0,
                                     temperature=1.0)
       tokens.append(next_token)
6. transcript = tokenizer.decode(tokens[1:-1])
```

Beam search uses standard length penalty and a repetition penalty (FireRedASR's chosen anti-repeat trick — same idea as Hugging Face's `repetition_penalty`, applied to logits before softmax).

#### 6.3 FireRedASR-LLM-L inference

```
1. Load + resample. Compute fbank 80-dim. Apply CMVN.
2. enc = ConformerEncoder(feat)                                    # [T/4, d_enc]
3. spliced = frame_splice(enc, factor=2)                           # [T/8, 2·d_enc]
4. speech_emb = Linear(768→3584) ∘ ReLU ∘ Linear(2·d_enc→768)(spliced)   # [T/8, 3584]
5. Build Qwen2 chat prompt with N placeholder tokens (N = T/8).
6. prompt_ids = qwen_tokenizer(prompt_text)
7. inputs_embeds = Qwen2.embed_tokens(prompt_ids)
8. inputs_embeds = _merge_input_ids_with_speech_features(inputs_embeds, speech_emb)
9. Run Qwen2-7B-Instruct generate() with KV cache, beam or greedy:
       generation_config: max_new_tokens=..., repetition_penalty=3.0, length_penalty=0.6
10. transcript = qwen_tokenizer.decode(generated_ids, skip_special_tokens=True)
```

The Qwen2 forward path is exactly the standard Qwen2-7B forward — `SharpInference.Audio` should hand this off to `dotLLM`'s Qwen2 implementation rather than reimplement it.

### 7. Streaming

**SenseVoice** is fundamentally **non-streaming**: the CTC head sees the full encoder output and emits all tokens in parallel. The `forward_chunk` / KV-cache code path in `MultiHeadedAttentionSANM` exists but is unused by the public checkpoint. Practical streaming is done VAD-segmented or "pseudo-streaming" Whisper-style: chunk audio into ~10–30 s windows (FunASR pairs it with `fsmn-vad`, `max_single_segment_time=30000 ms`), run the full encoder per chunk, and merge transcripts.

**FireRedASR-AED-L** is non-streaming by design (60 s max utterance, full encoder forward, full AR decode). Same VAD-chunking approach applies.

**FireRedASR-LLM-L** is non-streaming and also has a hard **30 s** input limit because the speech embeddings have to fit into the Qwen2 prompt at the chosen frame rate (`30 s / 80 ms = 375 speech tokens`).

For all three, a real-time pipeline = `VAD → chunk → SenseVoice/FireRedASR encode+decode → concatenate`. None of them give true token-level streaming.

### 8. Benchmarks / Comparison

#### 8.1 SenseVoice-Small vs Whisper

From the FunAudioLLM paper and the model card:

| Benchmark | Metric | Whisper-Large-v3 | SenseVoice-Small | Notes |
|-----------|--------|------------------|------------------|-------|
| AISHELL-1 (zh) | CER | ~2.7 | **~2.0** | SenseVoice wins on zh |
| AISHELL-2 (zh) | CER | ~3.0 | **~2.8** | SenseVoice wins on zh |
| WenetSpeech (zh) | CER | ~5.5 | **~5.0** | SenseVoice wins on zh |
| Common Voice yue | CER | 38.97 (Whisper-Small) | **much lower** | SenseVoice dominates Cantonese |
| LibriSpeech (en) | WER | ~2.0 | ~3-4 | Whisper still wins on English |
| Japanese / Korean | CER | better | worse | Whisper wins JA/KO |
| Inference time (10 s audio, GPU) | ms | ~1050 | **~70** | 15× speedup |

#### 8.2 FireRedASR vs Whisper / Paraformer / Seed-ASR (Chinese)

From arXiv:2501.14350 Table 2 — average CER over { AISHELL-1, AISHELL-2, WenetSpeech-Net, WenetSpeech-Meeting }:

| Model | Params | Avg public Mandarin CER (%) |
|-------|--------|------------------------------|
| Whisper-Large-v3 | 1.55 B | ~7.7 |
| Paraformer-Large-v2 | ~220 M | ~4.5 |
| Qwen-Audio | ~8 B | ~4.0 |
| Seed-ASR (ByteDance, prior SOTA) | undisclosed | 3.33 |
| **FireRedASR-AED-L** | **1.1 B** | **3.18** |
| **FireRedASR-LLM-L** | **8.3 B** | **3.05**  ← new SOTA |

Per-set numbers for FireRedASR (CER %, lower = better):

| Set | AED-L | LLM-L |
|-----|-------|-------|
| AISHELL-1 | 0.55 | 0.76 |
| AISHELL-2 | 2.52 | 2.15 |
| WenetSpeech-Net | 4.88 | ~4.6 |
| WenetSpeech-Meeting | 4.76 | ~4.6 |

Notes:
- AED-L is **better than LLM-L on AISHELL-1** (clean read speech) — the LLM bias toward fluent rewriting hurts perfect-transcription tasks.
- LLM-L wins on the harder, more conversational sets.
- On the 19-set internal dialect benchmark: ~11.55 % (LLM) / 11.67 % (AED) avg CER.
- On singing lyrics: 50–67 % relative CERR vs industrial baselines.

### 9. C# Implementation Notes (SharpInference.Audio)

#### 9.1 Shared infrastructure to build first

- **Kaldi-native-fbank in pure C#**: `Audio.Frontend.KaldiFbank` — Hamming window, dither, log-mel filterbank, snip-edges. Used by both SenseVoice and FireRedASR (and we already need it for Paraformer/SenseVoice/whisper.cpp parity). Validate within 1e-3 of `kaldi-native-fbank` output on a fixed test waveform.
- **CMVN loader**: parse FunASR `am.mvn` and FireRedASR `cmvn.ark` formats; apply per-feature `(x - mean) * scale`. Two formats, one C# struct.
- **SentencePiece runtime (BPE only)**: load the `.model` protobuf, run encode/decode. Existing dotLLM SentencePiece works for Qwen2 tokenizer (LLM variant). The SenseVoice multilingual model and the FireRedASR `train_bpe1000.model` are vanilla SentencePiece — same loader.
- **Special-token table**: small static struct mapping `id → "<|...|>"` strings for the SenseVoice 4-prefix slots.

#### 9.2 SenseVoice-Small specifics

- **No Conformer needed for SenseVoice** — the encoder is `SANMEncoder`, not Conformer. Don't try to share the Conformer block we build for Parakeet/FireRedASR; build a separate `SANMEncoderLayer` class:
  - Pre-norm LayerNorm.
  - `MultiHeadedAttentionSANM(n_head=4, n_feat=512, kernel_size=11)` = standard MHA fused with parallel depthwise Conv1D (kernel 11, groups=512) along time; sum the two outputs before the residual.
  - `PositionwiseFeedForward(512→2048→512, ReLU)`.
- **Encoder is 50 layers deep but only 512 wide** — fits comfortably in fp16 on any modern GPU (param budget ≈ 50 × (4·512² + 2·512·2048 + 512·11) ≈ 130 M for encoder; plus 25 055 × 512 ≈ 13 M CTC head; plus embeddings; total ~234 M).
- **LFR preprocessing**: must be in pure C# in the frontend, before encoder. Trivially vectorizable.
- **CTC decode**: argmax + run-length-encode + blank-drop. One kernel, no beam search needed.
- **Special-prefix split**: just slice the first 4 logit rows separately.
- **Checkpoint source format**: PyTorch `.pt` from HF. We'll need a one-time converter to our tensor layout (see existing dotLLM converter pattern). Tensor naming follows FunASR convention: `encoder.encoders.{i}.self_attn.linear_q_k_v.weight`, `encoder.encoders.{i}.self_attn.fsmn_block.weight`, etc.

#### 9.3 FireRedASR-AED-L specifics

- **Reuses the Conformer encoder we're building for Parakeet** (same Macaron-FF / MHSA / GLU-Conv / Macaron-FF block structure, same Conv2dSubsampling ×4 frontend). Difference: FireRedASR uses **relative positional encoding** in MHSA (not rotary) and **depthwise kernel 33** (Parakeet uses 31). Make these configurable parameters on the shared block.
- **Decoder is a vanilla Transformer decoder** — standard pre-norm self-attn + cross-attn + FFN. Same structure as Whisper decoder; we should reuse the Whisper decoder block from `SharpInference.Audio.Whisper`, just parameterize `vocab_size = 7832`, `kv_cache`, `cross-attn target = encoder_out`.
- **Beam search** with length penalty + repetition penalty — already part of the Whisper decoder pipeline plan; share that infra.
- **Tokenizer**: hybrid Chinese-char + English-BPE decoder. Easiest implementation: keep a flat `int[] -> string` table of size 7 832 built once from `dict.txt + train_bpe1000.model`. No need to keep the SentencePiece runtime live at inference time (encode path only matters for training / not needed for inference). Decode: lookup, then on the English-BPE pieces apply standard "▁ = space, merge subwords" rule.
- **Audio limit 60 s** — enforce in the API or chunk.

#### 9.4 FireRedASR-LLM-L specifics

- **Encoder = identical to AED-L's encoder** — load AED-L encoder weights into the LLM-L encoder slot. Same C# code path.
- **Adapter is trivial**: `frame_splice` (reshape `[T,2D] → [T/2, 4D]`... actually concatenate pairs: `[T,D] → [T/2, 2D]`) then 2× Linear with ReLU between. ~30 M params.
- **Decoder = Qwen2-7B-Instruct** — **delegate to dotLLM**. We do NOT reimplement Qwen2 inside SharpInference. Define a `SharpInference.Audio.ILlmDecoder` interface, have the FireRedASR-LLM pipeline call into a `dotLLM.Qwen2Model` instance through that interface. The interface needs:
  - `Embed(int[] tokenIds) → Tensor[seq, hidden]`  (so we can build the embedding sequence ourselves and then splice in speech_emb at the placeholder positions).
  - `Forward(Tensor inputsEmbeds, KVCache cache) → Tensor logits`.
  - `Generate(...)` with the standard Qwen2 generation config.
- **Hidden dim coupling**: the adapter's final Linear output dim **must equal** the dotLLM-loaded Qwen2 model's `hidden_size` (3 584). Assert at model-load time.
- **Audio limit 30 s** (375 speech tokens at 80 ms) — enforce in API; chunk longer audio.
- **Prompt template**: Qwen2 chat format with a fixed user instruction `"请转写音频内容"` and `<|im_start|>/<|im_end|>` framing. Templated string in C#, no template engine needed.

#### 9.5 Package boundaries

- `SharpInference.Audio` — frontend (fbank + LFR + CMVN), SenseVoice model, FireRedASR-AED model, FireRedASR-LLM **pipeline orchestration**.
- `SharpInference.Audio` depends on `SharpInference.Core` (tensors, CUDA) and — for FireRedASR-LLM only — on `dotLLM` (Qwen2 decoder). That dotLLM dependency must be a **soft / optional** package reference so users who only want SenseVoice or FireRedASR-AED don't pay for it. Suggested split: `SharpInference.Audio.FireRedLlm` as a separate small package that pulls in `dotLLM`, while base `SharpInference.Audio` covers SenseVoice + FireRedASR-AED.
- Tokenizers: SentencePiece runtime already needed by dotLLM; promote it to a shared `SharpInference.Tokenizers` package (or reuse dotLLM's) to avoid duplication.

#### 9.6 Validation targets

Per the project rule "validate against references within documented tolerances":

| Module | Reference | Tolerance |
|--------|-----------|-----------|
| KaldiFbank | `kaldi-native-fbank` Python on a fixed 5 s clip | max abs err < 1e-3 per bin |
| LFR + CMVN | FunASR `WavFrontend` output | bit-exact (deterministic) |
| SANMEncoder forward (fp32) | FunASR `SenseVoiceSmall` forward | per-frame CTC logits, max abs err < 1e-2 |
| SenseVoice end-to-end | `funasr.AutoModel.generate` | 100 % token-id match on AISHELL-1 dev (first 100 clips) |
| ConformerEncoder forward | `FireRedASR` Python `ConformerEncoder.forward` | per-frame, max abs err < 1e-2 |
| FireRedASR-AED end-to-end | upstream Python `aed_inference.py` | exact transcript match on AISHELL-1 dev (first 100 clips) |
| FireRedASR-LLM end-to-end | upstream Python `llm_inference.py` | transcript match modulo tokenization whitespace |

All references are open-source Python; capture them once in a `tests/reference-outputs/` fixture dir to make CI deterministic.
