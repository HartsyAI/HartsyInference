# SenseVoice + FireRedASR — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (SenseVoice + FireRedASR pipelines)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

This document covers two strong Chinese / multilingual STT model families that we plan to wrap as pure-C# pipelines under `HartsyInference.Audio`:

- **SenseVoice** (Alibaba FunAudioLLM, July 2024) — non-autoregressive, CTC-based, encoder-only "speech understanding" model. The released **SenseVoice-Small** variant matches Whisper-Large-v3 quality on Chinese / Cantonese while running 15× faster (≈70 ms for 10 s of audio). One forward pass emits a 4-part output: `<emotion><language><event><text>`. Encoder uses **SAN-M** (Self-Attention with Memory) — vanilla multi-head attention augmented with an FSMN depthwise-conv "memory" block.
- **FireRedASR** (Xiaohongshu / FireRedTeam, Jan 2025) — industrial-grade Mandarin + dialect + English ASR. Two variants: **AED-L** (1.1 B params, Conformer encoder + Transformer decoder, Whisper-style autoregressive decode) and **LLM-L** (8.3 B params, same Conformer encoder + Linear-ReLU-Linear adapter + Qwen2-7B-Instruct decoder). Both share the same audio frontend (80-bin log-mel, kaldi fbank, CMVN, no LFR) and the same `train_bpe1000` SentencePiece tokenizer (7 832 entries: 1 000 English BPE + 6 827 Chinese chars + 5 special tokens). FireRedASR-LLM-L holds SOTA on Mandarin benchmarks (avg CER 3.05 % across AISHELL-1/2, WenetSpeech-Net/Meeting).

Mel preprocessing for both families is the standard 16 kHz / 80-bin / 25 ms-window / 10 ms-hop kaldi-native-fbank with CMVN. SenseVoice additionally applies LFR (Low Frame Rate) frame stacking with `lfr_m=7, lfr_n=6` — i.e. stack 7 consecutive frames and step by 6 — yielding a 560-dim input vector at a 60 ms frame rate before linear projection to `d_model`. FireRedASR does **not** apply LFR; downsampling happens inside the encoder via a Conv2dSubsampling stack (stride-2 ×2 → 4× downsample → 40 ms frames; FireRedASR-LLM further frame-splices ×2 inside the adapter → 80 ms frames).

Sources:
- SenseVoice: [GitHub FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice), [model.py](https://github.com/FunAudioLLM/SenseVoice/blob/main/model.py), [HuggingFace FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall), [FunAudioLLM paper arXiv:2407.04051](https://arxiv.org/abs/2407.04051), [SenseVoice.cpp port](https://github.com/lovemefan/SenseVoice.cpp), [DeepWiki FunASR SenseVoice page](https://deepwiki.com/modelscope/FunASR/5.2-sensevoice), [sherpa-onnx SenseVoice docs](https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html)
- FireRedASR: [GitHub FireRedTeam/FireRedASR](https://github.com/FireRedTeam/FireRedASR), [FireRedASR paper arXiv:2501.14350](https://arxiv.org/abs/2501.14350), [HuggingFace FireRedASR-AED-L](https://huggingface.co/FireRedTeam/FireRedASR-AED-L), [HuggingFace FireRedASR-LLM-L](https://huggingface.co/FireRedTeam/FireRedASR-LLM-L), [literature review (themoonlight.io)](https://www.themoonlight.io/en/review/fireredasr-open-source-industrial-grade-mandarin-speech-recognition-models-from-encoder-decoder-to-llm-integration), [FireRedASR2 (follow-up) arXiv:2603.10420](https://arxiv.org/abs/2603.10420)

## Variants Tables

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

## Multilingual / Output Format / Supported Tags

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

## Tokenizer

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

## Benchmarks / Comparison

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

## C# Implementation Notes (HartsyInference.Audio)

#### 9.1 Shared infrastructure to build first

- **Kaldi-native-fbank in pure C#**: `Audio.Frontend.KaldiFbank` — Hamming window, dither, log-mel filterbank, snip-edges. Used by both SenseVoice and FireRedASR (and we already need it for Paraformer/SenseVoice/whisper.cpp parity). Validate within 1e-3 of `kaldi-native-fbank` output on a fixed test waveform.
- **CMVN loader**: parse FunASR `am.mvn` and FireRedASR `cmvn.ark` formats; apply per-feature `(x - mean) * scale`. Two formats, one C# struct.
- **SentencePiece runtime (BPE only)**: load the `.model` protobuf, run encode/decode. Existing HartsyInference.LLM SentencePiece works for Qwen2 tokenizer (LLM variant). The SenseVoice multilingual model and the FireRedASR `train_bpe1000.model` are vanilla SentencePiece — same loader.
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
- **Checkpoint source format**: PyTorch `.pt` from HF. We'll need a one-time converter to our tensor layout (see existing HartsyInference.LLM converter pattern). Tensor naming follows FunASR convention: `encoder.encoders.{i}.self_attn.linear_q_k_v.weight`, `encoder.encoders.{i}.self_attn.fsmn_block.weight`, etc.

#### 9.3 FireRedASR-AED-L specifics

- **Reuses the Conformer encoder we're building for Parakeet** (same Macaron-FF / MHSA / GLU-Conv / Macaron-FF block structure, same Conv2dSubsampling ×4 frontend). Difference: FireRedASR uses **relative positional encoding** in MHSA (not rotary) and **depthwise kernel 33** (Parakeet uses 31). Make these configurable parameters on the shared block.
- **Decoder is a vanilla Transformer decoder** — standard pre-norm self-attn + cross-attn + FFN. Same structure as Whisper decoder; we should reuse the Whisper decoder block from `HartsyInference.Audio.Whisper`, just parameterize `vocab_size = 7832`, `kv_cache`, `cross-attn target = encoder_out`.
- **Beam search** with length penalty + repetition penalty — already part of the Whisper decoder pipeline plan; share that infra.
- **Tokenizer**: hybrid Chinese-char + English-BPE decoder. Easiest implementation: keep a flat `int[] -> string` table of size 7 832 built once from `dict.txt + train_bpe1000.model`. No need to keep the SentencePiece runtime live at inference time (encode path only matters for training / not needed for inference). Decode: lookup, then on the English-BPE pieces apply standard "▁ = space, merge subwords" rule.
- **Audio limit 60 s** — enforce in the API or chunk.

#### 9.4 FireRedASR-LLM-L specifics

- **Encoder = identical to AED-L's encoder** — load AED-L encoder weights into the LLM-L encoder slot. Same C# code path.
- **Adapter is trivial**: `frame_splice` (reshape `[T,2D] → [T/2, 4D]`... actually concatenate pairs: `[T,D] → [T/2, 2D]`) then 2× Linear with ReLU between. ~30 M params.
- **Decoder = Qwen2-7B-Instruct** — **delegate to HartsyInference.LLM**. We do NOT reimplement Qwen2 inside HartsyInference. Define a `HartsyInference.Audio.ILlmDecoder` interface, have the FireRedASR-LLM pipeline call into a `HartsyInference.LLM.Qwen2Model` instance through that interface. The interface needs:
  - `Embed(int[] tokenIds) → Tensor[seq, hidden]`  (so we can build the embedding sequence ourselves and then splice in speech_emb at the placeholder positions).
  - `Forward(Tensor inputsEmbeds, KVCache cache) → Tensor logits`.
  - `Generate(...)` with the standard Qwen2 generation config.
- **Hidden dim coupling**: the adapter's final Linear output dim **must equal** the HartsyInference.LLM-loaded Qwen2 model's `hidden_size` (3 584). Assert at model-load time.
- **Audio limit 30 s** (375 speech tokens at 80 ms) — enforce in API; chunk longer audio.
- **Prompt template**: Qwen2 chat format with a fixed user instruction `"请转写音频内容"` and `<|im_start|>/<|im_end|>` framing. Templated string in C#, no template engine needed.

#### 9.5 Package boundaries

- `HartsyInference.Audio` — frontend (fbank + LFR + CMVN), SenseVoice model, FireRedASR-AED model, FireRedASR-LLM **pipeline orchestration**.
- `HartsyInference.Audio` depends on `HartsyInference.Core` (tensors, CUDA) and — for FireRedASR-LLM only — on `HartsyInference.LLM` (Qwen2 decoder). That HartsyInference.LLM dependency must be a **soft / optional** package reference so users who only want SenseVoice or FireRedASR-AED don't pay for it. Suggested split: `HartsyInference.Audio.FireRedLlm` as a separate small package that pulls in `HartsyInference.LLM`, while base `HartsyInference.Audio` covers SenseVoice + FireRedASR-AED.
- Tokenizers: SentencePiece runtime already needed by HartsyInference.LLM; promote it to a shared `HartsyInference.ModelAssets.Tokenizers` package (or reuse HartsyInference.LLM's) to avoid duplication.

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
