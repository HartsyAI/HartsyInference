# NVIDIA Parakeet (CTC / RNN-T / TDT) — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Parakeet pipeline)

## Summary

NVIDIA Parakeet is a family of ASR models that pair a **FastConformer encoder** ([arXiv:2305.05084](https://arxiv.org/abs/2305.05084)) with one of three decoder heads: **CTC** (single linear projection), **RNN-T** (LSTM prediction net + joint), or **TDT** — Token-and-Duration Transducer ([arXiv:2304.06795](https://arxiv.org/abs/2304.06795)). All variants share the same encoder, mel-spectrogram preprocessor, and SentencePiece BPE/Unigram tokenizer; only the head differs. The flagship **Parakeet-TDT-0.6B-v2** (the XL FastConformer variant, 24 layers @ 1024 d_model) achieves 1.69% / 3.19% WER on LibriSpeech test-clean / test-other and an **RTFx ~3,386** on the Hugging Face Open ASR Leaderboard at batch size 128 — i.e. it transcribes ~3,000x faster than real-time on a single A100 while matching Whisper-large-v3 English quality. The newer **Parakeet-TDT-0.6B-v3** (Aug 2025) extends the family to **25 European languages** at WER 6.34% / RTFx 3,333 with a 8192-vocab unified tokenizer trained on the 670k-hour Granary dataset.

The architectural value of TDT is the **duration head**: instead of consuming one encoder frame per joint-network call (RNN-T) and emitting a sea of blanks, the joint network outputs `[V + 1 + D]` logits — vocab + blank + a duration distribution over `{0, 1, 2, 3, 4}` — and the decoder advances `duration` frames per emission. This collapses the inference loop down to roughly one joint call per emitted token rather than one per encoder frame, giving TDT a ~2.8x speedup over standard RNN-T at equal or better WER.

This file covers all three Parakeet variants. Mel preprocessing math is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) (the "Parakeet/Canary" column). Cache-aware streaming details are cross-referenced to [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) (to be written). Comparison to encoder-decoder transformer ASR is in [WHISPER_ARCHITECTURE.md](WHISPER_ARCHITECTURE.md).

Sources: [FastConformer paper](https://arxiv.org/abs/2305.05084), [TDT paper](https://arxiv.org/abs/2304.06795), [NVIDIA NeMo repo](https://github.com/NVIDIA-NeMo/NeMo), [Parakeet-TDT-0.6B-v2 HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [Parakeet-TDT-0.6B-v3 HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [Parakeet-TDT-1.1B HF](https://huggingface.co/nvidia/parakeet-tdt-1.1b), [Parakeet-CTC-1.1B HF](https://huggingface.co/nvidia/parakeet-ctc-1.1b), [Parakeet-CTC-0.6B HF](https://huggingface.co/nvidia/parakeet-ctc-0.6b), [Parakeet-RNNT-1.1B HF](https://huggingface.co/nvidia/parakeet-rnnt-1.1b), [Parakeet-RNNT-0.6B HF](https://huggingface.co/nvidia/parakeet-rnnt-0.6b), [Open ASR Leaderboard paper](https://arxiv.org/abs/2510.06961), [Cache-aware Conformer Streaming](https://arxiv.org/abs/2312.17279), [NeMo Parakeet deep-dive (qed42)](https://www.qed42.com/insights/nvidia-parakeet-tdt-0-6b-v2-a-deep-dive-into-state-of-the-art-speech-recognition-architecture), [Speechmatics TDT explainer](https://www.speechmatics.com/company/articles-and-news/token-duration-transducer-tdt-explained).

## Detailed Findings

### 1. Parakeet Variants Table

All numbers below are confirmed from each model's official HF card. WER is reported on LibriSpeech (LS). "Average WER" is the 8-dataset mean used on the Hugging Face Open ASR Leaderboard (LibriSpeech clean+other, AMI, Earnings-22, GigaSpeech, TEDLIUM-v3, VoxPopuli, CommonVoice). RTFx is measured on A100 80GB; higher is faster.

| Model (HF path) | Head | FC variant | Params | Vocab / Tokenizer | Languages | LS-clean | LS-other | Avg WER | RTFx | Release |
|---|---|---|---|---|---|---|---|---|---|---|
| `nvidia/parakeet-ctc-0.6b` | CTC | Large | 0.6B | 1024 SentencePiece Unigram | English | 1.87 | 3.76 | 7.69 | 4,282 | 2023 |
| `nvidia/parakeet-ctc-1.1b` | CTC | XXL | 1.1B | 1024 SP Unigram | English | 1.83 | 3.54 | 7.40 | 2,729 | 2023 |
| `nvidia/parakeet-rnnt-0.6b` | RNN-T | Large | 0.6B | 1024 SP Unigram | English | 1.63 | 3.06 | — | — | 2023 |
| `nvidia/parakeet-rnnt-1.1b` | RNN-T | XXL | 1.1B | 1024 SP Unigram | English | 1.46 | 2.47 | 7.12 | 2,053 | 2023 |
| `nvidia/parakeet-tdt-1.1b` | TDT | XXL | 1.1B | 1024 SP Unigram | English (lowercase) | 1.39 | 2.62 | 7.02 | 2,391 | 2024 |
| `nvidia/parakeet-tdt-0.6b-v2` | TDT | **XL** | 0.6B | ~1024 SP BPE | English (+ punct/cap) | **1.69** | **3.19** | **6.05** | **3,386** | May 2025 |
| `nvidia/parakeet-tdt-0.6b-v3` | TDT | XL | 0.6B | **8192** SP Unigram | **25 EU languages** | 1.93 | 3.59 | 6.34 | 3,333 | Aug 2025 |

**File sizes** (approximate, FP32 `.nemo` archives): 0.6B ≈ 2.4 GB, 1.1B ≈ 4.5 GB. HF mirrors `parakeet-tdt-0.6b-v2/.nemo` is ~2.4 GB; FP16 safetensors mirrors (e.g. the MLX / ONNX community ports) cut this in half.

**Note on the v2 vs v3 split**: v2 is English-only with mixed-case + punctuation output and was trained on the new **Granary** corpus (120k hours, 10k human + 110k pseudo-labeled). v3 reuses the same XL encoder but swaps in a unified 8192-token multilingual tokenizer and was fine-tuned across the 25 EU languages from Granary's 670k-hour multilingual subset.

**FC variant** column maps to the FastConformer model-size table (see §2): Large = 17 layers / 512 d_model / 8 heads / 120M-ish (encoder only); XL = 24 layers / 1024 d_model / 8 heads / 616M; XXL = 42 layers / 1024 d_model / 8 heads / 1.2B. The Parakeet "0.6B" name refers to total params (encoder + small decoder head); the 1.1B ditto.

### 2. FastConformer Encoder (shared across CTC / RNN-T / TDT)

The encoder is identical across all Parakeet variants — only the decoder head changes. The reference YAML is [`fast-conformer_transducer_bpe.yaml`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_transducer_bpe.yaml).

#### 2.1 Model-size table (canonical, from NeMo YAML)

| Variant | d_model | n_heads | head_dim | n_layers | FFN dim | Encoder params |
|---|---|---|---|---|---|---|
| Small | 176 | 4 | 44 | 16 | 704 | ~14M |
| Medium | 256 | 4 | 64 | 16 | 1024 | ~32M |
| **Large** | **512** | **8** | **64** | **17** | **2048** | **~120M** |
| **XL** | **1024** | **8** | **128** | **24** | **4096** | **~616M** |
| **XXL** | **1024** | **8** | **128** | **42** | **4096** | **~1.2B** |

FFN dim = `d_model * ff_expansion_factor` with `ff_expansion_factor = 4`. Head dim = `d_model / n_heads`.

Parakeet-TDT-0.6B-v2/v3 = **XL** (24 layers, 1024 dim). Parakeet-*-1.1B = **XXL** (42 layers, 1024 dim). Parakeet-*-0.6B (the older v1 generation) = **Large** with a wider decoder; the "0.6B" label there comes from a fattened decoder/joint rather than encoder.

#### 2.2 Convolutional subsampling (8x)

Input: log-mel `[B, 80, T]` at 16 kHz with `hop=160` (= 10 ms per mel frame, ≈100 Hz).

NeMo's `dw_striding` subsampler stacks **three** depthwise-separable Conv2D blocks, each with stride `(2, 2)` on `(time, freq)`. Stack of three strides multiplicatively gives `2 * 2 * 2 = 8x` time reduction and `2 * 2 * 2 = 8x` freq reduction. Channels are reduced from the historical 512 down to **256** (key FastConformer change vs vanilla Conformer; this is the main MAC saving).

```
Input log-mel:   [B, 1, 80, T]
Conv2D 1 (standard, 3x3, stride (2,2), in=1, out=256):  [B, 256, 40, T/2]
Activation: ReLU
Conv2D 2 (depthwise-separable 3x3 stride (2,2), 256->256): [B, 256, 20, T/4]
Activation: ReLU
Conv2D 3 (depthwise-separable 3x3 stride (2,2), 256->256): [B, 256, 10, T/8]
Activation: ReLU
Flatten freq into channels: [B, 256*10, T/8] = [B, 2560, T/8]
Linear projection to d_model: [B, d_model, T/8]
Transpose: [B, T/8, d_model]
```

**Effective frame rate after subsampling**: 100 Hz input mel → **12.5 Hz encoder frames** (one encoder vector per 80 ms of audio). This is the rate the decoder loop iterates over.

> Note: the original docs sometimes phrase it as "80 Hz mel → 10 Hz encoder" (Whisper-style 50 ms hop), but Parakeet's preprocessor explicitly uses `window_stride=0.01` → 100 Hz mel → 12.5 Hz encoder. Use 80 ms / frame as the planning constant.

`Conv2D 1` is a regular (non-separable) conv. `Conv2D 2/3` use the depthwise-separable factorisation: a depthwise `(3x3, groups=C)` followed by a pointwise `(1x1)` to mix channels. This is the change vs. the original Conformer's `vggnet`/`striding` subsamplers (which used regular Conv2D and 4x reduction).

#### 2.3 FastConformer block (one of `n_layers` identical blocks)

The block keeps Conformer's **Macaron-FFN sandwich** structure with **half-step residuals** but reduces the depthwise conv kernel from **31 → 9**:

```
x_in = input [B, T', d_model]

# (1) Macaron FFN1
y = LayerNorm(x_in)
y = FFN(y)                # Linear(d, 4d) -> Swish -> Dropout -> Linear(4d, d) -> Dropout
x = x_in + 0.5 * y        # half-step residual

# (2) Multi-Head Self-Attention with relative positions
y = LayerNorm(x)
y = RelPosMHA(y)          # Transformer-XL style, see §2.4
x = x + Dropout(y)

# (3) Convolution module
y = LayerNorm(x)
y = ConvModule(y)         # see §2.5
x = x + Dropout(y)

# (4) Macaron FFN2
y = LayerNorm(x)
y = FFN(y)
x = x + 0.5 * y

# (5) Final LayerNorm
x_out = LayerNorm(x)
```

All Linear and Conv layers use bias. FFN inner activation is **Swish** (= SiLU). Dropout = 0.1 throughout (attention dropout also 0.1). The `0.5` factor on FFN residuals is the Macaron half-step from the original Conformer paper.

Source: [`nemo/collections/asr/parts/submodules/conformer_modules.py`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/conformer_modules.py) — `ConformerLayer.forward`.

#### 2.4 Relative-position multi-head self-attention (Transformer-XL style)

Set via `self_attention_model: rel_pos` in the YAML. This is the Dai et al. 2019 Transformer-XL formulation (not RoPE):

```
Given Q, K, V in [B, H, T, d_h]:
  Q = q_proj(x)        # Linear(d, d), bias=True
  K = k_proj(x)        # Linear(d, d), bias=True
  V = v_proj(x)        # Linear(d, d), bias=True
  R = pos_proj(P)      # Linear(d, d), bias=False  -- positional embeddings

  # Two learnable biases u, v of shape [H, d_h]
  AC = (Q + u) @ K^T              # content score
  BD = (Q + v) @ R^T              # position score
  BD = relative_shift(BD)         # Dai et al. shift trick
  attn = softmax((AC + BD) / sqrt(d_h)) @ V
```

Positional embeddings `P[i]` are sinusoidal with `pos_emb_max_len=5000` and `xscaling=True` (input is scaled by `sqrt(d_model)` like the original Transformer). Bias terms `u`, `v` are learned parameters per head.

**Attention window** (`att_context_size`):
- Offline models (Parakeet-TDT-0.6B-v2, all v1 variants): `[-1, -1]` = full self-attention across the chunked encoder sequence. Memory scales O(T²); for a 24-min audio at 12.5 Hz that's 18,000 frames → ~330M attention entries per head. Manageable on A100 80GB; on smaller GPUs you must chunk.
- Streaming / cache-aware models (`stt_en_fastconformer_hybrid_large_streaming_multi`, `nemotron-speech-streaming-en-0.6b`): trained with a *list* of `att_context_size` values to support multiple latencies in one model. Standard values are `[70, 0]`, `[70, 1]`, `[70, 6]`, `[70, 13]` (left context = 70 frames = 5.6 s, right context = 0/1/6/13 frames = 0/80/480/1040 ms look-ahead). Switchable at inference with `change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])`.
- Local-attention mode for very long audio: `rel_pos_local_attn` with symmetric chunks (Longformer-style sliding window). Lets the XL model do up to ~3 h of audio in one pass on an A100.

#### 2.5 ConvolutionModule (the Conformer C-module)

```
x: [B, T', d_model]
y = LayerNorm(x)
y = transpose(y) -> [B, d_model, T']
y = pointwise_conv1d(d, 2d)        # 1x1, expands to 2*d
y = GLU(y, dim=1)                  # gated linear unit, output [B, d, T']
y = depthwise_conv1d(d, d,
                     kernel=9,     # FastConformer key change: 9 not 31
                     groups=d,
                     padding=4)    # SAME padding (causal in streaming variants)
y = BatchNorm1d(y)                 # offline; LayerNorm for streaming variants
y = Swish(y)
y = pointwise_conv1d(d, d)         # 1x1
y = Dropout(y)
return transpose(y) -> [B, T', d_model]
```

`conv_norm_type: batch_norm` is the default. Streaming/cache-aware variants use `layer_norm` because BatchNorm running stats break under chunked inference. The depthwise kernel size of **9** (vs the original Conformer's 31) is the second main MAC saving alongside the 256-channel subsampler.

Source: [`ConformerConvolution` class in conformer_modules.py](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/conformer_modules.py).

#### 2.6 Encoder forward summary

```
log_mel [B, 80, T] (100 Hz)
  -> Subsampler (3 strided Conv2D) -> [B, T/8, d_model] (12.5 Hz)
  -> Add scaled embedding (xscaling), prepare relative-pos buffer
  -> N x FastConformerBlock (each: Macaron-FFN, RelPosMHA, ConvModule, Macaron-FFN, LayerNorm)
  -> final_ln output [B, T_enc, d_model]
```

Output sequence length `T_enc = floor(T / 8)`. `d_model` is 512 / 1024 depending on Large/XL/XXL.

### 3. CTC Head (Parakeet-CTC)

Trivial. A single linear projection from the encoder output to vocabulary logits:

```
W_ctc: Linear(d_model, vocab_size + 1)   # +1 = blank token
logits = W_ctc(encoder_out)              # [B, T_enc, V+1]
log_probs = log_softmax(logits, dim=-1)
```

**Blank index**: the last position, `V` (i.e. `vocab_size`). For Parakeet-CTC-1.1B with 1024 vocab tokens, blank = 1024 and the output has 1025 channels.

**Decoding** — two options:

- **Greedy (collapse-and-dedupe)**: for each frame take `argmax`, then merge consecutive duplicates, then drop blanks. This is what the HF Open ASR Leaderboard runs Parakeet-CTC under to get RTFx > 2700.

  ```
  preds = argmax(log_probs, dim=-1)             # [B, T_enc]
  out = []
  prev = BLANK
  for p in preds:
      if p != prev and p != BLANK:
          out.append(p)
      prev = p
  ```

- **Beam search** (optional): standard CTC prefix beam search with language model rescoring. Marginal improvement on Parakeet's clean benchmarks; mostly useful for noisy / OOV settings.

**Implementation note**: the C# CTC head is `Linear -> LogSoftmax -> argmax -> collapse-dedupe-drop-blank`. No state, no LSTM, no joint network — three matmuls plus a scan. Build this first to validate the encoder end-to-end.

### 4. RNN-T Head (Parakeet-RNNT)

Standard Graves 2012 RNN-Transducer with an LSTM prediction network.

#### 4.1 Prediction Network ("decoder" in NeMo terms)

```
embed   = Embedding(V + 1, pred_hidden)   # +1 = blank/SOS, shared index
lstm    = LSTM(input=pred_hidden,
               hidden=pred_hidden,
               num_layers=pred_rnn_layers,
               batch_first=True)
optional projection: Linear(pred_hidden, pred_out) if pred_out != pred_hidden
```

Reference YAML defaults: `pred_hidden=640`, `pred_rnn_layers=1`. State = `(h, c)` each `[L, B, pred_hidden]`. SOS is the blank index by convention; embedding[blank] is the initial input.

For 1024 vocab and `pred_hidden=640`: embed = `1025 * 640` = 656k params; LSTM (1 layer, hidden=640): `4 * (640 + 640) * 640` = ~3.3M params. Very small relative to the 600M-1.2B encoder.

#### 4.2 Joint Network (`RNNTJoint`)

The joint fuses one encoder frame with one prediction step:

```
# enc_out [B, T_enc, d_enc], pred_out [B, U, d_pred]
enc_proj  = Linear(d_enc,  joint_hidden, bias=True)
pred_proj = Linear(d_pred, joint_hidden, bias=True)
out_proj  = Linear(joint_hidden, V + 1, bias=True)

f = enc_proj(enc_out)                       # [B, T_enc, 1, H]
g = pred_proj(pred_out)                     # [B, 1, U, H]
h = activation(f + g)                       # broadcast-add → [B, T_enc, U, H]
                                            # activation = ReLU (Parakeet default)
h = Dropout(h, p=0.2)
logits = out_proj(h)                        # [B, T_enc, U, V+1]
log_probs = log_softmax(logits, dim=-1)
```

Defaults: `joint_hidden=640`, activation `relu`, dropout 0.2.

The training loss is RNN-T loss (forward-backward over the [T_enc, U] lattice with blank-or-emit transitions); inference uses greedy or beam decoding.

Source: [`RNNTJoint.joint_after_projection`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/rnnt.py).

#### 4.3 Greedy RNN-T decoding loop

```
hyp = []
state = None
last_tok = BLANK    # SOS

g, state = predict(last_tok, state)         # one prediction step bootstraps
for t in range(T_enc):
    symbols_added = 0
    while symbols_added < max_symbols_per_step:   # safeguard, e.g. 10
        logits = joint(enc_out[t], g)             # one joint forward
        k = argmax(logits)
        if k == BLANK:
            break                                  # advance frame, keep state
        hyp.append(k)
        g, state = predict(k, state)               # extend prediction net
        last_tok = k
        symbols_added += 1
```

The `max_symbols_per_step` safeguard (NeMo default 10) prevents runaway non-blank loops on a single frame.

### 5. TDT Head (Token-and-Duration Transducer)

The headline architecture for Parakeet-TDT-0.6B-v2 and the SOTA decoder on the Open ASR Leaderboard. Paper: Xu et al. 2023, *Efficient Sequence Transduction by Jointly Predicting Tokens and Durations* ([arXiv:2304.06795](https://arxiv.org/abs/2304.06795)).

#### 5.1 What changes vs RNN-T

The prediction network is **identical** to RNN-T (same LSTM, same pred_hidden=640, same defaults). The only architectural change is in the joint network output and the inference loop:

- The joint network produces **two independent softmax distributions**: one over `V + 1` tokens (incl. blank), one over `D` durations.
- A *duration* is the number of encoder frames the decoder advances after emitting the chosen token.
- Standard duration vocab: `[0, 1, 2, 3, 4]` (D=5). Some configs use `[0, 1, 2, 4, 8]` for longer skips, but the canonical NeMo TDT and Parakeet-TDT configs use `[0..4]`.

#### 5.2 TDT Joint Network

```
# f, g identical to RNN-T joint setup
h          = activation(enc_proj(f) + pred_proj(g))   # [B, T_enc, U, H]
h          = Dropout(h, 0.2)

# Two independent output heads sharing the trunk
token_proj    = Linear(H, V + 1)                       # token logits
duration_proj = Linear(H, D)                           # duration logits
                                                       #   D = len(durations) = 5

token_log_probs    = log_softmax(token_proj(h),    dim=-1)   # [B, T, U, V+1]
duration_log_probs = log_softmax(duration_proj(h), dim=-1)   # [B, T, U, D]
```

Equivalently NeMo concatenates the two heads in one final projection of width `V + 1 + D` and splits at runtime — the *split-then-softmax independently* is the load-bearing detail.

Total joint params added vs RNN-T: `H * D = 640 * 5 = 3200` floats. Negligible.

#### 5.3 TDT loss (training only; for reference)

The RNN-T loss generalises to a "extended transducer loss" where each lattice transition contributes both a token probability and a duration probability:

```
P(emit token y, advance d frames) = P_tok(y) * P_dur(d)
```

Two regularizers from the paper, both implemented in NeMo:

- **Sigma trick** (`tdt_loss_kwargs.sigma`, typical 0.02–0.05): per-transition logit under-normalization. Each transition's contribution to the lattice is multiplied by `exp(-sigma)`. Because paths with more transitions accumulate more `sigma` penalty, the model is biased toward fewer, longer-duration transitions. This is what trains the model to actually *use* the duration head instead of always picking `d=1`.
- **Omega weight** (`tdt_loss_kwargs.omega`, typical 0.1): a small RNN-T loss term added to the TDT loss to stabilise training.

These don't matter at inference; HartsyInference only needs the forward pass and the decode loop.

#### 5.4 TDT greedy decoding loop (the inference algorithm)

```
durations = [0, 1, 2, 3, 4]            # config-supplied vocabulary

t = 0                                  # encoder frame cursor
hyp = []
state = None
last_tok = BLANK
g, state = predict(last_tok, state)    # bootstrap prediction

while t < T_enc:
    token_log_probs, dur_log_probs = joint(enc_out[t], g)   # 2 heads
    k = argmax(token_log_probs)
    d = durations[argmax(dur_log_probs)]

    if k == BLANK:
        # No token emitted at this frame.
        # We still advance by predicted duration.
        # If duration==0 we'd loop forever, so floor to 1.
        t += max(d, 1)
    else:
        hyp.append(k)
        g, state = predict(k, state)
        last_tok = k
        # If a non-blank token has duration 0, treat as 1
        # to avoid the same-frame loop. (NeMo also has a
        # max_symbols_per_step guard that does this implicitly.)
        if d > 0:
            t += d
        # else: stay at frame t and emit another token next iteration
        #       (bounded by max_symbols_per_step)
```

**Key observation**: where RNN-T calls the joint network `T_enc + N_tokens` times (one per frame plus one per emission), TDT calls it roughly `N_tokens + N_blank_segments` times — typically 3-5x fewer joint forwards on speech. That, plus the duration head being identical-cost, is the source of the 2.8x decoder speedup in the TDT paper and the 1.5-3x in Parakeet.

NeMo greedy implementation: `GreedyBatchedTDTLabelLoopingComputer` in [`rnnt_greedy_decoding.py`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py).

#### 5.5 Why TDT is fast on GPU (and why it matters for the C# port)

- The prediction-net LSTM step is the only sequential dependency in the decoder. The joint network forward is a small MLP on `[B, 1, 1, H]` per step; trivially batched.
- The encoder runs **once** per utterance up-front; it has no autoregressive dependency. On A100 the 24-layer XL encoder processes 24 minutes of 16 kHz audio in well under a second.
- The dominant cost on long audio is *encoder* compute, not decoder. Skipping frames in the decoder is essentially free; the speedup comes from doing many fewer prediction-net LSTM rollouts.

For the C# port: a fast LSTM kernel is the critical path. The Conformer pieces (attention, ConvModule, FFN) are all "we already have these" by the time we ship Parakeet. See §11.

### 6. Tokenizer

All English Parakeet models use a **SentencePiece** tokenizer trained on the model's transcript corpus:

| Model family | Type | Vocab | Special tokens |
|---|---|---|---|
| Parakeet-CTC/RNNT-0.6B / 1.1B | SentencePiece **Unigram** | 1,024 | blank (added at runtime, index = vocab_size) |
| Parakeet-TDT-1.1B | SentencePiece Unigram | 1,024 | blank |
| Parakeet-TDT-0.6B-v2 | SentencePiece BPE | ~1,024 | blank; tokenizer emits mixed-case + punctuation directly |
| Parakeet-TDT-0.6B-v3 | SentencePiece Unigram (unified, multilingual) | **8,192** | blank |

The tokenizer model file lives inside the `.nemo` archive as `tokenizer.model` (the SentencePiece binary) plus `vocab.txt` (the human-readable vocab). HartsyInference already needs SentencePiece for other models (T5, etc.); we reuse that loader. Important detail: **no language IDs**, no special start/end-of-transcript tokens. Output is plain token IDs in `[0, V-1]`, blank is `V`.

For v3 (multilingual), there's no language token either — the encoder/decoder identify language implicitly from acoustics + LM context.

### 7. Mel-Spectrogram Preprocessing

Cross-reference [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) — Parakeet uses the **NeMo `AudioToMelSpectrogramPreprocessor`** with the following parameters (from the canonical FastConformer YAMLs):

| Param | Value | Notes |
|---|---|---|
| `sample_rate` | 16,000 | mono 16 kHz |
| `window_size` | 0.025 s | 400 samples |
| `window_stride` | 0.01 s | 160 samples (hop) |
| `window` | `hann` (periodic) | |
| `n_fft` | 512 | |
| `features` (n_mels) | **80** for most models; **128** for the hybrid-TDT-CTC variant and v3 | |
| `lowfreq` (fmin) | 0 | |
| `highfreq` (fmax) | `null` → defaults to `sample_rate / 2` = 8000 | |
| `mag_power` | 2.0 | power spectrogram \|STFT\|² |
| `log` | `True` (natural log, `torch.log`) | clamped with `eps=1e-5` |
| `dither` | 1e-5 (training only) | turn off for inference; 0 ensures determinism |
| `normalize` | `per_feature` | per-utterance, per-mel-bin mean/std |
| `pad_to` | 16 | pad time axis up to multiple of 16 |
| `pad_value` | 0 | |
| Mel scale | librosa Slaney + `norm='slaney'` | librosa.filters.mel |

**`per_feature` normalization mathematics** (load-bearing for parity):

```
For each utterance in the batch independently:
  For each mel bin i in [0, n_mels):
    mean_i = mean(log_mel[i, t] for t in valid_timesteps)
    std_i  = sqrt(var(log_mel[i, t] for t in valid_timesteps) + epsilon)
    log_mel[i, t] = (log_mel[i, t] - mean_i) / std_i
```

This is **per-utterance** mean/std (computed at inference from the input itself, masked by the valid-length mask) — **NOT** stored training-corpus statistics. This matters: Parakeet is robust to gain/SNR shifts precisely because each clip is z-scored individually.

`epsilon` in NeMo's `FilterbankFeatures` is `1e-5`. The sum over `valid_timesteps` uses the audio-length mask so padding doesn't poison the statistics.

Mel filterbank library: librosa `librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000, norm='slaney')`. **Slaney mel scale + Slaney area normalization**. Same as Whisper for the scale and norm; different `n_fft` (Whisper uses 400, Parakeet 512).

### 8. Decoding Loops (cheat sheet)

Pseudo-C# for the three heads. Treat `encOut` as the encoder output `[T_enc, d_model]` for a single utterance (batch dimension elided for clarity).

#### CTC greedy

```csharp
Span<int> preds = stackalloc int[T_enc];
for (int t = 0; t < T_enc; t++)
    preds[t] = Argmax(W_ctc * encOut[t]);   // includes blank at index V

var hyp = new List<int>();
int prev = BLANK;
for (int t = 0; t < T_enc; t++)
{
    int p = preds[t];
    if (p != prev && p != BLANK) hyp.Add(p);
    prev = p;
}
```

#### RNN-T greedy

```csharp
var hyp = new List<int>();
LstmState state = LstmState.Zero(predNet);
Tensor g = predNet.Step(BLANK, ref state);   // bootstrap

for (int t = 0; t < T_enc; t++)
{
    int safety = 0;
    while (safety < MaxSymbolsPerStep)       // typ. 10
    {
        Tensor logits = joint.Forward(encOut[t], g);   // [V+1]
        int k = Argmax(logits);
        if (k == BLANK) break;
        hyp.Add(k);
        g = predNet.Step(k, ref state);
        safety++;
    }
}
```

#### TDT greedy

```csharp
int[] durations = { 0, 1, 2, 3, 4 };

var hyp = new List<int>();
LstmState state = LstmState.Zero(predNet);
Tensor g = predNet.Step(BLANK, ref state);

int t = 0;
while (t < T_enc)
{
    int safety = 0;
    while (safety < MaxSymbolsPerStep)
    {
        (Tensor tokLogits, Tensor durLogits) = joint.ForwardTDT(encOut[t], g);
        int k = Argmax(tokLogits);
        int d = durations[Argmax(durLogits)];

        if (k == BLANK)
        {
            t += Math.Max(d, 1);             // never stay on same frame after blank
            break;
        }

        hyp.Add(k);
        g = predNet.Step(k, ref state);
        safety++;

        if (d > 0) { t += d; break; }
        // d == 0: don't advance frame; emit another token from the same frame
    }

    if (safety == MaxSymbolsPerStep) t++;    // hard advance to avoid livelock
}
```

Build CTC greedy first; it validates the encoder + tokenizer pipeline end-to-end without any prediction-net machinery. Then add TDT (which is the same as RNN-T plus the duration argmax). Beam search comes last and is a notable WER win only on noisy audio with an external LM.

### 9. Streaming Inference

Cross-reference [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) (planned). Parakeet supports genuine streaming through NeMo's **cache-aware Conformer** mechanism ([arXiv:2312.17279](https://arxiv.org/abs/2312.17279)).

**Two changes vs offline FastConformer:**

1. **Limited-context relative attention** (`self_attention_model: rel_pos_local_attn`). Self-attention sees a fixed `[left, right]` window in *encoder frames*. Typical configs:
   - `[70, 13]` = 5.6 s left context, 1040 ms look-ahead (~best WER, highest latency)
   - `[70, 6]` = 5.6 s left, 480 ms look-ahead
   - `[70, 1]` = 5.6 s left, 80 ms look-ahead
   - `[70, 0]` = 5.6 s left, 0 look-ahead (fully causal)
   Multi-latency models are trained with all of these sampled per batch, switchable at inference.

2. **Causal convolution module**. The ConvModule's depthwise conv uses `padding = kernel - 1` on the *left only* (instead of SAME). Combined with a per-chunk cache of the left context for both attention K/V and the conv buffer, inference becomes incremental: feed one chunk, get one chunk of encoder output, retain the cache.

**Per-block streaming state:**
- Conv cache: `[kernel_size - 1, d_model]` per layer.
- Attention K/V cache: `[left_context_size, d_model]` per layer, ring-buffer.

**Step**: ingest a chunk of `chunk_size + right_context` mel frames → run subsampler over the new region → for each block, prepend cached K/V to current chunk's K/V, run windowed attention, run causal conv with prepended conv cache, update caches.

NeMo also publishes a fully-streaming model `nvidia/nemotron-speech-streaming-en-0.6b` that bakes these choices in. For HartsyInference's first Parakeet release, offline TDT is enough; streaming is a follow-up that adds maybe 200 LoC to the existing block + a state object.

The decoder side (RNN-T / TDT) is already stateful and naturally streams — you just keep `(LSTM state, last_token)` between chunks and call the joint network whenever fresh encoder frames arrive.

### 10. Memory and Performance

#### 10.1 VRAM (offline, full attention, batch=1, 30 s audio)

| Model | FP32 weights | FP16 weights | INT8 weights | KV / activation peak (FP16) |
|---|---|---|---|---|
| Parakeet-CTC-0.6B | 2.4 GB | 1.2 GB | 0.6 GB | ~0.4 GB |
| Parakeet-RNNT-0.6B | 2.4 GB | 1.2 GB | 0.6 GB | ~0.4 GB |
| Parakeet-TDT-0.6B-v2 (XL) | 2.4 GB | 1.2 GB | 0.6 GB | ~0.5 GB |
| Parakeet-TDT-0.6B-v3 (XL) | 2.4 GB | 1.2 GB | 0.6 GB | ~0.5 GB |
| Parakeet-*-1.1B (XXL) | 4.4 GB | 2.2 GB | 1.1 GB | ~0.8 GB |

FP16 inference is the default deployment on A100/H100/Blackwell. The official HF cards state minimum 2 GB GPU RAM to *load* the 0.6B model; longer audio needs more for the encoder activations (O(T²) for full attention). FP8 (Hopper/Blackwell) and INT8 quantisation work cleanly because there's no critical layer norm of awkward range — Conformer is well-behaved at FP8.

#### 10.2 RTFx (HF Open ASR Leaderboard, A100 80GB, batch 128)

| Model | RTFx | Avg WER | Notes |
|---|---|---|---|
| **Parakeet-CTC-0.6B** | **4,282** | 7.69 | Fastest open-source ASR |
| **Parakeet-TDT-0.6B-v2** | **3,386** | **6.05** | Best speed/quality combo on the board |
| Parakeet-TDT-0.6B-v3 | 3,333 | 6.34 | Multilingual cost ~0.3% WER |
| Parakeet-CTC-1.1B | 2,729 | 7.40 | |
| Parakeet-CTC-1.1B (different report) | 2,794 | 6.68 | Cited in HF blog |
| Parakeet-TDT-1.1B | 2,391 | 7.02 | |
| Parakeet-RNNT-1.1B | 2,053 | 7.12 | |
| Whisper-large-v3 | 68.56 | 6.43 | 99 languages, transformer enc-dec |
| Whisper-large-v3-turbo | ~200 | ~7.5 | 4-layer distilled decoder |
| Moonshine-base | ~600 (CPU) | ~9 | Edge-targeted, no mel preproc |
| Canary-1B-v2 | ~600 | 6.0–7.0 | 25 languages, enc-dec transformer |
| Canary-Qwen-2.5B | ~30 | **5.63** | LLM decoder, current WER leader |

The key story: **Parakeet-CTC/TDT delivers 30-100x the throughput of Whisper-large-v3 at equal or better English WER** on the leaderboard's 8-dataset average. The price is English-only (or 25-language only, for v3). Multilingual support for the long tail (99 languages, including low-resource) is still Whisper / MMS territory.

#### 10.3 Comparison to Whisper for the C# port

| Aspect | Whisper-large-v3 | Parakeet-TDT-0.6B-v2 |
|---|---|---|
| Params | 1.55B | 0.6B |
| Architecture | Transformer enc + transformer dec | FastConformer enc + LSTM-pred + TDT joint |
| Encoder layers | 32 | 24 |
| Decoder layers | 32 | 1 LSTM + small MLP joint |
| Mel bins | 128 | 80 |
| Effective enc frame rate | 50 Hz (20 ms) | 12.5 Hz (80 ms) — 4x fewer frames |
| Decode complexity | autoregressive over output tokens with cross-attn | greedy over encoder frames |
| Long-form | 30 s windows, hallucination-prone | full 24 min single pass |
| Languages | 99 | 1 (v2) / 25 (v3) |
| English WER | 6.43 (avg) | 6.05 (avg) |
| RTFx (A100 b=128) | 69 | 3,386 |

For HartsyInference, Whisper remains the multilingual choice and Parakeet becomes the high-throughput English (and multi-EU) choice. The two models share the mel preprocessor (modulo `n_fft` 400 vs 512 and Whisper's `(x+4)/4` vs Parakeet's per-feature z-score), so the audio frontend is reusable with parameter swaps.

### 11. C# Implementation Notes

What we already have when Parakeet lands (assuming Whisper and Kokoro are done):

| Component | Status | Reuse for Parakeet |
|---|---|---|
| Mel spectrogram (Slaney, librosa-compatible) | done (Whisper) | reuse with different `n_fft=512`, `hop=160`, `n_mels=80`, `normalize=per_feature` |
| LayerNorm, Linear, MatMul, GELU, SiLU/Swish | done | reuse |
| Multi-head attention (absolute pos) | done (Whisper) | needs **rel-pos variant** (new code, ~200 LoC) |
| Conv1D, Conv2D | done (UNets) | reuse for subsampler |
| LSTM | new (also needed for Kokoro) | Kokoro adds it; Parakeet reuses |
| SentencePiece tokenizer | done (T5) | reuse |
| Safetensors loader | done | needed *only if* HF mirrors safetensors versions |
| `.nemo` tar loader | **new** | see §11.2 |

#### 11.1 New code surface (estimated)

- **FastConformer encoder**: ~1500 LoC including subsampler + block + rel-pos MHA + ConvModule. The Macaron-FFN sandwich, rel-pos shift trick, and the depthwise-separable Conv2D subsampler are the only genuinely new pieces.
- **Relative positional MHA (Transformer-XL style)**: ~250 LoC. The `relative_shift` trick has a known closed-form (zero-pad column 0, reshape, slice) — implement it once with shape asserts.
- **CTC head + greedy decode**: ~80 LoC. Trivial.
- **RNN-T head (joint network)**: ~150 LoC.
- **TDT head**: ~30 LoC extra on top of RNN-T joint (split the final projection at runtime, two argmaxes in the loop).
- **Greedy TDT/RNN-T decoder**: ~120 LoC including the state-management and safety guards.
- **Tokenizer wiring + post-processing** (mixed case for v2): ~50 LoC.

Total new HartsyInference.Audio.Parakeet surface: roughly **2200 LoC** plus shared infrastructure (LSTM kernel from Kokoro, rel-pos MHA reusable for any future Conformer).

#### 11.2 `.nemo` file format

`.nemo` is a **tar archive** containing:

```
model_config.yaml       # Hydra/OmegaConf model config (encoder, decoder, joint, preprocessor, tokenizer)
model_weights.ckpt      # PyTorch state_dict (torch.save format) — for older NeMo
  OR
model_weights/          # Sharded checkpoint folder (NeMo 2.0+, distributed-friendly)
tokenizer.model         # SentencePiece binary
vocab.txt               # Human-readable vocab (optional)
```

Two viable approaches:

1. **Direct `.nemo` loading**: tar extractor (.NET has `System.Formats.Tar.TarReader`) + a torch-pickle reader (~1500 LoC of Python opcode interpretation, equivalent to ggerganov/llama.cpp's approach for `.pth`). Doable but unfun; we'd be one of very few C# projects that parses pickled tensors.
2. **Use the HF safetensors mirrors** where they exist: e.g. `istupakov/parakeet-tdt-0.6b-v3-onnx`, `NexaAI/parakeet-tdt-0.6b-v2-MLX`, `FluidInference/parakeet-tdt-0.6b-v3-coreml`. The MLX and ONNX ports include `safetensors` weight files. We can build a one-time *conversion* tool (Python or PyTorch C# port via TorchSharp staged for conversion only) that produces a `.safetensors + config.json + tokenizer.model` triple. Loaders for those formats already exist in HartsyInference.

**Recommendation**: ship a small **conversion script** (Python, separate from the C# runtime) that takes a `.nemo` URL or path and produces a `parakeet/<variant>/{model.safetensors, config.json, tokenizer.model}` directory. The C# runtime never sees pickle. This is the same path we took for Whisper (HF safetensors) and matches the project rule "pure C# only" — the conversion is a one-off offline step, not part of inference.

#### 11.3 Validation strategy

Reference: NeMo's `asr_model.transcribe(['file.wav'])` output, captured ahead of time on a small WAV set (LibriSpeech test-clean subset). Validation tolerance:
- Log-mel: 1e-4 per element vs NeMo's `AudioToMelSpectrogramPreprocessor` (same tolerance as Whisper validation).
- Encoder output: 1e-3 vs NeMo at FP32, 5e-3 at FP16 (Conformer is more numerically sensitive than vanilla transformer due to the conv module — expect some drift).
- CTC log-probs: 1e-3 at FP32.
- TDT token+duration logits: 1e-3 at FP32.
- Final transcripts: byte-exact match on greedy decoding; ≤0.1% WER drift acceptable on FP16 inference vs reference FP32.

Build order recommendation: encoder → CTC head (validates encoder) → tokenizer → RNN-T joint (validates LSTM pred net) → TDT joint (validates duration head) → greedy TDT decoder → optional beam search.

### 12. Open ASR Leaderboard Context (2026)

As of the Open ASR Leaderboard 2026 expansion ([arXiv:2510.06961v4](https://arxiv.org/abs/2510.06961)), the board now tracks ~60 models across English short-form, multilingual, and long-form tracks. Key competitive positioning for Parakeet:

**English short-form (the historical leaderboard):**

| Rank tier | Model | Avg WER | RTFx | Architecture |
|---|---|---|---|---|
| Best WER | **Canary-Qwen-2.5B** | **5.63** | ~30 | Conformer + Qwen LLM decoder |
| Best WER (Conformer-only) | Granite-Speech-3.3-8B, Phi-4-Multimodal | ~5.7 | ~20 | Conformer + LLM |
| Best WER/speed tradeoff | **Parakeet-TDT-0.6B-v2** | **6.05** | **3,386** | FastConformer + TDT |
| Fastest, English-only | Parakeet-CTC-0.6B | 7.69 | 4,282 | FastConformer + CTC |
| Multilingual best WER | Canary-1B-v2 | ~6.0 (en) | ~600 | Conformer + transformer dec, 25 langs |
| 99-language baseline | Whisper-large-v3 | 6.43 | 68.56 | Transformer enc-dec |
| Whisper distilled | Whisper-large-v3-turbo | ~7.5 | ~200 | 32-enc / 4-dec |
| Edge baseline | Moonshine-base | ~9 | ~600 (CPU) | No mel preproc, transformer |
| Open-weight Meta | MMS-1B-All | high (varies) | ~100 | Wav2vec2 + 1100-lang adapters |

**Headline takeaways:**
- **LLM-decoder ASR** (Canary-Qwen, Granite, Phi-4-Multimodal) currently wins WER outright but pays 100x in RTFx — they're impractical for high-throughput offline transcription.
- **Parakeet-TDT-0.6B-v2 sits at the Pareto frontier**: every model that beats its WER is at least 30x slower; every model faster than it loses 1-2 absolute WER points.
- **The Parakeet-CTC variants are still the absolute throughput champions** for English. Use CTC when you need 4000+ RTFx and can tolerate a 1.5% WER bump.
- **Multilingual ranking** (added 2025): Parakeet-TDT-0.6B-v3 covers 25 EU langs at the same RTFx as v2; Canary-1B-v2 also does 25 at ~6x lower RTFx but better WER; Whisper-large-v3 covers 99 at 50x lower RTFx. For non-EU languages, no member of the Parakeet family is competitive — fall back to Whisper.
- **Long-form track** (audios > 30 min): Parakeet's full-attention XL handles ~24 min single-pass; its local-attention mode reaches ~3 h. Whisper's 30 s windowing + hallucination tendency makes it weaker on long form despite multilingual coverage.

For HartsyInference's positioning: **Parakeet-TDT-0.6B-v2 is the default high-throughput English path; Parakeet-TDT-0.6B-v3 is the default high-throughput EU-multilingual path; Whisper-large-v3 remains the long-tail multilingual path.** All three share our FastConformer + Whisper-encoder + mel infrastructure once §2 lands.
