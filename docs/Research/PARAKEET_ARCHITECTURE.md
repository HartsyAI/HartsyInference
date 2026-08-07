# NVIDIA Parakeet (CTC / RNN-T / TDT) — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Parakeet pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

NVIDIA Parakeet is a family of ASR models that pair a **FastConformer encoder** ([arXiv:2305.05084](https://arxiv.org/abs/2305.05084)) with one of three decoder heads: **CTC** (single linear projection), **RNN-T** (LSTM prediction net + joint), or **TDT** — Token-and-Duration Transducer ([arXiv:2304.06795](https://arxiv.org/abs/2304.06795)). All variants share the same encoder, mel-spectrogram preprocessor, and SentencePiece BPE/Unigram tokenizer; only the head differs. The flagship **Parakeet-TDT-0.6B-v2** (the XL FastConformer variant, 24 layers @ 1024 d_model) achieves 1.69% / 3.19% WER on LibriSpeech test-clean / test-other and an **RTFx ~3,386** on the Hugging Face Open ASR Leaderboard at batch size 128 — i.e. it transcribes ~3,000x faster than real-time on a single A100 while matching Whisper-large-v3 English quality. The newer **Parakeet-TDT-0.6B-v3** (Aug 2025) extends the family to **25 European languages** at WER 6.34% / RTFx 3,333 with a 8192-vocab unified tokenizer trained on the 670k-hour Granary dataset.

The architectural value of TDT is the **duration head**: instead of consuming one encoder frame per joint-network call (RNN-T) and emitting a sea of blanks, the joint network outputs `[V + 1 + D]` logits — vocab + blank + a duration distribution over `{0, 1, 2, 3, 4}` — and the decoder advances `duration` frames per emission. This collapses the inference loop down to roughly one joint call per emitted token rather than one per encoder frame, giving TDT a ~2.8x speedup over standard RNN-T at equal or better WER.

This file covers all three Parakeet variants. Mel preprocessing math is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) (the "Parakeet/Canary" column). Cache-aware streaming details are cross-referenced to [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) (to be written). Comparison to encoder-decoder transformer ASR is in [WHISPER_ARCHITECTURE.md](WHISPER_ARCHITECTURE.md).

Sources: [FastConformer paper](https://arxiv.org/abs/2305.05084), [TDT paper](https://arxiv.org/abs/2304.06795), [NVIDIA NeMo repo](https://github.com/NVIDIA-NeMo/NeMo), [Parakeet-TDT-0.6B-v2 HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [Parakeet-TDT-0.6B-v3 HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [Parakeet-TDT-1.1B HF](https://huggingface.co/nvidia/parakeet-tdt-1.1b), [Parakeet-CTC-1.1B HF](https://huggingface.co/nvidia/parakeet-ctc-1.1b), [Parakeet-CTC-0.6B HF](https://huggingface.co/nvidia/parakeet-ctc-0.6b), [Parakeet-RNNT-1.1B HF](https://huggingface.co/nvidia/parakeet-rnnt-1.1b), [Parakeet-RNNT-0.6B HF](https://huggingface.co/nvidia/parakeet-rnnt-0.6b), [Open ASR Leaderboard paper](https://arxiv.org/abs/2510.06961), [Cache-aware Conformer Streaming](https://arxiv.org/abs/2312.17279), [NeMo Parakeet deep-dive (qed42)](https://www.qed42.com/insights/nvidia-parakeet-tdt-0-6b-v2-a-deep-dive-into-state-of-the-art-speech-recognition-architecture), [Speechmatics TDT explainer](https://www.speechmatics.com/company/articles-and-news/token-duration-transducer-tdt-explained).

## Parakeet Variants Table

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

## Tokenizer

All English Parakeet models use a **SentencePiece** tokenizer trained on the model's transcript corpus:

| Model family | Type | Vocab | Special tokens |
|---|---|---|---|
| Parakeet-CTC/RNNT-0.6B / 1.1B | SentencePiece **Unigram** | 1,024 | blank (added at runtime, index = vocab_size) |
| Parakeet-TDT-1.1B | SentencePiece Unigram | 1,024 | blank |
| Parakeet-TDT-0.6B-v2 | SentencePiece BPE | ~1,024 | blank; tokenizer emits mixed-case + punctuation directly |
| Parakeet-TDT-0.6B-v3 | SentencePiece Unigram (unified, multilingual) | **8,192** | blank |

The tokenizer model file lives inside the `.nemo` archive as `tokenizer.model` (the SentencePiece binary) plus `vocab.txt` (the human-readable vocab). HartsyInference already needs SentencePiece for other models (T5, etc.); we reuse that loader. Important detail: **no language IDs**, no special start/end-of-transcript tokens. Output is plain token IDs in `[0, V-1]`, blank is `V`.

For v3 (multilingual), there's no language token either — the encoder/decoder identify language implicitly from acoustics + LM context.

## Memory and Performance

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

## C# Implementation Notes

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
