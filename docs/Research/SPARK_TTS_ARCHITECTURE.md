# Spark-TTS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Spark-TTS pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Spark-TTS ([arXiv:2503.01710](https://arxiv.org/abs/2503.01710), [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS), Mar 2025, CC-BY-NC-SA 4.0) is an efficient LLM-based zero-shot voice-cloning TTS. It pairs a fine-tuned **Qwen2.5-0.5B causal LM** (hidden_size=896, 24 layers, GQA 14:2, vocab=166000 — extended from Qwen2.5's 151,936 with ~14k Spark-specific control + audio tokens) with **BiCodec**, a custom *single-stream* speech codec that decomposes 16 kHz audio into two complementary token sets:

- **Semantic tokens** — a time-varying stream at **50 Hz**, single VQ codebook of **8192 entries (8-D factorized)**, encoding linguistic content from frozen wav2vec2-XLSR-53 features (layers 11, 14, 16 averaged).
- **Global tokens** — a fixed-length set of **32 tokens** per utterance encoding time-invariant speaker timbre, produced by an ECAPA-TDNN over mel-spectrograms → PerceiverResampler → FSQ (6 dims × 4 levels = 4096 codebook).

Both token streams are predicted *autoregressively in a single sequence* by the Qwen LM. The same LM checkpoint handles (a) zero-shot voice cloning from a reference clip and (b) controllable generation from coarse attribute prompts (gender/pitch_label/speed_label) or fine-grained numeric controls (pitch_value 0-1000, speed_value 0-10), unified by a chain-of-thought style prompt format. Cross-lingual cloning (zh ↔ en, including code-switching mid-sentence) works because the LM is bilingual and the global tokens decouple speaker identity from language. Reconstruction is done by a **DAC-style HiFi-GAN-like wave generator** (Snake1d activations, transposed convs at rates [8, 5, 4, 2] for hop=320 → 16 kHz output) fed by Vocos/ConvNeXt backbones that act as prenet/postnet conditioned on the global speaker vector.

Training data is **VoxBox** (102.5k hours, 4.7M utterances across 29 corpora; 47.6k h zh + 54.9k h en) annotated with gender/age/emotion/pitch/speed. The full model ships at ~3.95 GB (LLM 2.03 GB BF16 + BiCodec 626 MB + wav2vec2 1.27 GB FP32). Performance on an L20 GPU via Triton/TensorRT-LLM: RTF 0.14 @ concurrency-1, dropping to RTF 0.07 @ concurrency-4. UTMOS=4.35 (beats CosyVoice2 and even ground truth at 4.08).

This document covers the model + pipeline. The BiCodec codec module is also cross-referenced from [AUDIO_CODECS.md](AUDIO_CODECS.md) (needs a new section). Mel preprocessing in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). HiFi-GAN-style wave generator background in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). The LM reuses the same Qwen2.5-0.5B kernels as the native HartsyInference.LLM package.

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

## Variants

| Variant | LM Params | Total Pkg Size | Languages | HF Path | License |
|---|---|---|---|---|---|
| **Spark-TTS-0.5B** (only official release) | 507M (Qwen2.5-0.5B family, modified vocab) | 3.95 GB | zh + en (zero-shot cross-lingual, code-switching) | [`SparkAudio/Spark-TTS-0.5B`](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | CC-BY-NC-SA 4.0 (non-commercial) |
| Community quantizations (4 listed on HF) | varies (GGUF Q4/Q5/Q8) | ~300-500 MB LM | same | various (e.g. mradermacher) | inherit CC-BY-NC-SA |
| Community fine-tunes (25 listed on HF) | 0.5B | 3.95 GB | varies (some add ja/ko/multilingual) | community | — |

There is **no official larger or smaller variant** as of the May-2026 cutoff. SparkAudio explicitly markets the 0.5B size as the "single, edge-deployable" model. Mirror at [HKUSTAudio/Spark-TTS-0.5B](https://huggingface.co/HKUSTAudio/Spark-TTS-0.5B) is byte-identical.

## Memory and Performance

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

## HuggingFace Files

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

## C# Implementation Notes

#### 8.1 Reuse from HartsyInference.LLM / existing HartsyInference

- **Qwen2.5-0.5B LM = standard Llama-style decoder transformer.** GQA (14:2), SwiGLU FFN, RMSNorm, RoPE θ=1M, tied embeddings. **Reuse HartsyInference.LLM's `Qwen2ForCausalLM` implementation as-is** — only the embedding/lm_head dimensions change (vocab 151,936 → 166,000). Make sure the safetensors loader accepts the wider embedding matrix.
  - HartsyInference.LLM patterns for KV-cache, sampling (temp/top-k/top-p), RoPE precomputation, and the GGUF-quantized variants all apply.
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
- **`HartsyInference.LLM`** — Spark's LM reuses the Qwen2 implementation; the only delta is vocab size = 166000.
- **[TEXT_ENCODERS.md](TEXT_ENCODERS.md)** — no Spark-specific text encoder; the Qwen tokenizer covers it.
- **[MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md)** — confirm support for n_fft=1024, win=640, hop=320, n_mels=128, fmin=10. Note the 50% overlap (win/hop=2.0).

#### 8.5 Build order

Suggested implementation phases (depends on what's already done):

1. **Phase A (prerequisites)**: HartsyInference.LLM Qwen2.5-0.5B BF16 inference + Qwen BPE tokenizer with added-token registration. Mel-spec kernel. (Likely already shipped.)
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
