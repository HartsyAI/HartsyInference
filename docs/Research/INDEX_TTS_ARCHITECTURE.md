# IndexTTS-2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (IndexTTS pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

IndexTTS is a family of zero-shot voice-cloning text-to-speech systems developed by the **Bilibili Index Team**. It started as a heavy rework of Tortoise/XTTS (single-codebook codec + GPT-style autoregressive LM + neural vocoder) and evolved into a fully cascaded three-stage system. The two open-weight versions in scope for HartsyInference are:

- **IndexTTS-1.5** (May 2025) — single-codebook DVAE codec at 25 Hz, GPT-2-style 24-layer / 1280-dim / 20-head decoder generating mel codes, Conformer-Perceiver speaker conditioner, BigVGAN-v2 vocoder synthesising 24 kHz waveform directly from GPT output. Chinese + English. ~3.66 GB on disk.
- **IndexTTS-2** (Sept 2025, [arXiv:2506.21619](https://arxiv.org/abs/2506.21619)) — three modules trained separately: (T2S) the same GPT decoder enlarged to support an additional emotion-conditioning Perceiver, (S2M) a non-autoregressive **flow-matching DiT** that turns semantic codec tokens + speaker embedding + GPT latents into an 80-band mel at 22 050 Hz, and (vocoder) NVIDIA's pretrained `bigvgan_v2_22khz_80band_256x`. A separate fine-tuned **Qwen-3 0.6B** ("qwen0.6bemo4-merge") provides text-to-emotion-distribution control. Chinese + English (training corpus 55 000 h covers Chinese, English, Japanese; output quality is calibrated for ZH+EN). ~5.9 GB on disk.

The hallmark new capabilities of IndexTTS-2 are (a) **explicit duration control** via a token-count input that fixes the AR generation length without distorting prosody, (b) **disentangled emotion vs. speaker identity** via a Gradient-Reversal-Layer trick during T2S training, and (c) **natural-language emotion prompts** distilled from DeepSeek-R1 into a tiny LoRA-fine-tuned Qwen-3 0.6B. The system tops Chinese TTS benchmarks (WER on test-zh beats F5-TTS by ~0.5 pp and MaskGCT by ~2 pp) while matching F5-TTS on English.

For HartsyInference, IndexTTS-2 is a complex pipeline that **reuses several blocks we already have or are planning** — a causal GPT-2 style decoder (identical in shape to the native `HartsyInference.LLM` transformer), a DiT block stack (already implemented for Flux/SD3 in `HartsyInference.Diffusion`), a flow-matching Euler scheduler (already in `HartsyInference.Diffusion`, see [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md)), a BigVGAN-v2 vocoder ([HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md)), and a Conformer encoder for the speaker-prompt perceiver. The genuinely new work is: a custom **semantic codec** (Vocos-style decoder over 8192 codebook entries, see [AUDIO_CODECS.md](AUDIO_CODECS.md)), a **wav2vec2-BERT** front end for semantic-token extraction during conditioning, the **Conformer-Perceiver** prompt module, and the BPE+pinyin **Chinese tokenizer**.

Sources: [IndexTTS-2 paper (arXiv 2506.21619)](https://arxiv.org/abs/2506.21619), [IndexTTS paper (arXiv 2502.05512)](https://arxiv.org/abs/2502.05512), [HF IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2), [HF IndexTeam/IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5), [index-tts repo](https://github.com/index-tts/index-tts), [DeepWiki: index-tts](https://deepwiki.com/index-tts/index-tts), [IndexTTS-2 demo page](https://index-tts.github.io/index-tts2.github.io/), [BigVGAN-v2 22kHz 80band 256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x), [Qwen3 0.6B-emo4-merge config](https://huggingface.co/IndexTeam/IndexTTS-2/tree/main/qwen0.6bemo4-merge).

## Variants

| Variant | Date | Params (active) | Output langs | Output SR | HF repo | Size on disk |
|---|---|---|---|---|---|---|
| **IndexTTS-1** (original) | Feb 2025 | ~750 M (GPT 700 M + DVAE 50 M + BigVGAN-v2 112 M) | ZH + EN | 24 kHz | [IndexTeam/Index-TTS](https://huggingface.co/IndexTeam/Index-TTS) | ~3.5 GB |
| **IndexTTS-1.5** | May 2025 | Same shape as v1; quality / English improvements via more training data | ZH + EN | 24 kHz | [IndexTeam/IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | 3.66 GB |
| **IndexTTS-2** | Sept 2025 | T2S GPT (~870 M) + emo Perceiver (~10 M) + S2M flow-DiT (~300 M) + BigVGAN-v2 (112 M) + Qwen-3 0.6B emo (~600 M) ≈ 1.9 B total | ZH + EN (training data also ja) | 22.05 kHz mel → vocoder | [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | 5.9 GB |
| IndexTTS-2.5 (technical report only, [arXiv 2601.03888](https://arxiv.org/abs/2601.03888)) | Dec 2025 | Codec compressed 50 Hz → 25 Hz; S2M U-DiT replaced with Zipformer; **2.28× faster RTF**; adds Japanese + Spanish in the official model | ZH + EN + ja + es | 22.05 kHz | not yet open-weights at time of writing | — |

The 1.5 GPT checkpoint is 1.17 GB (fp16/fp32 mix), the 2.0 GPT checkpoint is 3.48 GB because the embedding tables, emotion perceiver, and longer max-mel-tokens (1815 vs 800) all grew it. Parameter counts above are estimated from the config (see §2) and known checkpoint sizes; the paper does not publish a single "X parameters" headline.

## Reference Audio Requirements

- **Length.** 5–10 s of clean speech is the documented sweet spot. The Conformer-Perceiver imposes no hard cap, so multi-clip references (concatenate several utterances of the same speaker) measurably improve similarity; 30 s of reference is typical for production clones.
- **Quality.** Studio-clean is best. Music/background noise contaminates both timbre (speaker prefix) and prosody (the `prompt_semantic_tokens` that seed the GPT).
- **Sample rate.** Internally resampled to 24 kHz mono for the Conformer's mel front end. Any input SR works.
- **Language.** Per the official model card the *reference* can be **any language** even though the *output* is restricted to ZH+EN; this enables cross-lingual cloning (Spanish reference → English output in the target speaker's voice).
- **Emotion reference (v2 only).** A separate clip if you want to dictate emotion explicitly; otherwise omit and the emotion vector defaults to neutral.

## HuggingFace Files

**IndexTTS-2** (`IndexTeam/IndexTTS-2`, total 5.9 GB):

| File | Size | Purpose |
|---|---|---|
| `config.yaml` | 2.88 kB | All hyperparameters (mel front-end, GPT, semantic_codec, s2mel, vocoder) |
| `bpe.model` | 476 kB | SentencePiece BPE, 12 000 vocab, ZH/EN/pinyin tokens |
| `gpt.pth` | **3.48 GB** | T2S Transformer (24L × 1280, 20 heads) + Conformer-Perceiver speaker conditioner + 4-block emotion Perceiver |
| `s2mel.pth` | **1.20 GB** | Flow-matching DiT (13L × 512, 8 heads) + length regulator + WaveNet final layer + style encoder + semantic codec (Vocos decoder) |
| `wav2vec2bert_stats.pt` | 9.34 kB | Per-channel mean/std normalisation for w2v-BERT features (model weights pulled from HF on demand) |
| `feat1.pt` | 57.2 kB | `spk_matrix` — projection from 8-dim emotion vector to a speaker-side bias |
| `feat2.pt` | 375 kB | `emo_matrix` — projection from 8-dim emotion vector to the emotion Perceiver input |
| `qwen0.6bemo4-merge/model.safetensors` | **1.19 GB** | Qwen-3 0.6B fine-tuned (LoRA-merged) to predict 8-way emotion distributions from text |
| `qwen0.6bemo4-merge/config.json` | 727 B | Qwen-3 config: 28 layers, 1024 hidden, 16 heads, 8 KV-heads, head_dim 128, intermediate 3072, vocab 151 936, RoPE θ=1e6, bf16 |
| `qwen0.6bemo4-merge/tokenizer.json` | 11.4 MB | Qwen-3 tokenizer (BPE with merges) |
| `qwen0.6bemo4-merge/vocab.json` | 2.78 MB | Vocab |
| `qwen0.6bemo4-merge/merges.txt` | 1.67 MB | BPE merges |
| `qwen0.6bemo4-merge/added_tokens.json` | 707 B | Special tokens added during fine-tune |
| `qwen0.6bemo4-merge/chat_template.jinja` | 550 B | Chat template for emotion-prompt formatting |
| `qwen0.6bemo4-merge/special_tokens_map.json` | 616 B | Standard special-token mapping |
| `qwen0.6bemo4-merge/tokenizer_config.json` | 5.43 kB | Tokenizer config |
| `qwen0.6bemo4-merge/generation_config.json` | 117 B | Default generation params |
| `qwen0.6bemo4-merge/Modelfile` | 360 B | Ollama Modelfile |
| `LICENSE.txt`, `LICENSE_ZH.txt`, `README.md`, `.gitattributes` | 22 kB | Metadata |

Not in the repo: `bigvgan_v2_22khz_80band_256x` (the vocoder weights live at [`nvidia/bigvgan_v2_22khz_80band_256x`](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x), MIT licence, ~445 MB) and the w2v-BERT base weights (pulled from HF on first use; `facebook/w2v-bert-2.0`, ~580 M params).

**IndexTTS-1.5** (`IndexTeam/IndexTTS-1.5`, total 3.66 GB):

| File | Size | Purpose |
|---|---|---|
| `config.yaml` | 2.49 kB | Hyperparameters (mel, GPT, vqvae, bigvgan) |
| `bpe.model` | 476 kB | Same SentencePiece BPE as v2 |
| `unigram_12000.vocab` | 94.7 kB | Human-readable vocab dump (debug aid; not strictly needed at runtime) |
| `gpt.pth` | 1.17 GB | GPT (24L × 1280, 20 heads) + Conformer-Perceiver speaker conditioner |
| `dvae.pth` | 243 MB | DVAE codec (100-band mel → 8192-entry codebook at ~23 Hz) |
| `bigvgan_generator.pth` | 536 MB | Custom-trained BigVGAN-v2 24 kHz generator (100-band, 4×4×4×4×2×2 upsample = 1024× hop, gpt_dim=1280 conditioning, snake-beta activation, CQT discriminator on the training side) |
| `bigvgan_discriminator.pth` | 1.65 GB | Training-only; **not needed for inference** — can be omitted from the HartsyInference distribution |

## Memory and Performance

**Disk + VRAM (fp16/bf16):**

| Component | Disk | VRAM-resident (fp16) |
|---|---|---|
| **IndexTTS-1.5 total inference** | 1.95 GB (drop the discriminator) | ~2.5 GB |
| **IndexTTS-2 total inference** | 5.9 GB + ~445 MB BigVGAN-v2 + ~1.1 GB w2v-BERT = ~7.4 GB | ~6–8 GB |
| └ T2S GPT-2 (24L×1280) | 3.48 GB → 1.7 GB fp16 | ~1.8 GB + KV cache (peaks ~1.5 GB for max-len) |
| └ S2Mel flow-DiT (13L×512) | 1.2 GB → 0.6 GB fp16 | ~0.7 GB + flow activations |
| └ Qwen-3 0.6B-emo | 1.19 GB bf16 already | ~0.7 GB (optional) |
| └ BigVGAN-v2 vocoder | 445 MB → 224 MB fp16 | ~0.4 GB |
| └ w2v-BERT 2.0 | 1.1 GB → 580 MB fp16 | ~0.7 GB (only during reference-encode) |

A 12 GB GPU (RTX 3060, RTX 4070) runs IndexTTS-2 comfortably; an 8 GB card requires offloading w2v-BERT and Qwen-3 to CPU between uses.

**Latency / RTF** (paper + community measurements, A100/H100 class GPU, batch 1):

| Stage | RTF (audio_secs / processing_secs ⁻¹) |
|---|---|
| T2S (GPT autoregressive)  | ~0.232 — dominant cost; scales with output length |
| S2Mel (25-step flow ODE) | ~0.060 |
| BigVGAN-v2 vocoder | ~0.018 |
| **Total** | **~0.31** (A100) |

On a RTX 3060 12 GB the community reports RTF ≈ 13× without optimisation (one second of audio takes ~13 s wall-clock); with FlashAttention + bf16 + KV-cache tricks this falls below 2×. **First-audio latency** for a single sentence on an A100 is roughly 0.3–0.5 s (sentence of ~3 s); on consumer hardware expect 1–5 s.

**Training scale** (for reference, not implementer-relevant): 55 000 hours of multilingual ZH/EN/JA, 8× A100 80 GB, AdamW lr=2e-4, three weeks of training.

## C# Implementation Notes for HartsyInference

Mapping every IndexTTS-2 component onto HartsyInference packages:

#### 9.1 HartsyInference.Audio.IndexTTS (new package)

Top-level orchestrator class `IndexTtsPipeline` with constructor taking the model directory. Owns:

- `IndexTextTokenizer` — SentencePiece BPE loader (see [TOKENIZERS.md](TOKENIZERS.md)). Handle CJK whitespace injection and pinyin-bracket annotation expansion in pure C# (`Span<char>` based). The SentencePiece `.model` file is a flatbuffers blob; a hand-rolled loader is small (≈ 1 kLOC) and is already needed for several other models. Vocabulary size 12 000, special tokens hard-coded (`start_text=0`, `stop_text=1`).
- `SpeakerEncoder` — mel front-end (100-band, n_fft 1024, hop 256, 24 kHz; reuse [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md)) + 6-block Conformer + Perceiver-Resampler. The Conformer is new; budget ~600 LOC of C# (subsample-conv2d2 → MHSA + relative-pos + depthwise-conv + Macaron FFNs). Perceiver-Resampler is ~150 LOC.
- `EmotionEncoder` — same shape as `SpeakerEncoder` but 4 blocks / 4 heads / output 512. Can share the Conformer block implementation.
- `OptionalQwenEmotionModel` — gated behind a build flag. Reuse the **native `HartsyInference.LLM` Qwen3 implementation directly**: drop the 1.19 GB `model.safetensors` into the engine's config-driven Qwen3 decoder with `hidden=1024, layers=28, heads=16, kv_heads=8, head_dim=128, vocab=151936, intermediate=3072, rope_theta=1e6, tie_word_embeddings=true`. Inference does a single greedy/argmax decode of the 8-emotion-token output. Document the chat template format from `chat_template.jinja` for the prompt. **Recommendation: do not bundle by default** — most users will pass either a reference clip or an explicit 8-vector; ship Qwen-3-emo as a separately downloadable add-on.
- `T2sGptDecoder` — causal Transformer, 24 × 1280 / 20 heads. Architecture is a **plain GPT-2** (sinusoidal positional embeddings, no RoPE, no GQA, full attention). Reuse a shared GPT decoder block if the engine already has one; otherwise this is ~400 LOC of standard MHSA + FFN. Must support **prefix conditioning** (the speaker/emotion prefixes are continuous 1280-dim vectors inserted *between* the text tokens and the mel-start token — they are not in the embedding table). Implement the prefix path via a `prepend_continuous(Tensor prefix)` API on the KV-cache builder. Sampling: top-p, top-k, repetition penalty (Tortoise/XTTS defaults). Stop at `stop_mel_token = 8193` or at `max_mel_tokens = 1815`. Must export the **final-layer hidden state per emitted position** for downstream S2Mel — make this an opt-in tap to avoid extra memory traffic when not needed.
- `SemanticCodec` — Vocos-style decoder (12 ConvNeXt blocks, dim 384, intermediate 2048) over an 8192-entry codebook with codebook_dim 8. The encoder side runs only on the reference clip during conditioning. Both halves are pure-C# CNNs; reuse [AUDIO_CODECS.md](AUDIO_CODECS.md)'s Vocos decoder description (the SNAC / WavTokenizer pattern is structurally identical).
- `W2vBert2Extractor` — a 580 M-param wav2vec2-BERT encoder used only to extract continuous features that the semantic codec then quantises. **Pre-quantise the reference clip's semantic tokens once** at session start so this network does not run per-token. Architecture: a CNN feature extractor + 24-block Conformer-style Transformer; substantial work (~1500 LOC). Mitigation: cache reference-derived semantic tokens to disk keyed by audio hash.
- `S2MelFlowMatcher` — the 13-block × 512-dim DiT. **Reuse the Flux/SD3 DiT block implementation in HartsyInference.Diffusion** with the following deltas: (a) `in_channels=80` instead of image latent channels, (b) two skip-connection patterns (`long_skip_connection` + `uvit_skip_connection`) — both are simple add-after-N-blocks patterns, (c) WaveNet final layer (8 dilated-conv blocks, kernel 5, hidden 512, gated tanh×sigmoid activation), (d) style conditioning concatenated to every block's input rather than via AdaLN — read the s2mel checkpoint to confirm. Scheduler: `FlowMatchEulerDiscreteScheduler` already exists; 25 steps default. CFG is supported (`class_dropout_prob=0.1` trained the unconditional path) — reuse the velocity-field CFG combiner from [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md).
- `BigVGanV2Vocoder` — reuse the BigVGAN section of [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Snake-Beta activation is non-trivial CUDA work (we already need it for several other models). The 22 kHz / 80-band / 256× variant is the **NVIDIA pretrained** weight — separate small download manager class.

#### 9.2 Configuration loader

`config.yaml` is plain YAML. Use the existing YamlDotNet dependency (already in the solution per `NUGET_PACKAGE_DESIGN.md`). The config schema maps cleanly to a strongly-typed `IndexTtsConfig` record with nested `Gpt`, `SemanticCodec`, `S2Mel.Dit`, `S2Mel.Wavenet`, `Vocoder` records.

#### 9.3 Weight loading

Both `.pth` files are PyTorch pickle archives (zip-of-pickled-tensors). We already have a pickle reader in `HartsyInference.Common.Weights` (used for Kokoro and other PyTorch-checkpoint models). The state-dict keys follow standard PyTorch module-path convention — write a per-component key-prefix mapper (`gpt.*` → `T2sGptDecoder`, `conditioning_encoder.*` → `SpeakerEncoder`, `emo_encoder.*` → `EmotionEncoder`). The Qwen-3 sub-model is in `model.safetensors` (standard safetensors layout — reuse [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md)). `feat1.pt` and `feat2.pt` are simple 2-D float matrices — load them as `Tensor<float>` constants on the device and use them via a fused matmul in the emotion path.

#### 9.4 Chinese tokenizer notes

The SentencePiece BPE alone handles CJK fine **provided** the pre-tokenizer inserts whitespace around CJK characters before BPE — otherwise the BPE will merge punctuation/spaces into the Chinese tokens incorrectly. Implement `TokenizeByCjkChar(ReadOnlySpan<char>)` returning a `string` with spaces inserted at every CJK / non-CJK boundary, and the inverse for detokenisation. The pinyin range `[8474..10201]` is documented in `pinyin.vocab` (94.7 kB ASCII file in the v1.5 repo and shipped in `checkpoints/` in the GitHub repo); load it as a `Dictionary<string,int>` for explicit pinyin-bracket annotation expansion. Tone marks: numbers 1–5 follow each syllable (e.g. `zhong1 guo2`).

#### 9.5 Validation tolerances

Per project rule "validate against references": compare on three fronts:
- **GPT token output** against the Python reference for a fixed seed + fixed input → expect *bit-identical* mel-token sequence (deterministic argmax / fixed-seed top-p).
- **Mel spectrogram** out of the S2Mel ODE → within `1e-3` mean-abs-diff after CFM-Euler at the same step count (flow matching is sensitive to step roundoff; bf16 vs fp32 will diverge after step 10).
- **Final waveform PSNR** ≥ 35 dB vs the Python reference for the same speaker+text input. BigVGAN-v2 has a known small non-determinism in its CUDA path so a small numerical gap is acceptable.

#### 9.6 Phased build order

1. **Phase 1 — IndexTTS-1.5 path only.** Skip semantic codec, S2Mel, w2v-BERT, Qwen-3, emotion. This gets a working ZH+EN voice clone with ~25% of the implementation work. Reuses everything we already need for Kokoro plus the GPT-2 + Conformer-Perceiver + DVAE + BigVGAN-v2 (24 kHz custom variant). Output: 24 kHz mono.
2. **Phase 2 — Add IndexTTS-2 pipeline.** Implement semantic codec (Vocos-style), wav2vec2-BERT extractor, S2Mel flow-matching DiT, the NVIDIA 22 kHz BigVGAN-v2. Output: 22.05 kHz mono with non-emotional, non-duration-controlled cloning. Quality should already beat IndexTTS-1.5.
3. **Phase 3 — Emotion control surfaces.** Wire up the emotion Perceiver, `feat1.pt`/`feat2.pt` matrix paths, and the **vector** and **audio-prompt** emotion APIs. Skip Qwen-3 for now.
4. **Phase 4 (optional add-on) — Qwen-3 0.6B-emo text-to-emotion.** Ship as a separately-downloadable extension package `HartsyInference.Audio.IndexTTS.EmotionText` to keep the base distribution small.
5. **Phase 5 (when IndexTTS-2.5 open-weights arrive).** Replace S2Mel with the streaming Zipformer variant, drop semantic-codec frame rate to 25 Hz, expose true `IAsyncEnumerable<AudioChunk>` streaming.

#### 9.7 What we deliberately do not implement

- **Training/fine-tuning paths.** Discriminators, GAN losses, GRL+classifier heads, EMA codebook update, KL warm-up — all training-only, omit from the inference distribution.
- **vLLM-style paged attention.** Out of scope for HartsyInference's first release; revisit when HartsyInference.API adds batched serving.
- **CUDA fused kernels for Snake-Beta.** The NVIDIA BigVGAN repo provides one for 1.5–3× speedup; write a PTX equivalent only after the pure-C# pipeline lands and we have a baseline RTF to compare against.

---

## Appendix A — Cross-reference summary for HartsyInference

| External dep | HartsyInference doc |
|---|---|
| Mel spectrogram (24 kHz / 100-band, 22.05 kHz / 80-band) | [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) |
| Semantic codec (Vocos-style decoder), DVAE quantiser | [AUDIO_CODECS.md](AUDIO_CODECS.md) |
| BigVGAN-v2 vocoder (custom 24 kHz and NVIDIA 22 kHz) | [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) |
| Flow-matching ODE solver, CFG, sway-sampling option | [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) |
| Streaming abstractions (`IStreamingPipeline`, `AudioRingBuffer`) | [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) |
| SentencePiece BPE loader, CJK tokenization | [TOKENIZERS.md](TOKENIZERS.md) |
| Safetensors loader (for Qwen-3 emo sub-model) | [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md) |
| DiT block reuse from Flux/SD3 | [SD3_ARCHITECTURE.md](SD3_ARCHITECTURE.md), [FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md) |
| Qwen-3 0.6B (architecture identical to the native LLM path) | `HartsyInference.LLM` |
| (Comparison) StyleTTS-2-class non-AR TTS | [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md) |
