# ChatTTS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (ChatTTS pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

ChatTTS (2noise team, 2024) is a conversational text-to-speech model designed for dialogue and chat use cases, with native paralinguistic control (laughs, sighs, breaks, orality level) and Chinese + English support. The architecture is a four-stage neural pipeline: **text BPE tokens -> GPT (semantic LM) -> audio token sequence (4-codebook RVQ) -> DVAE Decoder / Decoder head -> 100-bin mel spectrogram -> Vocos vocoder -> 24 kHz waveform**. Both transformers are LLaMA-style causal decoders sharing the same `LlamaConfig` shape (hidden=768, layers=20, heads=12, intermediate=3072, max position=4096) but with two separate roles:

- **GPT** (the "semantic LM") generates the audio token sequence one frame at a time. Each frame is `num_vq=4` parallel codebook indices over a vocabulary of `num_audio_tokens=626`. It is conditioned on the BERT-tokenized text plus a `[spk_emb]` token whose embedding is replaced at runtime by a 768-dim projection of the 192-dim speaker latent.
- **DVAE Decoder head** (called `decoder` in the codebase, separate file from the DVAE proper) converts the 4-codebook latent stream into a 100-channel mel spectrogram. The "DVAE" file additionally contains a Grouped Finite-Scalar-Quantizer (GFSQ) with `levels=[5,5,5,5]`, `G=2`, `R=2`, used during training and as a fallback path.

Voice identity is controlled by a 768-dim speaker embedding sampled from a stored mean+std Gaussian (`spk_stat.pt`, ~4 KB). The base release does not include zero-shot voice cloning (the DVAE encoder is reserved in the official roadmap), but the community has shipped clones via `Embed.safetensors` and the GFSQ encoder path. Streaming is supported by emitting partial DVAE-decoded mel chunks to Vocos as the GPT generates them; first-audio latency is ~500 ms on a single 4090 with `stream_speed=12000` (samples per batch) and `pass_first_n_batches=2`.

This file covers the model architecture and pipeline. The Vocos vocoder details (ConvNeXt backbone, iSTFT head) live in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) under the "Vocos" section. Mel spectrogram preprocessing is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). Generic BPE tokenization is in [TOKENIZERS.md](TOKENIZERS.md). The audio codec / RVQ design is in [AUDIO_CODECS.md](AUDIO_CODECS.md).

Sources: [2noise/ChatTTS](https://github.com/2noise/ChatTTS), [2Noise/ChatTTS HF](https://huggingface.co/2Noise/ChatTTS), [config/gpt.yaml](https://huggingface.co/2Noise/ChatTTS/raw/main/config/gpt.yaml), [config/decoder.yaml](https://huggingface.co/2Noise/ChatTTS/raw/main/config/decoder.yaml), [config/dvae.yaml](https://huggingface.co/2Noise/ChatTTS/raw/main/config/dvae.yaml), [config/vocos.yaml](https://huggingface.co/2Noise/ChatTTS/raw/main/config/vocos.yaml), [Vocos paper (arXiv:2306.00814)](https://arxiv.org/abs/2306.00814).

## Key Numbers / Constants

| Constant | Value | Notes |
|---|---|---|
| Sample rate | 24,000 Hz | Vocos output, fixed |
| Mel channels (n_mels) | 100 | Vocos input, Decoder output |
| Mel hop length | 256 | Frame rate 93.75 Hz |
| Mel n_fft | 1024 | Vocos STFT |
| Mel frame rate | 93.75 frames/s | 24000 / 256 |
| `num_text_tokens` | 21,178 | BERT vocab + 61 specials |
| `num_audio_tokens` | 626 | FSQ levels 5^4 + 1 per group |
| `num_vq` | 4 | (G=2) x (R=2) codebooks per audio frame |
| GPT hidden_size | 768 | LLaMA model dim |
| GPT layers | 20 | Decoder-only causal LM |
| GPT attention heads | 12 | head_dim = 64 |
| GPT intermediate_size | 3072 | SwiGLU FFN inner |
| GPT max_position_embeddings | 4096 | RoPE context window |
| GPT max_new_token (InferCode) | 2048 | ~21.8 s of audio |
| GPT max_new_token (RefineText) | 384 | text -> tagged text |
| spk_emb_dim | 192 | Pre-projection latent dim |
| Speaker vector (user-facing) | 768 dim float16 | Post-projection, base16384(LZMA2) encoded |
| Decoder dim | 384 | Input/output dim of DVAEDecoder |
| Decoder hidden | 512 | Internal ConvNeXt width |
| Decoder n_layer | 12 | Dilated ConvNeXt blocks |
| Decoder bn_dim | 128 | Bottleneck dim |
| DVAE dim | 512 | DVAE decoder I/O |
| DVAE GFSQ levels | [5,5,5,5] | FSQ scalar levels per quantizer |
| DVAE GFSQ G | 2 | Codebook groups |
| DVAE GFSQ R | 2 | Residual quantizers per group |
| Vocos dim | 512 | ConvNeXt width |
| Vocos intermediate_dim | 1536 | FFN inner |
| Vocos num_layers | 8 | ConvNeXt blocks |
| Stream chunk samples | 12,000 | `stream_speed` |
| Stream warmup batches | 2 | `pass_first_n_batches` |
| GPT InferCode temperature | 0.3 | |
| GPT InferCode top_K | 20 | |
| GPT InferCode top_P | 0.7 | |
| GPT InferCode repetition_penalty | 1.05 | Per VQ codebook |
| GPT RefineText temperature | 0.7 | |
| GPT RefineText top_K | 20 | |
| GPT RefineText top_P | 0.7 | |
| GPT RefineText repetition_penalty | 1.0 | |

## Data Layouts / Formats

### Text Input Tokenization

```
User input:   "Hello [uv_break] world [laugh] cool."
After wrap:   "[Stts][spk_emb][speed_5]Hello [uv_break] world [laugh] cool.[Etts]"
After BERT:   [CLS] [Stts] [spk_emb] [speed_5] hello [uv_break] world [laugh] cool . [Etts] [SEP]
Token IDs:    (1, T_text) int64,  T_text typically 20-200
```

### Speaker Latent

```
Stored shape:    (768,) float16
Sampling:        randn(768) * std + mean   where std, mean from spk_stat.pt (768,) each
Transport:       base16384(LZMA2_preset9_extreme(float16_bytes))
                 -> ASCII string ~1.5 KB per voice
GPT projection:  Linear(192, 768)   # NOTE: per gpt.yaml spk_emb_dim=192, but
                                    # the stored vector is 768-dim and shape-matches
                                    # the embedding space directly; some forks project
                                    # 768->192->768. Verify with reference run.
```

### GPT Audio Token Stream

```
Per step output: (4,) int64  in [0, 625]
Full sequence:   (T_audio, 4) int64,  T_audio in [1, 2048]
Re-embedded:     sum_i emb_code[i](tok[:, i])  -> (T_audio, 768) float
```

### Mel Spectrogram (Decoder output / Vocos input)

```
Shape:           (1, 100, T_audio) float32
Frame rate:      93.75 Hz
Range:           log-mel scale (Vocos's MelSpectrogramFeatures normalization)
```

### Audio Output

```
Shape:           (num_samples,) float32, range ~[-1, 1]
Sample rate:     24,000 Hz
num_samples:     T_audio * 256
Format:          mono PCM, write to WAV via soundfile / NAudio / similar
```

### `spk_stat.pt`

```
Format: PyTorch pickle of (2, 768) float16 tensor
        OR: base16384-encoded compressed bytes in some forks
Layout: row 0 = mean (768,), row 1 = std (768,)
Size:   ~3 KB on disk; 4.26 KB with pickle overhead
```

## Implementation Notes for HartsyInference

1. **LlamaModel is standard.** We will have already built a causal LLaMA implementation in the native HartsyInference.LLM package (RoPE + RMSNorm + SwiGLU FFN, KV cache). The GPT here is a stock LLaMA with hidden=768, layers=20, heads=12 — reuse the HartsyInference.LLM Llama block directly. The only ChatTTS-specific wrapper is:
   - Replace `emb_tokens` with a switchable `emb_text` (21178x768) + 4x `emb_code[i]` (626x768 each). At each step pick the right embedding table based on which mode we're in.
   - Multi-head output: 4 `head_code[i]` (768 -> 626) for InferCode, 1 `head_text` (768 -> 21178) for RefineText. Sample each VQ head independently and concatenate.
   - Speaker injection: a single in-place embedding overwrite at `[spk_emb]` token positions. Use a span-find on token_ids before the first forward.

2. **KV cache is mandatory.** 4096-token context, 20 layers, 12 heads, head_dim=64, two tensors (K and V) per layer. Pre-allocate at session start (`KvCache.Allocate(batch=1, max_seq=4096, layers=20, heads=12, head_dim=64, dtype=fp16)` ~ 96 MB). The yaml says `use_cache: False` but that is a training-time default; we always enable cache at inference (`enable_cache=True` in core.py).

3. **Two model files, not three.** The "GPT" and "Decoder" share NO weights and load from separate safetensors:
   - `GPT.safetensors` (or sharded `gpt/`) -> Llama + emb_text + emb_code[0..3] + head_text + head_code[0..3] + emb_spk_proj.
   - `Decoder.safetensors` -> DVAEDecoder + input projection + output mel head.
   - `Vocos.safetensors` -> ConvNeXt + iSTFT head.
   - `Embed.safetensors` is *redundant* with weights inside GPT.safetensors — it's a convenience split for cloning experiments. Load only if a user supplies a custom Embed override.

4. **DVAEDecoder is a dilated ConvNeXt stack, NOT a transformer.** Implement as 12 residual blocks:
   ```
   block(x) = x + pointwise_conv(GELU(GroupNorm(depthwise_conv_d(x))))
   ```
   with depthwise kernel=7 and dilation per-block doubling from 1 up to 2^11. Bottleneck projects to `bn_dim=128` mid-block. Final `Conv1d(384, 100, 1)` projects to mel. Build this as `HartsyInference.Audio/Modules/DvaeDecoder.cs`.

5. **GFSQ for cloning path (optional v2).** Group-Residual Finite Scalar Quantization. The encoder maps mel -> 1024-dim latent -> reshape to `(G=2, R=2, 4)` where 4 is the FSQ-dim and the levels are `[5,5,5,5]`. FSQ quantization is `round(tanh(z) * (L-1)/2) -> int in [-(L-1)/2, (L-1)/2]`. Decode by reversing. Implement as `HartsyInference.Audio/Modules/GroupedResidualFSQ.cs` — see [AUDIO_CODECS.md](AUDIO_CODECS.md) for the math.

6. **Vocos reuses existing code.** We have a Vocos implementation planned for F5-TTS and other models — see [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) Vocos section. Same exact config (input_channels=100, dim=512, num_layers=8, intermediate_dim=1536, iSTFTHead with n_fft=1024, hop=256). Single shared Vocos module across ChatTTS / F5-TTS / Kokoro (note: Kokoro uses iSTFTNet not Vocos — different module).

7. **Tokenizer:** BertTokenizerFast is HF's WordPiece, not BPE. We need a pure-C# WordPiece tokenizer that supports Chinese character-splitting and 61 additional special tokens. Plan: a generic WordPiece in `HartsyInference.Core/Tokenization/WordPieceTokenizer.cs`. The `tokenizer.json` from the HF tokenizer directory contains the full vocab + merge rules + special token list — parse once at load time.

8. **Special token handling:** `[spk_emb]` is a single token ID that needs to be located in the encoded sequence (one or more positions); the embedding at that position is replaced. Cache the ID at tokenizer-load time. Same for `[empty_spk]`. All `[break_X]`, `[laugh_X]`, `[oral_X]`, `[speed_X]` are normal tokens that go through the embedding table — no special handling beyond standard tokenization.

9. **Speaker latent transport:** Implement base16384 + LZMA2 codec so users can paste/share voice strings. .NET has `System.IO.Compression` but no LZMA — use `LZMA-SDK` or `SharpCompress` (already an indirect dep via the safetensors loader? verify). base16384 is unusual — port the ~50 lines from pybase16384 to C#. Keep it pure-C# (no native deps).

10. **Sampling:** Each of the 4 audio heads samples *independently* per step with temperature=0.3, top_k=20, top_p=0.7, repetition_penalty=1.05. Use the shared `HartsyInference.Core.Sampling.LogitsSampler` — no special ChatTTS-only sampler needed. Repetition penalty: track the last N (N=64? confirm) generated tokens per codebook; multiply logits of those tokens by `1/1.05` before top-k.

11. **Streaming requires chunked Decoder + Vocos.** Plan: a `ChatTtsStreamingSession` that owns the GPT KV-cache, a ring buffer of recent audio tokens (size = chunk + overlap), and a callback fired on each chunk. The first 2 chunks are computed but suppressed; from chunk 3 onward each new chunk is decoded with an overlap window, the leading samples of the Decoder output are dropped, and the rest is yielded. Same overlap/discard pattern for Vocos (centered STFT has reflective padding artifacts).

12. **Speaker stat loading:** `spk_stat.pt` is a tiny PyTorch pickle. Either ship a pre-converted `spk_stat.bin` (raw 2x768 float16, 3 KB) at model packaging time, or implement a minimal pickle reader for the float16 tensor case. Recommend pre-conversion.

13. **Determinism:** RNG for speaker sampling and for sampling tokens uses PyTorch's Mersenne Twister. We need bit-exact reproducibility against the reference for validation. Plan: use a HartsyInference-specific seedable PRNG and only compare final waveforms within an audio-quality tolerance (PESQ > 4.0, mel-cepstral distance < 1.0) — bit-exact PyTorch RNG reproduction is not worth the effort.

14. **Validation tolerances:**
    - GPT logits: cosine similarity > 0.9999 vs reference at FP32, > 0.999 at FP16.
    - Decoder mel output: mean abs error < 0.01 dB per mel bin vs reference.
    - Vocos waveform: PCM MAE < 1e-3 at FP32, < 5e-3 at FP16. PESQ vs reference > 4.2.
    - End-to-end: mel-cepstral distance < 0.5 between HartsyInference output and PyTorch output for the same text + seed.

15. **Memory budget at FP16:**
    - GPT weights: ~440 MB
    - Decoder weights: ~52 MB
    - Vocos weights: ~13.5 MB
    - Speaker stat: ~3 KB
    - KV cache (4096 ctx): ~96 MB
    - Activations (one chunk): ~50 MB
    - Total resident: ~650 MB; with 1 GB safety margin we're at <2 GB VRAM for a single session.

## Reference Implementations

- [2noise/ChatTTS](https://github.com/2noise/ChatTTS) — Official Python/PyTorch reference.
- [2Noise/ChatTTS HF](https://huggingface.co/2Noise/ChatTTS) — Official model weights and configs.
- [lenML/ChatTTS-Forge](https://huggingface.co/spaces/lenML/ChatTTS-Forge) — Community fork with extra tooling, voice cloning experiments, and webUI.
- [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) — Reference Vocos implementation (we mirror its math).
- [Vocos paper (arXiv:2306.00814)](https://arxiv.org/abs/2306.00814) — Vocos: Closing the gap between time-domain and Fourier-based neural vocoders.
- [Finite Scalar Quantization (arXiv:2309.15505)](https://arxiv.org/abs/2309.15505) — FSQ paper (the basis of GFSQ).
- [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) — Reference `GroupedResidualFSQ` implementation that ChatTTS uses.
- [HuggingFace LlamaModel docs](https://huggingface.co/docs/transformers/model_doc/llama) — Reference for the GPT backbone.
