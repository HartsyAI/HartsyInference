# HeartMuLa / heartlib — Architecture

> Build-ready spec for HeartMuLa-oss-3B music generation (Apache 2.0). Sources: `github.com/HeartMuLa/heartlib`,
> HF `HeartMuLa/HeartMuLa-oss-3B` (config + index verified), `HeartMuLa/HeartCodec-oss-20260123`, arXiv 2601.10547.
> Fetched 2026-06-20. **Confirmed real + downloadable.**

## What it is
An autoregressive **codec-LM** structured as the **CSM/Sesame two-transformer pattern** applied to music: a
global backbone predicts codebook 0 per frame; a depth decoder predicts the remaining RVQ codebooks. The
*codec decoder* (HeartCodec) is flow-matching; the LM itself is AR.

## LM (config.json + index.json verified)
- `architectures: ["HeartMuLa"]`, `model_type: "heartmula"`.
- **Global backbone** `backbone_flavor: "llama-3B"`, **28 layers** (`backbone.layers.0..27`). Predicts codebook 0.
- **Depth decoder** `decoder_flavor: "llama-300M"`, **3 layers** (`decoder.layers.0..2`). Predicts codebooks 1..7.
- `audio_num_codebooks: 8`, `audio_vocab_size: 8197`, `text_vocab_size: 128256` (Llama-3.2 tokenizer; lyrics +
  `[intro]`/`[verse]`/`[chorus]` markers). Style cond: **MuQ-MuLan** embedding (`muq_dim: 512`, via `muq_linear`).
- Keys: `backbone.*`, `decoder.*`, `audio_embeddings.*`, `text_embeddings.*`, `audio_head.*`, `codebook0_head.*`,
  `muq_linear.*`, `projection.*`, `unconditional_text_embedding`. Layer internals Llama-style (`attn.{q,k,v}_proj`,
  `mlp.{w1,w2,w3}`). Sharded `.safetensors` (~15.75 GB, no tokenizer files). Variants: `-oss-3B`, `-RL-oss-3B`,
  `-happy-new-year`; 7B mentioned but NOT released.

## HeartCodec (config.json verified)
48 kHz stereo, **12.5 Hz**, RVQ `num_quantizers 8 / codebook_size 8192 / codebook_dim 32` (~1.30 kbps). Conv
multi-scale enc/dec `downsample [3,4,4,4,5]`, dim 512, 24 heads, causal, `ada_norm_single`. **Flow-matching
decoder** (SQ-Codec target); encoder fuses Whisper/WavLM semantic features.

## C# build status (`Models/HeartMula/`)
- [x] **LM reuses the verified `CsmModel`** (dual `Qwen2Model` + codebook heads) — HeartMuLa is exactly the CSM
  two-transformer shape. [`HeartMulaConfig`](../../src/HartsyInference.Audio/Models/HeartMula/HeartMulaConfig.cs)
  supplies the music config (Llama-3B/300M, 8 codebooks vocab 8197, 48 kHz) → a `CsmConfig`. **Config + construct
  tested**; the LM forward is covered by the CSM tests.
- [x] [`HeartMulaPipeline`](../../src/HartsyInference.Audio/Pipelines/HeartMulaPipeline.cs) — lyrics tokens → CSM
  AR frames (8 codebooks) → codebook grid.
- [ ] **Staged:** the **HeartCodec** flow-matching RVQ decoder (reuse `ConditionalCfm` + the conv codec work),
  the **MuQ-MuLan** style embedder (net-new conditioning encoder; currently lyrics-only), and the sharded
  safetensors loader / exact key reconciliation.

## Performance (RTX 3060, `3b-base`, 4 s clip, cond+uncond CFG)

The LM decodes codec frames autoregressively at **12.5 Hz**, so the metric is **ms/frame**. Per-frame time is
**almost entirely per-token weight streaming** (~360 GB/s): backbone 3B ≈ 33 ms + depth decoder ≈ 23 ms + heads
— HeartMuLa is **memory-bandwidth-bound, not launch-bound**. This shapes which optimizations pay off.

| Config | ms/frame | fr/s | Notes |
|---|---:|---:|---|
| bf16 baseline | 91.5 | 10.9 | ~0.87× realtime (AR decode only) |
| + CUDA-graph decode (`HARTSY_CSM_GRAPH`, default on) | ~86–90 | ~11.2–11.7 | **bit-identical**, ~5% |

- **CUDA-graph decode** ([`CsmModel`](../../src/HartsyInference.Audio/Models/Csm/CsmModel.cs)): the single-frame
  backbone step and each depth-decoder step are captured once (via
  [`GenericTransformer.ForwardGraphDecodeStepEmbeds`](../../src/HartsyInference.LLM/Transformer/GenericTransformer.cs))
  and replayed per frame — cond + uncond = up to 4 concurrent graphs. It removes the ~8 ms/frame of kernel-launch
  overhead → **~5%, the honest ceiling** for a bandwidth-bound model (a launch-bound model like the FX decoders
  gains 2×+ from the same technique). Depth uses persistent per-session KV caches reset each frame.
- **Weight quantization** ([`CsmWeightCache`](../../src/HartsyInference.Audio/Models/Csm/CsmWeightCache.cs),
  `HARTSY_HEARTMULA_QUANT=q8_0|q4_k`): quantizes the projection/head matrices (keeps embeds/norms F16) **once** to
  a disk GGUF cache (streaming convert → no OOM; must run **post-`Remap`** since the remap splits the combined
  audio embed/head tensors), then mmaps the ~4.5 GB Q8 cache. On paper this is the real lever (Q8 ≈ 2×, Q4 ≈ 3.6×
  less bandwidth), and it produces **valid audio** — but the engine's Q8 fused GEMV (`LaunchMulMatVecQ8_0F32`) is
  currently **~8× slower than cuBLAS bf16** at M=1 for these shapes, so quant is **not yet a win**. Realizing it
  needs a quant-GEMV kernel pass, not more wiring.
