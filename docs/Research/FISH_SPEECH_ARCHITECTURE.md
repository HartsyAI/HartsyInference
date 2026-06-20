# Fish-Speech 1.5 / OpenAudio S1 — Architecture

> Build-ready spec. Sources: `fishaudio/fish-speech` (`models/text2semantic/llama.py`, `inference.py`,
> `models/vqgan/modules/firefly.py`+`fsq.py` @ v1.5.0, `models/dac/modded_dac.py` @ main, `tokenizer.py`),
> HF `fish-speech-1.5` config. Fetched 2026-06-20. Two generations differ only in the codec.

## DualAR text2semantic (shared by both generations)
Llama-style "slow" backbone (RoPE/RMSNorm/SwiGLU/GQA) + a small "fast"/depth transformer (shared weights
across depth steps, RoPE) predicting the per-frame audio codebooks.

**fish-speech-1.5 (verified config):** dim 1024, 24 layers, 16 heads, 2 KV (GQA), head_dim 64, intermediate
4096, text vocab 102048, **codebook_size 1024, num_codebooks 8**, n_fast_layer 4, rope θ=1e6, eps 1e-6, untied,
max_seq_len 8192. **openaudio-s1-mini:** config HF-gated/unverified — DAC codec (9 books = 1 semantic@4096 + 8
residual@1024); get the real config.json before building.

**Embedding (verified):** input `[num_codebooks+1, T]`; row 0 = semantic/text → `embeddings`, rows 1..N → one
shared `codebook_embeddings` with per-book offset `i·codebook_size`; **summed**, masked to semantic positions,
×`1/√(N+1)`. **Fast:** `fast_embeddings`, `fast_norm`, `fast_output` (→ codebook_size), `fast_project_in` if
`fast_dim≠dim`; runs over `num_codebooks` depth positions, input = slow hidden (step 0) then prev codebook emb.
**Gen:** temp 1.0, top_p 0.9, top_k 30, rep-penalty 1.1; per frame slow→semantic, fast→codebooks 1..N; stop on
`<|im_end|>`. Keys: `embeddings`, `codebook_embeddings`, `layers.{i}.attention.wqkv`/`feed_forward.w1/w2/w3`
(fused), `norm`, `output`; fast `fast_{embeddings,layers,norm,output,project_in}`.

## Codec
- **fish-speech-1.5 = firefly-gan-vq:** `DownsampleFiniteScalarQuantize` (grouped-residual **FSQ** levels
  (8,5,5,5)=1000, 9 quantizers, downsample (2,2) via ConvNeXt) + **HiFi-GAN generator** (SiLU + tanh, upsample
  (8,8,2,2,2)/(16,16,8,2,2), up_init 512, ResBlock1 [3,7,11], 128 input ch). NO iSTFT. 44.1 kHz.
- **openaudio-s1 = modded-DAC:** `DownsampleResidualVectorQuantize` (1 semantic@4096 + 8 residual@1024,
  codebook_dim 8) + **DAC decoder** (Snake1d, decoder_rates [8,8,4,2], dim 1536, tanh). 44.1 kHz.

## C# build status (`Models/FishSpeech/`) — targeted fish-speech-1.5
- [x] **`FishSpeechDualAr`** — slow + fast both reuse `Qwen2Model` (headless); dual summed-codebook embedding
  (offset table, ×1/√(N+1)); slow head → semantic; fast depth AR → 8 codebooks. Synthetic-forward verified.
- [x] **`FireflyDecoder`** — codebook dequant → **reused `VitsHiFiGan`** (firefly upsample config) → 44.1 kHz.
  Synthetic-forward verified.
- [x] **`FishSpeechPipeline`** — text prefill → AR frames → firefly decode.
- [ ] **Staged/reconcile:** firefly's grouped-residual **FSQ** (levels (8,5,5,5)) + ConvNeXt resample + **SiLU**
  activation (current decoder uses learned codebook embeds + LeakyReLU), Fish's fused `wqkv`/`w1w2w3` key adapter
  (vs Qwen2 split keys), the encoder (ref→tokens), the BPE tokenizer + `<|semantic:i|>`/`<|im_end|>` ids, and
  the openaudio-s1 DAC-codec variant (Snake decoder + RVQ).
