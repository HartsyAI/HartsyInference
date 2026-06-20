# Zonos-v0.1 (Zyphra, transformer variant) — Architecture

> Build-ready spec for `Zyphra/Zonos-v0.1-transformer`. Sources: `Zyphra/Zonos` GitHub (`model.py`,
> `backbone/_torch.py`, `conditioning.py`, `autoencoder.py`, `codebook_pattern.py`, `sampling.py`) + verified
> HF config.json. Fetched 2026-06-19. ~2B params.

## Backbone (transformer)
Llama-style GQA decoder but **LayerNorm, not RMSNorm**: d_model 2048, 26 layers (all attention), 16 heads,
4 KV heads (GQA 4:1), head_dim 128, FFN 8192 fused-gate-up **SwiGLU**, LayerNorm eps 1e-5, **RoPE θ=10000
interleaved** (rotary dim = full 128), no bias (fused `in_proj` 2048→3072 = q2048+k512+v512, `out_proj`),
max pos 16384. Block: `x += mixer(norm(x)); x += mlp(norm2(x))`, final `norm_f`.

## Codec
DAC 44.1 kHz (`descript/dac_44khz`), 9 codebooks × 1024, ~86 Hz. **9 embeddings** `(1026, 2048)` summed
input (1024 codes + EOS 1024 + masked 1025); **9 heads** `(2048→1025)` stacked output (1024 + EOS).
Only codebook 0 may emit EOS. **Delay = roll codebook k by k+1 → effective [1..9]**, masked-token(1025)
pad; revert by `k+1` slice. EOS in cb0 → 9-step diagonal flush, then revert + zero codes ≥1024 + trim last 9.

## Conditioning prefix (7 conditioners, seq-axis concat → Linear(2048→2048) + LayerNorm)
1. espeak phonemes (Embedding; PAD0/UNK1/BOS2/EOS3, symbols from id 4 = punct+ASCII+IPA) — N tokens
2. speaker (Passthrough Linear 128→2048 + learned uncond) — 1 token
3. emotion (Fourier, 8-dim, sum-normalized) — 1
4. fmax (Fourier, min0/max24000) — 1
5. pitch_std (Fourier, min0/max400) — 1
6. speaking_rate (Fourier, min0/max40) — 1
7. language_id (Integer embed, 105 langs, en-us=24) — 1
`FourierConditioner`: `x_norm=(x-min)/(max-min)`, `f=2π·x_norm@Wᵀ`, out `[cos(f),sin(f)]`, W~N(0,1)·std.
Speaker encoder (separate download, not in main ckpt): ResNet293 + SimAM + ASP + LDA→128.

## Generation
CFG default **2.0**: `uncond + (cond-uncond)*2.0` (cond/uncond batched; cfg==1 unsupported). Sampling temp
1.0, **min_p 0.1**, rep_penalty 3.0 window 2 (top_p/top_k 0 by default; optional NovelAI unified sampler).
max_new 86*30=2580. Stop on cb0 EOS → 9-step flush.

## Weight keys
`backbone.layers.{i}.{norm, mixer.in_proj, mixer.out_proj, norm2, mlp.fc1, mlp.fc2}`, `backbone.norm_f`;
`embeddings.{0..8}`, `heads.{0..8}`; `prefix_conditioner.conditioners.{0..6}.*` (espeak `phoneme_embedder`,
speaker `project`+`uncond_vector`, Fourier `weight`+`uncond_vector`, language `int_embedder`+`uncond_vector`),
`prefix_conditioner.project`/`norm`; `autoencoder.dac.*` (use own DAC).

## C# reuse map
**Reuse:** **`DiaAttention`** (GQA + interleaved RoPE + no bias — split fused in_proj at load) + **`DiaMlp`**
(fused gate-up SwiGLU — map fc1/fc2) for every block; **DAC** `Dac44kHz`; **`MusicGenDelay`** (k+1 delays,
mask fill); **`NucleusSampler`** (min_p). **Net-new (built):** the LayerNorm `ZonosBlock`/`ZonosBackbone`,
`ZonosCodebooks` (9 summed embeds + 9 stacked heads), `ZonosFourierConditioner`, `ZonosPipeline` (cond/uncond
backbones + delayed-AR + CFG + DAC). **Deferred:** espeak phonemization + full prefix assembly (speaker/
integer/passthrough conditioners), ResNet293 speaker encoder, NovelAI unified sampler.
