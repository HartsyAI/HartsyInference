# Dia-1.6B (Nari Labs) — Architecture

> Build-ready spec for `nari-labs/Dia-1.6B`. Sources: HF `nari-labs/Dia-1.6B/config.json` + model card,
> `nari-labs/dia` (`dia/layers.py`, `dia/model.py`, `dia/audio.py`), HF transformers `modeling_dia.py`.
> Fetched 2026-06-19.

## Overall

Encoder-decoder transformer (T5/Whisper-style, `is_encoder_decoder=true`) generating **DAC** 44.1 kHz codes
(9 codebooks, code range 0..1023). ~1.6B params. Text bytes in → encoder → causal decoder with
cross-attention → 9-channel DAC code grid (delay pattern) → DAC decode → 44.1 kHz audio.

## Encoder (12 layers)
hidden 1024, 16 heads, **16 KV heads (full MHA)**, head_dim 128 (QKV up-project to 2048, o_proj back to 1024),
FFN 4096, **SwiGLU** (SiLU gate, fused `gate_up_proj`), **RMSNorm** pre-norm with BOTH `pre_sa_norm` and
`post_sa_norm`, RoPE θ=10000, eps 1e-5, max text 1024. **Bidirectional (non-causal)** self-attention.
Text = raw UTF-8 bytes, `src_vocab_size=256`, `text_pad=0`; speaker tags `[S1]`/`[S2]` are literal bytes.

## Decoder (18 layers)
hidden 2048, FFN 8192, SwiGLU, RMSNorm pre-norm (`pre_sa_norm`, `pre_ca_norm`, `pre_mlp_norm` + final).
- Self-attn: 16 query heads, **4 KV heads (GQA)**, head_dim 128.
- Cross-attn: 16 query heads, **16 KV heads (MHA)**, head_dim 128, KV source dim **1024** (encoder hidden).
- RoPE θ=10000, eps 1e-5, max audio 3072.
- **9 channels**: 9 separate `Embedding(1028, 2048)` tables **summed** for input; single fused output head
  `logits_dense: 2048 → (9, 1028)`.

## Codec stream tokens
valid codes 0..1023; **EOS=1024, PAD=1025, BOS=1026** (`tgt_vocab_size=1028`).
Delay pattern: **[0, 8, 9, 10, 11, 12, 13, 14, 15]** (`max_delay=15`). Apply: `out[t,c]=in[t-delay[c],c]`;
pre-delay positions → **BOS(1026)**, past-end → **PAD(1025)**, BOS precedence. Revert shifts back by +delay,
out-of-bounds → PAD.

## Generation
CFG default **3.0**: `cond + 3.0*(cond - uncond)`; uncond = encoder over empty/padded text. Sampling temp
**1.2**, top_p **0.95**, top_k (`cfg_filter_top_k`) **45**, per-channel on flattened `[B*9,1028]`. Prefill
decoder step 0 with BOS in all 9 channels. Terminate when **channel 0 emits EOS(1024)** or step ≥
`max_tokens - max_delay`; then run `max_delay`(15) flush steps. Audio prompt / voice cloning = prefix DAC codes.

## Weight keys (HF port)
`model.encoder.{embedding, layers.{i}.{pre_sa_norm, self_attention.{q,k,v,o}_proj, post_sa_norm,
mlp.{gate_up_proj, down_proj}}, norm}`; `model.decoder.{embeddings.<9>, layers.{i}.{pre_sa_norm,
self_attention.*, pre_ca_norm, cross_attention.*, pre_mlp_norm, mlp.*}, norm}`; top-level `logits_dense`.
Original repo uses `DenseGeneral` fused tensors (reshape on map). Config has two field-name sets (original
nested `model.encoder.n_layer…` vs HF flat `num_hidden_layers…`).

## C# reuse map
- **DAC** `DacConfig.Dac44kHz` — exactly Dia's vocoder (9 cb, 1024).
- **`MusicGenDelay`** — reuse for the delay grid, but needs a custom delay array `[0,8..15]` AND distinct
  pre-fill (BOS) / post-fill (PAD); extend `Apply` with a (preFill, postFill) overload.
- **`NucleusSampler`**, CFG helpers — reuse.
- **RoPE/RMSNorm/SwiGLU/GQA** primitives exist in the Qwen2 stack.
- **Net-new:** a non-causal **encoder** stack (with post-attn norm), a decoder **cross-attention** sublayer
  (Q from decoder 2048, K/V from encoder 1024) — `Qwen2Model` is decoder-only with no cross-attn — the
  9-channel summed embeddings + fused multi-channel head, and the enc-dec orchestration (encode once, cache
  encoder cross-KV, batched cond/uncond decode with delay + 15-step EOS flush).
