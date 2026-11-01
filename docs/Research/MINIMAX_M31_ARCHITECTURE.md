# MiniMax-M3.1-Flash — architecture and checkpoint notes

Source snapshot 2026-11-01: `MiniMaxAI/MiniMax-M3.1-Flash` (`model_type: minimax_m3_vl`, `MiniMaxM3SparseForConditionalGeneration`) and
transformers 5.22.1's native `minimax_m3_vl` (`modeling_minimax_m3_vl.py`, a library, not remote code). **Not implemented in the engine.** This
records what the checkpoint and the reference contain so the port starts from facts. Where the checkpoint or the reference and this note disagree, they are right.

## Why it is not built yet

The checkpoint is 252 GB in 48 shards (about 428B parameters, 23B active), so nothing here has been run on real weights, and the CPU backend
widens quantized weights to F32. The port needs three pieces the engine does not have (a sparse-attention indexer, two quantized weight
formats read from safetensors, and a vision tower); each is small next to the verification it needs.

## Text decoder

| Property | Value |
|---|---|
| Layers | 60; layers 0-2 dense (`dense_intermediate_size` 12288), 3-59 MoE (`moe_layer_freq`) |
| Width | hidden 6144; 64 query heads, 4 KV heads, head dim 128 |
| Norm | RMSNorm in Gemma form (`use_gemma_norm`), eps 1e-6; per-head QK norm (`qk_norm_type: per_head`) |
| RoPE | theta 1e7, `partial_rotary_factor` 0.5 (`rotary_dim` 64) on every layer; context 1,048,576 |
| MoE | 128 routed experts, top 4, sigmoid scoring with `e_score_correction_bias` (F32 `[128]`), `routed_scaling_factor` 2.0, one shared expert of width 3072 |
| Activation | `swigluoai`: `swiglu_alpha` 1.702, `swiglu_limit` 7.0 |
| Attention | MiniMax Sparse Attention on all 60 layers (below) |
| Vocabulary | 200064; untied `lm_head` |

### Sparse attention indexer

Per layer (`MiniMaxM3VLIndexer`): `index_q_proj` (4 heads × 128) and `index_k_proj` (one 128-wide key) with their own RMSNorms, RoPE on the first 128
dims of the shared table, scores `q·k` in F32, future keys masked to −∞, then **max-pooled over blocks of 128 keys**. Per query and per index head
(4 heads = the 4 KV groups) the top 16 blocks are kept, and the query's own and preceding block (`sparse_local_block` 1) are forced in by
boosting their score to +∞. The main attention then runs over only those blocks (the reference builds a dense additive mask; the deployment
kernel consumes left-packed block indices padded with −1). The indexer has no value path (`sparse_disable_index_value` is 1 everywhere) and keeps
its own key cache. Blocks are anchored to absolute key slots, so left padding changes the selection; the reference documents this limitation.

## Checkpoint layout

`language_model.model.layers.N.…` for the decoder, `vision_tower.vision_model.…`, `multi_modal_projector.*`, `patch_merge_mlp.*`; 89,632 tensors.

- **Attention, dense MLP, shared expert, index projections: MXFP8.** `X.weight` is F8_E4M3 `[out, in]` and `X.weight_scale_inv` is U8 `[out, in/32]`
  (`quant_method: mxfp8`, `weight_block_size: [1, 32]`). The scales are E8M0: `weight = e4m3 × 2^(byte − 127)`. On `q_proj` of layers 3 and 20 the scale bytes are 114-117 and the dequantized rows have a standard deviation of 0.026 and 0.042, against about 150 for the raw E4M3 values.
- **Routed experts: NVFP4-style (compressed-tensors names).** `block_sparse_moe.experts.E.{w1,w2,w3}.weight_packed` is U8 `[3072, 3072]` for `w1`
  (two FP4 values per byte over 6144), with `weight_scale`, `weight_global_scale` and `input_global_scale` (F32 scalars). `w1`/`w3` are gate/up, `w2` is down.
- **Unquantized:** router `gate.weight` (F32 `[128, 6144]`), `e_score_correction_bias`, norms (BF16), the vision tower and projectors.
- `dspark/` holds a separate 2.3 GB speculative-decoding draft head with its own config and index.

## Vision

CLIP-style tower: 32 layers, hidden 1280, 16 heads, patch 14, 3-D RoPE (`rope_mode: 3d`), 2×2 spatial and 2-frame temporal patch merge, projector to 6144.
Images are tiled on a pinpoint grid up to 2016×2016; video segments take at most 4 frames.

## Existing pieces to reuse

Sigmoid routing with a selection bias and a shared expert (DeepSeek-V3 and Muse Spark paths in `GgufConfigFactory`/`MoeFeedForward`), clamped SwiGLU
(GPT-OSS), NVFP4 handling (`Nvfp4HostReference` and the NVFP4-AWQ notes in `MINIMAX_H3.md`), the DeepSeek-V4.1 indexer and sparse-attention
references (`IndexerScoresReference`, `SparseLatentAttentionReference`), and the attention contract's exclusion tensor
([ATTENTION_CONTRACT.md](../ATTENTION_CONTRACT.md)). Missing: an MXFP8 reader, the block-max indexer with a block-index attention input, the cache for the
indexer's keys, the vision tower, and the `minimax_m3_vl` config and key mapping.

## Verification plan

1. A tiny random `minimax_m3_vl` from transformers' own modeling code (the Flint port builds its tiny reference the same way), with the indexer's block size and
   top-k shrunk in the config so a short sequence already prunes (at the released 128 × 16 the prompt would need more than 2048 tokens), then engine logits against it.
2. Dequantize one MXFP8 and one NVFP4 tensor from the real checkpoint and compare with the same rows' reference values.
3. Real-weight generation needs about 250 GB of host or device memory; plan GPU hardware first.

## License

MiniMax Community License (not an OSI license). Free for non-commercial use. A **Commercial Use** (paid products or services built on the model,
commercial API use, or any commercial deployment of a post-trained or fine-tuned derivative) must display "Built with MiniMax M3.1-Flash"; it needs a
one-time notice to `api@minimax.io`, and prior written authorization when the product or service earns more than US$20 million a year. An appendix
prohibits uses including military purposes, content harmful to minors, and deliberate disinformation. This is a summary; the `LICENSE` file in the model repository is the binding text. Flag every catalog entry accordingly before shipping.
