# Hy4 Flash (Tencent) architecture

Snapshot of 2026-11-24. Pins, licence terms, the constants the config and checkpoints carry, and the key map. The code is `src/HartsyInference.LLM/Hy4Flash/`;
status is in [MODEL_STATUS_LLM.md](../Checklists/MODEL_STATUS_LLM.md). The reference implementation is an **unmerged** transformers PR, so every statement of
maths below is "as that PR computes it", not "as upstream's serving engines compute it".

## Sources and pins

| What | Pin | URL |
|---|---|---|
| Instruct (BF16) | tencent/Hy4-flash @ `eccd3123eec73f2040f4c9ed0de1883593470510` (modified 2026-11-23) | https://huggingface.co/tencent/Hy4-flash |
| FP8 instruct | tencent/Hy4-flash-FP8 @ `8a1b19890d3533aefc730f8b41f4b55f50cc31c0` | https://huggingface.co/tencent/Hy4-flash-FP8 |
| Pre-trained base | tencent/Hy4-flash-base @ `95b9f8239e2f63cb837c91704f070ae47332ebdf` (BF16, 139 shards, 438.8 GB) | https://huggingface.co/tencent/Hy4-flash-base |
| GGUF (test only, per card) | AngelSlim/Hy4-flash-GGUF @ `b85056a1a1eacae21317df7481d3c0f06912ae02` (IQ2_AH, Q3_K, Q4_K_M, SPARK 4-bit) | https://huggingface.co/AngelSlim/Hy4-flash-GGUF |
| Official GitHub | Tencent-Hunyuan/Hy4-flash @ `7ab20f35602b1d3fc863f85aee20cb87eb8e53c5` | https://github.com/Tencent-Hunyuan/Hy4-flash |
| **Reference modelling code** | huggingface/transformers PR #50903 "Add Hy4-flash (`hy_v4_flash`)" head `eb1d5db464b0df81793f914849142e9b82e056c7`, **open, unmerged**: `src/transformers/models/hy_v4_flash/{modeling,modular,configuration,processing}_hy_v4_flash.py`, `hy_v4_vl/` for the tower, `conversion_mapping.py` for key renames | https://github.com/huggingface/transformers/pull/50903 |
| Serving references named by the card | vLLM image `vllm/vllm-openai:hy4-flash` (attention backend `FLASHMLA_SPARSE`), SGLang image `lmsysorg/sglang:hy4-flash` (not read) | https://recipes.vllm.ai/tencent/Hy4-flash , https://docs.sglang.io/cookbook/autoregressive/Tencent/Hy4-flash |

Findings that corrected the first leads:
- **The official GitHub repo holds no model code** (README, LICENSE, finetune/, assets/ only). The only open reference implementation is the unmerged transformers PR. `Tencent-Hunyuan/Hy4-preview` exists
  (pushed 2026-08-28) and is the earlier preview; `tencent/Hy4` (1.55 TB, `hy_v4_vl`) is the larger sibling that shares the vision tower and several modules, but it is **not** Flash.
- **The licence has no territory clause** (section 2). The "Model-as-a-Service above $50M revenue or 100M MAU" part is real.
- Released: HF repos created 2026-11-17 (03:44-03:46 UTC); the card News says Hy4 flash, Hy4 flash-FP8 and Hy4 flash-base are released.
- The transformers snippet on the card installs from `refs/pull/50903/head`; the PR is not merged, so a Python reference run needs that exact ref.

## Licence (Tencent Hy License, Version 1.0; file `LICENSE` is 13,942 bytes in all three repos)

Source: https://huggingface.co/tencent/Hy4-flash/blob/main/LICENSE . Derived from Apache-2.0 "with modifications, mainly the additional commercial terms for large
Model-as-a-Service ('MaaS') businesses in Section 10". Searched the full text for territory, region, EU, UK, Korea: **no territory restriction**.
Derivative works "must not remove, limit, or conflict with Section 10". Section 10, operative text:

- **10.1 MaaS** = "giving third parties remote access to model inference or fine-tuning (e.g., via API), allowing them to submit inputs or specify parameters or training
  data". Excludes end-user products with capabilities "confined to specific features or workflows, without general-purpose model access" and merely relaying requests to an
  independently operated third-party service.
- **10.2** A MaaS operator needs a separate written licence from Tencent (email is enough) if (a) aggregate worldwide gross revenue exceeds **US$50 million** in any
  consecutive 12 months, whether or not derived from the Work, or (b) before using the Work in or for a product or service with more than **100 million monthly active users**.
- **10.3** Internal use is exempt provided the Work, derivatives and outputs are not made available to third parties.
- **10.4** Contact hy_partners@tencent.com (card: hunyuan_opensource@tencent.com).

Catalog entry text: "Tencent Hy License 1.0 (Apache-2.0-derived). MaaS operators above US$50M revenue or with a product above 100M MAU need a separate licence; no territory limit."
A Hartsy-hosted repack would be redistribution by a service operator: flagged in the report; by default the catalog points at the upstream repos (no upload).

## Architecture constants

Class `HYV4FlashForConditionalGeneration`, `model_type` `hy_v4_flash` (HF remaps the vision sub-config's `hy_v4_flash` label to `hy_v4_vl_vision`). 220B total
(155.5B backbone + 63.5B Engram + 0.43B ViT + 0.85B DFly2 drafter), 11.3B text active + 0.43B vision per token.

### Text backbone (42 layers, hidden 3072, vocab 120,832, untied head kept F32)
- **Residual**: iHC "identity Hyper-Connection", `hc_mult` 4 streams `[tokens, 4, 3072]`. Per sublayer: `mixes = (fn @ flat) * rsqrt(mean(flat^2) + rms_norm_eps)`;
  `pre = sigmoid(mixes[:4]*scale[0] + base[:4]) + hc_eps`; `post = hc_magnitude(2.0) * sigmoid(mixes[4:]*scale[1] + base[4:]) + hc_eps`; collapse `sum_i pre_i * stream_i`;
  expand `stream'_i = post_i * sublayer_out + stream_i` (**no Sinkhorn, no comb matrix**). `fn` F32 `[8, 12288]`, `base` F32 `[8]`, `scale` F32 `[2]`; `hc_eps` 1e-6; `rms_norm_eps` 1e-5.
  Head `hc_head`: `fn [4,12288]`, `base [4]`, `scale [1]`: `pre = sigmoid((fn@flat)*rsqrt(..)*scale + base) + eps`, collapse, `model.norm`, then `lm_head` in F32.
- **Block order** (sandwich): `[engram add]` -> attn_hc collapse -> `input_layernorm` -> attention -> `post_self_attn_layernorm` -> attn_hc expand -> ffn_hc collapse -> `post_attention_layernorm` ->
  MLP/MoE -> (dense layer: `post_mlp_layernorm`; MoE layers: `combine_output_norm` inside the block) -> ffn_hc expand.
- **Attention**: gated MLA, `q_lora_rank` 1024, `kv_lora_rank` 512, `qk_nope` 192 + `qk_rope` 64 (qk head 256), `v_head_dim` 256, scale `256^-0.5`, RMS eps 1e-5, **no biases**, learned per-head sink `[heads]` F32
  (`learnable_sink_param`), sigmoid output gate from the hidden state (`linear_gate`, `[heads*256, 3072]`, applied after value mixing, before `o_proj`). The reference **expands** KV through `kv_b_proj` to full MHA; the absorbed MQA form the
  engine's DSA path uses is mathematically identical.
  - 12 **indexed** layers (0, 1, 5, 9, ..., 41): 32 heads, RoPE theta 1e7, compressed indexer below.
  - 30 **sliding** layers (the rest): **64 heads** (`swa_num_attention_heads`), RoPE theta 1e4, window 512 (511 preceding + self), same sink and gate, no indexer. Layer 2's weights confirm it: `q_b [16384,1024]`, `kv_b [28672,512]`, `o_proj [3072,16384]`.
  - RoPE: **interleaved 3D M-RoPE**, `mrope_section [12,10,10]` over 32 pair-frequencies (freq i takes the H plane if `i%3==1 && i<30`, the W plane if `i%3==2 && i<30`, else T), **interleaved-pair** rotation
    (even/odd pairs, cos repeated pairwise) on the trailing 64 channels of q and the shared k_rot. Text-only positions are one plane repeated.
- **Compressed indexer** (indexed layers only, `compressed_indexer_*`): `index_kpool` = `compressed_indexer_compress_ratio` 4, `index_topk` 2048 (512 groups), 16 heads, head dim 128, rope on the trailing 64.
  Token keys `token_key_norm(token_key_projection(h))` (RMS 128) are cached per token. Groups of 4 complete tokens are pooled in **F32**: `values = key_value_projection(group)`, `gates = key_gate_projection(group) + intra_group_ape[4,128]`,
  `pooled = sum_over_slots(values * softmax(gates over the 4 slots))`, then `key_norm`. Query `query_projection(q_resid)` (`q_resid` = q latent after `q_a_layernorm`), head weights `head_weight_projection(h) * 16^-0.5 * 128^-0.5`;
  queries and group keys get M-RoPE (a group key at its first token's positions). `logit = sum_h w_h * relu(q_h . k_group)`; groups >= the query's own group are masked; keep the top `512-1` complete groups plus the query's own group;
  attention then covers **every token of the recalled groups** (causal). **Below `index_topk` = 2048 keys the layer attends densely and skips the indexer.** No sliding window is added on indexed layers.
- **FFN**: layer 0 dense SwiGLU (intermediate 13,824). Layers 1-41 LatentMoE: router F32 `[512,3072]` on the full-width input, `sigmoid` scores, `e_score_correction_bias` F32 `[512]` added for selection only, `n_group` 1 / `topk_group` 1 (plain top-16),
  weights are the **unbiased** scores of the picked experts, normalised (`norm_topk_prob`, +1e-20), times `routed_scaling_factor` **3.998**. Routed experts run in a 1536-wide latent: `pre_expert_proj` `[1536,3072]` (checkpoint
  `routed_expert_down_proj`), experts `gate_up [512, 3072, 1536]` (chunk(2): gate half then up half per expert) and `down [512, 1536, 1536]`, activation **clamped plain SwiGLU**
  (`gate.clamp(max=10)`, `up.clamp(-10,10)`, `silu(gate)*up`; `swiglu_limit` 10.0), `post_expert_proj` `[3072,1536]` (checkpoint `routed_expert_up_proj`). One shared expert (SwiGLU, intermediate 1536) on the full-width input; the block returns
  `combine_output_norm(post_expert_proj(routed) + shared)` (RMS, 3072).
- **Engram** (decoder layers 1, 11, 21, 31 zero-based; config ids 2,12,22,32 are index+1): orders 2 and 3, 8 hash heads each, one prime-sized table per head in a flat per-layer
  `multi_head_embedding.embedding.weight [164,994,380, 96]` BF16 (31.7 GB per layer, 126.7 GB for four) with per-head row offsets. Token ids go through the frozen int32 `lookup_table [120818]` stored in the checkpoint,
  hashed as `XOR_k(compressed[pos-k] * multiplier_k) % prime_head` (pads before the sequence start; image/video/pad positions masked to zero). **Multipliers**:
  `numpy.random.default_rng(engram_seed + 10007*layer_id).integers(0, int64_max // compressed_vocab // 2, size=3, dtype=int64) * 2 + 1`, `compressed_vocab = max(lookup_table)+1`. Then per-head RMSNorm(96), flatten to 1536,
  per hc stream `key = RMSNorm(key_projs[i](emb) + bias)`, `query = RMSNorm(stream_i)`, `d = (key.query) * 3072^-0.5`, `gate = sigmoid(sqrt|d| * sign(d))`, `main = unified_pre_gamma * RMSNormNoAffine(value_proj(emb))`,
  `out_i = gate_i * main + unified_per_hc_scales[i] * main`, added to the four streams before attention. `engram_fp8_embedding` is **false** in the instruct and FP8 configs (the base config sets it true).
- **DFly2 drafter** (`model.dfly_layers.*`, 5 layers, 0.85B): tensors are in the checkpoint, ignored by the reference loader. Out of scope for the first series.

### Vision (HY4-VL tower reused; checkpoint prefix `vit.*`)
27 pre-LN ViT blocks (hidden 1152, 16 heads, MLP 4304 GELU, LayerNorm eps 1e-5, biased q/k/v/o), Conv2d patch embed 16x16 (`[1152,3,16,16]`), learned position grid 128x128 (+cls row, `[16385,1152]`) interpolated bilinearly
with `scale_factor=((size+0.1)/128, ...)` per image, **no 2D RoPE in the tower code** (the config's `rope_2d_*` fields are unused by the PR). Merger (`vit.perceive.*`): RMSNorm(1152), 2x2 stride-2 conv to 4608, GELU, 1x1 conv to 3072, plus a learned
`image_newline [3072]` after every merged row (an HxW merged grid emits `H*(W+1)` tokens). Image tokens replace the `<｜hy_image▁pad:opensource｜>` placeholders (id 120120; start 120118, end 120119, newline 120121). Positions: each
frame is `H` rows of `W+1` tokens, all three axes offset per frame block of `max(H, W+1)`, `next_text_position = max(ids)+1`. Preprocessor (`processor_config.json`): mean/std 0.5, patch 16, merge 2, `min_pixels` 4096, `max_pixels` 4,194,304, bilinear.
Video exists in the processor; the engine plans image input only.

### Chat/tokenizer
Instruct `chat_template.jinja` (17 KB) takes `reasoning_effort` (`high` default, `no_think`), tools (`<tool_calls:opensource>` family), `preserved_thinking`, `add_vision_id`; the base template (11.8 KB) has no chat roles.
Special tokens carry the fullwidth bar and `:opensource` suffix (instruct bos `<｜hy_start:opensource｜>`, eos `<｜hy_end:opensource｜>` id 120025; base `<｜hy_begin_of_sentence:opensource｜>` / `<｜hy_eod:opensource｜>`).
Generation defaults: temperature 0.9, top_p 1.0, top_k -1 (card: recommended `temperature=0.9`, `top_p=1.0`).

## Weight files and sizes

The header inventories of all three repos are committed as `tests/HartsyInference.LLM.Tests/Fixtures/Hy4Flash/*.tensors.json.gz`; `Hy4FlashWeightMapTests` pins every number below.

| Group | BF16 repo | FP8 repo |
|---|---|---|
| Engram tables (4 x 31.7 GB, BF16 in both) | 126.72 GB | 126.72 GB |
| Routed experts (41 layers, fused 3D tensors) | 297.16 GB (7.25 GB/layer) | 148.60 GB (3.62 GB/layer + BF16 `_scale_inv`) |
| Per-token weights (attention, router, shared experts, dense FFN, norms, projections) | 12.53 GB | 8.36 GB |
| embed + lm_head | 1.48 GB | 1.48 GB |
| ViT + merger | 0.93 GB | 0.93 GB |
| DFly2 drafter | 1.69 GB | 0.97 GB |
| Total (index `total_size`) | 440.5 GB (148 shards) | 287.1 GB (148 shards) |

FP8 layout: `quantization_config` sits **inside `text_config`** (`quant_method fp8`, `fmt e4m3`, `weight_block_size [128,128]`, dynamic activation). The routed experts are FP8 on every MoE layer
(`experts.gate_up_proj` F8_E4M3 `[512,3072,1536]` + `experts.gate_up_proj_scale_inv` BF16 `[512,24,12]`; `experts.down_proj` + `down_proj_scale_inv` `[512,12,12]`), and on layers 1-41 also `self_attn.o_proj`,
`q_b_proj`, `linear_gate` (`*.weight_scale_inv`, BF16). `modules_to_not_convert` (744 entries) keeps embed, head, hc fns, indexers, q_a/kv_a/kv_b, shared experts, router, engram, ViT in BF16. The scales are **BF16**, not the F32 grids the
engine's `Fp8BlockLinear` reads today: needs a BF16 grid reader. `hc_*`, `e_score_correction_bias`, sinks, the indexer pooling projections, key norm and `intra_group_ape` are F32 in both checkpoints; the router `mlp.gate.weight`, the Engram tables and the small Engram tensors are BF16; `lookup_table` is I32 (checked against the header inventories by `Hy4FlashWeightMap.Check`).

## Key map (checkpoint key to parameter)

Per decoder layer `model.layers.N.`: `input_layernorm`, `post_attention_layernorm`, `hc_attn_layer.hc_pre.{hc_fn,hc_base,hc_scale}` and `hc_mlp_layer.hc_pre.*` (iHC, F32),
`self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,o_proj,linear_gate,learnable_sink_param,post_proj_norm}`; indexed layers add
`self_attn.compressed_indexer.{query_projection,head_weight_projection,token_key_projection,token_key_norm,key_value_projection,key_gate_projection,key_norm,intra_group_ape}`;
layer 0 FFN is `mlp.{gate_proj,up_proj,down_proj}` + `mlp.post_fc2_norm`; layers 1-41 are `mlp.{gate.weight,gate.e_score_correction_bias,routed_expert_down_proj,routed_expert_up_proj,experts.gate_up_proj,experts.down_proj,shared_experts.*,combine_output_norm}`;
Engram layers 1, 11, 21, 31 add `engram_layer.{lookup_table,multi_head_embedding.embedding,lookup_per_head_norms.0-15,value_proj,key_projs.0-3,norm_key.0-3,norm_query.0-3,unified_pre_gamma,unified_per_hc_scales.0-3}`.
Global: `model.embed_tokens`, `model.norm`, `lm_head`, `model.hc_head.{hc_head_fn,hc_head_base,hc_head_scale}`, `vit.*` (tower and `vit.perceive.*` projector), `model.dfly_layers.*` (drafter, skipped).
Renames the reference applies at load (`conversion_mapping.py`): `learnable_sink_param` to `sinks`, `linear_gate` to `gate_proj`, `post_proj_norm` to `post_self_attn_layernorm`, `post_fc2_norm` to `post_mlp_layernorm`,
`routed_expert_down_proj` to `pre_expert_proj`, `routed_expert_up_proj` to `post_expert_proj`, `hc_*_layer.hc_pre.*` to `attn_hc.*` / `ffn_hc.*`. The engine keeps the checkpoint names.

## Open questions

- The transformers PR is unmerged and under review; the vLLM and SGLang implementations were not read. Only real weights or the HF classes run on a tiny config settle a disagreement.
- Shipped config keys the reference never reads: `qk_norm`, `gating_type`, `o_groups`, vision `rope_2d_*`, `compressed_indexer_skip_fine_select`, `dsa_skip_attn_post_norm`, `enable_lm_head_fp32`.
- Engram norms are `nn.RMSNorm(...)` without an eps argument: PyTorch then uses the dtype's machine epsilon, not `rms_norm_eps`. Treat the epsilon as a field until the HF run settles it.
- The Engram multiplier seed is the config layer id (2, 12, 22, 32), not the zero-based layer index; the draw needs `max(lookup_table) + 1`, which is in the checkpoint and has not been read.
- Engram n-grams in the pinned reference (`build_ngram_tokens`) shift raw ids with the pad id before a document start and zero the module's output at image, video and pad positions; unlike the DeepSeek-V4.1 hasher they do not block a lookback that crosses a dead token. The HF dump settles whether `forward` and `generate` differ here (they pass different document offsets).
- The hash multipliers are drawn with `engram_lookup_table_size` (120,818) when the module is built and redrawn with `max(lookup_table) + 1` after the checkpoint loads. They agree only if that maximum plus one is 120,818. Reading the four `lookup_table` tensors (about 483 KB each, an HTTP range read) would settle it; not done without the user's approval.
- `engram_fp8_embedding` is true in the base config and in the reference's default, false in the instruct and FP8 configs. When true the reference passes each looked-up row through a per-vector e4m3 round trip (`x + (q - x)`, value `q`); the engine implements it with the Engram module.

