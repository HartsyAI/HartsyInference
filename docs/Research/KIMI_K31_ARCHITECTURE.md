# Kimi K3.1 Pro: architecture, checkpoint layout and reference formulas

Sources: `moonshotai/Kimi-K3.1-Pro` (`config.json`, `model.safetensors.index.json`, safetensors headers, read
2026-11-07); NVIDIA NeMo Automodel `nemo_automodel/components/models/kimi_k3/` (`model.py`, `kda_fused/`); the
Attention Residuals paper (arXiv 2603.15031); the Kimi Linear paper (arXiv 2510.26692) and flash-linear-attention's KDA
code. Tensor shapes and dtypes are from the shard headers. Parameter totals are computed from shapes, not measured.
The only public K3.1 model code is NeMo's; vLLM and SGLang have no K3.1 model file on main (the vLLM recipe says the
sparse-MLA path is refused on one AMD path).

## Top level

- `model_type` `kimi_k31`, architecture `KimiK31ForConditionalGeneration`; text decoder `text_config`
  `model_type` `kimi_linear`, class `KimiLinearForCausalLM` (NeMo: `KimiK3ForCausalLM`).
- Tensor prefixes: `language_model.model.*`, `language_model.lm_head.weight`, `vision_tower.*`, `mm_projector.*`.
- 96 shards, 497,340 tensor entries, 1,561,217,377,280 bytes.

## Text decoder dimensions

| Field | Value |
|---|---|
| layers | 93 (tensor indices 0 to 92) |
| hidden | 7168; vocab 163840; max positions 1,048,576 |
| attention heads | 96 (KV heads 96) |
| MLA (full layers) | `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 128, `qk_rope_head_dim` 64, `v_head_dim` 128 |
| KDA (linear layers) | 96 heads, head dim 128, short conv kernel 4, `use_full_rank_gate` true, `gate_lower_bound` -5.0 |
| dense MLP | layer 0 only, `intermediate_size` 33792 (`first_k_dense_replace` 1) |
| MoE | 896 routed, 16 active, 2 shared; sigmoid, `noaux_tc`, `num_expert_group` 1, `topk_group` 1, `routed_scaling_factor` 1.0, `moe_renormalize` true |
| routed expert | `moe_intermediate_size` 3072, latent width `routed_expert_hidden_size` 3584, `latent_moe_use_norm` true |
| activation | `hidden_act` `situ`, `activation_situ_beta` 4.0, `activation_situ_linear_beta` 25.0 |
| attention residuals | `attn_res_block_size` 12 |
| DSA indexer (full layers) | 32 heads, head dim 128, `index_topk` 2048, `index_rope_dim` 0, LayerNorm key norm |

Layer types (0-based tensor indices; the config lists them 1-based): KDA at 0, 1, 2, 4, 5, 6, ...; full attention with
DSA at 3, 7, 11, ..., 91, 92 (24 layers). Layer 0 is dense; layers 1 to 92 are MoE.

## Tensor families (checkpoint shapes, `[out, in]`)

Routed experts (MXFP4, layers 1 to 92, 896 each): `block_sparse_moe.experts.E.w1.weight_packed` [3072, 1792] U8,
`w1.weight_scale` [3072, 112] U8 (E8M0, group 32); `w3` the same; `w2.weight_packed` [3584, 1536], `w2.weight_scale`
[3584, 96]. No global scale tensor.

Latent MoE projections (bf16, per MoE layer): `routed_expert_down_proj` [3584, 7168] (hidden to latent),
`routed_expert_up_proj` [7168, 3584] (latent to hidden), `routed_expert_norm` [3584].
Router `block_sparse_moe.gate.weight` [896, 7168] bf16 and `gate.e_score_correction_bias` [896] F32.
Shared experts (bf16): `shared_experts.{gate_proj,up_proj}` [6144, 7168], `down_proj` [7168, 6144].
Dense layer 0 (bf16): `mlp.{gate_proj,up_proj}` [33792, 7168], `mlp.down_proj` [7168, 33792].

KDA layers: `self_attn.{q,k,v}_proj` [12288, 7168] bf16; conv `{q,k,v}_conv1d` [12288, 1, 4] F32; low-rank forget gate
`f_a_proj` [128, 7168] and `f_b_proj` [12288, 128]; `b_proj` [96, 7168] (beta, one per head); `A_log` **[128] F32**;
`dt_bias` [12288] F32; `g_proj` [12288, 7168] (output gate); `o_norm` [128] F32; `o_proj` [7168, 12288].

Full-attention layers: MLA `q_a_proj` [1536, 7168], `q_a_layernorm` [1536], `q_b_proj` [18432, 1536] (96 heads of
128 nope + 64 rope), `kv_a_proj_with_mqa` [576, 7168] (512 latent + 64 rope), `kv_a_layernorm` [512], `kv_b_proj`
[24576, 512] (96 heads of 128 k-nope + 128 v), `o_proj` [7168, 12288], `g_proj` [12288, 7168].
DSA indexer: `indexer.wq_b` [4096, 1536] (32 x 128 from the q latent), `indexer.wk` [128, 7168] (one key head),
`indexer.weights_proj` [32, 7168], `indexer.k_norm.{weight,bias}` [128].

Attention residuals (bf16): per layer `self_attention_res_proj` [1, 7168] and `self_attention_res_norm` [7168];
`mlp_res_proj` [1, 7168] and `mlp_res_norm` [7168]; at the output `output_attn_res_proj` [1, 7168] and
`output_attn_res_norm`.

Embeddings and head: `embed_tokens` [163840, 7168], `lm_head` [163840, 7168], final `model.norm`.

Vision tower: `patch_embed.proj` [1024, 3, 14, 14], `patch_embed.pos_emb` [64, 64, 1024], 27 blocks of `wqkv` [4608, 1024],
`wo`, `mlp.fc0`, `mlp.fc1`, `norm0`, `norm1`; `encoder.final_layernorm`. Projector `mm_projector.proj.0` [4096, 4096],
`proj.2` [7168, 4096], `post_norm`.

## Quantization

`compressed-tensors`, `mxfp4-pack-quantized`, group 32, scales U8, symmetric. Only the routed `w1/w2/w3` are packed.
Everything else listed above is bf16 (or F32 where noted).

## Memory (computed from shapes)

- Routed experts: 33.0M params per expert, 17.5 MB as MXFP4 with scales; 92 x 896 experts = 1.44 TB (about 92% of
  the checkpoint).
- Expert-parallel degree 8: 112 experts per GPU x 92 layers = about 180 GB per GPU.
- Dense bf16 (attention, KDA, shared experts, embeddings, latent projections, layer 0, vision): about 114 GB total.
  Replicated on every GPU that is about 294 GB, above a B300's 288 GB. So dense weights shard with tensor parallel
  degree 8 (about 14 GB per GPU).

## Reference formulas (resolved)

**KDA forget gate.** Raw gate `f = f_b_proj(f_a_proj(x))` per channel. Bounded form (the config sets
`gate_lower_bound` = -5, and NeMo uses this branch whenever a bound is set):
`g = -5 · sigmoid(exp(A_log) · (f + dt_bias))`, so `g` lies in (-5, 0) and the state decay is `exp(g)`.
Source: NeMo `_torch_kda_gate`. The default unbounded form would be `-exp(A_log) · softplus(f + dt_bias)`.

**KDA recurrence** (per head, state `S` is `[dk, dv]`, decay per key channel): `S ← Diag(exp(g_t)) · S`;
`kv = Sᵀ k_t`; `δ = (v_t − kv) · β_t`; `S ← S + k_t δᵀ`; `o_t = Sᵀ q_t`. `β = sigmoid(b_proj(x))`. `q` and `k`
are L2-normalized with eps inside the sqrt, and `q` is scaled by `dk^-0.5`. Short conv (depthwise, kernel 4, SiLU)
runs on q, k and v before this. State is F32; chunk size 64 in NeMo's CUDA kernel.

**KDA output.** `o = RMSNorm(o) · weight · sigmoid(g_proj(x))` (the gate multiplies after the norm, inside
`FusedRMSNormGated`), then `o_proj`.

**A_log shape conflict.** NeMo declares `A_log` as `[num_heads]` and views it as `[num_heads, 1]`, but the checkpoint has
`[128]` (head dim). The implementation takes `A_log` as one value per key channel, broadcast over heads, which is the
only shape-consistent reading. Confirm on real weights.

**MLA (full layers) is NoPE.** No rotary embedding is applied. The 64-dim rope slice of `kv_a_proj_with_mqa` is split
off and concatenated, unrotated, to the 128-dim nope part: q head dim 192, k head dim 192 (the rope part shared across
heads), v head dim 128. The attention scale for 192-dim heads is not confirmed; NeMo's `softmax_scale` not read.
Output: `attn · sigmoid(g_proj(x))`, then `o_proj`.

**DSA indexer.** `q = wq_b(q_latent).view(32, 128)`; `k = LayerNorm(wk(x))` (one head); head weights
`w = weights_proj(x) · (32·128)^-0.5`. Score per causal key: `Σ_h relu(q_h · k) · w_h`, no softmax. Keep the top 2048
keys per query (ascending). If the query has at most 2048 causal keys, all are kept (short context is full attention).
The indexer runs frozen (NeMo doc).

**Attention residuals.** Blocks of `attn_res_block_size` = 12 layers (NeMo: a new block when `layer_idx % 12 == 0`),
so about 8 blocks. Sources for a sublayer are the completed block summaries plus the current partial sum (the embedding
is the first source). Keys are `RMSNorm(source)` with the learned norm weight; the logit is the dot of the normalized
key with the learned pseudo-query (`proj.weight`, zero-initialized in the paper); softmax over the sources; output is the
weighted sum of the raw sources. Each transformer layer has two of these (before attention, before the MLP), each with
its own query and norm. Output: the same mix over the blocks, then the final norm (NeMo `_apply_output_attn_res`).
The RMSNorm eps is not confirmed (1e-6 used).

**Router.** `s = sigmoid(logits)` in F32. Selection uses `s + e_score_correction_bias`; weights are the uncorrected `s`
of the chosen experts. Renormalized when top-k > 1, then scaled by `routed_scaling_factor` (1.0).

**Latent MoE order** (NeMo `KimiK3MoE.forward`): `x_latent = routed_expert_down_proj(x)` (7168 to 3584); experts run on
`x_latent`; `routed_expert_norm` on the result; `routed_expert_up_proj` back to 7168; the shared experts run on the
original hidden input and their output is added.

**Activation `situ`.** `gate, up = x.chunk(2)` in NeMo; with separate `w1` (gate) and `w3` (up) the gate is `w1`.
Value: `4·tanh(g/4)·sigmoid(g) · 25·tanh(u/25)`. Same scalar as the engine's SiTU-GLU (`SituGluReference.Scalar`, which
takes (value, gate)). NeMo's `situ.py` uses `tanh` soft bounds, not a hard clamp.

## Remaining unknowns (need weights or the K3 modeling file)

1. The `A_log` shape reading above.
2. The attention-residual RMSNorm eps and the output mix details; the NeMo forward code past the first 100k characters.
3. The attention scale for 192-dim MLA heads (NeMo `softmax_scale`).
4. Whether `g_proj` on full layers gates the attention output exactly as NeMo states (sigmoid then `o_proj`).
5. Vision tower RoPE variant (interleaved or rotate-half) and its exact merge order.
