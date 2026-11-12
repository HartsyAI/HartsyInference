# Kimi K3.1 Pro: architecture, checkpoint layout and reference formulas

Sources. Model repo `moonshotai/Kimi-K3.1-Pro` (read 2026-11-12): `config.json`, `LICENSE`, `configuration_kimi_k3.py`,
`kimi_k3_vision_processing.py`, `media_utils.py`, the HF model API (`/api/models/moonshotai/Kimi-K3.1-Pro`, `/tree/main`),
shard 2 safetensors header (range request) and its `A_log` bytes, and `model.safetensors.index.json`. NVIDIA NeMo Automodel
`nemo_automodel/components/models/kimi_k3/` at `main` (`model.py`, `situ.py`, `attn_res_triton.py`, `vision.py`,
`state_dict_adapter.py`). Attention Residuals paper, arXiv 2603.15031, TeX source (`https://arxiv.org/e-print/2603.15031`:
`3-attnres.tex`, `figures/code.tex`, `5-exp.tex`). flash-linear-attention at `main`
(`fla/modules/norm/fused_norm_gate.py`, `fla/ops/kda/chunk.py`). Tensor shapes and dtypes are from the shard headers.
Parameter totals are computed from shapes, not measured. The HF repo has no modeling file (`auto_map` lists only
`AutoConfig`), so every forward-pass fact below comes from NeMo's implementation or the paper, not from K3 release code.
vLLM and SGLang have no K3.1 model file on `main`.

## Top level

- `model_type` `kimi_k31`, architecture `KimiK31ForConditionalGeneration`; text decoder `text_config` `model_type`
  `kimi_linear` (NeMo: `KimiK3ForCausalLM`).
- Tensor prefixes: `language_model.model.*`, `language_model.lm_head.weight`, `vision_tower.*`, `mm_projector.*`.
- 96 safetensors shards. Shard sizes sum to 1,561,293,160,568 bytes; the whole repo (112 files, including the 16
  non-weight files) is 1,561,355,966,953 bytes (HF tree API, 2026-11-12). The earlier figure 1,561,217,377,280 bytes does
  not match this and is withdrawn. The "497,340 tensor entries" count is not re-verified here.

## Text decoder dimensions

| Field | Value |
|---|---|
| layers | 93 (tensor indices 0 to 92) |
| hidden | 7168; vocab 163840; max positions 1,048,576 |
| attention heads | 96 (KV heads 96) |
| MLA (full layers) | `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 128, `qk_rope_head_dim` 64, `v_head_dim` 128, `mla_use_nope` true, `mla_use_output_gate` true |
| KDA (linear layers) | 96 heads, head dim 128, short conv kernel 4, `use_full_rank_gate` true, `gate_lower_bound` -5.0 |
| dense MLP | layer 0 only, `intermediate_size` 33792 (`first_k_dense_replace` 1) |
| MoE | 896 routed, 16 active, 2 shared; sigmoid, `noaux_tc`, `num_expert_group` 1, `topk_group` 1, `routed_scaling_factor` 1.0, `moe_renormalize` true |
| routed expert | `moe_intermediate_size` 3072, latent width `routed_expert_hidden_size` 3584, `latent_moe_use_norm` true |
| activation | `hidden_act` `situ`, `activation_situ_beta` 4.0, `activation_situ_linear_beta` 25.0 |
| attention residuals | `attn_res_block_size` 12 (see the AttnRes section for the convention) |
| `rms_norm_eps` | 1e-05 (`text_config`, `config.json`) |
| DSA indexer (full layers) | 32 heads, head dim 128, `index_topk` 2048, `index_rope_dim` 0, LayerNorm key norm (`index_k_norm_type` "layernorm") |

Layer types. The config lists `kda_layers` and `full_attn_layers` 1-based (`linear_attn_config`, `config.json`):
`full_attn_layers` is 4, 8, ..., 92 and 93. In 0-based tensor indices, full attention (MLA with DSA) is at 3, 7, 11, ..., 91,
92 (24 layers), and KDA is everywhere else (69 layers). Layer 0 is dense; layers 1 to 92 are MoE.

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
`f_a_proj` [128, 7168] and `f_b_proj` [12288, 128]; `b_proj` [96, 7168] (beta, one per head); `A_log` **F32 [128] stored,
of which the first 96 entries are used (one per head) and entries 96 to 127 are zero padding** (see the A_log section);
`dt_bias` [12288] F32; `g_proj` [12288, 7168] (output gate); `o_norm` [128] F32; `o_proj` [7168, 12288].
Shapes above are verified from the layer 1 header of shard 2 (`f_a_proj` BF16 [128, 7168], `f_b_proj` BF16 [12288, 128],
`A_log` F32 [128], `dt_bias` F32 [12288]).

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

`compressed-tensors`, `mxfp4-pack-quantized`, group 32, scales U8, symmetric (`config.json`, `quantization_config`). Only
the routed `w1/w2/w3` are packed. The `ignore` list keeps attention (`self_attn`), shared experts, dense MLPs, the router,
the latent projections, `lm_head`, the vision tower and the projector in bf16.

## Memory (computed from shapes)

- Routed experts: 33.0M params per expert, 17.5 MB as MXFP4 with scales; 92 x 896 experts = 1.44 TB (about 92% of
  the checkpoint).
- Expert-parallel degree 8: 112 experts per GPU x 92 layers = about 180 GB per GPU.
- Dense bf16 (attention, KDA, shared experts, embeddings, latent projections, layer 0, vision): about 114 GB total.
  Replicated on every GPU that is about 294 GB, above a B300's 288 GB. So dense weights shard with tensor parallel
  degree 8 (about 14 GB per GPU).

## Reference formulas

### KDA forget gate (verified)

Raw gate `f = f_b_proj(f_a_proj(x))` per channel (no bias in the projections). NeMo `model.py:137-162`
(`_torch_kda_gate`):

```
gate = gate.float() + dt_bias.float().view(num_heads, head_dim)
decay = a_log.float().view(num_heads, 1).exp()
if lower_bound is not None:
    return lower_bound * torch.sigmoid(decay * gate)
return -decay * F.softplus(gate)
```

So `g = -5 · sigmoid(exp(A_log) · (f + dt_bias))` with `lower_bound` = `gate_lower_bound` = -5.0 (`config.json`,
`linear_attn_config`). The state decay is `exp(g)`. The unbounded branch (`-exp(A_log) · softplus(f + dt_bias)`) is not used
by K3 (the config sets the bound). NeMo's FLA fused path (`model.py:164-184`) is used when available and receives the same
`lower_bound`.

### KdaForgetRank (verified: 128, derived, not a config key)

`config.json` has no forget-rank key. NeMo builds `f_a_proj = nn.Linear(hidden_size, self.head_dim)` (`model.py:1312`), so
the rank is `linear_attn_config.head_dim` = 128. The shard header confirms `f_a_proj` [128, 7168]. The engine hard-codes
`KdaForgetRank = 128` (`src/HartsyInference.LLM/Kimi/KimiK31HfConfig.cs:36-37`), which is correct.

### A_log (resolved: one value per head, not per key channel)

The doc previously said `A_log` is per key channel. That is wrong. Evidence:

- The checkpoint stores 128 F32 entries. Layer 1 bytes (`model-00002-of-000096.safetensors`, data offsets 3584 to 4096,
  read by range request) hold 96 non-zero values (entry 0 is -0.02843385, entry 95 is 0.38348693) followed by 32 exact
  zeros (entries 96 to 127).
- NeMo's `state_dict_adapter.py:362-370` truncates a stored `A_log` to `num_heads` entries and its `_pad_checkpoint_a_log`
  (lines 372-383) restores the 128-entry layout, so the zero padding is a storage artifact.
- `KimiKDAFp32Params` declares `A_log` as `torch.empty(num_heads)` (`model.py:1224`), and the decay is
  `a_log.view(num_heads, 1).exp()` (`model.py:159`), i.e. one decay per head broadcast over that head's 128 channels.

Consequence for the engine: `KimiK31WeightLoader.cs:103-106` reads 128 entries and uses `aLogHead[ch % KdaHeadDim]`, which
indexes by channel-within-head and pulls the zero padding in for channels 96 to 127 of each head. It should read the first
96 entries and use `aLog[ch] = aLogHead[ch / KdaHeadDim]`. See the engine findings below. Only layer 1 was byte-checked; the
padding pattern for other layers is inferred from NeMo's adapter, not measured.

### KDA recurrence

Per head, state `S` is `[dk, dv]`, decay per key channel: `S ← Diag(exp(g_t)) · S`; `kv = Sᵀ k_t`; `δ = (v_t − kv) · β_t`;
`S ← S + k_t δᵀ`; `o_t = Sᵀ q_t`. `β = sigmoid(b_proj(x))` (`model.py:1481`). Short conv (depthwise, kernel 4, SiLU) runs on
q, k and v (`model.py:1282-1309`). q and k are L2-normalized in the kernel by default (`kda_use_qk_l2norm_in_kernel`,
`model.py:1493`); when the kernel does not do it, NeMo uses `F.normalize(..., eps=1e-6)` (`model.py:1495-1496`).
Score scale: NeMo passes no `scale` to the KDA kernel (the only `scale=` uses are in MLA), so FLA's default
`scale = K ** -0.5` applies (`fla/ops/kda/chunk.py` line 222 `scale: float | None = None`; lines 494-495 `if scale is None: scale = K ** -0.5`).
The FLA kernel body was not read to confirm where the scale is applied. State is F32; chunk size 64 in NeMo's CUDA kernel
(not re-checked in this pass).

### KDA output gate and norm (verified: gate after the norm)

NeMo `model.py:1322-1330` builds `o_norm = FusedRMSNormGated(head_dim, eps=config.rms_norm_eps, activation="sigmoid")`.
`model.py:1552-1556` computes `gate = self.g_proj(hidden_states)` then `o = self.o_norm(o, gate)` then `o_proj`. FLA's
`fla/modules/norm/fused_norm_gate.py` forward kernel (lines 94-104) computes the normalized row first (`b_y = b_x_hat * b_w`)
and then multiplies by `sigmoid(g)` for `activation == "sigmoid"`. So `o = RMSNorm(o) · weight · sigmoid(g_proj(x))`, then
`o_proj`. The norm eps is `config.rms_norm_eps` = 1e-5 (see the RMSNorm eps section).

### RMSNorm eps (resolved: 1e-5 in K3; 1e-6 is only a Python default)

- `config.json` `text_config.rms_norm_eps` = `1e-05`.
- NeMo passes `config.rms_norm_eps` to every norm it builds: KDA output (`model.py:1325`), decoder norms (`model.py:1832-1833`),
  AttnRes norms (`model.py:1839-1844`), the routed-expert latent norm (`model.py:1701`), the indexer k-norm (`model.py:457-459`).
- `1e-6` appears only as the default argument of `KimiRMSNorm.__init__` (`model.py:343`) and of `KimiK3Config`
  (`configuration_kimi_k3.py:22`, `rms_norm_eps=1e-6`). `config.json` overrides both.
- The q/k L2-norm eps of 1e-6 (above) is a separate constant.

The engine uses `_config.RmsNormEps` (1e-5) for KDA and decoder norms (`KimiK31Model.cs:284, 300`) but not for AttnRes (see
the engine findings).

### MLA (full layers): NoPE and scale (verified)

- `config.json`: `mla_use_nope` true. NeMo raises if it is false (`model.py:606-607`).
- No rotary embedding is applied. `q_rot` and `k_rot` are split off (`model.py:753-756`), `k_rot` is shared across heads
  (`:761-762`), and both are concatenated unrotated to the nope parts (`:764-765`). There is no rotary call in MLA.
- Head dim 192 (128 nope + 64 rope) for q and k; v head dim 128.
- Scale: `self.scaling = self.q_head_dim**-0.5` (`model.py:608`), i.e. `192^-0.5`. Passed as `softmax_scale`/`scale` to every
  MLA path (`model.py:667, 783, 786, 789, 1030, 1049`). No YaRN `mscale` term appears in the file.
- Layers: MLA at 0-based 3, 7, ..., 91, 92 (from `full_attn_layers`; NeMo selects KDA by `(layer_idx + 1) in kda_layers`,
  `configuration_kimi_k3.py` `is_kda_layer`).
- Output gate: `attn_output = attn_output * self.g_proj(hidden_states).sigmoid()` (`model.py:843-844`), then
  `return self.o_proj(attn_output)` (`model.py:845`). The same order holds on the other two MLA paths (`:904-905`, `:1053-1054`).

### DSA indexer (unchanged, not re-verified in this pass)

`q = wq_b(q_latent).view(32, 128)`; `k = LayerNorm(wk(x))` (one head); head weights `w = weights_proj(x) · (32·128)^-0.5`
(NeMo `model.py:451, 502`: `score_scale = (head_dim * num_heads) ** -0.5`). Score per causal key: `Σ_h relu(q_h · k) · w_h`,
no softmax. Keep the top 2048 keys per query (ascending). If the query has at most 2048 causal keys, all are kept. The
indexer runs frozen (NeMo doc).

### Attention residuals (verified against the paper's TeX source and NeMo)

Paper equations (`3-attnres.tex`, eq. "attn-weights", "qkvattn", "full-attn"): the weight of source `i` for layer `l` is a
softmax over sources of `φ(q_l, k_i) = exp(q_lᵀ RMSNorm(k_i))`, with a learned per-layer pseudo-query `q_l = w_l` and
sources `v_0 = h_1` (the embedding) and `v_i = f_i(h_i)`. The output is `Σ_i α_{i→l} v_i` over the raw sources.

Block AttnRes (`3-attnres.tex` lines 50-73; `figures/code.tex` lines 12-49): block `n` sum `b_n = Σ_{j∈B_n} f_j(h_j)`;
`b_0 = h_1` is always a source. Within a block, the partial sum `b_n^i` is the running sum of sublayer outputs. The
attention mix for the first layer of a block uses `[b_0..b_{n-1}]`; later layers add the partial sum.

NeMo implementation (verified line by line):

- Per sublayer: entries `[block_residual..., prefix_sum]` (`attn_res_triton.py` docstring lines 17-26 and `situ.py:670-690`,
  `_apply_attn_res`). Score per entry: `s_j = <RMSNorm(x_j) ⊙ norm_weight, proj_weight>`, i.e.
  `rsqrt(mean(x_j²) + eps) · Σ_d x_jd · nw_d · pw_d` (kernel `attn_res_triton.py:124-147`). Softmax over the `k+1` entries
  per token (`:148-151`). Output `Σ_j p_j x_j` over the raw (un-normalized) entries (`:153-166`). The mix is a convex
  combination: it does not rescale beyond the softmax weights.
- Block boundary (`model.py:1932-1937`): `if self.layer_idx % self.attn_res_block_size == 0:` the current stream is appended
  to `block_residual` and `prefix_sum` is set to `None`. The next partial sum is the attention output alone.
- Layer 0: `block_residual` starts empty (`model.py:2256-2260`, shape `[tokens, 0, hidden]`). The attention mix is skipped
  when `block_residual.shape[1] == 0` (`model.py:1924`), so the attention input is the embedding; the embedding is then
  appended as `b_0` at layer 0 (`:1932`).
- Attention input: `input_layernorm(mix(prefix_sum, block_residual))` (`:1925-1930, 1939-1940`). MLP input:
  `post_attention_layernorm(mix(prefix_sum, block_residual))` (`:1946-1952`). The residual stream returned is the raw
  `prefix_sum + mlp_output` (`:1959`).
- Output: `hidden = self.norm(mix(hidden, block_residual))` with `output_attn_res_proj` and `output_attn_res_norm`
  (`model.py:2284-2286, 2288-2300`).

Answers to the open questions:

- Is a zero partial source included? No. In NeMo there is no zero entry at layer 0 (the mix is skipped), and at a block
  boundary the partial sum is replaced by the sublayer output, not by a zero vector added to the softmax. The paper's listing
  does the same (`partial_block = None` then `partial_block = attn_out`, `code.tex` lines 35-40).
- Does a zero entry scale the mix? Only if included. A zero source gets score 0 and weight `exp(0)/Z`, so it shrinks the
  mixed state by its weight. Neither NeMo nor the paper does this.
- Eps of the AttnRes RMSNorm: NeMo passes `config.rms_norm_eps` (1e-5). The paper gives no epsilon (no `epsilon` string in
  the TeX source).
- Initialization: the paper states "all pseudo-query vectors must be initialized to zero" (`5-exp.tex` line 8). NeMo's
  `init_weights` uses `nn.init.normal_` (`model.py:1971-1972`). Training-time only; it does not change inference.
- Block convention: the paper's listing uses `block_size // 2` (`code.tex` line 34; the comment on line 33 says "block_size counts ATTN + MLP"), which
  for K3's 12 would be 6 transformer layers per block (about 16 blocks). NeMo uses 12 transformer layers per block, giving 8
  block entries (layers 0, 12, ..., 84), and its kernel comment says "Kimi-K3 (93 layers, block size 12) reaches 8"
  (`attn_res_triton.py:70-71`). The paper's text says "we fix the number of blocks to ≈ 8" (`5-exp.tex`). NeMo's convention matches
  that text. The checkpoint shapes do not distinguish the two (both give the same tensor shapes).

### Router (unchanged, not re-verified in this pass)

`s = sigmoid(logits)` in F32. Selection uses `s + e_score_correction_bias`; weights are the uncorrected `s` of the chosen
experts. Renormalized when top-k > 1, then scaled by `routed_scaling_factor` (1.0).

### Latent MoE order (unchanged, not re-verified in this pass)

`x_latent = routed_expert_down_proj(x)` (7168 to 3584); experts run on `x_latent`; `routed_expert_norm` on the result;
`routed_expert_up_proj` back to 7168; the shared experts run on the original hidden input and their output is added.

### Activation `situ` (unchanged, not re-verified in this pass)

`gate, up = x.chunk(2)` in NeMo; with separate `w1` (gate) and `w3` (up) the gate is `w1`. Value:
`4·tanh(g/4)·sigmoid(g) · 25·tanh(u/25)`. NeMo's `situ.py` uses `tanh` soft bounds, not a hard clamp.

## Vision tower (verified against NeMo `vision.py`, not against weights)

- Patchify (`media_utils.py:331-336`, `navit_patchify`): the image is reshaped to `(T, H/p, p, W/p, p, C)`, transposed to
  `(T, H/p, W/p, C, p, p)`, so patches are in raster order (row-major over the patch grid), each patch as `[C, p, p]`.
- RoPE (`vision.py:163-181`, `apply_rope`): `torch.view_as_complex(xq.float().view(..., -1, 2))`, i.e. complex rotation of
  adjacent pairs `(2i, 2i+1)`. This is INTERLEAVED, not rotate-half.
- 2D frequencies (`vision.py:359-381`, `Rope2DPosEmbRepeated._precompute_freqs_cis`): `x_pos = flat % max_width` (column)
  and `y_pos = flat // max_width` (row); the code interleaves them as `cat([x_cis, y_cis], -1)` (lines 378-380), so the
  even slot `2i` holds the WIDTH (column) frequency and the odd slot `2i+1` the HEIGHT frequency. The docstring (lines 363-364)
  says the reverse. The code is what runs. Head dim is `qkv_hidden_size / num_heads` = 1536 / 12 = 128 (`vision.py:541`,
  frequencies `theta_base` 10000).
- Patch merge (`vision.py:579-600`, `tpool_patch_merger`, merge type `sd2_tpool`): for each `(t, h, w)`, reshape to
  `(t, h/2, 2, w/2, 2, d)`, permute to `(t, h/2, w/2, 2, 2, d)`, mean over `t` (temporal pooling, a no-op for images), then
  `view(new_h·new_w, 4, d)`. Within each 2x2 group the four tokens are ordered `[(kh=0,kw=0), (0,1), (1,0), (1,1)]`
  (top-left, top-right, bottom-left, bottom-right), and the merged vectors are concatenated in that order along channels
  (`view(B, -1, 4·1024)` in `PatchMergerMLPV2`, `vision.py:756`). The merged grid is row-major.
- Projector (`vision.py:730-758`, `PatchMergerMLPV2`, selected by `mm_projector_type` "patchmergerv2"): `proj` =
  `Linear(4096, 4096, bias=False)` · exact GELU · `Linear(4096, 7168, bias=False)`, then `post_norm = RMSNorm(7168,
  eps = projector_ln_eps = 1e-5)`. No `pre_norm`, which matches the checkpoint keys (`proj.0`, `proj.2`, `post_norm`).
- Position embedding: `pos_emb_interpolation_mode` "bilinear" from `config.json` (`vision.py:630` reads the config value).
- Encoder: 27 layers, `vt_hidden_size` 1024, `qkv_hidden_size` 1536 (12 heads of 128), `vt_intermediate_size` 4096,
  `patch_size` 14, `merge_kernel_size` [2, 2] (`config.json` `vision_config`).

## License and gating (verified)

- Repo: `gated` is `false` and `private` is `false` (HF model API, 2026-11-12). Tags include `license:other`;
  `cardData.license_name` is `kimi-k3.1-pro`. The license file is `LICENSE` (3,077 bytes, "Kimi K3.1-Pro License",
  Copyright (c) 2026 Moonshot AI), linked from `README.md` section 6.
- Terms, quoted from `LICENSE` (`https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/LICENSE`):
  - Permission: "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software"; the
    Software includes "the model weights, parameters, configuration files, inference and training code".
  - Section 1: the notice "shall be included in all copies or substantial portions of the Software".
  - Section 2 (Model as a Service): applies to a Model as a Service operator whose "aggregate revenue ... exceeds 20 million US
    dollars (or the equivalent in other currencies) in total over any consecutive 12 months" must "enter into a separate
    agreement with Moonshot AI before using the Software ... for any commercial purpose". MaaS excludes "end-user products
    with model capabilities solely embedded within specific features or harnesses".
  - Section 3: where the Software is used in a commercial product with "more than 100 million monthly active users, or more
    than 20 million US dollars ... in monthly revenue", "Kimi K3.1-Pro" "must be prominently displayed on the user interface".
  - Section 4: Sections 2 and 3 do not apply to internal use (use that does not make the Software, its outputs, or its
    capabilities available to third parties), or to use "accessed through Moonshot AI's official products or certified
    inference partners".
  - Section 5: "AS IS" with no warranty.
  - Territory: the license has no territory or geographic restriction.
- Practical reading (not legal advice): a self-hosted, internal or commercial product below the thresholds needs the notice
  kept; above them, Section 2 and Section 3 apply.

## Engine findings (not changed in this PR)

Found while checking the engine's Kimi code against the reference above. None of these were changed here. The rule for this
PR is that engine code changes only for a provably wrong constant with a test, and the items below are not that, so they
need their own PR with decoder-level tests.

1. AttnRes includes a zero partial source at layer 0 and at every block boundary (wrong).
   `src/HartsyInference.LLM/Kimi/KimiK31Model.cs:220-228` clears `state.Partial` at layer 0 and after each block push, and
   `:266-276` (`Mix`) stacks `Blocks` plus the partial, so the attention mix at those layers includes a zero row. At layer 0
   the attention input becomes `p·embedding` instead of the embedding (NeMo `model.py:1924`: no mix when no blocks; paper
   `code.tex`: the first layer's sources are `[b_0]`). At a block boundary the engine pushes the partial and then mixes with an
   extra zero row (NeMo mixes before the push, over `[b_0..b_{n-1}, partial]`).
2. AttnRes RMSNorm eps is 1e-6, not the config's 1e-5 (constant mismatch against NeMo `model.py:1839-1844`).
   `KimiK31Model.cs:275` calls `AttentionResidual.Mix(...)` without the eps argument, so `AttentionResidual.cs:19` uses its
   default `eps = 1e-6f`. The fix is to pass `_config.RmsNormEps`. Not applied because finding 1 sits on the same path, and a
   test for the wiring needs a decoder-level reference.
3. A_log folded per channel from the wrong entries (wrong).
   `src/HartsyInference.LLM/Kimi/KimiK31WeightLoader.cs:103-106` reads `c.KdaHeadDim` (128) entries and uses
   `aLogHead[ch % c.KdaHeadDim]`. It should read the first `num_heads` (96) entries and use `aLogHead[ch / c.KdaHeadDim]`
   (see the A_log section). The comment there ("one value per head-channel, shared by every head") and the description at
   `src/HartsyInference.LLM/Kimi/KimiK31Reference.cs:23-26` ("the channel's A_log entry") are wrong.
4. `KdaForgetRank = 128` is correct (`KimiK31HfConfig.cs:36-37`); the comment matches the verified derivation.

## Resolved and open unknowns

| # | Question | Status | Evidence |
|---|---|---|---|
| 1 | AttnRes block-summary and pseudo-query formulas | Verified (paper TeX source and NeMo) | arXiv e-print 2603.15031 `3-attnres.tex`, `figures/code.tex`; NeMo `model.py:1832-1959`, `situ.py:632-690`, `attn_res_triton.py:124-171` |
| 1a | Zero partial source at layer 0 and block boundaries | Verified: not included in NeMo or the paper. The engine includes it (finding 1) | As above; `KimiK31Model.cs:220-228, 266-276` |
| 1b | Output mix and final norm | Verified | NeMo `model.py:2284-2300` |
| 1c | AttnRes RMSNorm eps | Verified for NeMo (1e-5 from config); paper silent | NeMo `model.py:1839-1844`; `config.json` |
| 1d | Block-count convention (12 transformer layers vs `block_size // 2` in the paper listing) | NeMo convention adopted (8 entries, matches "≈ 8" in the paper). Not confirmed on weights | NeMo `model.py:1932`, `attn_res_triton.py:70-71`; `5-exp.tex`, `code.tex` line 34 |
| 2 | KDA gate formula with `gate_lower_bound` -5 | Verified: `g = -5 · sigmoid(exp(A_log) · (f + dt_bias))` | NeMo `model.py:137-162`; `config.json` `linear_attn_config.gate_lower_bound` |
| 2a | A_log shape and semantics | Resolved: per head, 96 used plus 32 zero padding (layer 1 byte-checked). The engine is wrong (finding 3) | Shard 2 header and bytes; NeMo `state_dict_adapter.py:362-383`, `model.py:1224, 159` |
| 2b | KdaForgetRank | Verified: 128 = `head_dim`, not a config key | NeMo `model.py:1312`; shard header `f_a_proj` [128, 7168] |
| 2c | RMSNorm eps | Verified: 1e-5 (config); 1e-6 is only a Python default | `config.json` `rms_norm_eps`; NeMo `model.py:343, 1325`; `configuration_kimi_k3.py:22` |
| 2d | Output gate after the norm | Verified | NeMo `model.py:1552-1556`; FLA `fused_norm_gate.py` lines 101-104 |
| 2e | KDA q scale `dk^-0.5` | Scale default verified in FLA `chunk_kda` (`K ** -0.5`, lines 222, 494-495); kernel body not read | `fla/ops/kda/chunk.py`; NeMo passes no `scale` |
| 3 | MLA attention scale for 192-dim heads | Verified: `192^-0.5`, no mscale | NeMo `model.py:608, 667, 783, 789` |
| 3b | NoPE on full-attention layers (0-based 3, 7, ..., 91, 92) | Verified | `config.json` `mla_use_nope`; NeMo `model.py:606-607, 753-765`; `full_attn_layers` |
| 3c | `g_proj` gates MLA attention output before `o_proj` | Verified against NeMo only | NeMo `model.py:843-845` |
| 4 | Vision RoPE variant | Verified against NeMo: interleaved pairs; even slot is width, odd is height | NeMo `vision.py:163-181, 359-381` |
| 4b | Vision patch-merge order | Verified against NeMo: raster patches; 2x2 groups in TL, TR, BL, BR order, channel-concatenated | `media_utils.py:331-336`; NeMo `vision.py:579-600, 730-758` |
| 5 | Checkpoint file list and size | Verified from the HF tree API: 96 shards, 112 files, 1,561,355,966,953 bytes in total | HF `/api/models/moonshotai/Kimi-K3.1-Pro/tree/main?recursive=true` |
| 5b | License and gating | Verified: `gated` false; "Kimi K3.1-Pro License" (see above) | HF model API; `LICENSE` |
| - | Whether any of the forward-pass facts match the released weights | Open. Only NeMo's implementation and the paper were checked. No K3 modeling file is in the repo | Would be settled by a real-weight generation check (CPU reference or NeMo forward) on the pod |
| - | `A_log` padding in layers other than 1 | Open. Inferred from NeMo's adapter | Byte-check more layers' `A_log` tails |
