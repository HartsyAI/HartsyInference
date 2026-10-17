# NVIDIA Nemotron Dory 1.0 — looped normalized-GPT decoder

Source snapshot 2026-10-17. Upstream: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B` (HF, gated; BF16, MXFP8 and NVFP4
derivatives), `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF` (one file, `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`,
3,673,288,864 bytes, sha256 `96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0`). Reference code:
vLLM PR [vllm-project/vllm#60149](https://github.com/vllm-project/vllm/pull/60149) (`vllm/model_executor/models/dory.py`,
`vllm/transformers_utils/configs/dory.py`, branch `shan18:add-dory`) and llama.cpp PR
[ggml-org/llama.cpp#30306](https://github.com/ggml-org/llama.cpp/pull/30306) (`src/models/dory.cpp`, `conversion/dory.py`,
branch `brianchmielNV:dory-support`, unmerged on 2026-10-17). The two agree on every operation below. Engine code:
`src/HartsyInference.LLM/Looped/`.

## Structure

- Three blocks of physical layers: input (I), recurrent (R), output (O). The card counts them in attention+MLP
  pairs (8/20/8); the checkpoint's `num_hidden_layers` counts single sublayers (vLLM's default config: 72 layers,
  `layer_types` = `mlp` on odd layers, `full_attention` where `i % 8 == 6`, else `sliding_attention`).
- Every physical layer is ONE sublayer: sliding-window attention (`S`), full attention (`*`) or a gated MLP (`-`).
  The GGUF stores the pattern as a string (`dory.layer_pattern`) with one character per physical layer.
- The recurrent block is applied `n_recurrent_loops` times. Per pass: `h = norm(state + input_h)`, run R,
  `state = norm(h)`. `input_h` is the input block's output, captured once (`initial_state = input_copy`,
  `input_transform = residual`, `output_transform = replace`). The trained range is 1 to 4 loops.
- No final norm (`ngpt_enable_final_norm = false`); the residual leaves every layer on the unit sphere.
- Logits: `lm_head(h) ⊙ logit_scale / init_std`, with a learned per-vocabulary `lm_head.logit_scale` vector.
  Embeddings are untied.

## Layer math (nGPT, `fngpt` variant)

`norm(x) = x / sqrt(sum(x²) + 1e-6)` over the last dimension. Scales are stored as `value · init_scaling / init`
and undone at inference: `init_std = init_method_std` (0.02 by default), `alpha_init = ngpt_alpha_init` (0.05),
`gate_init = attention_output_gate_logit_scale_init` or `sqrt(hidden_size)` when unset.

Attention (GQA, bias-free):

```
q, k, v, g = Wq x, Wk x, Wv x, Wg x
q, k = rope(q), rope(k)                       # NEOX rotate-half, full head, base per layer kind
q_h = (sqk_h · sqrt(d) / init_std)² ⊙ norm(q_h)   # per head, per dimension
k_h = norm(k_h)
a = softmax(q·kᵀ / sqrt(d)) v                   # causal; sliding layers see the last swa_window_size keys
a = norm(a ⊙ sigmoid(g ⊙ (s_gate · sqrt(gate_init) / init_std)²))   # norm over all heads at once
out = Wo a
```

MLP: `out = Wdown · norm(silu(Wgate x ⊙ suv_gate · sqrt(hidden)) ⊙ (Wup x ⊙ suv_up · sqrt(hidden)))`.

Residual (both kinds): `h' = norm(norm(h) + |alpha · alpha_init / init_std| ⊙ (norm(out) − norm(h)))`.

The engine bakes every scale to its effective value at load (`LoopedLayer.LoadWeights`) and expresses `norm` as
`RmsNorm` with a constant `1/sqrt(n)` weight and `eps/n` (`HypersphereOps`), which is exact.

## Key/value cache modes (`recurrent_kv_cache_mode`)

- `per_loop` (checkpoint default, what the weights were trained under): one cache per recurrent attention layer
  per loop. Chunk-invariant. Slots for 4 loops: the input and output blocks' attention layers plus 4× the
  recurrent block's.
- `last_loop`: one cache per recurrent attention layer, overwritten by every pass for the tokens of the current
  forward. Loop k sees loop-k keys for this chunk and last-loop keys for earlier chunks, so the output depends on
  how the prompt was chunked (llama.cpp's PR notes the same). Fixed memory whatever the loop count.

The engine assigns slots so that a cache sized for the maximum loop count serves every smaller one
(`LoopedSchedule`): input attention slots, then output attention slots, then one group per loop. A sequence
carries its loop count (`LoopedSequenceState`); a retained prefix is only reused by the same count.

## GGUF metadata (llama.cpp PR #30306)

| Key | Meaning |
|---|---|
| `dory.block_count` | physical layers (= `num_hidden_layers`) |
| `dory.input_layer_count`, `dory.recurrent_layer_count`, `dory.output_layer_count` | block sizes |
| `dory.num_loops` | default loop count (`n_recurrent_loops`); overridable at load with `--override-kv dory.num_loops=int:N` in llama.cpp, per request here |
| `dory.layer_pattern` | `S`/`*`/`-` per physical layer (`hybrid_layer_pattern`) |
| `dory.rope.profile` | uint32 per physical layer, 1 = `rope.freq_base`, 2 = `rope.freq_base_2` |
| `dory.rope.freq_base`, `dory.rope.freq_base_2` | the two bases (`rope_theta`, `rope_theta_2`) |
| `dory.attention.sliding_window` | `swa_window_size` |
| `dory.recurrent_kv_cache_mode` | `per_loop` or `last_loop` |
| `dory.init_std`, `dory.alpha_init`, `dory.gate_init` | the nGPT constants above |
| `dory.attention.key_length` = `value_length` = `rope.dimension_count` | head dimension (partial rotary unsupported) |

Tensors: `blk.N.{attn_q,attn_k,attn_v,attn_output,attn_gate}.weight` or `blk.N.{ffn_gate,ffn_up,ffn_down}.weight`,
the 1-D scales `blk.N.residual_alpha`, `blk.N.attn_qk_scale`, `blk.N.attn_gate_scale`, `blk.N.ffn_gate_scale`,
`blk.N.ffn_up_scale` (kept F32 in quantized files, BF16/F16 in those file types) and `output_scale`; HF names
`backbone.layers.N.mixer.*`, `backbone.layers.N.attn_alpha` / `mlp_alpha`, `mixer.sqk`, `mixer.s_gate`,
`mixer.suv_gate`, `mixer.suv_up`, `lm_head.logit_scale`. RoPE type is NEOX.

## Not done / open

- The HF repo is gated, so `config.json` was not read; the GGUF metadata carries everything the engine needs.
- NVFP4 (W4A16, the native pre-training format) and MXFP8 safetensors are not loaded: the engine reads GGUF only.
  The CUDA backend already dequantizes ModelOpt NVFP4 weights per GEMM (`Nvfp4ResidentCodec`), so a loader for the
  `backbone.*` safetensors layout is the missing piece, not a kernel.
- Real-weight verification and a llama.cpp speed comparison (no Dory-capable llama.cpp build existed on the
  swarm box on 2026-10-17).
- llama.cpp refuses K-shift (context shifting) for Dory because of the post-RoPE normalization; the engine has no
  context shifting to refuse.
