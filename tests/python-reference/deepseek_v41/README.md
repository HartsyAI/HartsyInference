# DeepSeek-V4.1-Flash reference harness

Offline Python reference for the DeepSeek-V4.1-Flash port. Python is used only here; the engine stays pure C#.
Frozen facts and open reference gaps: [DEEPSEEK_V41_ARCHITECTURE.md](../../../docs/Research/DEEPSEEK_V41_ARCHITECTURE.md).

## Setup

```bash
tests/python-reference/deepseek_v41/setup_env.sh
```

Builds a venv at `~/dsv41-ref/.venv` (override with `DSV41_REF_ROOT`), installs `requirements.txt`, and fetches the
non-weight files of the official repo at the pinned revision into `~/dsv41-ref/upstream`, verified against `upstream.sha256`.
Re-check without downloading: `python fetch_upstream.py --verify-only`.

## Files

| File | Purpose |
|---|---|
| `kernel_ports.py` | Pure-torch stand-in for the tilelang `kernel.py`, so the unmodified upstream `model.py` runs on CPU |
| `test_kernel_ports.py` | Checks each port against an independent scalar or float64 implementation |
| `run_small_config.py` | Runs upstream `Transformer` on two small seeded configs and dumps every intermediate |
| `dump_engram_constants.py` | Dumps the Engram token map, hash multipliers, primes and offsets with SHA-256 |
| `dump_moe_route_fixture.py` | Runs the upstream `Gate` (sqrtsoftplus, bias) and writes `fixtures/moe_route_sqrtsoftplus.json` |
| `dump_latent_fixtures.py` | Writes `fixtures/latent_quant_bytes.json` (FP8/FP4 quantize bytes), `sparse_latent_attention.json` (float64 `sparse_attn_exact` with sink and -1 skipping) and `indexer_scores.json` (mid-group compress-length masking) |
| `dump_hc_rope_window_fixtures.py` | Writes `fixtures/hc_mix.json` (Sinkhorn 1/3/20 iters, `hc_pre`, `hc_post`), `rope_interleaved_offset.json` (offset rotary and its inverse) and `window_indices.json` (`get_window_topk_idxs` prefill/decode) |
| `dump_parser_reference.py` | Runs upstream `encoding.py` `parse_message_from_completion_text` on golden, canonical and malformed completions and writes `parser_reference/parser_reference.json` |
| `cross_check_recipe.py` | Compares that fixture with the `deepseek-recipe` stream parser at ~40 split points per case (`pip install deepseek-recipe`) |
| `dump_rope_table_fixture.py` | Writes `fixtures/rope_table.json`: upstream `precompute_freqs_cis`, plain and YaRN, as cos/sin tables |
| `dump_compressor_fixture.py` | Writes `fixtures/compressor_pool.json`: upstream `Compressor` prefill + decode pooling with the norm bypassed |
| `dump_candidate_select_fixture.py` | Writes `fixtures/candidate_select.json`: `select_candidate_blocks` and the indexer's final sorted top-k |
| `dump_moe_exec_fixture.py` | Writes `fixtures/moe_exec.json`: upstream `MoE` (gate, routed experts, shared expert) with float32 weights |
| `dump_engram_module_fixture.py` | Writes `fixtures/engram_module.json`: upstream `Engram` with a bf16-rounded float table in place of the FP8 lookup |
| `dump_hyper_connection_fixture.py` | Writes `fixtures/hyper_connection.json`: `Block.hc_mixes` / `hc_pre` / `hc_post` called unbound |
| `dump_attention_fixture.py` | Writes `fixtures/attention_stack.json` (with an exact-softmax `sparse_attn`, see below): six upstream `Attention` layers (every mode) through prefill and decode, with the indices each index-source layer chose |
| `dump_model_fixture.py` | Writes `fixtures/model_forward.json`: a small upstream `Transformer` (all layer modes, float32, no Engram/vision/draft) with every parameter, each block's output stream, the final normed hidden states and last-position logits through prefill and decode |
| `dump_derivative_quant_fixtures.py` | Writes `fixtures/derivative_quant_codecs.json`: independent numpy decoders for ModelOpt NVFP4 and Quark MXFP4, and `mx.quantize`/`mx.dequantize` for MLX affine 4/8-bit gs64 (needs `pip install mlx[cpu]`, used only by this script) |

## Commands

```bash
~/dsv41-ref/.venv/bin/python -m unittest -v test_kernel_ports
~/dsv41-ref/.venv/bin/python run_small_config.py --config default --seed 0 --prefill 24 --decode 6
~/dsv41-ref/.venv/bin/python run_small_config.py --config modes --seed 0 --prefill 24 --decode 6
~/dsv41-ref/.venv/bin/python dump_engram_constants.py
```

Run the tests from this directory. Output goes to `deepseek_v41_ref/{default,modes,engram}` (gitignored): raw little-endian `.bin`
files plus a `manifest.tsv` of `name, kind, dtype, shape, file`. Names are `<step>.<module path>` with steps `prefill`, `decode0`, ….

## Limits

- Weights are seeded random. Upstream allocates with `torch.empty`, so no small checkpoint exists.
- Local runs use bf16 dense weights and no FP4 experts. Agreement of the ports with the tilelang kernels is not established here.
- `modes` keeps the real layer-mode pattern in miniature (SWA-only, ratio-2 sources, ratio-1 candidate source with re-index and reuse layers, Engram, three draft layers).
- The attention and model fixtures replace the port's `sparse_attn` with `sparse_attn_exact`. The port rounds attention probabilities to bf16, as the real kernel does; against it the float32 host reference differs by a few 1e-3 per layer, which compounds through routing and index choices. With the exact softmax the host reference matches to about 1e-5.
