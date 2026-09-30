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
