# LongCat-Video DiT: per-stage reference dump

Reference for phase B1 (`LongCatTransformer`, `LongCatBlock`, the conditioning-aware attention plan). The dumper runs the pinned reference's own
modules, `meituan-longcat/LongCat-Video` @ `58eb336edc5c8c63389838727e40b8c18434e54e`, on CPU in F32 with flash-attn replaced by torch SDPA (the only
change), on a tiny random model: 2 blocks, hidden 128, 4 heads of 32, caption width 48, adaLN width 64, text length 7 of which 5 are valid.

Run on WSL or a pod, never in the Windows venv:

```
python dump_longcat_dit.py --repo /path/to/LongCat-Video --out tests/python-reference/longcat_dit_reference_tensors
```

Needs torch >= 2.6 (CPU), einops, safetensors, numpy (diffusers is stubbed) and a checkout of the pinned commit. No CUDA, triton, flash-attn, transformers or real
weight. Output (gitignored by `*_reference_tensors/`): `ckpt/` (the tiny model as a two-shard diffusers-named safetensors index plus `config.json`, read
by the C# loader exactly as the real folder is), `<scenario>.<stage>.f32` raw little-endian F32 per stage, and `meta.json` with every shape and each
scenario's conditioning.

Scenarios: `t2v` (no conditioning, non-square grid, distinct per-frame timesteps), `i2v` (1 conditioning frame), `vc` (2, other aspect),
`avatar_ref_skip` (reference + 2 context frames, RoPE index 4, skip window fits), `avatar_ref_fallback` (index 10: window does not fit, full attention),
`avatar_ref_i2v` (reference as the single conditioning frame). The avatar cases swap the avatar model's `Attention` (with its `ref_img_index` RoPE
and `mask_frame_range` skip) into the same blocks; the audio cross-attention is not part of this dump.

Stages per scenario: `t_freq` (the sinusoidal features, before the MLP), `x_embedder`, `t_embedder`, `y_embedder` (after the zero-pad mask), `rope_cos`/`rope_sin`, and per block `modulation`, `attn_in`,
`self_attn`, `after_self`, `cross_attn_full` (zeros on the conditioning rows; the C# probe is the noisy rows only), `after_cross`, `ffn`, `out`, then
block 0's `q`/`k` after norm and RoPE and `v`, `final_layer` and `velocity`.

C# side: set `HARTSY_LONGCAT_DIT_DIR` to the output directory and run
`dotnet test tests/HartsyInference.Diffusion.Tests --filter FullyQualifiedName~LongCatDitReferenceParityTests`. Tolerance relRMS 1e-5 per stage (1e-6
for the RoPE tables), checked twice: cumulatively (the whole C# forward) and isolated (each block fed the reference's own saved inputs), so the
report names where a difference starts. The test is skipped, and so is not evidence, until that variable is set. Run on 2026-11-24 in the `hartsy-ref:cpu` image (see `PARITY_VERIFICATION.md`); the test also writes `parity-report.txt` with every number.
