# LTX-2.5 diffusion decoder — ComfyUI layer-diff harness

The reference half of `LtxVideo25ReferenceLayerDiffTests`. It runs ComfyUI's `NADiffusionDecoder` — the reference
implementation of this exact VAE — on a **file-supplied latent and noise** and dumps raw f32 at every stage, so
the C# decoder can be diffed against it stage by stage with the pipeline out of the picture.

This is what found all three correctness bugs in the decoder (see `benchmarks/scoreboards/VIDEO.md`). Structural
inspection of the same code found none of them, so reach for this before re-reading the port.

## Running it

```bash
COMFY="/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI"
cd "$COMFY"
./venv/bin/python <repo>/benchmarks/ltx25_layerdiff/ref_dump.py /tmp/refdump

cd <repo>
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LTX25_REFDUMP=/tmp/refdump \
  dotnet test tests/HartsyInference.Diffusion.Tests -c Release \
  --filter "FullyQualifiedName~LtxVideo25ReferenceLayerDiffTests"
```

The C# side reads the geometry from the dump's `shapes.json`, so changing `T_LAT, H_LAT, W_LAT` in `ref_dump.py`
is all that is needed to move geometry. **Use a non-square latent** (e.g. `3, 8, 4`): the first run used
`3, 4, 4`, where H == W hides any H/W transposition and both axes are shorter than the 7-wide attention kernel,
so the sliding-window path is never exercised at stage 0.

Optional deeper bisects:
- `ref_attn.py <out> <refdump>` dumps the first attention module's internals (qkv, q/k norm, post-rope q/k, na3d,
  proj) plus the per-block stage-0 outputs. Point the test at it with `LTX25_REFATTN`.
- `LTX25_OURDUMP=<dir>` mirrors the C# side's own tensors so a divergence can be dissected in numpy — which rows,
  which channels, what weight the output implies — rather than through summary statistics.
- `na3d_rule.py` compares our `Na3dWindowStart` rule against `comfy_kitchen.na3d` directly, in both the clamped
  and genuinely-sliding regimes. Runs on CPU; needs no checkpoint.

## The trap that will waste your session

`ref_dump.py` asserts the state-dict load is exact — **zero missing keys, `type_emb` the only unexpected one**.
Do not relax that. A `strict=False` load that silently leaves an `nn.Linear` at random init gives you a
confident-looking reference that "diverges at block 1", and you will chase a phantom.

Second trap: cast the checkpoint's BF16 weights to F32 on **both** sides. Leaving the C# side in BF16 puts a
~1.6e-3 floor under every comparison, which is large enough to hide a real defect.
