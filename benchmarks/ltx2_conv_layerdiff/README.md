# LTX-2 convolutional VAE decoder — ComfyUI layer-diff harness

The reference half of `LtxVideo2ConvVaeReferenceLayerDiffTests`. It runs ComfyUI's
`causal_video_autoencoder.Decoder` — the reference implementation of this exact VAE — on a file-supplied
latent and dumps raw f32 at every stage, so `LtxVideo2VaeDecoder` can be diffed against it stage by stage
with the pipeline out of the picture.

This is what found the decoder's two config bugs (spurious upsampler residual, reflect-instead-of-zeros
spatial padding). Structural inspection of the C# alone found neither, because both look like deliberate
choices in the source — they are only wrong against the config the checkpoint itself ships.

## Running it

```bash
COMFY="/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI"
cd "$COMFY"
./venv/bin/python <repo>/benchmarks/ltx2_conv_layerdiff/ref_dump.py /tmp/refA

cd <repo>
CUDA_VISIBLE_DEVICES="" LTX2CONV_REFDUMP=/tmp/refA \
  dotnet test tests/HartsyInference.Diffusion.Tests -c Release -f net10.0 \
  --filter "FullyQualifiedName~LtxVideo2ConvVaeReferenceLayerDiffTests"
```

Both sides run on the CPU — the geometry is tiny and the decoder is backend-generic, so this needs no GPU
(~100 s for the C# half). The C# side reads the geometry from `shapes.json`, so changing `T_LAT, H_LAT,
W_LAT` in `ref_dump.py` is all that is needed to move it. `LTX2CONV_TOL=0` makes a passing run print its
table anyway.

## Run it as TWO reference runs, not one

A single diff cannot separate two independent defects: reflect-vs-zeros makes `conv_in` diverge at once and
contaminates every stage below it, hiding everything else. So:

- **Run A — truth**: `ref_dump.py /tmp/refA`. The reference as the checkpoint's own metadata configures it.
  This is the one that answers "are we correct".
- **Run B — reference-as-misconfigured**: `ref_dump.py /tmp/refB --misconfig`. The same reference forced into
  *our* configuration. If ours matches THAT to ~1e-6 at every stage, the port is faithful and the entire
  delta is the flags — no third bug hiding downstream. `--reflect` / `--residual` isolate one flag each.

## The traps

- **The strict-load assertion is mandatory.** A `strict=False` load that silently leaves a conv at random
  init gives a confident-looking reference that "diverges at block 1", and you chase a phantom.
- **Take the config from the file's `__metadata__`, not `get_default_config(version)`.** `sd.py`'s version
  probe returns 2 for this checkpoint, and the built-in version-2 default (5/5/5/5 res blocks,
  `timestep_conditioning=True`) does not describe it. The strict assert is what catches that.
- **Cast both sides to F32.** Production runs the decode in BF16; leaving ours there puts a ~1.6e-3 floor
  under every comparison, large enough to hide a real defect.
- **`mark_conv3d_ended` before every conv-owning module, and `causal=False` on every call.** `forward_orig`
  does both; block forwards default to `causal=True`, and without the ended mark the non-causal *trailing*
  replicate frame is never produced, so the last frames diverge for the wrong reason. The temporal caches
  live on the modules and are only popped by `forward()` — build one decoder per process.
- **Use a non-square, non-power-of-two latent** (`3, 6, 4`): a square one hides an H/W transposition, and
  both axes have to exceed the reflect-pad width.
- The C# assert on `InferredStructure` is our equivalent of the strict load: `CountResnets` stops at the
  first missing index, so a converter mis-mapping would silently shrink a stage and every remaining stage
  would still "match" in shape.
