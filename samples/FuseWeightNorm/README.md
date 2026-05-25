# fuse-weight-norm

Offline weight-norm fusion tool. Scans a safetensors file for `*.weight_g` / `*.weight_v`
pairs (PyTorch's [weight_norm](https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html)
reparametrization) and writes a new safetensors file with each pair fused into a single
`*.weight` tensor.

Runtime fusion via `WeightNormFusion.Fuse` already works on every codec we ship
(EnCodec / DAC / SNAC / WavTokenizer / BiCodec / Mimi). This tool is for production
deployments that want to skip the per-load fusion cost — pre-fused safetensors load
instantly with the same key conventions any vanilla `nn.Conv1d` checkpoint would use.

## Usage

```sh
dotnet run --project samples/FuseWeightNorm -- INPUT.safetensors OUTPUT.safetensors
```

Or use the published assembly:

```sh
fuse-weight-norm INPUT.safetensors OUTPUT.safetensors
```

## What it does

For each `*.weight_g` tensor in the input:

1. Looks up the companion `*.weight_v`.
2. Determines the conv direction by matching `weight_g`'s element count against the
   conv's `out_channels` axis:
   - If `weight_g` size matches axis 0 → `nn.Conv1d` / `nn.Conv2d` / `nn.Linear`
     → fuse via `WeightNormFusion.Fuse`.
   - If `weight_g` size matches axis 1 on a rank-3 tensor → `nn.ConvTranspose1d`
     → fuse via `WeightNormFusionT.Fuse`.
3. Writes the result under `*.weight` (drops both `weight_g` and `weight_v`).

All other tensors are copied through unchanged.

## When you'd use it

- **Production model packaging.** Ship pre-fused safetensors with your model so the
  consumer doesn't pay a (typically 50-200 ms) fusion cost at load time.
- **Inspection / debugging.** Some tooling expects vanilla `.weight` keys and can't
  handle the `weight_g` / `weight_v` pair directly.
- **Hashing / signing.** Pre-fused checkpoints are byte-stable across SharpInference
  versions; the `weight_g` / `weight_v` representation isn't (PyTorch can choose to
  normalize along different axes depending on minor version changes).
