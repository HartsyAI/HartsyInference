# CoLa3D geometry decoder: scoping

Scope: `Models/CoLa3D/geometry_decoder.pt`, read with `PytorchPickleLoader` (no pickle execution). Shapes are the loader's
view of the file. The key map and shapes are checked by `CoLa3DGeometryDecoderKeyMapTests` (Integration).

## Top-level key groups

292 tensors, 474,231,463 elements. Block indices are shown as `#`; `blocks.{s}` is the stage, and the block index inside
a stage runs from 0 to the stage's block count.

| Key group | Tensors | Shapes (per stage where they differ) |
|---|---|---|
| `from_latent.{weight,bias}` | 2 | `[1024, 32]`, `[1024]` |
| `blocks.{s}.{n}` `SparseConvNeXtBlock3d` | 4 + 16 + 8 + 4 blocks | `conv.weight` `[C, 3, 3, 3, C]`, `conv.bias` `[C]`, `norm.{weight,bias}` `[C]`, `mlp.0` `[4C, C]`, `mlp.2` `[C, 4C]`, `to_subdiv` `[8, C]`; C = 1024, 512, 256, 128 |
| `blocks.{s}.{n}` `SparseResBlockC2S3d` (one per stage) | 4 | `norm1` `[C]`, `conv1` `[8C/2, 3, 3, 3, C]`, `conv2` `[C/2, 3, 3, 3, C/2]`, `to_subdiv` `[8, C]`; C = 1024, 512, 256, 128 |
| `output_layer.{weight,bias}` | 2 | `[7, 64]`, `[7]` |

Stage 4 (64 channels) has no blocks in this checkpoint, matching `num_blocks = [4, 16, 8, 4, 0]`.

## Relation to the shape VAE decoder

- All 292 keys and shapes equal `ShapeVaeKeyMap.Decoder(ShapeVaeConfig.NextDcF16C32.Decoder)`, the existing shape-VAE
  decoder map. No new key map is needed.
- All 292 values differ from `TRELLIS2-ShapeVAE/shape_dec_next_dc_f16c32_fp16.safetensors` (max absolute difference
  0.0270593, at `blocks.1.16.to_subdiv.weight`). CoLa3D finetuned the decoder, so the two are different weight sets.
- The reference demo (`scripts/inference/demo.py`) passes `geometry_decoder.pt` as `--geo-ckpt`. The alternative is
  `--pretrained-geo`, which loads the pretrained shape-VAE safetensors.

## Consequence for the C# pipeline (not changed here)

`CoLa3DCheckpointConverter` loads `shape_dec_next_dc_f16c32_fp16.safetensors`, and `CoLa3DPipeline` decodes object latents
with it. That is the pretrained decoder, not the finetuned geometry decoder the reference demo uses by default. The
no-prompt output is therefore not the reference's object geometry. This predates the prompt work.

## What remains

1. Decide whether to decode with `geometry_decoder.pt`. The change is in the loader only: the same `ShapeVaeDecoder` takes
   these tensors, since names and shapes match. It changes the no-prompt output, so it needs a decision.
2. No reference decode fixture exists. The parity record covers the shape VAE decoder on 12 voxels and does not cover
   the geometry decoder's forward. A Python dump of `DecomposableVaeDecoder` decode on fixed latents is needed before
   claiming parity.
3. CPU cost is unmeasured. Real object latents at the 1024 grid can have far more voxels than the 12-voxel parity case,
   and the decoder is 474M parameters in F32 (about 1.9 GB in memory after load).
