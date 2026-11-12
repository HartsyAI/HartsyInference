# TRELLIS.2 stage parity against upstream, on the CPU

One real run of the C# pipeline is captured stage by stage; the same inputs go through upstream's own code on the CPU (fp32, naive
attention) and the outputs are compared. Nothing from upstream is copied here: you point the scripts at a clone.

## What runs

| Stage | C# side | Upstream side |
|---|---|---|
| Structure decoder (`ss_dec`) | `SparseStructureDecoder` on a seeded `[1, 8, 16, 16, 16]` latent | `SparseStructureDecoder` |
| Shape decode (7 channels, per-level child logits) | `ShapeVaeDecoder` via the pipeline | `FlexiDualGridVaeDecoder` up to the head (`SparseUnetVaeDecoder.forward(return_subs=True)`) |
| Dual-grid mesher | `DualGridMesher` on the C# field | `o_voxel.convert.flexible_dual_grid_to_mesh` on the same field, and on the upstream field |
| Texture flow, one forward | `Trellis2TextureFlow.Forward` on the captured shape latent and conditioning, seeded noise, t = 500 | `SLatFlowModel` with `concat_cond` |
| Texture decode | `Trellis2TextureDecoderStage` | `SparseUnetVaeDecoder` with `guide_subs` from the upstream shape decode, `* 0.5 + 0.5` |

The CUDA-only libraries are replaced by stand-ins in `shims/`: `flex_gemm` (a gather-GEMM submanifold sparse conv), `flash_attn`
(per-sequence softmax attention), `o_voxel` (upstream's `flexible_dual_grid_to_mesh`, read from the clone at import, with its hashmap
swapped for a sorted-key lookup) and an empty `cumesh`. The sparse conv stand-in assumes the kernel index runs (kd, kh, kw) with kd along
x, as the C# engine does, so the axis convention is not independently checked.

## How to run

1. Capture (GPU, one process at a time; about 3 minutes at 512): `dotnet run -c Release --project csharp/Capture -- <models/3D/trellis2>
   <image.rgba> 512 <src/HartsyInference.Cuda/Ptx> <capture dir> 12`. The image is raw RGBA, square. It writes `shape_decode.tns`,
   `mesh.tns`, `texture_flow.tns`, `texture_decode.tns` and `times.txt` (per-stage seconds).
2. Forward inputs and outputs (GPU, about a minute): `dotnet run -c Release --project csharp/Forward -- <models/3D/trellis2> <capture dir>
   <Ptx dir> <capture dir>/forward.tns cuda`.
3. Compare on the CPU:
   `python -I run_stage_parity.py <TRELLIS.2 clone> <models/3D/trellis2> <shape decoder path without extension> <capture dir>` and
   `python -I run_forward_parity.py <same four arguments> [--flow-only] [--bf16]`.

The capture reads each stage's tensors back to the host to write them, which offloads the device copies of the conditioning and the shape latent the next stage still uses, so a stage time can include one re-upload (milliseconds at 2,634 tokens). `run_stage_parity.py` takes the grid resolution from `mesh.tns`; for the 1024 and 1536 cascades check it against the decoder's grid before trusting the comparison.

Needs `torch`, `numpy`, `safetensors`. The shape and texture decodes of a 700k-voxel field take about 1 to 2 minutes each on 8 threads.
Results and the numbers measured on 2026-11-12 are in `docs/Checklists/MODEL_STATUS_3D.md` (TRELLIS.2, stage parity).
