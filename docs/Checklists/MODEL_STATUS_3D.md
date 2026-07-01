# 3D Models — status

Concise status for 3D-generation (image → mesh) models. Build detail lives in
[PHASE_11_THREED.md](PHASE_11_THREED.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **TripoSR** | Real-weight image → mesh on the RTX 3060 (`stabilityai/TripoSR`, 2026-06-30). Every stage matches the upstream `tsr` reference — DINO ViT-B/16 tokens corr ~1.0, Transformer1D backbone scene_codes **corr 1.00000000** (std 409.33 vs 409.33), NeRF decoder density/color to 1.6e-2/5e-6 — and the density field (1.2% > iso 25) meshes into a coherent 84k-tri chair `.glb`. Two bugs fixed (see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)): DINO pos-embed interp `+0.1` fudge, and a CUDA activation-cache reshape-identity bug in the backbone. **Caveat:** background removal (rembg + resize_foreground + gray-0.5 composite) is still done in Python — the C# pipeline expects an already-composited RGB input; port it for raw-RGBA CLI use. |

## Built, validation-pending (🔧)

| Model | Notes |
|---|---|
| **Hunyuan3D-2** | image → mesh pipeline on the shared 3D foundation. Reference dump `dump_hunyuan3d_full_forward.py` still has unfilled `TODO[VG]` (model construction + forward hooks) — build the oracle before running. |

## Foundation

The `HartsyInference.ThreeD` package provides the reusable mesh / splat / triplane foundation: marching
cubes plus glTF / OBJ / PLY export. Both models above are structural; the remaining work is a real-weight
checkpoint download + numeric validation pass for each.
