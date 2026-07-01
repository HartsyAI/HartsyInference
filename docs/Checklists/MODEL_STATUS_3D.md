# 3D Models — status

Concise status for 3D-generation (image → mesh) models. Build detail lives in
[PHASE_11_THREED.md](PHASE_11_THREED.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **TripoSR** | Real-weight image → mesh, verified on **both CPU F32 and the RTX 3060** vs the upstream `tsr` reference (`stabilityai/TripoSR`, 2026-06-30). Every stage matches: DINO ViT-B/16 image tokens maxAbs **8.5e-6** (corr ~1.0), Transformer1D backbone scene_codes **corr 1.00000000** (std 409.33 vs 409.33; CPU maxAbs 3.4e-3), NeRF decoder density/color to 1.6e-2 / 5e-6, 64³ density grid corr **1.0** — and the density field (1.2% > iso 25) meshes into a coherent 84k-tri chair `.glb`. The scaffold was a wrong-architecture guess and was rewritten; two bugs fixed (see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)): DINO pos-embed interp `+0.1` fudge, and a CUDA activation-cache reshape-identity bug in the backbone. **Caveat:** background removal (rembg + resize_foreground + gray-0.5 composite) is still done in Python — the C# pipeline expects an already-composited RGB input; port it for raw-RGBA CLI use. |

## Built, validation-pending (🔧)

| Model | Notes |
|---|---|
| **Hunyuan3D-2** | image → mesh. Architecture fully reverse-engineered: **Flux-lineage MMDiT** (16 double + 32 single stream blocks, QK-RMSNorm, no RoPE) conditioned by a **DINOv2-giant** (1536-dim, 40 layers, SwiGLU; bundled in the ckpt) + a VecSet **ShapeVAE** (`post_kl` → 16 qk-normed self-attn resblocks → cross-attn `geo_decoder` w/ 51-dim Fourier point queries → `output_proj`) + FlowMatchEuler scheduler. **Conditioner verified** (DINOv2-giant SwiGLU: CPU corr 1.0, CUDA corr 0.99999). DiT + VAE + pipeline are the wrong architecture in the C# scaffold and need a from-scratch rebuild reusing the engine's Flux blocks (validate on CUDA). Rebuild spec in [PHASE_11_THREED.md](PHASE_11_THREED.md) § 2. |

## Foundation

The `HartsyInference.ThreeD` package provides the reusable mesh / splat / triplane foundation: marching
cubes plus glTF / OBJ / PLY export. Both models above are structural; the remaining work is a real-weight
checkpoint download + numeric validation pass for each.
