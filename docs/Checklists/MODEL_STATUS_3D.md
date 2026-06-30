# 3D Models — status

Concise status for 3D-generation (image → mesh) models. Build detail lives in
[PHASE_11_THREED.md](PHASE_11_THREED.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **TripoSR** | Real-weight numeric parity vs the upstream `tsr` library (CPU F32): DINO ViT-B/16 image tokens maxAbs **8.5e-6**, Transformer1D scene_codes maxAbs **3.4e-3** (corr 1.0), NeRF decoder probe density+color + 64³ density grid corr **1.0**. The scaffold was a wrong-architecture guess and was rewritten (see [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md) § Bugs). |

## Built, validation-pending (🔧)

| Model | Notes |
|---|---|
| **Hunyuan3D-2** | image → mesh. Architecture fully reverse-engineered from the real checkpoint: it is a **Flux-lineage MMDiT** (16 double + 32 single stream blocks, QK-RMSNorm, no RoPE) conditioned by a **DINOv2-giant** (1536-dim, 40 layers, SwiGLU; bundled in the ckpt) + a VecSet **ShapeVAE** (`post_kl` → 16 qk-normed self-attn resblocks → cross-attn `geo_decoder` w/ 51-dim Fourier point queries → `output_proj`) + FlowMatchEuler scheduler. The current C# scaffold is the wrong architecture (simple AdaLN DiT) and needs a from-scratch rebuild reusing the engine's Flux blocks. |

## Foundation

The `HartsyInference.ThreeD` package provides the reusable mesh / splat / triplane foundation: marching
cubes plus glTF / OBJ / PLY export. Both models above are structural; the remaining work is a real-weight
checkpoint download + numeric validation pass for each.
