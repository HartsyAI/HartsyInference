# 3D Models — status

Concise status for 3D-generation (image → mesh) models. Build detail lives in
[PHASE_11_THREED.md](PHASE_11_THREED.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **TripoSR** | Real-weight image → mesh, verified on **both CPU F32 and the RTX 3060** vs the upstream `tsr` reference (`stabilityai/TripoSR`, 2026-06-30). Every stage matches: DINO ViT-B/16 image tokens maxAbs **8.5e-6** (corr ~1.0), Transformer1D backbone scene_codes **corr 1.00000000** (std 409.33 vs 409.33; CPU maxAbs 3.4e-3), NeRF decoder density/color to 1.6e-2 / 5e-6, 64³ density grid corr **1.0** — and the density field (1.2% > iso 25) meshes into a coherent 84k-tri chair `.glb`. The scaffold was a wrong-architecture guess and was rewritten; two bugs fixed (see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)): DINO pos-embed interp `+0.1` fudge, and a CUDA activation-cache reshape-identity bug in the backbone. **Caveat:** background removal (rembg + resize_foreground + gray-0.5 composite) is still done in Python — the C# pipeline expects an already-composited RGB input; port it for raw-RGBA CLI use. |

## Components verified, full mesh pending (🔬)

| Model | Notes |
|---|---|
| **Hunyuan3D-2** | image → mesh. Architecture fully reverse-engineered + **rewritten to the real arch** (2026-06-30) — the original scaffold was a wrong-architecture guess. All three model components are now **numerically verified vs the hy3dgen reference on CUDA**: **DINOv2-giant conditioner** (1536-dim, 40 layers, SwiGLU) last_hidden_state **corr 1.0**; **Flux-lineage DiT** (16 double + 32 single stream blocks, QK-RMSNorm, **no RoPE**) velocity **corr 0.99999738** (bug fixed: timestep `max_period = time_factor = 1000`, not 10000 — see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)); **VecSet ShapeVAE** (`post_kl` → 16 qk-normed self-attn resblocks → cross-attn `geo_decoder` w/ 51-dim Fourier point queries → `output_proj`) occupancy **corr 0.99999518**. The pipeline (DINOv2-giant cond → FlowMatchEuler ascending-sigma flow-match + 2-way CFG → `latents/scale_factor` → ShapeVAE grid decode → marching cubes) is **wired and runs e2e** (`steps=1` produces + saves a `.glb`). **Not yet ✅:** multi-step runs hit a runtime VRAM/native-crash issue — a full coherent mesh needs a crash bisect + a GPU-residency block rewrite (the DiT/VAE block glue is CPU-resident → heavy VRAM + ~62 s/step). **Model correctness is done; the remaining work is runtime/perf.** Tests: `Hunyuan3DDinoParityTests`, `Hunyuan3DDitParityTests`, `Hunyuan3DVaeParityTests`, `Hunyuan3DGenerationTests`. Detail + next steps in [PHASE_11_THREED.md](PHASE_11_THREED.md) § 2 and [E2E_3D_WORKLOG.md](E2E_3D_WORKLOG.md). |

## Deferred / not started (❌)

| Model | Notes |
|---|---|
| **TRELLIS** (image → Gaussian splat + mesh) | Not implemented — needs sparse 3D conv/attention (no backend op yet) + flexicubes + splat rendering. The `GaussianSplatCloud` type + PLY splat export are already in place as foundation. |
| **Hunyuan3D Paint** (texture/PBR) | Out of scope for the shape pipelines (multiview diffusion + UV bake). |

## Foundation

The `HartsyInference.ThreeD` package provides the reusable mesh / splat / triplane foundation: marching
cubes plus glTF / OBJ / PLY export, shared by both built models.
