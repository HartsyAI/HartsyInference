# 3D Models — status

Concise status for 3D-generation (image → mesh) models. Build detail lives in
[PHASE_11_THREED.md](PHASE_11_THREED.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

None yet. Both pipelines are built structurally; neither has a real-weight, output-confirmed run.

## Built, validation-pending (🔧)

| Model | Notes |
|---|---|
| **Hunyuan3D-2** | image → mesh pipeline on the shared 3D foundation. |
| **TripoSR** | image → mesh (triplane → marching cubes). |

## Foundation

The `HartsyInference.ThreeD` package provides the reusable mesh / splat / triplane foundation: marching
cubes plus glTF / OBJ / PLY export. Both models above are structural; the remaining work is a real-weight
checkpoint download + numeric validation pass for each.
