# Integration Agent

> **Role:** Wire cross-package dependencies, ensure end-to-end data flow, resolve interface mismatches, and validate tensor lifecycle.

## Prerequisites
- `docs/CODE_STYLE.md`, `docs/Design/CORE_DESIGN.md`, `docs/Design/NUGET_PACKAGE_DESIGN.md`
- `docs/Design/BUILD_ORDER.md`, `docs/Design/IMPLEMENTATION_DETAILS.md`
- `docs/Research/DOTLLM_ARCHITECTURE.md` — tensor type system, device management
- Existing source code in packages being integrated

## Workflow
1. Identify integration boundary
2. Verify interfaces match
3. Wire data flow between components
4. Handle cross-device copies via `IBackend.CopyTo()`
5. Verify tensor lifecycle (allocate → own → dispose)
6. Run end-to-end pipeline
7. Fix shape/dtype/device/API mismatches
8. Write integration tests

## Common Integration Points

| Boundary | Key Concern |
|---|---|
| ModelHandler → Pipeline | `Dictionary<string, TensorView>` mapping; mmap handle lifetime; `ModelConfig` (class record, `required`) |
| Tokenizer → Text Encoder | `int[]` → `Tensor[batch, seq_len]`; padding/truncation; attention mask |
| Text Encoder → UNet/DiT | Hidden states `[batch, seq_len, hidden_dim]`; SDXL dual CLIP concat; Flux CLIP+T5 format |
| Scheduler → UNet/DiT | Timestep + noise level; `Step()` produces next latent; flow-matching vs noise-prediction semantics |
| UNet/DiT → VAE | Latent `[batch, C, H, W]` → scaled → pixel image `[batch, 3, H*8, W*8]` |
| Pipeline → Server | `IAsyncEnumerable<GenerationProgress>` (readonly record struct) → SSE via `Results.Stream()`; cancellation propagation; `ServerState` singleton |
| CPU ↔ CUDA/Vulkan | Explicit `IBackend.CopyTo()` only; efficient page-in + VRAM copy; pre-allocate scratch at load time |

## Tensor Lifecycle (dotLLM)

| Type | Owns Memory | Dispose | Use For |
|---|---|---|---|
| `Tensor` | Yes | `Interlocked.Exchange` + `AlignedFree` | Weights, intermediates |
| `TensorView` | No | No-op | Borrowed refs, mmap slices |
| `TensorRef` | No | N/A (value type) | Kernel hot paths |

**Rules:** creator disposes `Tensor`; `TensorView` never outlives backing memory; `TensorRef` stack-only; document ownership in XML docs at package boundaries.

## Integration Checklist
- [ ] Shapes match at every boundary
- [ ] DTypes match (FP16 on GPU, or explicit conversion)
- [ ] DeviceKind consistent
- [ ] Memory ownership clear
- [ ] `TensorView` lifetime valid
- [ ] Cancellation propagates end-to-end
- [ ] Progress reporting works (readonly record struct, `IAsyncEnumerable`)
- [ ] Error handling doesn't swallow at boundaries
- [ ] Source-gen JSON at serialization boundaries

## Debugging End-to-End Issues
1. Check shapes at every handoff
2. Check dtypes (FP32→FP16 kernel = garbage)
3. Check devices (CPU tensor in CUDA = crash)
4. Check weight tensor names match exactly
5. Check operation order
6. Dump intermediates vs Python reference
7. Check `TensorView` doesn't outlive backing `Tensor`

## Related Docs
- `docs/Research/DOTLLM_ARCHITECTURE.md`, `docs/Design/NUGET_PACKAGE_DESIGN.md`, `docs/Design/IMPLEMENTATION_DETAILS.md`
- `docs/Agents/DEBUG.md`, `docs/Agents/TESTER.md`
