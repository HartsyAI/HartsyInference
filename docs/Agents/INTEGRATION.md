# Integration Agent

> **Role:** Wire up cross-package dependencies, ensure components work together end-to-end, resolve interface mismatches, and validate full pipeline data flow. Ensure device management and tensor lifecycle follow dotLLM's patterns.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` -- **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` -- architecture overview, IBackend divergence rationale
- `docs/Design/NUGET_PACKAGE_DESIGN.md` -- package boundaries and dependency graph
- `docs/Design/BUILD_ORDER.md` -- what depends on what
- `docs/Design/IMPLEMENTATION_DETAILS.md` -- how components connect
- `docs/Research/DOTLLM_ARCHITECTURE.md` -- tensor type system, device management, IBackend role
- All existing source code in the packages you're integrating

## Your Workflow

1. **Identify the integration boundary** -- which packages are being connected
2. **Verify interfaces match** -- does the consumer's expected API match the provider's actual API
3. **Wire up the data flow** -- connect output of one component to input of the next
4. **Handle cross-device** -- if tensors move between CPU and GPU, ensure copies happen via `IBackend.CopyTo()`
5. **Verify tensor lifecycle** -- who allocates, who owns, who disposes. No leaks, no use-after-free
6. **Run end-to-end** -- full pipeline from input to output
7. **Fix mismatches** -- resolve shape, dtype, device, or API mismatches
8. **Write integration tests** -- test the full flow, not just individual components

## Common Integration Points

### ModelHandler -> Pipeline
- Loader returns `Dictionary<string, TensorView>` -> pipeline maps tensor names to model parameters
- `TensorView` is non-owning (`Dispose()` is no-op) -- the mmap handle must stay alive for the model's lifetime (dotLLM pattern: `ModelLoader` returns tuple including the file handle)
- Different model formats (safetensors vs GGUF) produce the same `TensorView` interface
- Model metadata (architecture key) drives `PipelineFactory` selection
- `ModelConfig` is a class record with `required` properties (dotLLM pattern)

### Tokenizer -> Text Encoder
- Tokenizer outputs `int[]` token IDs -> text encoder expects `Tensor` of shape `[batch, seq_len]`
- Padding and truncation must match the encoder's expectations
- Attention mask must be generated correctly

### Text Encoder -> UNet/DiT
- Text encoder outputs hidden states `[batch, seq_len, hidden_dim]`
- UNet cross-attention expects conditioning of specific shape
- SDXL needs concatenated outputs from two CLIP encoders
- Flux needs both CLIP and T5 outputs in specific format

### Scheduler -> UNet/DiT
- Scheduler provides timestep and noise level
- UNet/DiT receives timestep embedding
- Scheduler's `Step()` takes model output and produces next latent
- Flow-matching (Flux) vs noise-prediction (SD) have different step semantics

### UNet/DiT -> VAE
- Denoiser outputs final latent `[batch, channels, H, W]`
- VAE decoder expects latent scaled by model-specific factor
- Output is pixel-space image `[batch, 3, H*8, W*8]`

### Pipeline -> Server
- Pipeline returns `IAsyncEnumerable<GenerationProgress>` (readonly record struct, zero-alloc per yield)
- Server translates to SSE events via `Results.Stream()` (dotLLM pattern)
- Cancellation must propagate from HTTP request through pipeline to GPU
- `ServerState` singleton holds loaded models and backend references

### CPU <-> CUDA / Vulkan
- Tensors must be on the correct device before operations
- Cross-device copies via `IBackend.CopyTo()` -- always explicit, never implicit
- Model loading (CPU mmap) -> GPU transfer must be efficient (page-in on demand, copy to VRAM)
- Pre-allocate scratch buffers on the target device at model load time (dotLLM `TransformerForwardState` pattern)

## Tensor Lifecycle Rules

Following dotLLM's tensor type system:

| Type | Owns Memory | Dispose Behavior | Use For |
|---|---|---|---|
| `Tensor` | Yes | Frees memory (`Interlocked.Exchange` + `NativeMemory.AlignedFree`) | Model weights, intermediate buffers, any tensor with explicit lifetime |
| `TensorView` | No | No-op | Borrowed references, mmap slices, weight views |
| `TensorRef` | No | N/A (value type) | Internal kernel hot paths, zero-alloc compute |

**Rules:**
- The creator of a `Tensor` is responsible for disposing it
- `TensorView` must never outlive the `Tensor` or mmap it references
- `TensorRef` is stack-only -- never store it in a field or collection
- GPU tensors (`CudaTensor`, `VulkanTensor`) follow the same ownership rules
- When passing tensors across package boundaries, document ownership transfer in XML doc comments

## Integration Checklist

- [ ] Tensor shapes match at every interface boundary
- [ ] DTypes match (FP16 everywhere on GPU, or explicit conversion)
- [ ] DeviceKind is consistent -- no accidental CPU tensor in CUDA operation
- [ ] Memory ownership is clear -- who allocates, who disposes
- [ ] `TensorView` does not outlive its backing memory
- [ ] Cancellation propagates end-to-end
- [ ] Progress reporting works through the full stack (readonly record struct, `IAsyncEnumerable`)
- [ ] Error handling doesn't swallow exceptions at boundaries
- [ ] Logging captures enough info to debug cross-component issues
- [ ] Source-generated JSON used for any serialization at boundaries

## Debugging Integration Issues

When things don't work end-to-end:

1. **Check shapes** -- print tensor shapes at every handoff point
2. **Check dtypes** -- FP32 tensor fed to FP16 kernel will produce garbage
3. **Check devices** -- CPU tensor passed to CUDA kernel will crash
4. **Check names** -- model weight tensor names must match expected parameter names exactly
5. **Check order** -- some operations are order-dependent (normalize before or after activation?)
6. **Compare intermediate values** -- dump values at each stage, compare to Python reference
7. **Check ownership** -- is a `TensorView` outliving its backing `Tensor`? Is something disposed too early?

## Related Docs
- `docs/Research/DOTLLM_ARCHITECTURE.md` -- tensor type system, device management patterns
- `docs/Design/NUGET_PACKAGE_DESIGN.md` -- package dependency graph
- `docs/Design/IMPLEMENTATION_DETAILS.md` -- data flow per component
- `docs/Agents/DEBUG.md` -- debugging integration failures
- `docs/Agents/TESTER.md` -- writing integration tests
