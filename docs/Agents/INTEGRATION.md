# Integration Agent

> **Role:** Wire up cross-package dependencies, ensure components work together end-to-end, resolve interface mismatches, and validate full pipeline data flow.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` — architecture overview
- `docs/Design/NUGET_PACKAGE_DESIGN.md` — package boundaries and dependency graph
- `docs/Design/BUILD_ORDER.md` — what depends on what
- `docs/Design/IMPLEMENTATION_DETAILS.md` — how components connect
- All existing source code in the packages you're integrating

## Your Workflow

1. **Identify the integration boundary** — which packages are being connected
2. **Verify interfaces match** — does the consumer's expected API match the provider's actual API
3. **Wire up the data flow** — connect output of one component to input of the next
4. **Handle cross-device** — if tensors move between CPU and GPU, ensure copies happen
5. **Run end-to-end** — full pipeline from input to output
6. **Fix mismatches** — resolve shape, dtype, device, or API mismatches
7. **Write integration tests** — test the full flow, not just individual components

## Common Integration Points

### ModelHandler → Pipeline
- Loader returns `Dictionary<string, TensorView>` → pipeline maps tensor names to model parameters
- Different model formats (safetensors vs GGUF) produce the same `TensorView` interface
- Model metadata (architecture key) drives `PipelineFactory` selection

### Tokenizer → Text Encoder
- Tokenizer outputs `int[]` token IDs → text encoder expects `Tensor` of shape `[batch, seq_len]`
- Padding and truncation must match the encoder's expectations
- Attention mask must be generated correctly

### Text Encoder → UNet/DiT
- Text encoder outputs hidden states `[batch, seq_len, hidden_dim]`
- UNet cross-attention expects conditioning of specific shape
- SDXL needs concatenated outputs from two CLIP encoders
- Flux needs both CLIP and T5 outputs in specific format

### Scheduler → UNet/DiT
- Scheduler provides timestep and noise level
- UNet/DiT receives timestep embedding
- Scheduler's `Step()` takes model output and produces next latent
- Flow-matching (Flux) vs noise-prediction (SD) have different step semantics

### UNet/DiT → VAE
- Denoiser outputs final latent `[batch, channels, H, W]`
- VAE decoder expects latent scaled by model-specific factor
- Output is pixel-space image `[batch, 3, H*8, W*8]`

### Pipeline → Server
- Pipeline returns `IAsyncEnumerable<GenerationProgress>`
- Server translates to SSE events or final response
- Cancellation must propagate from HTTP request through pipeline to GPU

### CPU ↔ CUDA
- Tensors must be on the correct device before operations
- Cross-device copies must be explicit
- Model loading (CPU mmap) → GPU transfer must be efficient

## Integration Checklist

- [ ] Tensor shapes match at every interface boundary
- [ ] DTypes match (FP16 everywhere on GPU, or explicit conversion)
- [ ] DeviceKind is consistent — no accidental CPU tensor in CUDA operation
- [ ] Memory ownership is clear — who allocates, who disposes
- [ ] Cancellation propagates end-to-end
- [ ] Progress reporting works through the full stack
- [ ] Error handling doesn't swallow exceptions at boundaries
- [ ] Logging captures enough info to debug cross-component issues

## Debugging Integration Issues

When things don't work end-to-end:

1. **Check shapes** — print tensor shapes at every handoff point
2. **Check dtypes** — FP32 tensor fed to FP16 kernel will produce garbage
3. **Check devices** — CPU tensor passed to CUDA kernel will crash
4. **Check names** — model weight tensor names must match expected parameter names exactly
5. **Check order** — some operations are order-dependent (normalize before or after activation?)
6. **Compare intermediate values** — dump values at each stage, compare to Python reference

## Related Docs
- `docs/Design/NUGET_PACKAGE_DESIGN.md` — package dependency graph
- `docs/Design/IMPLEMENTATION_DETAILS.md` — data flow per component
- `docs/Agents/DEBUG.md` — debugging integration failures
- `docs/Agents/TESTER.md` — writing integration tests
