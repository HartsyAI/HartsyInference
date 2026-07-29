# Add-a-Model Agent

> Bring a new model (image / audio / video / world / 3D / vision / LLM) up to **verified end-to-end**.
> Assumes you've read `AGENTS.md` (shared rules + Core Engine Patterns) and `docs/CODE_STYLE.md`.
> `docs/Checklists/TROUBLESHOOTING.md` is your bring-up debugging bible — read it before you start.

## Workflow

1. **Map the data flow with shapes** at every boundary (this table is the highest-value artifact):
   tokenizer→encoder `int[]`→`[B,seq]`+mask · encoder→DiT `[B,seq,hidden]` · scheduler `Step()` (flow-match
   vs ε-pred) · denoiser→VAE latent `[B,C,H,W]`→`[B,3,H·8,W·8]`.
2. **Pick the package** a sibling model already lives in; write the forward pass **against `IBackend` only**.
3. **Convert weights at load time** (repack/transpose once, not per-forward).
4. **Parity-test** vs a Python reference; start `SyntheticSmoke`, graduate to `Unit` once real-weight parity
   is documented in `PARITY_VERIFICATION.md`.
5. **Wire it in**: one `IArchitectureRecipe` registered in `RecipeRegistry` + a catalog entry. Update the
   modality's `MODEL_STATUS_*` doc.

## Backend & pipeline shape

```csharp
// ✅ model code takes IBackend; math routes through it; reuse the shared utilities
public sealed class Sd15RecipePipeline : IRecipePipeline   // src/HartsyInference.Engine/Recipes/Image/
{
    private readonly IBackend _backend;
    // GenerateFromTokens(...) → CfgHelper.ApplyCfg / DtypeCastHelper.EnsureF32 / Img2ImgSetup.Prepare
}
// ❌ leaking a concrete backend, or re-deriving CFG/img2img/dtype-cast inline per pipeline
public void Generate(CudaBackend backend) { for (...) { /* hand-rolled CFG slice */ } }
```

```csharp
// ✅ progress is a synchronous per-step callback; pipelines inherit DiffusionPipelineBase
public (byte[] rgb, int w, int h, int seed) GenerateFromTokens(..., Action<GenerationProgress>? onProgress);
// ❌ there is NO IDiffusionPipeline and NO IAsyncEnumerable pipeline — both were deleted (no pipeline used them)
```

```csharp
// ✅ register the recipe (last-registered wins) — this is the whole "wire a model in" step
RecipeRegistry.Register(new Sd15Recipe());          // RecipeRegistry.cs: _recipes.Insert(0, recipe)
```

## Weight conversion

- **Never execute pickle** — use the safe-subset parser. Validate converted output vs the original before
  shipping; preserve architecture/config/tokenizer metadata; report precision loss.
- Mixed-precision default: UNet/DiT `Q8_0`, VAE `FP16`, text encoders `FP16` (T5 `Q8_0` only if safe).
- Inspect the safetensors/GGUF header **before** writing the loader — single-file checkpoints fuse QKV,
  rename scale companions, drop suffixes. 10 seconds of inspection saves hours (see TROUBLESHOOTING §Quant).

## Parity testing

```csharp
// ✅ real-weight test is env-gated and skips cleanly on a machine without the checkpoint
string? path = Environment.GetEnvironmentVariable("HARTSY_MODEL_GGUF_PATH");
if (path is null) return;                             // skips on hosted CI
Assert.True(cosine >= 0.99);                          // vs a saved HF/diffusers reference dump
```

- Compare against a Python reference using **Python's saved noise/embeddings** — never match RNG seeds
  (C# Box-Muller ≠ PyTorch). Bisect layer-by-layer to the first divergent layer, then sub-op decompose.
- FP32 tolerance ladder (10×+ over = real bug): element-wise `<1e-7`, LayerNorm/GroupNorm `<1e-6`,
  GEMM `<1e-5`, attention block `<1e-4`, full DiT/UNet `<1e-3`.
- **"Tests pass" ≠ correct** — a garbled image can have perfectly finite tensors. Visually inspect; keep
  known-good reference images. Tag the forward test `[Trait("Category","SyntheticSmoke")]`; drop the trait
  to graduate to the Unit gate only after documented real-weight parity.

## The recurring bring-up traps (full list in TROUBLESHOOTING.md)

- diffusers `attention_head_dim` is head **count**, not dim, when `num_attention_heads` is absent.
- `Shape[N]` returns 0 for an uninitialized dim — a rank-2 tensor's `Shape[2]==0` silently zeroes matmul.
- GEGLU/SwiGLU split on the **last dim**, never the flat midpoint (garbled-but-finite output).
- Don't skip CLIP `final_layer_norm` (5× amplified conditioning); check timestep `flip_sin_to_cos`.
- Mixed-dtype checkpoints: never `(float*)weight.DataPointer` a BF16/F16 tensor — cast at load.
