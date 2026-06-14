# GGUF Quantizer — Usage Guide

> Take a HartsyInference safetensors checkpoint and write a quantized GGUF file. Mirrors llama.cpp's `quantize` tool but operates on safetensors input and is C#-native (no native dependency).

## Quick start

### CLI (most users)

```bash
# Build the converter once
cd /path/to/HartsyInference
dotnet build samples/ConvertSafetensorsToGguf/ConvertSafetensorsToGguf.csproj -c Release

# Convert any safetensors to GGUF (architecture is required so loaders pick the right key mapper)
samples/ConvertSafetensorsToGguf/bin/Release/net10.0/convert-safetensors-to-gguf \
    flux1-schnell.safetensors  flux1-schnell-Q4_K_M.gguf  q4_k_m  flux

# Output:
# [convert-safetensors-to-gguf] flux1-schnell.safetensors → flux1-schnell-Q4_K_M.gguf [Q4_K_M] arch=flux
#   passthrough: 12
#   cast (→ F16/F32): 468
#   quantized: 304
#     Q4_K: 228 tensors
#     Q6_K: 76 tensors
#   output: 7138 MB
#   elapsed: 142.3s
```

### C# API (programmatic)

```csharp
using HartsyInference.ModelHandler.Gguf;

GgufQuantizationReport report = GgufQuantizer.ConvertSafetensorsToGguf(
    safetensorsPath: "flux1-schnell.safetensors",
    outputGgufPath:  "flux1-schnell-Q4_K_M.gguf",
    policy:          GgufQuantPolicy.Q4_K_M,
    architecture:    "flux");

Console.WriteLine($"Quantized {report.QuantizedCount} tensors, output {report.OutputBytes / (1024*1024)} MB.");
```

## What you can write

| Policy | Backbone | High-fidelity bump | Approx size vs F16 | Recommended for |
|---|---|---|---|---|
| `q8_0` | Q8_0 (8-bit) | none | ~50% | Conservative — no quality concern |
| `q4_k_s` | Q4_K (4-bit) | none | ~25% | Smallest 4-bit, fastest to write |
| `q4_k_m` | Q4_K + Q6_K on V/output | mix | ~30% | **Most popular default** — matches llama.cpp's `_M` policy |
| `q5_k_m` | Q5_K + Q6_K on V/output | mix | ~37% | Better quality than Q4_K_M |
| `q6_k` | Q6_K (6-bit) | none | ~44% | Near-lossless |

Norms / biases / token embeddings / position embeddings / register tokens are always kept at F16. Quantizing those wrecks output (visible posterization, broken cross-attention) and saves negligible bytes anyway.

VAE is **never** quantized below F16 — quality-critical, small enough that the savings aren't worth the artifacts.

## What you can't write (yet)

| Policy you might want | Status | Why |
|---|---|---|
| Q4_0 / Q4_1 / Q5_0 / Q5_1 | not supported | Legacy 32-block quants superseded by K-quants. We **read** them (city96/unsloth dumps include them), but writing isn't useful since `q4_k_m` is strictly better. |
| Q8_1 | not supported | Activation quant, rarely on disk |
| Q2_K / Q3_K | not supported | Too aggressive for diffusion. Would visibly degrade output. |
| IQ4_NL / IQ4_XS / IQ2_* / IQ3_* / IQ1_* | not supported | i-quant family needs ggml's importance-weighted lookup tables. Low priority for diffusion. |
| TQ1_0 / TQ2_0 | not supported | Ternary, rare |

**Workaround**: if you need a non-supported quant type, use llama.cpp's `quantize` tool. HartsyInference can read its output (verified against city96 dumps which use llama.cpp internally).

## Architecture argument — what it does

The 4th CLI argument (`architecture`) becomes the GGUF metadata field `general.architecture`. This is what the GGUF loader uses to pick the right key mapper when reading the file back. Recognized values (case-insensitive):

```
flux | flux2 | sdxl | sd3 | sd15 | flite | chroma | auraflow | zimage |
ernie_image | hunyuan_image | qwen_image | llama | passthrough
```

Pass `passthrough` for an unknown architecture; the reader will fall back to a key-pattern heuristic. For known architectures, set the right value — it's a no-op when reading HartsyInference-internal naming, but matters when the GGUF is consumed by `ComfyUI-GGUF` or other downstream tools that route by `general.architecture`.

## What gets quantized vs kept at F16

The policy's predicate logic decides per tensor:

1. **Always F16**: any 1D tensor (norms, biases). Element count below `MinElementsToQuantize` (default 256). Names containing `.norm`, `layernorm`, `rmsnorm`, `embed_tokens`, `token_embd`, `pos_embed`, `register_tokens`. Custom predicate via `QualityProfile.ShouldKeepF16`.
2. **High-fidelity DType** (in `_M` policies): names matching attention V projection (`.attn_v.`, `.to_v.`, `.v_proj.`) or output projection (`output.weight`, `lm_head.weight`, `final_proj.weight`, `proj_out.weight`). These get the bumped quant (typically Q6_K).
3. **Backbone DType**: everything else. The bulk of the model.

You can override any of this by constructing a custom `GgufQuantPolicy`:

```csharp
GgufQuantPolicy custom = new()
{
    BackboneDType    = DType.Q5_K,
    HighFidelityDType = DType.Q8_0,
    IsHighFidelity   = name => name.Contains("img_attn"),  // bump only image-attention paths
    ShouldKeepF16    = (name, t) => name.Contains("custom_metric"),
    MinElementsToQuantize = 1024,
};

GgufQuantizer.ConvertSafetensorsToGguf("input.safetensors", "output.gguf", custom, "myarch");
```

## Restrictions & constraints

1. **Block alignment**: K-quants have a 256-element super-block; legacy quants have a 32-element block. The inner dimension of every quantized weight must be a multiple of 256 (K-quants) or 32 (legacy). For Flux (hidden=3072), SDXL (320..1280), SD3 (1536..2432), Z-Image (3840) — all are 256-aligned, so this is rarely a problem in practice. If you hit a tensor whose inner dim isn't a multiple of 256, the quantizer will throw at write time. **Workaround**: add that tensor's name to `ShouldKeepF16`.

2. **Quality vs llama.cpp**: HartsyInference's writer uses a simplified `make_qkx2_quants` (initial pass without iterative refinement). PPL gap to llama.cpp's `quantize` is ~5%. For bit-identical quality, use llama.cpp's tool. For most diffusion users, the gap is invisible.

3. **Memory at write time**: peak ~2× input file size (mmap of source + anonymous heap for quantized output). For Flux Schnell at 24 GB F16 → Q4_K_M (~7 GB), peak is ~31 GB. Make sure your machine can handle it.

4. **Per-tensor overrides**: if you need to force specific tensors to specific DTypes (e.g., keep one specific attention block at F16 for debugging), use the C# API and supply your own `IsHighFidelity` / `ShouldKeepF16` lambdas.

5. **VAE**: don't quantize. The default policies skip it via the `embed_tokens` / norm heuristics partially, but if you build a custom policy for a model that doesn't fit the standard naming, **manually exclude any VAE tensors** with `ShouldKeepF16 = (name, t) => name.StartsWith("vae.")`.

## Round-trip example: quantize, then read back

```csharp
// 1. Quantize
GgufQuantizer.ConvertSafetensorsToGguf(
    "myflux.safetensors", "myflux-Q4_K_M.gguf",
    GgufQuantPolicy.Q4_K_M, architecture: "flux");

// 2. Read back into a HartsyInference pipeline
(FluxCheckpointConverter.ConvertedWeights converted,
 GgufModelLoader.LoadedGgufModel handle) =
    GgufConverterBridge.LoadGguf(
        "myflux-Q4_K_M.gguf", DType.F16, FluxCheckpointConverter.Convert);

using (handle)
{
    // converted.Transformer / .ClipL / .T5 / .Vae are now ready for FluxTransformer.LoadWeights etc.
}
```

The reader uses the same codecs as the writer (forward direction), so a HartsyInference-written GGUF round-trips losslessly through dequant → original-quality F16.

## Verifying the output

After conversion, the file passes any of these checks:

```bash
# 1. File magic + version
xxd -l 8 myflux-Q4_K_M.gguf
# expect: 4747 5546 0300 0000  ("GGUF" + version 3 LE)

# 2. ComfyUI-GGUF can load it (validates against the canonical reader)
# Drop into your ComfyUI's models/unet/ folder and try a workflow.

# 3. HartsyInference can round-trip it
dotnet test tests/HartsyInference.ModelHandler.Tests/HartsyInference.ModelHandler.Tests.csproj \
    --filter "FullyQualifiedName~GgufRoundTripTests"
```

## Further reading

- [`GGUF_BACKEND.md`](GGUF_BACKEND.md) — full architecture of the GGUF reader, codec registry, key mappers, GPU dequant kernels.
- [`GGUF_FORMAT.md`](GGUF_FORMAT.md) — the binary format itself (header, metadata, tensor descriptors, alignment rules).
- llama.cpp [`quantize` tool docs](https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md) — for bit-identical output if quality margin matters.
- ggml [`ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c) — canonical quant block layouts and `make_qkx2_quants` reference.
