# convert-safetensors-to-gguf

CLI utility for converting any SharpInference-compatible safetensors checkpoint to a quantized GGUF file.

## Quick reference

```bash
convert-safetensors-to-gguf <input.safetensors> <output.gguf> <policy> [architecture]
```

| `<policy>` | Output size | Use when |
|---|---|---|
| `q8_0` | ~50% of F16 | Need conservative quant, no quality concern |
| `q4_k_s` | ~25% | Smallest 4-bit, fastest write |
| `q4_k_m` | ~30% | **Most popular default** (matches llama.cpp `_M`) |
| `q5_k_m` | ~37% | Better than Q4_K_M, still 4-bit-ish backbone |
| `q6_k` | ~44% | Near-lossless |

`<architecture>` (case-insensitive): `flux | flux2 | sdxl | sd3 | sd15 | flite | chroma | auraflow | zimage | ernie_image | hunyuan_image | qwen_image | llama | passthrough`

## Build

```bash
dotnet build samples/ConvertSafetensorsToGguf/ConvertSafetensorsToGguf.csproj -c Release
```

## Run example

```bash
samples/ConvertSafetensorsToGguf/bin/Release/net10.0/convert-safetensors-to-gguf \
    /models/flux1-schnell.safetensors  /models/flux1-schnell-Q4_K_M.gguf  q4_k_m  flux
```

## Restrictions

- Only Q8_0 / Q4_K / Q5_K / Q6_K can be authored. Other types are read-only.
- Inner dim of quantized tensors must be 256-aligned (K-quants) or 32-aligned (legacy). Common diffusion arches all satisfy this.
- VAE is not quantized below F16 in the built-in policies (correct default).
- ~5% PPL quality gap vs llama.cpp's `quantize` tool (we skip the iterative refinement pass).

## Full docs

See [`docs/Research/GGUF_QUANTIZER_USAGE.md`](../../docs/Research/GGUF_QUANTIZER_USAGE.md) for:
- The C# API for programmatic use
- How to build a custom `GgufQuantPolicy`
- Per-tensor F16/quant override examples
- Round-trip validation steps
- Memory profile at write time
