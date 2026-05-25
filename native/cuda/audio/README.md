# Audio codec CUDA kernels

CUDA C++ source files for the audio backend ops added in Phase 5:

- **`audio_activations_f32.cu`** — `sigmoid`, `tanh`, `elu`, `snake`, `snake_beta`
- **`conv1d_f32.cu`** — `conv1d`, `conv_transpose1d`

These compile to PTX modules that the C# CUDA backend loads via
`CudaModule.LoadFromFile`. Until the build pipeline picks them up, `CudaBackend`
throws `NotSupportedException` on these ops with a "use CpuBackend" hint.

## Build

Requires NVIDIA CUDA Toolkit 11.0+ (for `sm_80` target — adjust per your minimum-spec
GPU).

```sh
nvcc --ptx -arch=sm_80 audio_activations_f32.cu -o audio_activations_f32.ptx
nvcc --ptx -arch=sm_80 conv1d_f32.cu -o conv1d_f32.ptx
cp audio_activations_f32.ptx conv1d_f32.ptx ../../src/SharpInference.Cuda/Ptx/
```

After copying the `.ptx` artifacts into `src/SharpInference.Cuda/Ptx/`, update
`CudaKernels.cs` to load them at startup and replace the `NotSupportedException`
stubs in `CudaBackend.cs` with kernel launches.

## Coverage

| Op | F32 | F16 | BF16 |
|---|---|---|---|
| sigmoid | ✅ source | ⏳ | ⏳ |
| tanh | ✅ source | ⏳ | ⏳ |
| elu | ✅ source | ⏳ | ⏳ |
| snake | ✅ source | ⏳ | ⏳ |
| snake_beta | ✅ source | ⏳ | ⏳ |
| conv1d | ✅ source | ⏳ | ⏳ |
| conv_transpose1d | ✅ source | ⏳ | ⏳ |

F16 / BF16 variants will follow the same convention as the existing
`elementwise_f16.ptx` / `elementwise_bf16.ptx` files: include `<cuda_fp16.h>` or
`<cuda_bf16.h>`, cast in and out per-element. Audio codecs primarily run in F32
so F16 variants are lower priority.
