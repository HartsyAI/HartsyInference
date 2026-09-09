# HartsyInference

C#/.NET inference for text, images, video, audio, vision, 3D, and world models, using CUDA, Vulkan, or CPU.
No Python runtime is required for inference. GPU backends require compatible drivers and userspace libraries.
Libraries target .NET 8 and .NET 10.

**Alpha: pin an exact package version.** Coverage varies by model, format, backend, and feature;
implementation does not imply real-weight verification.

## Install and generate

```bash
dotnet add package HartsyInference --prerelease
```

The meta-package includes Engine and the modality/backends. InferenceEngine owns model loading and generation:

```csharp
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

using InferenceEngine engine = new("cuda");
ImageResult image = await engine.Images.GenerateAsync(
    new ModelSpec { Requested = "sdxl", Modality = Modality.Image, LocalPath = "sdxl.safetensors" },
    new ImageRequest { Prompt = "a castle on a mountain at sunset", Steps = 25 });
// image.Rgb contains image.Width * image.Height * 3 RGB24 bytes.
```

## Documentation

[Repository and usage](https://github.com/HartsyAI/HartsyInference) ·
[SwarmUI extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend) ·
[Model status](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Checklists/MODEL_STATUS.md) ·
[Benchmarks](https://github.com/HartsyAI/HartsyInference/tree/main/benchmarks/scoreboards) ·
[Multi-GPU](https://github.com/HartsyAI/HartsyInference/blob/main/docs/MULTI_GPU.md)

Code/packages are MIT licensed. Model weights have separate publisher licenses.
