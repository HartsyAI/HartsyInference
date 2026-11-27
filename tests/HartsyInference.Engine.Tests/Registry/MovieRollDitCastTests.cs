using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Engine.Tests.Registry;

/// <summary>The MovieRoll DiT's load-time F16 cast is chosen per backend: the released FP32 (and BF16) weights go to F16 on CUDA only. The CPU backend
/// keeps its F32 weights, because the F16 path is not proven there. The CUDA case needs a device and is skipped without one.</summary>
[Trait("Category", "Integration")]
public sealed class MovieRollDitCastTests
{
    [Fact]
    public void CpuBackend_KeepsTheFp32DitWeights()
    {
        using CpuBackend cpu = new();
        Assert.Empty(MovieRollRecipe.DitHalfCastSources(cpu));
    }

    [CudaDeviceFact]
    public void CudaBackend_CastsTheFp32AndBf16DitWeightsToF16()
    {
        using IBackend cuda = GpuDevices.OpenCuda();
        Assert.Equal([DType.BF16, DType.F32], MovieRollRecipe.DitHalfCastSources(cuda));
    }
}
