using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Engine.Tests.Registry;

/// <summary>The MovieRoll DiT's load-time F16 cast is chosen per backend: BF16 goes to F16 on every backend, as before; FP32 (the released file) goes to
/// F16 on CUDA only, and the CPU backend keeps those weights in F32. The CPU case runs everywhere.</summary>
[Trait("Category", "Integration")]
public sealed class MovieRollDitCastTests
{
    [Fact]
    public void CpuBackend_CastsOnlyBf16ToF16_KeepingTheFp32DitWeights()
    {
        using CpuBackend cpu = new();
        Assert.Equal([DType.BF16], MovieRollRecipe.DitHalfCastSources(cpu));
    }
}

/// <summary>The CUDA side of <see cref="MovieRollDitCastTests"/>: FP32 and BF16 DiT weights go to F16 on CUDA. Runs as a GPU integration test
/// (Category=GpuIntegration); it is skipped where no CUDA device is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollDitCastGpuTests
{
    [CudaDeviceFact]
    public void CudaBackend_CastsTheFp32AndBf16DitWeightsToF16()
    {
        using IBackend cuda = GpuDevices.OpenCuda();
        Assert.Equal([DType.BF16, DType.F32], MovieRollRecipe.DitHalfCastSources(cuda));
    }
}
