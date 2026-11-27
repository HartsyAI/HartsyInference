using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Engine.Tests.Registry;

/// <summary>The MovieRoll DiT's load-time F16 cast is chosen per backend: BF16 goes to F16 on every backend, as before; FP32 (the released file) goes to
/// F16 on CUDA only, and the CPU backend keeps those weights in F32. The CUDA case needs a device and is skipped without one.</summary>
[Trait("Category", "Integration")]
public sealed class MovieRollDitCastTests
{
    [Fact]
    public void CpuBackend_CastsOnlyBf16ToF16_KeepingTheFp32DitWeights()
    {
        using CpuBackend cpu = new();
        Assert.Equal([DType.BF16], MovieRollRecipe.DitHalfCastSources(cpu));
    }

    [CudaDeviceFact]
    public void CudaBackend_CastsTheFp32AndBf16DitWeightsToF16()
    {
        using IBackend cuda = GpuDevices.OpenCuda();
        Assert.Equal([DType.BF16, DType.F32], MovieRollRecipe.DitHalfCastSources(cuda));
    }
}
