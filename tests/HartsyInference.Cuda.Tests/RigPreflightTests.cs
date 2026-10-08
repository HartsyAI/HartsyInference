using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Run first on the rig. The GPU suites return early when CUDA is unavailable and xUnit counts that as a pass, so this test is
/// the one that fails when there is no usable device: a readiness run is not evidence until it passes.
/// </summary>
[Trait("Category", "GpuIntegration")]
public sealed class RigPreflightTests
{
    [Fact]
    public void CudaIsAvailable_SoEveryGpuSuiteBelowActuallyRuns()
    {
        Assert.True(CudaContext.IsAvailable(), "CUDA is not usable on this machine; the GPU suites would pass without running.");
    }
}
