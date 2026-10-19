using HartsyInference.Cuda;
using HartsyInference.Engine;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary><c>auto</c> must not pick a CUDA device below the kernel baseline: such a device now constructs a backend with its
/// sm_80 kernels unbound, so choosing it would fail on the first model op instead of running on Vulkan. The device's compute
/// capability is stood in for through the test seam; this class runs alone because that seam is process-wide.</summary>
[Collection(CollectionName)]
public sealed class BackendKernelBaselineTests
{
    public const string CollectionName = "BackendKernelBaselineSerial";

    private readonly ITestOutputHelper _output;
    public BackendKernelBaselineTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Auto_PassesOverACudaDeviceBelowTheKernelBaseline_AndSaysWhy()
    {
        if (!CudaContext.IsAvailable() || CudaContext.GetDeviceCount() == 0) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        Func<int, int>? previous = BackendFactory.QuerySmForTests;
        try
        {
            BackendFactory.QuerySmForTests = _ => 75;
            Assert.NotEqual("cuda", BackendFactory.Resolve("auto"));
            Assert.Contains("SM 7.5", BackendFactory.CudaUnavailableReason, StringComparison.Ordinal);

            BackendFactory.QuerySmForTests = _ => CudaKernels.BaselineSm;
            Assert.Equal("cuda", BackendFactory.Resolve("auto"));
            Assert.Null(BackendFactory.CudaUnavailableReason);
        }
        finally
        {
            BackendFactory.QuerySmForTests = previous;
        }
    }
}

[CollectionDefinition(BackendKernelBaselineTests.CollectionName, DisableParallelization = true)]
public sealed class BackendKernelBaselineCollection { }
