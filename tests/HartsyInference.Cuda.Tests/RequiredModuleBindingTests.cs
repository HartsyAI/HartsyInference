using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>On the card this runs on, every module the engine requires binds from the shipped PTX. A required module left unbound
/// refuses the card at probe time, so this test is what catches a Turing card being refused for want of one module.</summary>
public sealed class RequiredModuleBindingTests
{
    private readonly ITestOutputHelper _output;
    public RequiredModuleBindingTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void EveryRequiredModuleBindsFromTheShippedPtxOnTheRunningCard()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptx = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptx))
            ptx = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaContext context = new CudaContext(0);
        using CudaKernels kernels = new CudaKernels(ptx, context);
        string unbound = kernels.UnboundRequiredModules.Count == 0 ? "none" : string.Join(", ", kernels.UnboundRequiredModules);
        _output.WriteLine($"SM {kernels.Sm}: unbound required modules: {unbound}");
        Assert.Empty(kernels.UnboundRequiredModules);
    }
}
