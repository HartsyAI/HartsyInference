using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>On the card this runs on, every module the engine requires binds from the shipped PTX. A required module left unbound
/// refuses the card at probe time, so this test is what catches a Turing card being refused for want of one module.</summary>
[Collection("CudaSerial")]
public sealed class RequiredModuleBindingTests
{
    private readonly ITestOutputHelper _output;
    public RequiredModuleBindingTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void EveryRequiredModuleBindsFromTheShippedPtxOnTheRunningCard()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaContext context = new CudaContext(0);
        using CudaKernels kernels = new CudaKernels(ShippedPtxDir(), context);
        string unbound = kernels.UnboundRequiredModules.Count == 0 ? "none" : string.Join(", ", kernels.UnboundRequiredModules);
        _output.WriteLine($"SM {kernels.Sm}: unbound required modules: {unbound}");
        Assert.Empty(kernels.UnboundRequiredModules);
    }

    /// <summary>A card below the floor leaves required modules unbound, which is what makes the probe refuse it. The gate stands
    /// in for a card below Turing, so this runs on any CUDA device.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void BelowTheFloor_ARequiredModuleIsReportedUnbound()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        int previous = CudaKernels.GateSmForTests;
        try
        {
            CudaKernels.GateSmForTests = 70;
            using CudaContext context = new CudaContext(0);
            using CudaKernels kernels = new CudaKernels(ShippedPtxDir(), context);
            Assert.NotEmpty(kernels.UnboundRequiredModules);
            _output.WriteLine($"gate 70 unbound required modules: {string.Join(", ", kernels.UnboundRequiredModules)}");
        }
        finally
        {
            CudaKernels.GateSmForTests = previous;
        }
    }

    private static string ShippedPtxDir()
    {
        string ptx = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptx))
            ptx = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptx;
    }
}
