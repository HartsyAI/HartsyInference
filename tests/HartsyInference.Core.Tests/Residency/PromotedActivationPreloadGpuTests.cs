using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Residency;

/// <summary>
/// A preload also promotes tensors an op produced on the device (Kandinsky5's text conditioning is one) into weights without reading them back, so
/// their device buffer may be the only copy of the values. The give-back that runs when the device reports itself full must only free weights that
/// went up from the host, or the next op reads zeros.
/// </summary>
[Trait("Category", "GpuIntegration")]
public sealed class PromotedActivationPreloadGpuTests
{
    private const int Count = 4096;

    private readonly ITestOutputHelper _output;
    public PromotedActivationPreloadGpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ReleaseForOvercommit_KeepsAPromotedActivation_AndFreesHostWeights()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend backend = new(0, PtxDir());
        IBackend device = backend;
        using Tensor a = Filled(2f);
        using Tensor b = Filled(3f);
        using Tensor produced = new(new TensorShape(Count), DType.F32);
        device.Add(produced, a, b);
        device.Sync();
        using Tensor hostWeight = Filled(4f);

        bool promotedRegistered = GpuTransferHelper.PreloadWeight(produced, out bool promotedFromHost);
        bool hostRegistered = GpuTransferHelper.PreloadWeight(hostWeight, out bool hostFromHost);
        Assert.True(promotedRegistered);
        Assert.False(promotedFromHost);
        Assert.True(hostRegistered);
        Assert.True(hostFromHost);

        List<Tensor> fromHost = [hostWeight];
        List<Tensor> rollback = [produced, hostWeight];
        GpuTransferHelper.ReleaseForOvercommit(fromHost, rollback);

        Assert.Empty(fromHost);
        Assert.Equal([produced], rollback);

        // The promoted tensor still has its values on the device: 5 + 2, not the zeros of its never-written host buffer.
        using Tensor result = new(new TensorShape(Count), DType.F32);
        device.Add(result, produced, a);
        device.Sync();
        Assert.All(result.AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(7f, v));
        device.FreeWeights([produced]);
    }

    private static Tensor Filled(float value)
    {
        Tensor t = new(new TensorShape(Count), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }

        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test output directory.");
    }
}
