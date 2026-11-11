using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Residency;

/// <summary>
/// The contract the TRELLIS.2 stage sweeps (<c>Trellis2DeviceMemory</c>) rely on: <c>FreeActivations</c> and <c>FreeAllDeviceMemory</c> do not
/// read back, so a tensor an op produced on the device and nobody read on the host loses its values, and <c>OffloadActivation</c> is what
/// keeps them. Losing them once sent a zero-filled DINOv3 conditioning into a flow and produced copper, metallic busts.
/// </summary>
[Trait("Category", "GpuIntegration")]
public sealed class DeviceSweepOffloadGpuTests
{
    private const int Count = 4096;

    private readonly ITestOutputHelper _output;
    public DeviceSweepOffloadGpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void DeviceOnlyResult_SurvivesBothSweeps_WhenOffloadedFirst()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend backend = new(0, PtxDir());
        IBackend device = backend;
        using Tensor a = Filled(2f);
        using Tensor b = Filled(3f);
        using Tensor sum = new(new TensorShape(Count), DType.F32);
        device.Add(sum, a, b);
        device.Sync();

        device.OffloadActivation(sum);
        device.FreeActivations();
        device.FreeAllDeviceMemory();

        Assert.All(sum.AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(5f, v));
    }

    [Fact]
    public void DeviceOnlyResult_IsLostBySweeps_WhenNotOffloaded()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend backend = new(0, PtxDir());
        IBackend device = backend;
        using Tensor a = Filled(2f);
        using Tensor b = Filled(3f);
        using Tensor sum = new(new TensorShape(Count), DType.F32);
        device.Add(sum, a, b);
        device.Sync();

        device.FreeActivations();

        // The host buffer was never written: this is the hazard the offload guards against. If a backend ever starts reading back here,
        // the stage sweeps become safe without the offload and this test says so.
        Assert.All(sum.AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(0f, v));
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
