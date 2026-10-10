using System.Reflection;
using HartsyInference.Core.Backends;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The parts of the backend contract a new backend inherits without writing any code, and the size of the
/// floor it has to implement before it can run at all. Both are cheap to get wrong in a way no individual backend's
/// tests would catch, because the failure lives in the DEFAULT — the behaviour a backend gets by not overriding
/// something — or in the abstract surface, which no single backend can observe growing.</summary>
public sealed class BackendContractTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _out = output;

    /// <summary>A device added to the enum must not read as a CPU. This gated on an explicit CUDA-or-Vulkan list, so
    /// every caller asking "is this a GPU?" — same-device serialization, weight preloading, VRAM reclamation — would
    /// have answered "no" for a ROCm or Metal device and silently skipped it.</summary>
    [Theory]
    [InlineData(DeviceType.Cuda)]
    [InlineData(DeviceType.Rocm)]
    public void Every_Non_Cpu_Device_Is_A_Gpu(DeviceType type)
    {
        Assert.True(new DeviceKind(type, 0).IsGpu);
    }

    [Fact]
    public void Cpu_Is_Not_A_Gpu()
    {
        Assert.False(DeviceKind.Cpu.IsGpu);
    }

    /// <summary>A software rasterizer is its own vendor. llvmpipe reports the silicon vendor of the machine it runs
    /// on, so a vendor check alone would file it as Intel or AMD hardware and let it into a hardware comparison.</summary>
    [Fact]
    public void Software_Is_Distinct_From_Every_Hardware_Vendor()
    {
        Assert.NotEqual(GpuVendor.Software, GpuVendor.Intel);
        Assert.NotEqual(GpuVendor.Software, GpuVendor.Amd);
        Assert.NotEqual(GpuVendor.Software, GpuVendor.None);
    }

    /// <summary>The floor: members with no default implementation, which a new backend MUST write before it compiles.
    /// Everything else on the interface has a host fallback it inherits.
    ///
    /// <para>Pinned as a number rather than a list because the number is the claim that matters — adding an abstract
    /// member breaks every backend at once, including ones outside this repo, so it should be a deliberate act that
    /// updates this test rather than something noticed later. The names are printed on failure so the diff is
    /// obvious.</para></summary>
    [Fact]
    public void The_Abstract_Surface_Is_The_Floor_A_New_Backend_Implements()
    {
        MethodInfo[] abstractMethods = [.. typeof(IBackend)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Where(m => m.IsAbstract)
            .OrderBy(m => m.Name, StringComparer.Ordinal)];

        string[] names = [.. abstractMethods.Select(m => m.Name).Distinct(StringComparer.Ordinal)];
        _out.WriteLine($"{names.Length} abstract member names, {abstractMethods.Length} overloads:");
        foreach (string name in names) _out.WriteLine($"  {name}");

        // A backend that implements these runs; everything beyond them is speed, not correctness.
        Assert.InRange(names.Length, 25, 40);
        Assert.Contains("MatMul", names);
        Assert.Contains("Conv2D", names);
        Assert.Contains("ScaledDotProductAttention", names);

        // The levers and capability queries must NOT be abstract: a backend that has not thought about one inherits
        // the safe answer instead of being forced to invent one.
        Assert.DoesNotContain("get_CacheWeightCasts", abstractMethods.Select(m => m.Name));
        Assert.DoesNotContain("get_NativeFp8Gemm", abstractMethods.Select(m => m.Name));
        Assert.DoesNotContain("get_HighPrecisionGemm", abstractMethods.Select(m => m.Name));
    }
}
