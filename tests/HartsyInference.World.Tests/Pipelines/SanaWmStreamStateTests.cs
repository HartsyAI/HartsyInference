using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.World.Pipelines.SanaWm;
using Xunit;

namespace HartsyInference.World.Tests.Pipelines;

/// <summary><see cref="SanaWmStreamState.Resolve"/>: the explicit modes in any case and spacing, an unknown mode refused, and
/// <c>auto</c> on the CPU backend (memory unknown, so not a small device) holding the state as computed. The device-size rule
/// itself is exercised on the GPUs: <c>auto</c> picks BF16 on the 12 GB 3060 and F32 on the 24 GB 4090.</summary>
public sealed class SanaWmStreamStateTests
{
    [Theory]
    [InlineData("f32", false)]
    [InlineData(" F32 ", false)]
    [InlineData("bf16", true)]
    [InlineData("BF16", true)]
    public void ExplicitModes_PickTheirDtype(string mode, bool bf16)
    {
        using CpuBackend cpu = new();
        Assert.Equal(bf16 ? DType.BF16 : null, SanaWmStreamState.Resolve(cpu, mode));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("auto")]
    [InlineData("")]
    public void Auto_OnABackendOfUnknownMemory_HoldsTheStateAsComputed(string? mode)
    {
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        Assert.True(backend.GetVramInfo().TotalBytes <= 0 || backend.GetVramInfo().TotalBytes > SanaWmStreamState.AutoBf16AtOrBelowBytes,
            "The CPU backend reports a device size inside the BF16 range; this test's premise no longer holds.");
        Assert.Null(SanaWmStreamState.Resolve(cpu, mode));
    }

    [Fact]
    public void UnknownMode_IsRefused()
    {
        using CpuBackend cpu = new();
        Assert.Throws<ArgumentException>(() => SanaWmStreamState.Resolve(cpu, "fp8"));
    }
}
