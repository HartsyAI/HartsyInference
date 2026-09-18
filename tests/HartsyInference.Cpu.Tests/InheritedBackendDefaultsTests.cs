using HartsyInference.Core.Backends;
using Xunit;

namespace HartsyInference.Cpu.Tests;

/// <summary>What a backend that overrides none of the optional contract actually reports. <see cref="CpuBackend"/>
/// implements the abstract floor and nothing beyond it, so reading these off it is reading the interface's own
/// defaults through a real implementation — which is what a future ROCm or Metal backend inherits on day one.</summary>
public sealed class InheritedBackendDefaultsTests
{
    [Fact]
    public void A_Backend_That_Overrides_Nothing_Reports_The_Conservative_Answers()
    {
        using IBackend backend = new CpuBackend();

        // Caching weight casts is the fast path every backend that HAS a cast cache wants; a backend without one
        // is unaffected either way.
        Assert.True(backend.CacheWeightCasts);

        // Claiming an fp8 GEMM that does not exist would let a recipe keep weights packed for a GEMM that cannot
        // read them — so absence has to be the default.
        Assert.False(backend.NativeFp8Gemm);

        // Neither a vendor nor a VRAM size can be invented; a planner reading these must see "unknown", not a guess.
        Assert.Equal(GpuVendor.None, backend.Capabilities.Vendor);
        Assert.Equal(0, backend.Capabilities.TotalVramBytes);

        Assert.False(backend.Device.IsGpu);
    }

    /// <summary>Writing a lever a backend does not implement is a no-op, not a throw. Callers apply these across
    /// every placement backend — including the CPU one holding a text encoder — without asking which of them care.</summary>
    [Fact]
    public void Setting_An_Unimplemented_Lever_Is_Harmless()
    {
        using IBackend backend = new CpuBackend();

        backend.CacheWeightCasts = false;
        backend.HighPrecisionGemm = true;
        backend.SeamlessTilingX = true;

        Assert.True(backend.CacheWeightCasts);      // the write was ignored, and did not throw
    }
}
