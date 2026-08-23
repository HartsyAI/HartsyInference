using Xunit;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>The driving-cache dtype resolution order: env override, then the global low-VRAM policy, then the
/// measured fits-check. Silent when wrong in both directions — a bad F32 pick OOMs a fresh install with no
/// explanation, a bad BF16 pick quietly costs cache precision on a card with room to spare — and the CPU branch
/// is what keeps every parity test on exact numerics now that unset means auto rather than F32.</summary>
public sealed class WanAnimate2DrivingCachePolicyTests
{
    private const long Gib = 1L << 30;

    [Theory]
    [InlineData("1", true)]
    [InlineData("true", true)]
    [InlineData("TRUE", true)]
    [InlineData("on", true)]
    [InlineData("0", false)]
    [InlineData("false", false)]
    [InlineData("off", false)]
    public void ParseEnv_KeepsTheHistoricalBooleanTokens(string value, bool expected)
    {
        Assert.Equal(expected, WanAnimate2DrivingCachePolicy.ParseEnv(value, log: false));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    [InlineData("auto")]
    [InlineData("AUTO")]
    [InlineData("banana")]   // unrecognized warns and falls back to auto, like LowVramPolicy
    public void ParseEnv_UnsetAutoAndGarbage_MeanAuto(string? value)
    {
        Assert.Null(WanAnimate2DrivingCachePolicy.ParseEnv(value, log: false));
    }

    [Fact]
    public void EnvOverride_BeatsTheGlobalLowVramPolicy_BothWays()
    {
        // Forced F32 wins even under ForceOn, and forced BF16 wins even under ForceOff — the env var is the
        // operator's last word; LowVramPolicy is consulted only inside auto.
        Assert.False(WanAnimate2DrivingCachePolicy.ResolveCore(
            envForced: false, LowVramMode.ForceOn, freeBytes: 0, f32DemandBytes: long.MaxValue, out string by));
        Assert.Contains(WanAnimate2DrivingCachePolicy.EnvironmentVariable, by);
        Assert.True(WanAnimate2DrivingCachePolicy.ResolveCore(
            envForced: true, LowVramMode.ForceOff, freeBytes: long.MaxValue, f32DemandBytes: 1, out _));
    }

    [Fact]
    public void GlobalPolicy_ForcesTheDtype_WithoutMeasuring()
    {
        // ForceOn → BF16 even with room for F32; ForceOff → F32 even with none (operator sized their own
        // workload; the recipe pre-flight is what fails loudly when it does not fit).
        Assert.True(WanAnimate2DrivingCachePolicy.ResolveCore(
            envForced: null, LowVramMode.ForceOn, freeBytes: long.MaxValue, f32DemandBytes: 1, out _));
        Assert.False(WanAnimate2DrivingCachePolicy.ResolveCore(
            envForced: null, LowVramMode.ForceOff, freeBytes: 1, f32DemandBytes: long.MaxValue, out _));
    }

    [Fact]
    public void Auto_KeepsF32_WhenTheF32DemandFits()
    {
        Assert.False(WanAnimate2DrivingCachePolicy.ResolveCore(
            envForced: null, LowVramMode.Auto, freeBytes: 23 * Gib, f32DemandBytes: 15 * Gib, out string by));
        Assert.Equal("measured", by);
        // Exactly at the boundary still fits — the demand already carries its own headroom terms.
        Assert.False(WanAnimate2DrivingCachePolicy.ResolveCore(
            envForced: null, LowVramMode.Auto, freeBytes: 15 * Gib, f32DemandBytes: 15 * Gib, out _));
    }

    [Fact]
    public void Auto_DropsToBf16_WhenTheF32DemandDoesNotFit()
    {
        Assert.True(WanAnimate2DrivingCachePolicy.ResolveCore(
            envForced: null, LowVramMode.Auto, freeBytes: 23 * Gib, f32DemandBytes: 37 * Gib, out string by));
        Assert.Equal("measured", by);
    }

    [Fact]
    public void Auto_OnBackendsWithNoVramReport_ResolvesToF32()
    {
        // GetVramInfo() defaults to (0, 0) on CPU — F32 keeps the parity tests on exact numerics.
        Assert.False(WanAnimate2DrivingCachePolicy.ResolveCore(
            envForced: null, LowVramMode.Auto, freeBytes: 0, f32DemandBytes: long.MaxValue, out string by));
        Assert.Equal("no VRAM report", by);
    }
}
