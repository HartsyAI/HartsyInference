using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The device-sized query tile (<c>numerics.tileFromDevice</c>): the same shape and device give the same tile whatever the free memory was,
/// and the knob is off unless set.</summary>
public sealed class SdpaTileFromDeviceTests
{
    private const long Heads = 12, Tokens = 32528;
    private const ulong Card24Gb = 24UL << 30, Card8Gb = 8UL << 30;

    [Fact]
    public void TheKnobIsOffUnlessSet()
    {
        Assert.Null(EngineKnobs.TileFromDevice.Value);
    }

    [Fact]
    public void TheRowsFollowTheDeviceRungAndNotFreeMemory()
    {
        long perRow = Heads * Tokens * sizeof(float);
        Assert.Equal((1L << 30) / perRow, CudaBackend.TileRowsFromDevice(Card24Gb, Heads, Tokens, Tokens, forced: null));
        Assert.Equal((512L << 20) / perRow, CudaBackend.TileRowsFromDevice(Card8Gb, Heads, Tokens, Tokens, forced: null));
    }

    [Fact]
    public void AForcedHeightAndTheQueryRowsStillBound()
    {
        Assert.Equal(256, CudaBackend.TileRowsFromDevice(Card24Gb, Heads, Tokens, Tokens, forced: 256));
        Assert.Equal(100, CudaBackend.TileRowsFromDevice(Card24Gb, Heads, 100, Tokens, forced: null));
    }

    [Fact]
    public void TheOneGateBudgetIsHalfTheTotalOrTheFreeWhenThatIsLess()
    {
        // Ample free memory: the gate reads the device total only, so the path is the same on every run of one card.
        Assert.Equal(Card24Gb / 2, CudaBackend.DeviceGateBudget(Card24Gb, freeBytes: 20UL << 30));
        // Free below half the total: the free clamp applies (a buffer the card does not have free is not chosen).
        Assert.Equal(6UL << 30, CudaBackend.DeviceGateBudget(Card24Gb, freeBytes: 6UL << 30));
    }

    [Fact]
    public void TheGateBudgetIsNotTheTileRung()
    {
        // A 1 GiB rung on a 24 GB card: a 2 GiB score matrix is under the gate, so it stays materialized.
        Assert.Equal(1L << 30, CudaBackend.TileRungBytes(Card24Gb));
        Assert.True(2UL << 30 <= CudaBackend.DeviceGateBudget(Card24Gb, freeBytes: 20UL << 30));
    }

    [Theory]
    [InlineData(20UL << 30, 1L << 30)]
    [InlineData((20UL << 30) - 1, 512L << 20)]
    [InlineData(18_000_000_000UL, 512L << 20)]
    public void TheRungBoundaryIs20GiB(ulong total, long expected)
    {
        Assert.Equal(expected, CudaBackend.TileRungBytes(total));
    }
}
