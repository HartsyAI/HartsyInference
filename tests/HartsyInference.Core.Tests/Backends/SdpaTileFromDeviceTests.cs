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
}
