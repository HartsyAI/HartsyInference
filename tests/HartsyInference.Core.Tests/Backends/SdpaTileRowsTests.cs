using HartsyInference.Core.Backends.Attention;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The height of the query tile the CUDA F32 attention loop runs. It is a function of the shape and the device's total VRAM only: free
/// memory moves between runs and changed the tile height, and so the rounding of the tiled GEMMs, so the same call gave different video.</summary>
public sealed class SdpaTileRowsTests
{
    private const long Heads = 12, Tokens = 32528;
    private static readonly long PerRowBytes = Heads * Tokens * sizeof(float);
    private const ulong Card24Gb = 24UL << 30, Card8Gb = 8UL << 30;

    [Theory]
    [InlineData(Card24Gb)]
    [InlineData(Card8Gb)]
    public void TheTileDependsOnTheDeviceNotOnFreeMemory(ulong deviceTotal)
    {
        // Two runs on one device give one tile: the height is a fixed function of (heads, rows, keys, device total).
        long first = CudaBackend.TileRows(deviceTotal, Heads, Tokens, Tokens, forced: null);
        long second = CudaBackend.TileRows(deviceTotal, Heads, Tokens, Tokens, forced: null);
        Assert.Equal(first, second);
    }

    [Fact]
    public void ACardOf20GbOrMoreGetsTheOneGibRung()
    {
        Assert.Equal(1L << 30, CudaBackend.TileRungBytes(20UL << 30));
        Assert.Equal(1L << 30, CudaBackend.TileRungBytes(Card24Gb));
        Assert.Equal(512L << 20, CudaBackend.TileRungBytes(Card8Gb));
        Assert.Equal(512L << 20, CudaBackend.TileRungBytes((20UL << 30) - 1));
    }

    [Fact]
    public void TheRungSetsTheRowsOfTheTile()
    {
        Assert.Equal((1L << 30) / PerRowBytes, CudaBackend.TileRows(Card24Gb, Heads, Tokens, Tokens, forced: null));
        Assert.Equal((512L << 20) / PerRowBytes, CudaBackend.TileRows(Card8Gb, Heads, Tokens, Tokens, forced: null));
    }

    [Fact]
    public void TheTileNeverExceedsTheQueryRowsOrAForcedHeight()
    {
        Assert.Equal(100, CudaBackend.TileRows(Card24Gb, Heads, 100, Tokens, forced: null));
        Assert.Equal(256, CudaBackend.TileRows(Card24Gb, Heads, Tokens, Tokens, forced: 256));
        Assert.Equal(100, CudaBackend.TileRows(Card24Gb, Heads, 100, Tokens, forced: 256));
    }

    [Fact]
    public void OneRowPastTheSoftmaxAddressIsRefused()
    {
        Assert.Throws<AttentionUnsupportedException>(() =>
            CudaBackend.TileRows(Card24Gb, totalHeads: 1L << 20, sq: 10, skv: 1L << 20, forced: null));
    }
}
