using HartsyInference.Core.Backends.Attention;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The height of the query tile the CUDA F32 attention loop runs: it follows free VRAM, but a free-memory reading that is far too low
/// (a Windows card a process has filled past about 8 GiB reports none) must not shrink it to one query row, which is one cuBLAS launch per head
/// per row.</summary>
public sealed class SdpaTileRowsTests
{
    private const long Heads = 12, Tokens = 32528;
    private static readonly long PerRowBytes = Heads * Tokens * sizeof(float);

    [Fact]
    public void ZeroFreeMemoryStillGetsTheFloorTile()
    {
        long rows = CudaBackend.TileRows(freeBytes: 0, Heads, Tokens, Tokens, forced: null);

        Assert.Equal(CudaBackend.MinTileScoreBytes / PerRowBytes, rows);
        Assert.True(rows > 50, $"A tile of {rows} rows would be one launch per few rows: {Tokens / rows} tiles per call.");
    }

    [Theory]
    [InlineData(1UL << 20)]
    [InlineData(64UL << 20)]
    [InlineData(400UL << 20)]
    public void ALowReadingNeverGoesBelowTheFloor(ulong freeBytes)
    {
        long rows = CudaBackend.TileRows(freeBytes, Heads, Tokens, Tokens, forced: null);

        // Every reading under 512 MB free puts a quarter of it under the floor, so the tile is exactly the floor's.
        Assert.Equal(CudaBackend.MinTileScoreBytes / PerRowBytes, rows);
    }

    [Fact]
    public void AGenerousReadingKeepsAQuarterOfFreeMemoryAsBefore()
    {
        ulong free = 8UL << 30;

        long rows = CudaBackend.TileRows(free, Heads, Tokens, Tokens, forced: null);

        Assert.Equal((long)(free / 4) / PerRowBytes, rows);
    }

    [Fact]
    public void TheTileNeverExceedsTheQueryRowsOrAForcedHeight()
    {
        Assert.Equal(100, CudaBackend.TileRows(ulong.MaxValue / 8, Heads, 100, Tokens, forced: null));
        Assert.Equal(256, CudaBackend.TileRows(0, Heads, Tokens, Tokens, forced: 256));
        Assert.Equal(100, CudaBackend.TileRows(0, Heads, 100, Tokens, forced: 256));
    }

    [Fact]
    public void ARememberedCapLimitsTheTileLikeTheFreeMemoryReadingDoes()
    {
        long capBytes = 40 * PerRowBytes;

        // A card that really is full: the floor would ask for more than fits, the cap says what did.
        Assert.Equal(40, CudaBackend.TileRows(0, Heads, Tokens, Tokens, forced: null, capBytes));
        Assert.Equal(40, CudaBackend.TileRows(8UL << 30, Heads, Tokens, Tokens, forced: null, capBytes));
        // No cap, and a cap larger than the budget, change nothing; a forced height still wins.
        Assert.Equal(CudaBackend.TileRows(0, Heads, Tokens, Tokens, null), CudaBackend.TileRows(0, Heads, Tokens, Tokens, null, 0));
        Assert.Equal(CudaBackend.TileRows(0, Heads, Tokens, Tokens, null), CudaBackend.TileRows(0, Heads, Tokens, Tokens, null, 1L << 40));
        Assert.Equal(256, CudaBackend.TileRows(0, Heads, Tokens, Tokens, forced: 256, capBytes));
        // A cap below one row still gives one.
        Assert.Equal(1, CudaBackend.TileRows(0, Heads, Tokens, Tokens, forced: null, capBytes: 1));
    }

    [Fact]
    public void ATileOfOneRowPastTheAddressableScoresIsRefused()
    {
        // 2^32 / (heads * keys) < 1: even one query row of every head is more scores than the softmax addresses.
        Assert.Throws<AttentionUnsupportedException>(() => CudaBackend.TileRows(0, 1024, 8_000_000, 8_000_000, forced: null));
    }
}
