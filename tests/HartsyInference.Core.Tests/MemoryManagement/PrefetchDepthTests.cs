using HartsyInference.Core.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Proves the shared depth calculation reproduces each pipeline's own formula exactly, including the three places they deliberately differed.</summary>
/// <remarks>The risk in hoisting this was averaging six near-identical formulas into one that is subtly wrong for
/// all of them — a prefetch that is one too deep overcommits VRAM at exactly the geometries streaming exists for.
/// So the sweeps below re-derive each original inline and assert equality rather than asserting chosen constants.</remarks>
public sealed class PrefetchDepthTests
{
    private const long Mb = 1024 * 1024;

    public static TheoryData<long, long> Budgets()
    {
        TheoryData<long, long> data = [];
        foreach (long avail in new long[] { -1, 0, 1, 100 * Mb, 512 * Mb, 1024 * Mb, 4096 * Mb, long.MaxValue / 4 })
        {
            foreach (long perBlock in new long[] { 0, 1, 64 * Mb, 384 * Mb, 512 * Mb })
            {
                data.Add(avail, perBlock);
            }
        }
        return data;
    }

    /// <summary>Krea2 / ChromaRadiance / QwenImage: <c>perBlock > 0 ? Clamp(avail/perBlock - 2, 0, 2) : 0</c>.</summary>
    [Theory]
    [MemberData(nameof(Budgets))]
    public void MatchesTheInlineFormula(long avail, long perBlock)
    {
        int legacy = perBlock > 0 ? Math.Clamp((int)(avail / perBlock) - 2, 0, 2) : 0;
        // The inline sites never guarded avail <= 0 separately; the clamp floor already handled it.
        Assert.Equal(legacy, PrefetchDepth.Choose(avail, perBlock));
    }

    /// <summary>Flux / HunyuanImage returned 1 for an unmeasurable block instead of 0, and bailed on a non-positive budget first.</summary>
    [Theory]
    [MemberData(nameof(Budgets))]
    public void MatchesTheHelperFormulaWithItsUnknownBlockFallback(long avail, long perBlock)
    {
        int legacy;
        if (avail <= 0) legacy = 0;
        else if (perBlock <= 0) legacy = 1;
        else legacy = Math.Clamp((int)(avail / perBlock) - 2, 0, 2);
        Assert.Equal(legacy, PrefetchDepth.Choose(avail, perBlock, maxDepth: 2, unknownBlockDepth: 1));
    }

    /// <summary>Ideogram 4 runs two DiTs at once, so it halves the budget and caps the depth at one.</summary>
    [Theory]
    [MemberData(nameof(Budgets))]
    public void MatchesIdeogramsTwoWindowSplit(long avail, long perBlock)
    {
        int legacy = perBlock > 0 ? Math.Clamp((int)(avail / 2 / perBlock) - 2, 0, 1) : 0;
        Assert.Equal(legacy, PrefetchDepth.Choose(avail / 2, perBlock, maxDepth: 1));
    }

    /// <summary>The -2 is the point of sharing this: the window transiently holds prefetch+2 blocks, so a budget of
    /// exactly N blocks must not authorize a depth of N.</summary>
    [Fact]
    public void LeavesRoomForTheTransientOverlap()
    {
        Assert.Equal(0, PrefetchDepth.Choose(2 * Mb, 1 * Mb));
        Assert.Equal(1, PrefetchDepth.Choose(3 * Mb, 1 * Mb));
        Assert.Equal(2, PrefetchDepth.Choose(4 * Mb, 1 * Mb));
        Assert.Equal(2, PrefetchDepth.Choose(400 * Mb, 1 * Mb));
    }

    [Fact]
    public void RejectsANegativeCap()
        => Assert.Throws<ArgumentOutOfRangeException>(() => PrefetchDepth.Choose(Mb, Mb, maxDepth: -1));
}
