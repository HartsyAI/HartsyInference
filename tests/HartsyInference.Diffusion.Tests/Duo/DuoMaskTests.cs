using HartsyInference.Diffusion.Models.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

public sealed class DuoMaskTests
{
    private static DuoGroup Group(int sample, int block, DuoGroupKind kind, int start, int end) =>
        new(sample, block, kind, kind is DuoGroupKind.Text or DuoGroupKind.Shared or DuoGroupKind.Reference ? DuoModality.None : DuoModality.Video, start, end);

    [Fact]
    public void Visible_FollowsTheCausalRulesOfTheReferenceVisibility()
    {
        DuoGroup reference = Group(0, -1, DuoGroupKind.Reference, 0, 4);
        DuoGroup shared = Group(0, -1, DuoGroupKind.Shared, 4, 6);
        DuoGroup text0 = Group(0, 0, DuoGroupKind.Text, 6, 8);
        DuoGroup clean0 = Group(0, 0, DuoGroupKind.Clean, 8, 12);
        DuoGroup noisy0 = Group(0, 0, DuoGroupKind.Noisy, 12, 16);
        DuoGroup text1 = Group(0, 1, DuoGroupKind.Text, 16, 18);
        DuoGroup noisy1 = Group(0, 1, DuoGroupKind.Noisy, 18, 22);

        // Reference and shared rows see everything of their own kind, and every row sees them.
        Assert.True(DuoMask.Visible(text0, reference, DuoAttentionMode.Causal, refiner: false));
        Assert.True(DuoMask.Visible(noisy1, shared, DuoAttentionMode.Causal, refiner: false));
        Assert.False(DuoMask.Visible(reference, shared, DuoAttentionMode.Causal, refiner: false));
        Assert.False(DuoMask.Visible(shared, text0, DuoAttentionMode.Causal, refiner: false));

        // Text sees only its own block's text; noisy sees only its own block's noisy rows.
        Assert.True(DuoMask.Visible(text0, text0, DuoAttentionMode.Causal, refiner: false));
        Assert.False(DuoMask.Visible(text1, text0, DuoAttentionMode.Causal, refiner: false));
        Assert.True(DuoMask.Visible(noisy0, text0, DuoAttentionMode.Causal, refiner: false));
        Assert.False(DuoMask.Visible(text0, noisy0, DuoAttentionMode.Causal, refiner: false));
        Assert.False(DuoMask.Visible(noisy1, noisy0, DuoAttentionMode.Causal, refiner: false));

        // Noisy rows see earlier clean history but never the clean copy of their own block.
        Assert.True(DuoMask.Visible(noisy1, clean0, DuoAttentionMode.Causal, refiner: false));
        Assert.False(DuoMask.Visible(noisy0, clean0, DuoAttentionMode.Causal, refiner: false));
        Assert.True(DuoMask.Visible(clean0, clean0, DuoAttentionMode.Causal, refiner: false));
        Assert.False(DuoMask.Visible(clean0, noisy1, DuoAttentionMode.Causal, refiner: false));
    }

    [Fact]
    public void Visible_NeverCrossesSamplesAndPairsPaddingOnlyWithPadding()
    {
        DuoGroup a = Group(0, -1, DuoGroupKind.Reference, 0, 4);
        DuoGroup b = Group(1, -1, DuoGroupKind.Reference, 4, 8);
        DuoGroup padding = new(-1, -1, DuoGroupKind.Padding, DuoModality.None, 8, 16);
        Assert.False(DuoMask.Visible(b, a, DuoAttentionMode.Causal, refiner: false));
        Assert.False(DuoMask.Visible(a, padding, DuoAttentionMode.Causal, refiner: false));
        Assert.True(DuoMask.Visible(padding, padding, DuoAttentionMode.Causal, refiner: false));
    }

    [Fact]
    public void Visible_BidirectionalSeesAllSameSampleRowsAndRefusesClean()
    {
        DuoGroup text = Group(0, 0, DuoGroupKind.Text, 0, 2);
        DuoGroup noisy = Group(0, 1, DuoGroupKind.Noisy, 2, 6);
        Assert.True(DuoMask.Visible(text, noisy, DuoAttentionMode.Bidirectional, refiner: false));
        DuoGroup clean = Group(0, 0, DuoGroupKind.Clean, 6, 8);
        Assert.Throws<InvalidOperationException>(() => DuoMask.Visible(noisy, clean, DuoAttentionMode.Bidirectional, refiner: false));
    }

    [Fact]
    public void Visible_RefinerSeesSharedRowsAndOwnBlockText()
    {
        DuoGroup shared = Group(0, -1, DuoGroupKind.Shared, 0, 2);
        DuoGroup text0 = Group(0, 0, DuoGroupKind.Text, 2, 4);
        DuoGroup text1 = Group(0, 1, DuoGroupKind.Text, 4, 6);
        Assert.True(DuoMask.Visible(text0, shared, DuoAttentionMode.Causal, refiner: true));
        Assert.True(DuoMask.Visible(text0, text0, DuoAttentionMode.Causal, refiner: true));
        Assert.False(DuoMask.Visible(text1, text0, DuoAttentionMode.Causal, refiner: true));
        Assert.False(DuoMask.Visible(shared, text0, DuoAttentionMode.Causal, refiner: true));
    }

    [Fact]
    public void Build_MergesAdjacentVisibleKeysIntoOneRectangle()
    {
        // A noisy query of block 1 sees reference, shared, clean block 0 and its own noisy rows, which are contiguous.
        List<DuoGroup> keys =
        [
            Group(0, -1, DuoGroupKind.Reference, 0, 4),
            Group(0, -1, DuoGroupKind.Shared, 4, 6),
            Group(0, 0, DuoGroupKind.Clean, 6, 10),
            Group(0, 1, DuoGroupKind.Noisy, 10, 14),
        ];
        IReadOnlyList<DuoMaskRect> rects = DuoMask.Build([keys[3]], keys, DuoAttentionMode.Causal);
        Assert.Equal([new DuoMaskRect(10, 14, 0, 14)], rects);
    }

    [Fact]
    public void Build_SplitsAroundAHiddenKeyAndDenseMatchesTheRules()
    {
        // A clean copy of block 1 sits between two shared-visible spans; a block 0 query must not see it.
        List<DuoGroup> keys =
        [
            Group(0, -1, DuoGroupKind.Shared, 0, 2),
            Group(0, 1, DuoGroupKind.Clean, 2, 4),
            Group(0, -1, DuoGroupKind.Reference, 4, 6),
        ];
        IReadOnlyList<DuoGroup> queries = [Group(0, 0, DuoGroupKind.Text, 6, 8)];
        IReadOnlyList<DuoMaskRect> rects = DuoMask.Build(queries, keys, DuoAttentionMode.Causal);
        Assert.Equal([new DuoMaskRect(6, 8, 0, 2), new DuoMaskRect(6, 8, 4, 6)], rects);

        bool[,] dense = DuoMask.Dense(rects, 8, 6);
        for (int q = 6; q < 8; q++)
        {
            Assert.False(dense[q, 2]);
            Assert.False(dense[q, 3]);
            Assert.True(dense[q, 0]);
            Assert.True(dense[q, 5]);
        }
    }

    [Fact]
    public void Build_RefusesAQueryThatSeesNothing()
    {
        IReadOnlyList<DuoGroup> keys = [Group(0, 1, DuoGroupKind.Noisy, 0, 4)];
        IReadOnlyList<DuoGroup> queries = [Group(0, 0, DuoGroupKind.Text, 4, 6)];
        Assert.Throws<InvalidOperationException>(() => DuoMask.Build(queries, keys, DuoAttentionMode.Causal));
    }
}
