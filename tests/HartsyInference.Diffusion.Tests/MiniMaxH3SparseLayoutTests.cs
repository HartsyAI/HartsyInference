using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Structural tests for H3's generation-scoped segment-pure / 4x4x4 VSA layout.</summary>
public sealed class MiniMaxH3SparseLayoutTests
{
    /// <summary>Prefix blocks never cross segment classes, and every live row belongs to exactly one block.</summary>
    [Fact]
    public void Build_SplitsPrefixBySegmentAndCoversEveryRowOnce()
    {
        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(
            textLen: 70, latentT: 5, latentH: 10, latentW: 12, audioT: 35,
            keyframes: [new MiniMaxH3Keyframe { ResolvedFrameIndex = 0 }], frameCount: 17);

        VideoSparseAttentionPlan plan = MiniMaxH3SparseLayoutBuilder.Build(
            layout, VideoSparseAttentionProfileKind.ComfySol64V1);

        Assert.Equal(layout.SequenceLength, plan.SequenceLength);
        Assert.True(plan.PrefixSinkBlocks > 0);
        bool[] seen = new bool[layout.SequenceLength];
        for (int block = 0; block < plan.BlockOffsets.Length - 1; block++)
        {
            int start = plan.BlockOffsets[block];
            int stop = plan.BlockOffsets[block + 1];
            Assert.InRange(stop - start, 1, 64);
            foreach (int source in plan.SourceIndices[start..stop])
            {
                Assert.False(seen[source]);
                seen[source] = true;
                MiniMaxH3Segment segment = layout.Segments.Single(candidate =>
                    source >= candidate.Start && source < candidate.Stop);
                Assert.Equal((int)segment.Kind, plan.SegmentClasses[block]);
            }
        }
        Assert.All(seen, Assert.True);
    }

    /// <summary>Target-video rows are traversed as spatial-temporal 4x4x4 cubes, including ragged edges.</summary>
    [Fact]
    public void Build_TargetVideoUsesFourByFourByFourCubes()
    {
        const int latentT = 5;
        const int latentH = 10;
        const int latentW = 12;
        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(
            textLen: 1, latentT, latentH, latentW, audioT: 1);
        VideoSparseAttentionPlan plan = MiniMaxH3SparseLayoutBuilder.Build(
            layout, VideoSparseAttentionProfileKind.FastVideoVsa64V1);
        MiniMaxH3Segment video = layout.Segments.Last();

        int rowsH = latentH / 2;
        int rowsW = latentW / 2;
        List<int> expectedFirstCube = new List<int>(64);
        for (int t = 0; t < 4; t++)
        {
            for (int y = 0; y < 4; y++)
            {
                for (int x = 0; x < 4; x++)
                {
                    expectedFirstCube.Add(video.Start + (t * rowsH + y) * rowsW + x);
                }
            }
        }
        int firstVideoBlock = plan.PrefixSinkBlocks;
        int start = plan.BlockOffsets[firstVideoBlock];
        int stop = plan.BlockOffsets[firstVideoBlock + 1];
        Assert.Equal(expectedFirstCube, plan.SourceIndices[start..stop]);
        Assert.Equal(8, plan.SegmentClasses.Length - plan.PrefixSinkBlocks);
    }

    /// <summary>The profile changes routing semantics, not the source layout.</summary>
    [Fact]
    public void Build_ProfileKindsShareTheSameImmutableGeometry()
    {
        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(3, 5, 8, 8, 4);
        VideoSparseAttentionPlan comfy = MiniMaxH3SparseLayoutBuilder.Build(
            layout, VideoSparseAttentionProfileKind.ComfySol64V1);
        VideoSparseAttentionPlan fast = MiniMaxH3SparseLayoutBuilder.Build(
            layout, VideoSparseAttentionProfileKind.FastVideoVsa64V1);

        Assert.Equal(comfy.BlockOffsets, fast.BlockOffsets);
        Assert.Equal(comfy.SourceIndices, fast.SourceIndices);
        Assert.Equal(comfy.SegmentClasses, fast.SegmentClasses);
        Assert.Equal(comfy.PrefixSinkBlocks, fast.PrefixSinkBlocks);
    }
}
