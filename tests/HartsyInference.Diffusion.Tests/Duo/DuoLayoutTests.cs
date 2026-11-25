using HartsyInference.Diffusion.Models.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

public sealed class DuoLayoutTests
{
    private static DuoSampleInput OneBlock(double startMs = 0.0, int bodyRows = 2, int descriptionRows = 2, int cueRows = 1,
        IReadOnlyList<int>? noisy = null, int clean = 0) => new()
    {
        SampleId = 7,
        SharedDescriptionRows = descriptionRows,
        SharedCueRows = cueRows,
        Reference = new DuoGrid(1, 4, 4),
        Blocks =
        [
            new DuoBlockInput
            {
                StartMs = startMs,
                BodyTextRows = bodyRows,
                Video = new DuoGrid(2, 4, 4),
                AudioSteps = 3,
                FrameStartsMs = [0.0, 100.0],
            },
        ],
        CleanBlocks = clean,
        NoisyBlocks = noisy ?? [0],
    };

    [Fact]
    public void CompactOrder_PacksReferenceSharedTextAndNoisyMediaThenPadsTo128()
    {
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);

        Assert.Equal(new[] { DuoGroupKind.Reference, DuoGroupKind.Shared, DuoGroupKind.Shared, DuoGroupKind.Text, DuoGroupKind.Noisy,
            DuoGroupKind.Noisy, DuoGroupKind.Padding }, layout.Groups.Select(g => g.Kind));
        Assert.Equal(23, layout.LiveRows);
        Assert.Equal(128, layout.Length);
        Assert.Equal(new DuoGroup(0, -1, DuoGroupKind.Reference, DuoModality.Video, 0, 4), layout.Groups[0]);
        Assert.Equal(new DuoGroup(0, 0, DuoGroupKind.Noisy, DuoModality.Video, 9, 17), layout.Groups[4]);
        Assert.Equal(new DuoGroup(0, 0, DuoGroupKind.Noisy, DuoModality.Audio, 17, 23), layout.Groups[5]);
    }

    [Fact]
    public void Tags_AreVideoZeroTextOneAudioTwoAndPaddingMinusOne()
    {
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        Assert.Equal(0, layout.Tags[0]);
        Assert.Equal(1, layout.Tags[7]);
        Assert.Equal(0, layout.Tags[9]);
        Assert.Equal(2, layout.Tags[17]);
        Assert.Equal(-1, layout.Tags[23]);
    }

    [Fact]
    public void VideoAndAudioRowsAreListedInGroupOrder()
    {
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        Assert.Equal(Enumerable.Range(0, 4).Concat(Enumerable.Range(9, 8)), layout.VideoRows);
        Assert.Equal(Enumerable.Range(17, 6), layout.AudioRows);
        Assert.Equal(new[] { 4, 5, 6, 7, 8 }, layout.TextRows);
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, layout.TextSourceRows);
    }

    [Fact]
    public void TimeIds_GiveNoisyRowsTheirOneBasedBlockIndexAndEveryOtherRowZero()
    {
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        Assert.Equal(0, layout.TimeIds[0]);
        Assert.Equal(0, layout.TimeIds[7]);
        Assert.Equal(1, layout.TimeIds[9]);
        Assert.Equal(1, layout.TimeIds[22]);
        Assert.Equal(0, layout.TimeIds[23]);
        Assert.Equal(3, layout.AdalnRow(9));                         // time 1 * 3 + video 0
        Assert.Equal(1, layout.AdalnRow(7));                         // time 0 * 3 + text 1
        Assert.Equal(5, layout.AdalnRow(17));                        // time 1 * 3 + audio 2
    }

    [Fact]
    public void VideoPositions_UseTheFrameTimeAndTheSpatialAxesWithWFastest()
    {
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        Assert.Equal(new double[] { 0, 0, 0 }, Row(layout, 9));
        Assert.Equal(new double[] { 0, 0, 16 }, Row(layout, 10));
        Assert.Equal(new double[] { 0, 16, 0 }, Row(layout, 11));
        Assert.Equal(new double[] { 0, 16, 16 }, Row(layout, 12));
        // Frame 1 starts at 100 ms = 4 clock units; F32 storage keeps that exact.
        Assert.Equal(new double[] { 4, 0, 0 }, Row(layout, 13));
    }

    [Fact]
    public void AudioPositions_CountStepsOnTheBlockClockWithChannelOneOnTheWideEdge()
    {
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock(startMs: 1000.0)], DuoRopePolicy.Old);
        Assert.Equal(new double[] { 40, 0, 0 }, Row(layout, 17));
        Assert.Equal(new double[] { 41, 0, 0 }, Row(layout, 18));
        Assert.Equal(new double[] { 42, 0, 0 }, Row(layout, 19));
        Assert.Equal(new double[] { 40, 0, 16 }, Row(layout, 20));
    }

    [Fact]
    public void Positions_OldPolicyPinsSharedAtMinusOneAndBlockTextAtItsStart()
    {
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock(startMs: 1000.0)], DuoRopePolicy.Old);
        Assert.Equal(-1, Row(layout, 4)[0]);
        Assert.Equal(-1, Row(layout, 6)[0]);
        Assert.Equal(40, Row(layout, 7)[0]);
        Assert.Equal(40, Row(layout, 8)[0]);
    }

    [Fact]
    public void Positions_NewPolicyPacksSharedSpansToEndAtMinusOneAndCountsBlockTokens()
    {
        // Description has 2 rows, cue has 3: the spans are [-5,-4] and [-3,-2,-1].
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock(startMs: 1000.0, descriptionRows: 2, cueRows: 3, bodyRows: 2)], DuoRopePolicy.New);
        // Rows: reference 0-3, description 4-5, cue 6-8, body 9-10, then the noisy video from row 11.
        Assert.Equal(-5, Row(layout, 4)[0]);
        Assert.Equal(-4, Row(layout, 5)[0]);
        Assert.Equal(-3, Row(layout, 6)[0]);
        Assert.Equal(-1, Row(layout, 8)[0]);
        Assert.Equal(40, Row(layout, 9)[0]);
        Assert.Equal(41, Row(layout, 10)[0]);
    }

    [Fact]
    public void RowOrder_ClusterEachCleanBlockBeforeItsIntentAndTheNoisyBlockLast()
    {
        DuoSampleInput sample = new()
        {
            SampleId = 1,
            SharedDescriptionRows = 1,
            SharedCueRows = 1,
            Reference = new DuoGrid(1, 2, 2),
            Blocks =
            [
                new DuoBlockInput { StartMs = 0, BodyTextRows = 1, Video = new DuoGrid(1, 2, 2), AudioSteps = 0, FrameStartsMs = [0.0] },
                new DuoBlockInput { StartMs = 500, BodyTextRows = 1, Video = new DuoGrid(1, 2, 2), AudioSteps = 0, FrameStartsMs = [0.0] },
            ],
            CleanBlocks = 1,
            NoisyBlocks = [1],
        };
        DuoLayout layout = DuoLayoutBuilder.Build([sample], DuoRopePolicy.New);
        // Reference 0, shared 1-2, clean block 0 at 3, its Intent 4, block 1 Intent 5, noisy block 1 at 6.
        Assert.Equal(new[] { DuoGroupKind.Reference, DuoGroupKind.Shared, DuoGroupKind.Shared, DuoGroupKind.Clean, DuoGroupKind.Text,
            DuoGroupKind.Text, DuoGroupKind.Noisy, DuoGroupKind.Padding }, layout.Groups.Select(g => g.Kind).ToArray());
        Assert.Equal(new DuoGroup(0, 0, DuoGroupKind.Clean, DuoModality.Video, 3, 4), layout.Groups[3]);
        Assert.Equal(0, layout.TimeIds[3]);
        Assert.Equal(1, layout.TimeIds[6]);
    }

    [Fact]
    public void MultipleConversations_NumberNoisyBlocksAcrossThePackAndSourceTheirIntentRows()
    {
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock(), OneBlock()], DuoRopePolicy.Old);
        Assert.Equal(1, layout.TimeIds[9]);
        int secondNoisy = layout.Groups.Single(g => g.Sample == 1 && g.Kind == DuoGroupKind.Noisy && g.Modality == DuoModality.Video).Start;
        Assert.Equal(2, layout.TimeIds[secondNoisy]);
        // Sample 1's intent rows follow sample 0's five; its first shared row is source 5.
        Assert.Equal(5, layout.TextSourceRows[5]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public void Build_RefusesAGridWithAnOddOrEmptyDimension(int height)
    {
        DuoSampleInput sample = OneBlock() with { Reference = new DuoGrid(1, height, 4) };
        Assert.ThrowsAny<ArgumentException>(() => DuoLayoutBuilder.Build([sample], DuoRopePolicy.Old));
    }

    [Fact]
    public void Build_RefusesFrameStartsThatDoNotMatchTheFrameCount()
    {
        DuoSampleInput sample = OneBlock() with
        {
            Blocks = [new DuoBlockInput { StartMs = 0, BodyTextRows = 1, Video = new DuoGrid(2, 4, 4), AudioSteps = 0, FrameStartsMs = [0.0] }],
        };
        Assert.Throws<ArgumentException>(() => DuoLayoutBuilder.Build([sample], DuoRopePolicy.Old));
    }

    [Fact]
    public void SpatialAxes_MatchTheClosedFormForANonSquareGrid()
    {
        (double[] h, double[] w) = DuoLayoutBuilder.SpatialAxes(2, 6);
        Assert.Single(h);
        Assert.Equal(3, w.Length);
        Assert.InRange(h[0], 6.7623959 - 1e-5, 6.7623959 + 1e-5);
        double[] expectedW = [-11.712813, 6.7623959, 25.237603];
        for (int i = 0; i < 3; i++)
        {
            Assert.InRange(w[i], expectedW[i] - 1e-5, expectedW[i] + 1e-5);
        }
    }

    private static double[] Row(DuoLayout layout, int row) =>
        [layout.Positions[row * 3], layout.Positions[row * 3 + 1], layout.Positions[row * 3 + 2]];
}
