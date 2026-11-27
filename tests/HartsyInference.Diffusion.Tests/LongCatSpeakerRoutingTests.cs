using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>How the multi-speaker path routes: the boxes to token masks, the reference map to the queries' 1D positions, the tracks to the audio
/// keys' positions, the map itself against a direct computation, and the refusals. The numbers against the reference are in
/// <c>LongCatAvatarMultiSpeakerParityTests</c>.</summary>
public sealed class LongCatSpeakerRoutingTests
{
    /// <summary>A 16 x 32 portrait generated at its own size, on a 2 x 4 token grid: token (row, column) samples pixel (8·row, 8·column).</summary>
    private static (int Regions, float[] Masks) Masks(LongCatSpeakerBox? first, LongCatSpeakerBox? second, params LongCatSpeakerBox[] others) =>
        LongCatSpeakerMasks.TokenMasks(16, 32, first, second, others, 16, 32, 2, 4);

    [Fact]
    public void EachBoxMarksItsOwnTokensAndTheBackgroundIsTheRest()
    {
        (int regions, float[] masks) = Masks(new LongCatSpeakerBox(0, 0, 16, 8), new LongCatSpeakerBox(8, 24, 16, 32),
            new LongCatSpeakerBox(0, 8, 8, 16));

        Assert.Equal(4, regions);
        Assert.Equal([1, 0, 0, 0, 1, 0, 0, 0], masks.Take(8));              // first speaker: the left column
        Assert.Equal([0, 0, 0, 0, 0, 0, 0, 1], masks.Skip(8).Take(8));      // second speaker: bottom right
        Assert.Equal([0, 1, 1, 1, 0, 1, 1, 0], masks.Skip(16).Take(8));     // background: outside both speakers, the silent person included
        Assert.Equal([0, 1, 0, 0, 0, 0, 0, 0], masks.Skip(24).Take(8));     // the silent person
    }

    [Fact]
    public void TheDefaultBoxesAreTheHalvesInsetByATenth()
    {
        (LongCatSpeakerBox first, LongCatSpeakerBox second) = LongCatSpeakerMasks.DefaultBoxes(100, 201);

        Assert.Equal(new LongCatSpeakerBox(10, 10, 90, 90), first);
        Assert.Equal(new LongCatSpeakerBox(10, 110, 90, 190), second);
        // Without boxes the masks are those of the default ones.
        Assert.Equal(LongCatSpeakerMasks.TokenMasks(100, 200, first, second, [], 64, 128, 4, 8).Masks,
            LongCatSpeakerMasks.TokenMasks(100, 200, null, null, [], 64, 128, 4, 8).Masks);
    }

    [Fact]
    public void ARegionTheCentreCropRemovesHasNoToken()
    {
        // A 16 x 96 portrait at 16 x 32 keeps columns 32 to 63.
        (_, float[] masks) = LongCatSpeakerMasks.TokenMasks(16, 96, new LongCatSpeakerBox(0, 40, 16, 56), new LongCatSpeakerBox(0, 0, 16, 24), [],
            16, 32, 2, 4);

        Assert.Contains(1f, masks.Take(8));
        Assert.All(masks.Skip(8).Take(8), value => Assert.Equal(0f, value));
    }

    [Fact]
    public void OneBoxAloneAndAnEmptyBoxAreRefused()
    {
        Assert.Throws<ArgumentException>(() => Masks(new LongCatSpeakerBox(0, 0, 8, 8), null));
        Assert.Throws<ArgumentException>(() => Masks(new LongCatSpeakerBox(4, 0, 4, 8), new LongCatSpeakerBox(0, 8, 8, 16)));
        LongCatSpeakerBox left = new(0, 0, 16, 16), right = new(0, 16, 16, 32);
        Assert.Throws<ArgumentException>(() => Masks(left, right, new LongCatSpeakerBox(0, -1, 8, 8)));
        // The reference cannot combine silent people with its default boxes.
        Assert.Throws<ArgumentException>(() => Masks(null, null, new LongCatSpeakerBox(0, 0, 8, 8)));
    }

    [Fact]
    public void EachTokenTakesThePositionOfTheRegionItAttendsToMost()
    {
        // Three regions, five tokens. Tokens: first, second, background, first (a tie with the second goes to the first), second.
        float[] map =
        [
            0.9f, 0.1f, 0.2f, 0.5f, 0.3f,
            0.2f, 0.8f, 0.1f, 0.5f, 0.6f,
            0.1f, 0.2f, 0.7f, 0.0f, 0.1f,
        ];

        float[] positions = LongCatSpeakerRope.QueryPositions(map, 3, 5);

        // A speaker's values are scaled by their range over every token: the first speaker's into [0, 4], the second's into [20, 24].
        Assert.Equal((0.9f - 0.1f) / (0.9f - 0.1f + 1e-8f) * 4f, positions[0]);
        Assert.Equal((0.8f - 0.1f) / (0.8f - 0.1f + 1e-8f) * 4f + 20f, positions[1]);
        Assert.Equal(12f, positions[2]);
        Assert.Equal((0.5f - 0.1f) / (0.9f - 0.1f + 1e-8f) * 4f, positions[3]);
        Assert.Equal((0.6f - 0.1f) / (0.8f - 0.1f + 1e-8f) * 4f + 20f, positions[4]);
        Assert.InRange(positions[4], 20f, 24f);
    }

    [Fact]
    public void ASilentPersonsRegionCountsAsBackgroundAtTheFarPosition()
    {
        // Four regions, two tokens: the background wins the first, the silent person the second.
        float[] map = [0.1f, 0.1f, 0.2f, 0.2f, 0.9f, 0.1f, 0.3f, 0.8f];

        Assert.Equal([100f, 100f], LongCatSpeakerRope.QueryPositions(map, 4, 2));
    }

    [Theory]
    [InlineData(2, 3, new float[] { 2, 2, 22, 22 })]
    [InlineData(3, 3, new float[] { 2, 2, 22, 22, 12, 12 })]
    [InlineData(3, 4, new float[] { 2, 2, 22, 22, 100, 100 })]
    public void TheAudioKeysSitAtTheirSpeakersPositions(int tracks, int regions, float[] expected) =>
        Assert.Equal(expected, LongCatSpeakerRope.KeyPositions(tracks, 2, regions));

    [Fact]
    public void TheTablesHoldOneAnglePerPair()
    {
        (Tensor cos, Tensor sin) = LongCatSpeakerRope.Tables([0f, 22f], 4);
        using (cos)
        using (sin)
        {
            Assert.Equal([1f, 1f, 1f, 1f], cos.AsReadOnlySpan<float>()[..4].ToArray());
            float slow = 22f * (1f / MathF.Pow(10000f, 0.5f));
            Assert.Equal([MathF.Cos(22f), MathF.Cos(22f), MathF.Cos(slow), MathF.Cos(slow)], cos.AsReadOnlySpan<float>()[4..].ToArray());
            Assert.Equal([MathF.Sin(22f), MathF.Sin(22f), MathF.Sin(slow), MathF.Sin(slow)], sin.AsReadOnlySpan<float>()[4..].ToArray());
        }
    }

    [Fact]
    public void TracksJoinFrameByFrameAndTheReferenceSlotCanBeAddedBeforeOrAfter()
    {
        // Two frames of two values per track.
        float[] one = [1, 2, 3, 4], two = [5, 6, 7, 8];

        Assert.Equal([1, 2, 5, 6, 3, 4, 7, 8], LongCatAvatarAudio.JoinTracks([one, two], 2));
        Assert.Same(one, LongCatAvatarAudio.JoinTracks([one], 2));
        Assert.Equal(
            LongCatAvatarChainPlan.ReferenceSlotTokens(LongCatAvatarAudio.JoinTracks([one, two], 2), 4),
            LongCatAvatarAudio.JoinTracks(
                [LongCatAvatarChainPlan.ReferenceSlotTokens(one, 2), LongCatAvatarChainPlan.ReferenceSlotTokens(two, 2)], 3));
        Assert.Throws<ArgumentException>(() => LongCatAvatarAudio.JoinTracks([one, [1, 2]], 2));
    }

    [Theory]
    [InlineData(4, 5)]
    [InlineData(5, 4096)]
    public void TheReferenceMapIsTheRegionMeanOfTheAttentionWhateverTheTileSize(int heads, int tileRows)
    {
        const int HeadDim = 8, ReferenceTokens = 6, Rows = 13, FirstRow = 2, Regions = 3;
        Random rng = new(11);
        float[] q = LongCatTestData.RandomValues(rng, (FirstRow + Rows) * heads * HeadDim);
        float[] k = LongCatTestData.RandomValues(rng, heads * ReferenceTokens * HeadDim);
        // The second region is empty.
        float[] masks = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
        using CpuBackend backend = new();
        using LongCatSpeakerPass pass = new(new LongCatSpeakers(2, Regions, 2, 3, masks), heads, HeadDim, 4, 1, tileRows);
        AttentionLayer layer = new(new AttentionLayerConfig
        {
            Heads = heads, KvHeads = heads, HeadDim = HeadDim, Projection = AttentionProjection.PerWeight, ComputeDType = DType.F32,
        });
        using Tensor queries = new(new TensorShape(FirstRow + Rows, heads, HeadDim), DType.F32);
        q.CopyTo(queries.AsSpan<float>());
        using Tensor keys = new(new TensorShape(1, heads, ReferenceTokens, HeadDim), DType.F32);
        k.CopyTo(keys.AsSpan<float>());

        float[] map = pass.ReferenceMap(backend, layer, queries, FirstRow, keys);

        // Two groups of heads/2 heads: an odd last head is left out, as the reference leaves it.
        int used = 2 * (heads / 2);
        float[] expected = new float[Regions * Rows];
        for (int row = 0; row < Rows; row++)
            for (int h = 0; h < used; h++)
            {
                double[] scores = new double[ReferenceTokens];
                for (int key = 0; key < ReferenceTokens; key++)
                    for (int d = 0; d < HeadDim; d++)
                        scores[key] += q[((FirstRow + row) * heads + h) * HeadDim + d] * (double)k[(h * ReferenceTokens + key) * HeadDim + d]
                            / Math.Sqrt(HeadDim);
                double max = scores.Max(), sum = scores.Sum(score => Math.Exp(score - max));
                for (int r = 0; r < Regions; r++)
                {
                    double inside = 0, count = 0;
                    for (int key = 0; key < ReferenceTokens; key++)
                    {
                        inside += masks[r * ReferenceTokens + key] * Math.Exp(scores[key] - max) / sum;
                        count += masks[r * ReferenceTokens + key];
                    }
                    expected[r * Rows + row] += (float)(inside / Math.Max(count, 1) / used);
                }
            }
        Assert.True(LongCatTestData.RelRms(map, expected) < 1e-5, $"relRMS {LongCatTestData.RelRms(map, expected):E2}");
        Assert.All(map.Skip(Rows).Take(Rows), value => Assert.Equal(0f, value));
    }

    [Fact]
    public void ConditioningRefusesBadShapes()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LongCatSpeakers(1, 3, 1, 1, new float[3]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LongCatSpeakers(2, 2, 1, 1, new float[2]));
        Assert.Throws<ArgumentException>(() => new LongCatSpeakers(2, 3, 1, 1, new float[4]));
        Assert.Throws<ArgumentException>(() => new LongCatSpeakers(2, 3, 1, 1, [0f, 0.5f, 1f]));
        Assert.Throws<ArgumentException>(() => new LongCatSpeakerPass(new LongCatSpeakers(2, 3, 1, 1, new float[3]), 2, 2, 4, 1));
    }
}
