using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Models.Utonia;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Utonia;

public sealed class UtoniaEncoderTests
{
    [Fact]
    public void ForwardLevel4_ReturnsLastStageFeaturesAndItsCoarseGridWithoutBatch()
    {
        UtoniaConfig config = UtoniaTestData.TinyConfig();
        using UtoniaEncoder encoder = new(new CpuBackend(), config, UtoniaTestData.RandomWeights(config, 7));
        (int[] coords, int batchSize) = RandomCoords(seed: 11, grid: 16, perSample: new[] { 12, 7 });

        UtoniaFeatures result = encoder.ForwardLevel4(coords, batchSize);

        // Two pooling steps of stride 2 take a 16 grid to 4, so the last stage is the input index shifted by 2 bits.
        int[] expectedGrid = ExpectedCoarseGrid(coords, shift: 2);
        Assert.Equal((long)(expectedGrid.Length / 3), result.Feats.Shape[0]);
        Assert.Equal(36L, result.Feats.Shape[1]);
        Assert.Equal(expectedGrid, result.GridCoord);
        Assert.Equal(36, encoder.BottleneckChannels);
        Assert.All(UtoniaTestData.Read(result.Feats), v => Assert.True(float.IsFinite(v)));
        result.Feats.Dispose();
    }

    [Fact]
    public void ForwardLevel4_IsDeterministicForFixedWeights()
    {
        UtoniaConfig config = UtoniaTestData.TinyConfig();
        using UtoniaEncoder encoder = new(new CpuBackend(), config, UtoniaTestData.RandomWeights(config, 9));
        (int[] coords, int batchSize) = RandomCoords(seed: 4, grid: 16, perSample: new[] { 9 });

        UtoniaFeatures first = encoder.ForwardLevel4(coords, batchSize);
        UtoniaFeatures second = encoder.ForwardLevel4(coords, batchSize);

        Assert.Equal(UtoniaTestData.Read(first.Feats), UtoniaTestData.Read(second.Feats));
        first.Feats.Dispose();
        second.Feats.Dispose();
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Forward_RowChunksMatchTheUnchunkedRunExactly(bool encoderOnly)
    {
        UtoniaConfig config = UtoniaTestData.TinyConfig(encoderOnly);
        (int[] coords, int batchSize) = RandomCoords(seed: 31, grid: 16, perSample: new[] { 13, 6 });
        using UtoniaEncoder whole = new(new CpuBackend(), config, UtoniaTestData.RandomWeights(config, 5), UtoniaRowChunks.ChunkBytes);
        using UtoniaEncoder chunked = new(new CpuBackend(), config, UtoniaTestData.RandomWeights(config, 5), chunkBytes: 96);

        UtoniaFeatures expected = whole.Forward(coords, batchSize);
        UtoniaFeatures actual = chunked.Forward(coords, batchSize);

        Assert.Equal(expected.GridCoord, actual.GridCoord);
        float[] e = UtoniaTestData.Read(expected.Feats), a = UtoniaTestData.Read(actual.Feats);
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits(e[i]), BitConverter.SingleToInt32Bits(a[i]));
        }

        expected.Feats.Dispose();
        actual.Feats.Dispose();
    }

    [Fact]
    public void ForwardLevel4_EmptyInputReturnsZeroRows()
    {
        UtoniaConfig config = UtoniaTestData.TinyConfig();
        using UtoniaEncoder encoder = new(new CpuBackend(), config, UtoniaTestData.RandomWeights(config, 1));

        UtoniaFeatures result = encoder.ForwardLevel4(ReadOnlySpan<int>.Empty, 1);

        Assert.Equal(0, result.Feats.Shape[0]);
        Assert.Empty(result.GridCoord);
        result.Feats.Dispose();
    }

    [Fact]
    public void Forward_WithDecoder_RestoresTheInputGridInInputOrder()
    {
        UtoniaConfig config = UtoniaTestData.TinyConfig(encoderOnly: false);
        using UtoniaEncoder encoder = new(new CpuBackend(), config, UtoniaTestData.RandomWeights(config, 13));
        (int[] coords, int batchSize) = RandomCoords(seed: 21, grid: 16, perSample: new[] { 6, 5 });
        int n = coords.Length / 4;

        UtoniaFeatures result = encoder.Forward(coords, batchSize);

        Assert.Equal((long)n, result.Feats.Shape[0]);
        Assert.Equal(12L, result.Feats.Shape[1]);
        int[] inputGrid = new int[n * 3];
        for (int i = 0; i < n; i++)
        {
            inputGrid[i * 3] = coords[i * 4 + 1];
            inputGrid[i * 3 + 1] = coords[i * 4 + 2];
            inputGrid[i * 3 + 2] = coords[i * 4 + 3];
        }

        Assert.Equal(inputGrid, result.GridCoord);
        result.Feats.Dispose();
    }

    [Fact]
    public void DirectGrid_CentersEachSampleInXYAndStartsZAtZero()
    {
        int[] coords = { 0, 0, 0, 0, 0, 10, 20, 30 };

        UtoniaPoint point = UtoniaDirectGrid.Build(coords, batchSize: 1, colors: null, normals: null);

        float[] expected = { -0.05f, -0.1f, 0f, 0.05f, 0.1f, 0.3f };
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], point.Coord[i], 5);
        }

        float[] feat = UtoniaTestData.Read(point.Feat);
        Assert.Equal(new float[] { -0.05f, -0.1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f }, feat.Take(9).ToArray(), new FloatComparer(1e-5f));
        point.Dispose();
    }

    [Fact]
    public void Validate_RejectsHeadDimensionsThat3DRopeCannotSplit()
    {
        UtoniaConfig config = UtoniaTestData.TinyConfig() with { EncNumHeads = [4, 4, 6] };

        Assert.Throws<HartsyInferenceException>(() => config.Validate());
    }

    [Fact]
    public void Validate_RejectsUnknownSerializationOrder()
    {
        UtoniaConfig config = UtoniaTestData.TinyConfig() with { Order = ["z", "morton"] };

        Assert.Throws<HartsyInferenceException>(() => config.Validate());
    }

    private static (int[] Coords, int BatchSize) RandomCoords(int seed, int grid, int[] perSample)
    {
        Random rng = new(seed);
        int total = perSample.Sum();
        int[] coords = new int[total * 4];
        int row = 0;
        for (int b = 0; b < perSample.Length; b++)
        {
            for (int i = 0; i < perSample[b]; i++, row++)
            {
                coords[row * 4] = b;
                coords[row * 4 + 1] = rng.Next(grid);
                coords[row * 4 + 2] = rng.Next(grid);
                coords[row * 4 + 3] = rng.Next(grid);
            }
        }

        return (coords, perSample.Length);
    }

    /// <summary>Distinct (x, y, z) cells after shifting, in (batch, x, y, z) order, without the batch column.</summary>
    private static int[] ExpectedCoarseGrid(int[] coords, int shift)
    {
        List<(int B, int X, int Y, int Z)> cells = new();
        for (int i = 0; i < coords.Length / 4; i++)
        {
            cells.Add((coords[i * 4], coords[i * 4 + 1] >> shift, coords[i * 4 + 2] >> shift, coords[i * 4 + 3] >> shift));
        }

        List<int> flat = new();
        foreach ((int _, int x, int y, int z) in cells.Distinct().OrderBy(c => c.B).ThenBy(c => c.X).ThenBy(c => c.Y).ThenBy(c => c.Z))
        {
            flat.Add(x);
            flat.Add(y);
            flat.Add(z);
        }

        return flat.ToArray();
    }

    private sealed class FloatComparer : IEqualityComparer<float>
    {
        private readonly float _tolerance;

        public FloatComparer(float tolerance) => _tolerance = tolerance;

        public bool Equals(float x, float y) => Math.Abs(x - y) <= _tolerance;

        public int GetHashCode(float obj) => 0;
    }
}
