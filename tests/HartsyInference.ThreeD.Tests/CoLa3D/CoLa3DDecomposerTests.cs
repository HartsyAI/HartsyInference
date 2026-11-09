using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.CoLa3D;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

public sealed class CoLa3DDecomposerTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void FixedQueries_ProduceFiniteOutputsWithTheDeclaredShapes()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 3);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [5, 4], seed: 4);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            using CoLa3DOutput output = decomposer.Forward(backend, latent, utonia);

            Assert.Equal(new TensorShape(9, 16), output.PointFeatures.Feats.Shape);
            Assert.Equal(new TensorShape(9, 3), output.MaskLogits.Shape);
            Assert.Equal(new TensorShape(2, 3, 16), output.QueryFeatures.Shape);
            AssertFinite(output.PointFeatures.Feats);
            AssertFinite(output.MaskLogits);
            AssertFinite(output.QueryFeatures);
            Assert.True(output.MaskLogits.AsReadOnlySpan<float>().ToArray().Any(v => v != 0f), "Mask logits are all zero.");
            Assert.True(output.QueryFeatures.AsReadOnlySpan<float>().ToArray().Any(v => v != 0f), "Query features are all zero.");
            Assert.Equal(6, output.ActiveSlots.Count);
            Assert.All(output.ActiveSlots, active => Assert.True(active));
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Fact]
    public void ActiveSlots_CannotBeCastBackToTheInternalArrayOrWrittenThrough()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 3);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [5, 4], seed: 4);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            using CoLa3DOutput output = decomposer.Forward(backend, latent, utonia);

            Assert.False(output.ActiveSlots is bool[]);
            Assert.Throws<NotSupportedException>(() => ((IList<bool>)output.ActiveSlots)[0] = false);
            Assert.All(output.ActiveSlots, active => Assert.True(active));
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Fact]
    public void LearnableQueries_RunForward()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed)
            with { QueryInit = CoLa3DConfig.QueryInitKind.Learnable };
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 35);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [5, 4], seed: 36);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            using CoLa3DOutput output = decomposer.Forward(backend, latent, utonia);

            Assert.Equal(new TensorShape(9, 3), output.MaskLogits.Shape);
            Assert.Equal(new TensorShape(2, 3, 16), output.QueryFeatures.Shape);
            AssertFinite(output.PointFeatures.Feats);
            AssertFinite(output.MaskLogits);
            AssertFinite(output.QueryFeatures);
            Assert.True(output.QueryFeatures.AsReadOnlySpan<float>().ToArray().Any(v => v != 0f), "Query features are all zero.");
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Fact]
    public void BatchedForward_MatchesEachSampleRunAlone()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 5);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [5, 4], seed: 6);
        (SparseTensor first, Tensor firstUtonia) = Sample(latent, utonia, start: 0, count: 5);
        (SparseTensor second, Tensor secondUtonia) = Sample(latent, utonia, start: 5, count: 4);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            using CoLa3DOutput batched = decomposer.Forward(backend, latent, utonia);
            using CoLa3DOutput alone0 = decomposer.Forward(backend, first, firstUtonia);
            using CoLa3DOutput alone1 = decomposer.Forward(backend, second, secondUtonia);

            AssertRowsClose(batched.PointFeatures.Feats, 0, alone0.PointFeatures.Feats, 0, 5);
            AssertRowsClose(batched.PointFeatures.Feats, 5, alone1.PointFeatures.Feats, 0, 4);
            AssertRowsClose(batched.MaskLogits, 0, alone0.MaskLogits, 0, 5);
            AssertRowsClose(batched.MaskLogits, 5, alone1.MaskLogits, 0, 4);
            AssertRowsClose(batched.QueryFeatures, 0, alone0.QueryFeatures, 0, 3);
            AssertRowsClose(batched.QueryFeatures, 3, alone1.QueryFeatures, 0, 3);
        }
        finally
        {
            first.Feats.Dispose();
            firstUtonia.Dispose();
            second.Feats.Dispose();
            secondUtonia.Dispose();
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Theory]
    [InlineData(CoLa3DConfig.PointFeatureSourceKind.Latent)]
    [InlineData(CoLa3DConfig.PointFeatureSourceKind.LatentUtonia)]
    public void PointPrompts_ValidSlotsMatchTheSubsetRun(CoLa3DConfig.PointFeatureSourceKind source)
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.PromptPoint) with { PointFeatureSource = source };
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 7);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [6], seed: 8);
        int[] c0 = Coords(latent, 0), c2 = Coords(latent, 2), c4 = Coords(latent, 4);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            // Slot 1 is invalid and sits on no voxel: it must be skipped, not looked up.
            CoLa3DQueryPrompts threeSlots = new()
            {
                Batch = 1,
                Queries = 3,
                Points = [c0[0], c0[1], c0[2], 0, 0, 0, c4[0], c4[1], c4[2]],
                Valid = [true, false, true],
            };
            CoLa3DQueryPrompts twoSlots = new()
            {
                Batch = 1,
                Queries = 2,
                Points = [c0[0], c0[1], c0[2], c4[0], c4[1], c4[2]],
            };
            using CoLa3DOutput masked = decomposer.Forward(backend, latent, utonia, threeSlots);
            using CoLa3DOutput subset = decomposer.Forward(backend, latent, utonia, twoSlots);

            AssertRowsClose(masked.PointFeatures.Feats, 0, subset.PointFeatures.Feats, 0, 6);
            AssertRowsClose(masked.QueryFeatures, 0, subset.QueryFeatures, 0, 1);
            AssertRowsClose(masked.QueryFeatures, 2, subset.QueryFeatures, 1, 1);
            AssertRowsClose(masked.MaskLogits, 0, subset.MaskLogits, 0, 6, columnMap: [(0, 0), (2, 1)]);
            AssertRowsZero(masked.QueryFeatures, 1);
            AssertColumnZero(masked.MaskLogits, 1);
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Fact]
    public void MaskPrompts_RunWithCenterEncodingAndZeroInvalidSlots()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.PromptMask);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 9);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [5, 4], seed: 10);
        Tensor masks = new(new TensorShape(2, 3, 8, 8), DType.F32);
        Random random = new(11);
        Span<float> maskValues = masks.AsSpan<float>();
        for (int i = 0; i < maskValues.Length; i++)
            maskValues[i] = random.Next(2);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            CoLa3DQueryPrompts prompts = new()
            {
                Batch = 2,
                Queries = 3,
                Masks = masks,
                Valid = [true, true, false, true, false, true],
            };
            using CoLa3DOutput output = decomposer.Forward(backend, latent, utonia, prompts);

            Assert.Equal(new TensorShape(2, 3, 16), output.QueryFeatures.Shape);
            AssertFinite(output.QueryFeatures);
            AssertFinite(output.MaskLogits);
            Assert.Equal(new bool[] { true, true, false, true, false, true }, output.ActiveSlots.ToArray());
            // Sample 0 (voxel rows 0-4) has slot 2 invalid; sample 1 (rows 5-8) has slot 1 invalid.
            AssertRowsZero(output.QueryFeatures, 2);
            AssertRowsZero(output.QueryFeatures, 4);
            AssertColumnZeroInRows(output.MaskLogits, column: 2, fromRow: 0, toRow: 5);
            AssertColumnZeroInRows(output.MaskLogits, column: 1, fromRow: 5, toRow: 9);
        }
        finally
        {
            masks.Dispose();
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Fact]
    public void Forward_RejectsPromptsForFixedQueriesAndMissingPromptsForPromptSources()
    {
        CoLa3DConfig fixedConfig = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        CoLa3DConfig pointConfig = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.PromptPoint);
        Dictionary<string, Tensor> fixedWeights = CoLa3DTestSupport.RandomWeights(fixedConfig, seed: 12);
        Dictionary<string, Tensor> pointWeights = CoLa3DTestSupport.RandomWeights(pointConfig, seed: 12);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(fixedConfig, [3], seed: 13);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer fixedDecomposer = new(fixedConfig, fixedWeights);
            using CoLa3DDecomposer pointDecomposer = new(pointConfig, pointWeights);
            CoLa3DQueryPrompts prompts = new() { Batch = 1, Queries = 1, Points = [0, 0, 0] };

            Assert.Throws<HartsyInferenceException>(() => fixedDecomposer.Forward(backend, latent, utonia, prompts));
            Assert.Throws<HartsyInferenceException>(() => pointDecomposer.Forward(backend, latent, utonia));
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(fixedWeights.Values);
            CoLa3DTestSupport.Dispose(pointWeights.Values);
        }
    }

    [Fact]
    public void Forward_RejectsBatchesThatAreNotGroupedFromZero()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 14);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [2, 2], seed: 15);
        latent.Coords[4] = 2;
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => decomposer.Forward(backend, latent, utonia));
            Assert.Contains("contiguous batch index", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Fact]
    public void Forward_RejectsAPointPromptWithNoVoxelAtItsCoordinate()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.PromptPoint);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 16);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [1], seed: 17);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            int[] missing = [7, 7, 7];
            if (Coords(latent, 0).SequenceEqual(missing))
                missing = [0, 0, 0];
            CoLa3DQueryPrompts prompts = new() { Batch = 1, Queries = 1, Points = missing };

            HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => decomposer.Forward(backend, latent, utonia, prompts));
            Assert.Contains("no active voxel", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Theory]
    [InlineData(8, 0, 0)]
    [InlineData(0, -1, 0)]
    [InlineData(0, 0, 8)]
    public void Forward_RejectsAPointPromptOutsideTheGrid(int x, int y, int z)
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.PromptPoint);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 37);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [3], seed: 38);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            CoLa3DQueryPrompts prompts = new() { Batch = 1, Queries = 1, Points = [x, y, z] };

            HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => decomposer.Forward(backend, latent, utonia, prompts));
            Assert.Contains("outside", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Fact]
    public void PointPrompts_SkipInvalidSlotsWhateverTheirCoordinates()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.PromptPoint);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 39);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [6], seed: 40);
        int[] c0 = Coords(latent, 0);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            // Slot 0 is invalid and lies outside the grid; only the valid slot is checked and gathered.
            CoLa3DQueryPrompts prompts = new()
            {
                Batch = 1,
                Queries = 2,
                Points = [config.LatentResolution, 0, 0, c0[0], c0[1], c0[2]],
                Valid = [false, true],
            };
            using CoLa3DOutput output = decomposer.Forward(backend, latent, utonia, prompts);

            Assert.Equal(new bool[] { false, true }, output.ActiveSlots.ToArray());
            AssertRowsZero(output.QueryFeatures, 0);
            AssertColumnZero(output.MaskLogits, 0);
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Fact]
    public void PointPrompts_WithEverySlotInvalidGiveZeroQueriesAndPointFeatures()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.PromptPoint);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 41);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [5, 4], seed: 42);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, weights);
            CoLa3DQueryPrompts prompts = new()
            {
                Batch = 2,
                Queries = 2,
                Points = new int[12],
                Valid = [false, false, false, false],
            };
            using CoLa3DOutput output = decomposer.Forward(backend, latent, utonia, prompts);

            Assert.Equal(new TensorShape(9, 16), output.PointFeatures.Feats.Shape);
            AssertFinite(output.PointFeatures.Feats);
            Assert.All(output.ActiveSlots, active => Assert.False(active));
            for (int row = 0; row < 4; row++)
                AssertRowsZero(output.QueryFeatures, row);
            Assert.All(output.MaskLogits.AsReadOnlySpan<float>().ToArray(), value => Assert.Equal(0f, value));
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Fact]
    public void Forward_AfterDispose_ThrowsObjectDisposedException()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 43);
        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [3], seed: 44);
        try
        {
            using CpuBackend backend = new();
            CoLa3DDecomposer decomposer = new(config, weights);
            decomposer.Dispose();

            Assert.Throws<ObjectDisposedException>(() => decomposer.Forward(backend, latent, utonia));
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    private static int[] Coords(SparseTensor latent, int row) => [latent.Coords[row * 4 + 1], latent.Coords[row * 4 + 2], latent.Coords[row * 4 + 3]];

    /// <summary>Copies rows <c>[start, start + count)</c> into a standalone single-sample latent with batch index 0.</summary>
    private static (SparseTensor Latent, Tensor Utonia) Sample(SparseTensor latent, Tensor utonia, int start, int count)
    {
        int channels = latent.Channels, width = (int)utonia.Shape[1];
        Tensor feats = new(new TensorShape(count, channels), DType.F32);
        latent.Feats.AsReadOnlySpan<float>().Slice(start * channels, count * channels).CopyTo(feats.AsSpan<float>());
        Tensor ut = new(new TensorShape(count, width), DType.F32);
        utonia.AsReadOnlySpan<float>().Slice(start * width, count * width).CopyTo(ut.AsSpan<float>());
        int[] coords = new int[count * 4];
        for (int r = 0; r < count; r++)
            for (int a = 1; a < 4; a++)
                coords[r * 4 + a] = latent.Coords[(start + r) * 4 + a];
        return (new SparseTensor(feats, coords, latent.Resolution), ut);
    }

    private static void AssertFinite(Tensor tensor)
    {
        foreach (float value in tensor.AsReadOnlySpan<float>())
            Assert.True(float.IsFinite(value), "Non-finite value in CoLa3D output.");
    }

    /// <summary>Compares <paramref name="rows"/> rows of a rank-2 or rank-3 tensor, row by row over the flattened width.</summary>
    private static void AssertRowsClose(Tensor a, int aRow, Tensor b, int bRow, int rows, (int From, int To)[]? columnMap = null)
    {
        int width = (int)a.Shape[a.Shape.Rank - 1];
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>(), y = b.AsReadOnlySpan<float>();
        if (columnMap is null)
        {
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < width; c++)
                    Assert.True(MathF.Abs(x[(aRow + r) * width + c] - y[(bRow + r) * width + c]) <= Tolerance,
                        $"Row {aRow + r} column {c} differs.");
            return;
        }

        int bWidth = (int)b.Shape[b.Shape.Rank - 1];
        for (int r = 0; r < rows; r++)
            foreach ((int from, int to) in columnMap)
                Assert.True(MathF.Abs(x[(aRow + r) * width + from] - y[(bRow + r) * bWidth + to]) <= Tolerance,
                    $"Row {aRow + r} column {from} differs from column {to}.");
    }

    private static void AssertRowsZero(Tensor tensor, int row)
    {
        int width = (int)tensor.Shape[tensor.Shape.Rank - 1];
        ReadOnlySpan<float> values = tensor.AsReadOnlySpan<float>().Slice(row * width, width);
        foreach (float value in values)
            Assert.Equal(0f, value);
    }

    private static void AssertColumnZero(Tensor logits, int column) =>
        AssertColumnZeroInRows(logits, column, fromRow: 0, toRow: (int)logits.Shape[0]);

    private static void AssertColumnZeroInRows(Tensor logits, int column, int fromRow, int toRow)
    {
        int width = (int)logits.Shape[1];
        ReadOnlySpan<float> values = logits.AsReadOnlySpan<float>();
        for (int r = fromRow; r < toRow; r++)
            Assert.Equal(0f, values[r * width + column]);
    }
}
