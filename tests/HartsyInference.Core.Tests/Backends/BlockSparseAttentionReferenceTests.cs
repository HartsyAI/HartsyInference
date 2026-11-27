using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The block-sparse attention reference against dense attention under a mask this class builds itself, against the outputs of
/// LongCat-Video's kernel (<see cref="BlockSparseAttentionVectors"/>), and on the rules the kernel does not state.</summary>
public sealed class BlockSparseAttentionReferenceTests(ITestOutputHelper log)
{
    private const double Tolerance = 1e-5;

    [Theory]
    [InlineData(1, 2, 3, 8, new[] { 4, 8, 12 }, new[] { 4, 8, 12 }, new[] { 4, 4, 4 }, new[] { 4, 4, 4 }, 0.5, 0f)]
    [InlineData(2, 1, 2, 16, new[] { 4, 8, 8 }, new[] { 8, 8, 12 }, new[] { 4, 4, 4 }, new[] { 4, 4, 4 }, 0.75, 0.7f)]
    [InlineData(3, 2, 1, 5, new[] { 2, 6, 4 }, new[] { 4, 4, 8 }, new[] { 2, 3, 2 }, new[] { 4, 2, 4 }, 0.4, 0f)]
    [InlineData(4, 1, 2, 12, new[] { 4, 8, 8 }, new[] { 4, 8, 8 }, new[] { 4, 4, 4 }, new[] { 4, 4, 4 }, 0.0, 0f)]
    [InlineData(5, 1, 1, 8, new[] { 8, 8, 8 }, new[] { 8, 8, 8 }, new[] { 4, 4, 4 }, new[] { 4, 4, 4 }, 0.875, 0f)]
    public void MatchesDenseAttentionUnderTheSameMask(int seed, int batch, int heads, int dim, int[] queryGrid, int[] keyGrid,
        int[] queryBlock, int[] keyBlock, double sparsity, float scale)
    {
        using CpuBackend backend = new();
        BlockSparseAttentionOptions options = new()
        {
            QueryGrid = Box(queryGrid), KeyGrid = Box(keyGrid), QueryBlock = Box(queryBlock), KeyBlock = Box(keyBlock),
            Sparsity = sparsity, Scale = scale == 0f ? null : scale,
        };
        int queryTokens = (int)options.QueryGrid.Tokens, keyTokens = (int)options.KeyGrid.Tokens;
        Random random = new(seed);
        using Tensor query = Noise(random, batch, heads, queryTokens, dim);
        using Tensor key = Noise(random, batch, heads, keyTokens, dim);
        using Tensor value = Noise(random, batch, heads, keyTokens, dim);
        using Tensor output = new(query.Shape, DType.F32);
        using Tensor expected = new(query.Shape, DType.F32);
        using Tensor selected = new(new TensorShape(batch, heads, options.QueryBlocks, options.SelectedBlocks), DType.I32);
        using Tensor bias = new(new TensorShape(batch, heads, queryTokens, keyTokens), DType.F32);

        int[] chosen = FillMask(bias, query, key, options);
        backend.ScaledDotProductAttention(expected, query, key, value, bias, scale == 0f ? 1f / MathF.Sqrt(dim) : scale);
        backend.BlockSparseAttention(output, query, key, value, options, selected);

        Assert.Equal(chosen, selected.AsReadOnlySpan<int>().ToArray());
        double error = BlockSparseAttentionCase.RelativeRms(output.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
        log.WriteLine($"relative RMS against dense attention: {error:E3}");
        Assert.True(error <= Tolerance, $"relative RMS {error:E3}");
    }

    [Theory]
    [InlineData("Shipped")]
    [InlineData("CrossLength")]
    [InlineData("Batched")]
    [InlineData("UnequalBlocks")]
    [InlineData("HeadDim128")]
    [InlineData("Truncated")]
    [InlineData("NoneSelected")]
    public void MatchesTheReferenceKernel(string name)
    {
        using CpuBackend backend = new();
        BlockSparseAttentionCase reference = BlockSparseAttentionCase.Named(name);
        (uint[] bits, int[] blocks) = name switch
        {
            "Shipped" => (BlockSparseAttentionVectors.ShippedOutput, BlockSparseAttentionVectors.ShippedSelected),
            "CrossLength" => (BlockSparseAttentionVectors.CrossLengthOutput, BlockSparseAttentionVectors.CrossLengthSelected),
            "Batched" => (BlockSparseAttentionVectors.BatchedOutput, BlockSparseAttentionVectors.BatchedSelected),
            "UnequalBlocks" => (BlockSparseAttentionVectors.UnequalBlocksOutput, BlockSparseAttentionVectors.UnequalBlocksSelected),
            "HeadDim128" => (BlockSparseAttentionVectors.HeadDim128Output, BlockSparseAttentionVectors.HeadDim128Selected),
            "Truncated" => (BlockSparseAttentionVectors.TruncatedOutput, BlockSparseAttentionVectors.TruncatedSelected),
            _ => (BlockSparseAttentionVectors.NoneSelectedOutput, []),
        };

        (float[] output, int[] selected) = reference.Run(backend);

        Assert.Equal(blocks, selected);
        float[] expected = Array.ConvertAll(bits, BitConverter.UInt32BitsToSingle);
        double error = BlockSparseAttentionCase.RelativeRms(reference.SampledRows(output), expected);
        log.WriteLine($"{name}: relative RMS against the reference kernel {error:E3} over {expected.Length} values");
        Assert.True(error <= Tolerance, $"{name}: relative RMS {error:E3}");
    }

    [Theory]
    [InlineData(20, 12, 8, 0.9, 2)]
    [InlineData(8, 8, 8, 0.9375, 0)]
    [InlineData(12, 44, 80, 0.9375, 41)]
    [InlineData(4, 4, 4, 0.0, 1)]
    [InlineData(4, 4, 4, 1.0, 0)]
    public void SelectedBlockCountTruncatesInDouble(int frames, int height, int width, double sparsity, int expected)
    {
        BlockSparseAttentionOptions options = new() { QueryGrid = new(4, 4, 4), KeyGrid = new(frames, height, width), Sparsity = sparsity };
        Assert.Equal(expected, options.SelectedBlocks);
    }

    [Fact]
    public void EqualScoresGoToTheLowerBlock()
    {
        using Fixture f = new(sparsity: 0.5);
        f.Key.AsSpan<float>().Fill(0.25f);
        f.Run();
        Assert.All(f.SelectedPerQueryBlock(), static blocks => Assert.Equal([0, 1], blocks));
    }

    [Fact]
    public void ANaNKeyRanksFirstAndReachesTheOutput()
    {
        using Fixture f = new(sparsity: 0.75);
        Span<float> key = f.Key.AsSpan<float>();
        key[^1] = float.NaN;
        f.Run();
        Assert.All(f.SelectedPerQueryBlock(), static blocks => Assert.Equal([3], blocks));
        Assert.All(f.Output.AsReadOnlySpan<float>().ToArray(), static v => Assert.True(float.IsNaN(v)));
    }

    [Fact]
    public void NoSelectedBlockWritesZeros()
    {
        using Fixture f = new(sparsity: 0.9);
        Assert.Equal(0, f.Options.SelectedBlocks);
        f.Run();
        Assert.All(f.Output.AsReadOnlySpan<float>().ToArray(), static v => Assert.Equal(0f, v));
    }

    [Fact]
    public void MalformedCallsAreRefusedBeforeAnyWrite()
    {
        using Fixture f = new(sparsity: 0.5);
        BlockSparseAttentionOptions good = f.Options;
        foreach (BlockSparseAttentionOptions bad in (BlockSparseAttentionOptions[])
        [
            good with { QueryBlock = new(4, 4, 3) },
            good with { KeyBlock = new(0, 4, 4) },
            good with { KeyGrid = new(4, 8, 12) },
            good with { Sparsity = double.NaN },
            good with { Sparsity = 1.5 },
            good with { Sparsity = -0.1 },
            good with { Scale = float.PositiveInfinity },
        ])
            Assert.ThrowsAny<ArgumentException>(() => f.Backend.BlockSparseAttention(f.Output, f.Query, f.Key, f.Value, bad));
        Assert.Throws<ArgumentException>(() => f.Backend.BlockSparseAttention(f.Query, f.Query, f.Key, f.Value, good));
        Assert.Throws<ArgumentException>(() => f.Backend.BlockSparseAttention(f.Output, f.Query, f.Key, f.Query, good));
        using Tensor half = new(f.Query.Shape, DType.F16);
        Assert.Throws<ArgumentException>(() => f.Backend.BlockSparseAttention(f.Output, half, f.Key, f.Value, good));
        using Tensor wrongReport = new(new TensorShape(1, 1, 2, 1), DType.I32);
        Assert.Throws<ArgumentException>(() => f.Backend.BlockSparseAttention(f.Output, f.Query, f.Key, f.Value, good, wrongReport));
        Assert.All(f.Output.AsReadOnlySpan<float>().ToArray(), static v => Assert.Equal(-7f, v));
    }

    private static TokenGrid3D Box(int[] extents) => new(extents[0], extents[1], extents[2]);

    private static Tensor Noise(Random random, int batch, int heads, int tokens, int dim)
    {
        Tensor tensor = new(new TensorShape(batch, heads, tokens, dim), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 4f - 2f;
        return tensor;
    }

    /// <summary>The reference's mask, written from its definition and sharing no code with the operation: block means in double, the
    /// highest dot products, and a bias of 0 on the pairs of a chosen pair of blocks, negative infinity elsewhere. Returns the chosen
    /// blocks, ascending, as <c>[B, H, queryBlock, selected]</c>.</summary>
    private static int[] FillMask(Tensor bias, Tensor query, Tensor key, BlockSparseAttentionOptions options)
    {
        int sequences = (int)(query.Shape[0] * query.Shape[1]), dim = (int)query.Shape[3];
        int queryTokens = (int)query.Shape[2], keyTokens = (int)key.Shape[2];
        int queryBlocks = options.QueryBlocks, keyBlocks = options.KeyBlocks;
        int count = (int)((1.0 - options.Sparsity) * keyBlocks);
        List<int> chosen = [];
        Span<float> mask = bias.AsSpan<float>();
        for (int s = 0; s < sequences; s++)
        {
            ReadOnlySpan<float> queryRows = query.AsReadOnlySpan<float>().Slice(s * queryTokens * dim, queryTokens * dim);
            ReadOnlySpan<float> keyRows = key.AsReadOnlySpan<float>().Slice(s * keyTokens * dim, keyTokens * dim);
            double[][] q = Means(queryRows, options.QueryGrid, options.QueryBlock, dim);
            double[][] k = Means(keyRows, options.KeyGrid, options.KeyBlock, dim);
            bool[,] read = new bool[queryBlocks, keyBlocks];
            for (int m = 0; m < queryBlocks; m++)
            {
                double Score(int n) => q[m].Zip(k[n], static (a, b) => a * b).Sum();
                foreach (int n in Enumerable.Range(0, keyBlocks).OrderByDescending(Score).Take(count).Order())
                {
                    read[m, n] = true;
                    chosen.Add(n);
                }
            }
            for (int i = 0; i < queryTokens; i++)
                for (int j = 0; j < keyTokens; j++)
                {
                    int m = BlockOf(i, options.QueryGrid, options.QueryBlock), n = BlockOf(j, options.KeyGrid, options.KeyBlock);
                    mask[(s * queryTokens + i) * keyTokens + j] = read[m, n] ? 0f : float.NegativeInfinity;
                }
        }
        return [.. chosen];
    }

    private static int BlockOf(int row, TokenGrid3D grid, TokenGrid3D block)
    {
        int frame = row / (grid.Height * grid.Width), y = row / grid.Width % grid.Height, x = row % grid.Width;
        return (frame / block.Frames * (grid.Height / block.Height) + y / block.Height) * (grid.Width / block.Width) + x / block.Width;
    }

    private static double[][] Means(ReadOnlySpan<float> rows, TokenGrid3D grid, TokenGrid3D block, int dim)
    {
        double[][] means = new double[grid.Tokens / block.Tokens][];
        for (int n = 0; n < means.Length; n++) means[n] = new double[dim];
        for (int row = 0; row < grid.Tokens; row++)
            for (int d = 0; d < dim; d++)
                means[BlockOf(row, grid, block)][d] += rows[row * dim + d] / (double)block.Tokens;
        return means;
    }

    /// <summary>One head, four query tokens' worth of blocks over four key blocks, with an output primed to −7.</summary>
    private sealed class Fixture : IDisposable
    {
        public readonly CpuBackend Backend = new();
        public readonly Tensor Query, Key, Value, Output;
        public readonly BlockSparseAttentionOptions Options;
        private readonly Tensor _selected;

        public Fixture(double sparsity)
        {
            Options = new() { QueryGrid = new(4, 4, 8), KeyGrid = new(4, 8, 8), Sparsity = sparsity };
            Random random = new(9);
            Query = Noise(random, 1, 1, 128, 4);
            Key = Noise(random, 1, 1, 256, 4);
            Value = Noise(random, 1, 1, 256, 4);
            Output = new Tensor(Query.Shape, DType.F32);
            Output.AsSpan<float>().Fill(-7f);
            _selected = new Tensor(new TensorShape(1, 1, Options.QueryBlocks, Options.SelectedBlocks), DType.I32);
        }

        public void Run() => Backend.BlockSparseAttention(Output, Query, Key, Value, Options, _selected);

        public int[][] SelectedPerQueryBlock() =>
            [.. _selected.AsReadOnlySpan<int>().ToArray().Chunk(Options.SelectedBlocks)];

        public void Dispose()
        {
            foreach (Tensor tensor in (ReadOnlySpan<Tensor>)[Query, Key, Value, Output, _selected]) tensor.Dispose();
            Backend.Dispose();
        }
    }
}
