using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>One case of tests/python-reference/longcat_bsa/dump_longcat_bsa.py, with its operands built as the script builds them.</summary>
internal sealed record BlockSparseAttentionCase(string Name, int Batch, int Heads, int Dim, TokenGrid3D QueryGrid, TokenGrid3D KeyGrid,
    TokenGrid3D QueryBlock, TokenGrid3D KeyBlock, double Sparsity, uint Seed, int RowStep)
{
    public static readonly BlockSparseAttentionCase[] All =
    [
        new("Shipped", 1, 2, 32, new(8, 16, 16), new(8, 16, 16), new(4, 4, 4), new(4, 4, 4), 0.9375, 11, 67),
        new("CrossLength", 1, 2, 64, new(4, 16, 16), new(12, 16, 16), new(4, 4, 4), new(4, 4, 4), 0.9375, 23, 61),
        new("Batched", 2, 3, 16, new(4, 32, 16), new(4, 32, 16), new(4, 4, 4), new(4, 4, 4), 0.875, 37, 127),
        new("UnequalBlocks", 1, 2, 16, new(8, 16, 16), new(8, 16, 32), new(4, 4, 4), new(4, 4, 8), 0.9375, 41, 67),
        new("HeadDim128", 1, 1, 128, new(8, 16, 16), new(8, 16, 16), new(4, 4, 4), new(4, 4, 4), 0.9375, 71, 127),
        new("Truncated", 1, 1, 32, new(4, 8, 8), new(20, 12, 8), new(4, 4, 4), new(4, 4, 4), 0.9, 53, 11),
        new("NoneSelected", 1, 1, 16, new(8, 8, 8), new(8, 8, 8), new(4, 4, 4), new(4, 4, 4), 0.9375, 67, 29),
    ];

    public static BlockSparseAttentionCase Named(string name) => All.Single(c => c.Name == name);

    public BlockSparseAttentionOptions Options => new()
    {
        QueryGrid = QueryGrid, KeyGrid = KeyGrid, QueryBlock = QueryBlock, KeyBlock = KeyBlock, Sparsity = Sparsity,
    };

    /// <summary>Runs the case on <paramref name="backend"/>; returns the output and, ascending, each query block's key blocks.</summary>
    public (float[] Output, int[] Selected) Run(IBackend backend)
    {
        int queryTokens = (int)QueryGrid.Tokens, keyTokens = (int)KeyGrid.Tokens;
        using Tensor query = Operand(queryTokens, Seed, 2f);
        using Tensor key = Operand(keyTokens, Seed + 1, 2f);
        using Tensor value = Operand(keyTokens, Seed + 2, 1f);
        using Tensor output = new(query.Shape, DType.F32);
        BlockSparseAttentionOptions options = Options;
        using Tensor selected = new(new TensorShape(Batch, Heads, options.QueryBlocks, options.SelectedBlocks), DType.I32);
        output.AsSpan<float>().Fill(-7f);
        backend.BlockSparseAttention(output, query, key, value, options, selected);
        // A report of no blocks holds no bytes to read.
        int[] blocks = options.SelectedBlocks > 0 ? selected.AsReadOnlySpan<int>().ToArray() : [];
        return (output.AsReadOnlySpan<float>().ToArray(), blocks);
    }

    /// <summary>Every <see cref="RowStep"/>-th query row of a full output, in the vectors' order.</summary>
    public float[] SampledRows(ReadOnlySpan<float> output)
    {
        List<float> rows = [];
        int queryTokens = (int)QueryGrid.Tokens;
        for (int sequence = 0; sequence < Batch * Heads; sequence++)
            for (int row = 0; row < queryTokens; row += RowStep)
                rows.AddRange(output.Slice((sequence * queryTokens + row) * Dim, Dim));
        return [.. rows];
    }

    /// <summary>The RMS of the difference over the RMS of <paramref name="expected"/>; the plain difference when that is zero.</summary>
    public static double RelativeRms(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double difference = (double)actual[i] - expected[i];
            error += difference * difference;
            norm += (double)expected[i] * expected[i];
        }
        return norm > 0 ? Math.Sqrt(error / norm) : Math.Sqrt(error / expected.Length);
    }

    /// <summary>The script's values in [-1, 1), a 32-bit LCG exact in F32, times <paramref name="factor"/>.</summary>
    private Tensor Operand(int tokens, uint seed, float factor)
    {
        Tensor tensor = new(new TensorShape(Batch, Heads, tokens, Dim), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        uint state = seed;
        for (int i = 0; i < values.Length; i++)
        {
            state = unchecked(state * 1664525u + 1013904223u);
            values[i] = (((state >> 8) & 0xFFFF) / 32768f - 1f) * factor;
        }
        return tensor;
    }
}
