using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Block-sparse attention on CUDA against the CPU reference and against LongCat-Video's kernel, held to the bar of
/// docs/BACKEND_OPS.md: the same selected blocks; an F32 output within a relative RMS of 1e-5; a BF16 output that is the kernel's own
/// F32 result rounded to nearest even, and within a relative RMS of 1e-4 of the reference's.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class BlockSparseAttentionGpuTests(ITestOutputHelper log)
{
    private const double WideTolerance = 1e-5, NarrowTolerance = 1e-4;

    [CudaTheory]
    [InlineData("Shipped")]
    [InlineData("CrossLength")]
    [InlineData("Batched")]
    [InlineData("UnequalBlocks")]
    [InlineData("HeadDim128")]
    [InlineData("Truncated")]
    [InlineData("NoneSelected")]
    public void MatchesTheReferenceKernelsVectors(string name)
    {
        using CudaBackend cuda = Open();
        BlockSparseAttentionCase reference = BlockSparseAttentionCase.Named(name);
        (uint[] bits, int[] blocks) = reference.Vectors;

        (float[] output, int[] selected) = reference.Run(cuda);

        Assert.Equal(blocks, selected);
        float[] expected = Array.ConvertAll(bits, BitConverter.UInt32BitsToSingle);
        double error = BlockSparseAttentionCase.RelativeRms(reference.SampledRows(output), expected);
        log.WriteLine($"{name}: CUDA relative RMS against the reference kernel {error:E3}");
        Assert.True(error <= WideTolerance, $"{name}: relative RMS {error:E3}");
    }

    [CudaTheory]
    [InlineData("Shipped")]
    [InlineData("CrossLength")]
    [InlineData("Batched")]
    [InlineData("UnequalBlocks")]
    [InlineData("HeadDim128")]
    [InlineData("Truncated")]
    [InlineData("NoneSelected")]
    public void MatchesTheCpuReferenceInEveryDtype(string name)
    {
        using CudaBackend cuda = Open();
        BlockSparseAttentionCase reference = BlockSparseAttentionCase.Named(name);
        (Tensor query, Tensor key, Tensor value) = reference.Operands();
        using (query)
        using (key)
        using (value)
            foreach (DType operands in (DType[])[DType.F32, DType.BF16])
                foreach (DType result in (DType[])[DType.F32, DType.BF16])
                    Compare(cuda, query, key, value, reference.Options, operands, result, name);
    }

    /// <summary>Query blocks shorter than a tile and longer than one, key chunks that straddle blocks, every compiled head dim,
    /// several sequences, an explicit scale, and every block read.</summary>
    [CudaTheory]
    [InlineData(1, 2, 3, 16, new[] { 2, 6, 4 }, new[] { 4, 4, 8 }, new[] { 2, 3, 2 }, new[] { 4, 2, 4 }, 0.4, 0f)]
    [InlineData(2, 1, 2, 32, new[] { 4, 8, 16 }, new[] { 8, 8, 8 }, new[] { 4, 4, 8 }, new[] { 4, 4, 4 }, 0.75, 0.7f)]
    [InlineData(3, 2, 1, 64, new[] { 4, 4, 12 }, new[] { 4, 8, 12 }, new[] { 4, 4, 6 }, new[] { 1, 2, 3 }, 0.9, 0f)]
    [InlineData(4, 1, 2, 128, new[] { 4, 8, 8 }, new[] { 4, 8, 8 }, new[] { 4, 4, 4 }, new[] { 4, 4, 4 }, 0.0, 0f)]
    [InlineData(5, 1, 3, 16, new[] { 3, 5, 7 }, new[] { 3, 5, 7 }, new[] { 3, 5, 7 }, new[] { 1, 1, 1 }, 0.33, 0f)]
    [InlineData(6, 1, 1, 128, new[] { 5, 10, 10 }, new[] { 2, 6, 6 }, new[] { 5, 5, 5 }, new[] { 1, 3, 3 }, 0.5, 0f)]
    public void RandomShapesMatchTheCpuReference(int seed, int batch, int heads, int dim, int[] queryGrid, int[] keyGrid,
        int[] queryBlock, int[] keyBlock, double sparsity, float scale)
    {
        using CudaBackend cuda = Open();
        BlockSparseAttentionOptions options = new()
        {
            QueryGrid = Box(queryGrid), KeyGrid = Box(keyGrid), QueryBlock = Box(queryBlock), KeyBlock = Box(keyBlock),
            Sparsity = sparsity, Scale = scale == 0f ? null : scale,
        };
        Random random = new(seed);
        using Tensor query = Noise(random, batch, heads, (int)options.QueryGrid.Tokens, dim);
        using Tensor key = Noise(random, batch, heads, (int)options.KeyGrid.Tokens, dim);
        using Tensor value = Noise(random, batch, heads, (int)options.KeyGrid.Tokens, dim);
        foreach (DType operands in (DType[])[DType.F32, DType.BF16])
            Compare(cuda, query, key, value, options, operands, DType.F32, $"seed {seed}");
    }

    /// <summary>The 2640 key blocks of a 48 × 44 × 80 token grid, which the selection searches for every query block.</summary>
    [CudaFact]
    public void SelectsAsTheReferenceDoesAmongManyKeyBlocks()
    {
        using CudaBackend cuda = Open();
        BlockSparseAttentionOptions options = new() { QueryGrid = new(4, 4, 8), KeyGrid = new(48, 44, 80), Sparsity = 0.9375 };
        Random random = new(71);
        using Tensor query = Noise(random, 1, 2, 128, 16);
        using Tensor key = Noise(random, 1, 2, (int)options.KeyGrid.Tokens, 16);
        using Tensor value = Noise(random, 1, 2, (int)options.KeyGrid.Tokens, 16);
        Assert.Equal(165, options.SelectedBlocks);
        Compare(cuda, query, key, value, options, DType.F32, DType.F32, "2640 key blocks");
    }

    /// <summary>Ties at the last block taken, a NaN key, a NaN query row and logits that are all negative infinity.</summary>
    [CudaTheory]
    [InlineData("ties")]
    [InlineData("nan key")]
    [InlineData("nan query")]
    [InlineData("excluded")]
    public void SpecialScoresMatchTheCpuReference(string name)
    {
        using CudaBackend cuda = Open();
        BlockSparseAttentionOptions options = new() { QueryGrid = new(4, 4, 8), KeyGrid = new(8, 8, 12), Sparsity = 0.75 };
        Random random = new(19);
        using Tensor query = Noise(random, 1, 1, 128, 16);
        using Tensor key = Noise(random, 1, 1, 768, 16);
        using Tensor value = Noise(random, 1, 1, 768, 16);
        Span<float> q = query.AsSpan<float>(), k = key.AsSpan<float>();
        switch (name)
        {
            case "ties":
                // Three scores shared by four blocks each, so every query block takes three of the four that score highest.
                for (int block = 0; block < options.KeyBlocks; block++)
                    for (int offset = 0; offset < 64; offset++)
                        k.Slice(options.KeyRow(block, offset) * 16, 16).Fill(block % 3 * 0.25f);
                break;
            case "nan key":
                k[700 * 16 + 3] = float.NaN;
                break;
            case "nan query":
                q[5 * 16 + 1] = float.NaN;
                break;
            default:
                q.Fill(1e20f);
                k.Fill(-1e20f);
                break;
        }
        Compare(cuda, query, key, value, options, DType.F32, DType.F32, name);
    }

    [CudaFact]
    public void ResidentOperandsAreReadAndNothingIsSyncedUntilAResultIs()
    {
        using CudaBackend cuda = Open();
        BlockSparseAttentionCase reference = BlockSparseAttentionCase.Named("Shipped");
        BlockSparseAttentionOptions options = reference.Options;
        (Tensor query, Tensor key, Tensor value) = reference.Operands();
        using (query)
        using (key)
        using (value)
        using (Tensor residentQuery = new(query.Shape, DType.F32))
        using (Tensor residentKey = new(key.Shape, DType.F32))
        using (Tensor residentValue = new(value.Shape, DType.F32))
        {
            cuda.Scale(residentQuery, query, 1f);
            cuda.Scale(residentKey, key, 1f);
            cuda.Scale(residentValue, value, 1f);
            using Tensor output = new(query.Shape, DType.F32);
            using Tensor selected = new(new TensorShape(1, 2, options.QueryBlocks, options.SelectedBlocks), DType.I32);
            cuda.Sync();
            cuda.ResetD2hSyncCount();

            cuda.BlockSparseAttention(output, residentQuery, residentKey, residentValue, options, selected);

            Assert.Equal(0, cuda.GetD2hSyncCount());
            Assert.Equal(BlockSparseAttentionVectors.ShippedSelected, selected.AsReadOnlySpan<int>().ToArray());
            float[] expected = Array.ConvertAll(BlockSparseAttentionVectors.ShippedOutput, BitConverter.UInt32BitsToSingle);
            double error = BlockSparseAttentionCase.RelativeRms(reference.SampledRows(output.AsReadOnlySpan<float>()), expected);
            Assert.True(error <= WideTolerance, $"relative RMS {error:E3}");
        }
    }

    [CudaFact]
    public void AnUncompiledHeadDimAndTooManyKeyBlocksAreRefusedBeforeAnyWrite()
    {
        using CudaBackend cuda = Open();
        Refused(cuda, new() { QueryGrid = new(4, 4, 4), KeyGrid = new(4, 4, 4), Sparsity = 0.0 }, 24);
        Refused(cuda, new() { QueryGrid = new(4, 4, 4), KeyGrid = new(4, 4, 4 * 12289), Sparsity = 0.9 }, 16);
    }

    private static void Refused(CudaBackend cuda, BlockSparseAttentionOptions options, int dim)
    {
        using Tensor query = new(new TensorShape(1, 1, options.QueryGrid.Tokens, dim), DType.F32);
        using Tensor key = new(new TensorShape(1, 1, options.KeyGrid.Tokens, dim), DType.F32);
        using Tensor value = new(key.Shape, DType.F32);
        using Tensor output = new(query.Shape, DType.F32);
        output.AsSpan<float>().Fill(-7f);
        Assert.Throws<NotSupportedException>(() => cuda.BlockSparseAttention(output, query, key, value, options));
        cuda.Sync();
        Assert.All(output.AsReadOnlySpan<float>().ToArray(), static v => Assert.Equal(-7f, v));
    }

    /// <summary>Runs the CPU reference and CUDA on the operands cast to <paramref name="operands"/> and holds CUDA to the bar.</summary>
    private void Compare(CudaBackend cuda, Tensor query, Tensor key, Tensor value, BlockSparseAttentionOptions options,
        DType operands, DType result, string name)
    {
        using CpuBackend cpu = new();
        using Tensor? q = operands == DType.F32 ? null : query.CastTo(operands);
        using Tensor? k = operands == DType.F32 ? null : key.CastTo(operands);
        using Tensor? v = operands == DType.F32 ? null : value.CastTo(operands);
        TensorShape report = new(query.Shape[0], query.Shape[1], options.QueryBlocks, options.SelectedBlocks);
        using Tensor expected = new(query.Shape, result), actual = new(query.Shape, result);
        using Tensor expectedSelected = new(report, DType.I32), actualSelected = new(report, DType.I32);
        cpu.BlockSparseAttention(expected, q ?? query, k ?? key, v ?? value, options, expectedSelected);
        if (result == DType.F32) actual.AsSpan<float>().Fill(-7f);
        else actual.AsSpan<ushort>().Fill(0xC0E0);

        cuda.BlockSparseAttention(actual, q ?? query, k ?? key, v ?? value, options, actualSelected);

        if (options.SelectedBlocks > 0)
            Assert.Equal(expectedSelected.AsReadOnlySpan<int>().ToArray(), actualSelected.AsReadOnlySpan<int>().ToArray());
        float[] want = Floats(expected), got = Floats(actual);
        double error = 0, norm = 0, largest = 0;
        for (int i = 0; i < want.Length; i++)
        {
            if (float.IsNaN(want[i]))
            {
                Assert.True(float.IsNaN(got[i]), $"{name} {operands}->{result}: element {i} is {got[i]}, the reference NaN");
                continue;
            }
            Assert.False(float.IsNaN(got[i]), $"{name} {operands}->{result}: element {i} is NaN, the reference {want[i]}");
            largest = Math.Max(largest, Math.Abs((double)got[i] - want[i]));
            error += ((double)got[i] - want[i]) * ((double)got[i] - want[i]);
            norm += (double)want[i] * want[i];
        }
        if (result == DType.BF16)
        {
            using Tensor wide = new(query.Shape, DType.F32);
            cuda.BlockSparseAttention(wide, q ?? query, k ?? key, v ?? value, options);
            ReadOnlySpan<float> unrounded = wide.AsReadOnlySpan<float>();
            ReadOnlySpan<ushort> rounded = actual.AsReadOnlySpan<ushort>();
            for (int i = 0; i < unrounded.Length; i++)
                if (!float.IsNaN(unrounded[i]))
                    Assert.True(rounded[i] == TensorCasts.F32ToBf16BitsRounded(unrounded[i]),
                        $"{name} {operands}->BF16: element {i} is {got[i]}, not {unrounded[i]} rounded to nearest even");
        }
        double relative = norm > 0 ? Math.Sqrt(error / norm) : Math.Sqrt(error / want.Length);
        log.WriteLine($"{name} {operands}->{result}: relative RMS {relative:E3} over {want.Length} values, largest difference {largest:E3}");
        Assert.True(relative <= (result == DType.F32 ? WideTolerance : NarrowTolerance), $"{name} {operands}->{result}: relative RMS {relative:E3}");
    }

    private static float[] Floats(Tensor tensor)
    {
        if (tensor.DType == DType.F32) return tensor.AsReadOnlySpan<float>().ToArray();
        return Array.ConvertAll(tensor.AsReadOnlySpan<ushort>().ToArray(), static b => BitConverter.UInt32BitsToSingle((uint)b << 16));
    }

    private static TokenGrid3D Box(int[] extents) => new(extents[0], extents[1], extents[2]);

    private static Tensor Noise(Random random, int batch, int heads, int tokens, int dim)
    {
        Tensor tensor = new(new TensorShape(batch, heads, tokens, dim), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 4f - 2f;
        return tensor;
    }

    private static CudaBackend Open()
    {
        string beside = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(beside)) return new CudaBackend(0, beside);
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return new CudaBackend(0, candidate);
        }
        throw new DirectoryNotFoundException("No CUDA PTX directory beside the test assembly or in the repository.");
    }

    private sealed class CudaFactAttribute : FactAttribute
    {
        public CudaFactAttribute()
        {
            if (!CudaContext.IsAvailable()) Skip = "SKIPPED: CUDA unavailable.";
        }
    }

    private sealed class CudaTheoryAttribute : TheoryAttribute
    {
        public CudaTheoryAttribute()
        {
            if (!CudaContext.IsAvailable()) Skip = "SKIPPED: CUDA unavailable.";
        }
    }
}
