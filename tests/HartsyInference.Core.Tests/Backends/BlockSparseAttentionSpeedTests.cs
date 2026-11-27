using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Times CUDA block-sparse attention beside CUDA dense attention on the same operands, on the token grids of LongCat-Video's
/// 720p refinement (benchmarks/attention/reports/block-sparse-3060.md). It asserts nothing about speed.</summary>
[Trait("Category", "GpuBenchmark")]
public sealed class BlockSparseAttentionSpeedTests(ITestOutputHelper log)
{
    private const int HeadDim = 128, Timed = 3;

    [CudaTheory]
    [InlineData(12, 44, 80, 32)]
    [InlineData(48, 44, 80, 8)]
    public void BlockSparseBesideDense(int frames, int height, int width, int heads)
    {
        string beside = Path.Combine(AppContext.BaseDirectory, "Ptx");
        using CudaBackend cuda = new(0, beside);
        BlockSparseAttentionOptions options = new()
        {
            QueryGrid = new(frames, height, width), KeyGrid = new(frames, height, width), Sparsity = 0.9375,
        };
        TensorShape shape = new(1, heads, options.QueryGrid.Tokens, HeadDim);
        float scale = 1f / MathF.Sqrt(HeadDim);
        log.WriteLine($"grid {frames}x{height}x{width}, {options.QueryGrid.Tokens} tokens, {heads} heads, head dim {HeadDim}; "
            + $"{options.KeyBlocks} key blocks, {options.SelectedBlocks} read");
        foreach (DType type in (DType[])[DType.F32, DType.BF16])
        {
            using Tensor query = Resident(cuda, shape, type, 1), key = Resident(cuda, shape, type, 2), value = Resident(cuda, shape, type, 3);
            using Tensor output = new(shape, type);
            double sparse = Median(cuda, () => cuda.BlockSparseAttention(output, query, key, value, options));
            double dense = Median(cuda, () => cuda.ScaledDotProductAttention(output, query, key, value, null, scale));
            log.WriteLine($"{type}: block-sparse {sparse:F1} ms, dense {dense:F1} ms, ratio {dense / sparse:F2}");
        }
    }

    /// <summary>The median of <see cref="Timed"/> synchronized calls after one untimed call.</summary>
    private static double Median(CudaBackend cuda, Action call)
    {
        call();
        cuda.Sync();
        double[] times = new double[Timed];
        for (int i = 0; i < Timed; i++)
        {
            Stopwatch watch = Stopwatch.StartNew();
            call();
            cuda.Sync();
            times[i] = watch.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);
        return times[Timed / 2];
    }

    private static Tensor Resident(CudaBackend cuda, TensorShape shape, DType type, int seed)
    {
        using Tensor host = new(shape, DType.F32);
        Span<float> values = host.AsSpan<float>();
        Random random = new(seed);
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 4f - 2f;
        Tensor resident = new(shape, type);
        if (type == DType.F32) cuda.Scale(resident, host, 1f);
        else cuda.CastToBf16(resident, host);
        cuda.Sync();
        return resident;
    }

    private sealed class CudaTheoryAttribute : TheoryAttribute
    {
        public CudaTheoryAttribute()
        {
            if (!CudaContext.IsAvailable()) Skip = "SKIPPED: CUDA unavailable.";
        }
    }
}
