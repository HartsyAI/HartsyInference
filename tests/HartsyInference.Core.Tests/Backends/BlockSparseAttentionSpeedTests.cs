using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Times a device's block-sparse attention, and on CUDA its dense attention on the same operands, on token grids of
/// LongCat-Video's 720p refinement (benchmarks/attention/reports/block-sparse-3060.md). It asserts nothing about speed.</summary>
[Trait("Category", "GpuBenchmark")]
public sealed class BlockSparseAttentionSpeedTests(ITestOutputHelper log)
{
    private const int HeadDim = 128, Timed = 3;

    [CudaTheory]
    [InlineData(12, 44, 80, 32)]
    [InlineData(48, 44, 80, 8)]
    public void BlockSparseBesideDense(int frames, int height, int width, int heads)
    {
        using CudaBackend cuda = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
        Record(cuda, frames, height, width, heads, dense: true);
    }

    /// <summary>No dense attention here: on a Windows RTX 3060 it has lost the device on these grids. On a card driving a
    /// display, run the cases one at a time, smallest first.</summary>
    [VulkanTheory]
    [InlineData(4, 44, 80, 1)]
    [InlineData(4, 44, 80, 4)]
    [InlineData(12, 44, 80, 8)]
    [InlineData(12, 44, 80, 32)]
    public void VulkanBlockSparse(int frames, int height, int width, int heads)
    {
        using VulkanBackend vulkan = new();
        Record(vulkan, frames, height, width, heads, dense: false);
    }

    private void Record(IBackend backend, int frames, int height, int width, int heads, bool dense)
    {
        BlockSparseAttentionOptions options = new()
        {
            QueryGrid = new(frames, height, width), KeyGrid = new(frames, height, width), Sparsity = 0.9375,
        };
        TensorShape shape = new(1, heads, options.QueryGrid.Tokens, HeadDim);
        float scale = 1f / MathF.Sqrt(HeadDim);
        log.WriteLine($"{backend.Device} grid {frames}x{height}x{width}, {options.QueryGrid.Tokens} tokens, {heads} heads, head dim {HeadDim}; "
            + $"{options.KeyBlocks} key blocks, {options.SelectedBlocks} read");
        foreach (DType type in (DType[])[DType.F32, DType.BF16])
        {
            using Tensor query = Resident(backend, shape, type, 1), key = Resident(backend, shape, type, 2);
            using Tensor value = Resident(backend, shape, type, 3);
            using Tensor output = new(shape, type);
            double sparse = Median(backend, () => backend.BlockSparseAttention(output, query, key, value, options));
            log.WriteLine($"{type}: block-sparse {sparse:F1} ms");
            if (!dense) continue;
            try
            {
                double full = Median(backend, () => backend.ScaledDotProductAttention(output, query, key, value, null, scale));
                log.WriteLine($"{type}: dense {full:F1} ms, ratio {full / sparse:F2}");
            }
            catch (NotSupportedException refusal)
            {
                log.WriteLine($"{type}: dense refused: {refusal.Message}");
            }
        }
    }

    /// <summary>The median of <see cref="Timed"/> synchronized calls after one untimed call.</summary>
    private static double Median(IBackend backend, Action call)
    {
        call();
        backend.Sync();
        double[] times = new double[Timed];
        for (int i = 0; i < Timed; i++)
        {
            Stopwatch watch = Stopwatch.StartNew();
            call();
            backend.Sync();
            times[i] = watch.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);
        return times[Timed / 2];
    }

    private static Tensor Resident(IBackend backend, TensorShape shape, DType type, int seed)
    {
        using Tensor host = new(shape, DType.F32);
        Span<float> values = host.AsSpan<float>();
        Random random = new(seed);
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 4f - 2f;
        Tensor resident = new(shape, type);
        if (type == DType.F32) backend.Scale(resident, host, 1f);
        else backend.CastToBf16(resident, host);
        backend.Sync();
        return resident;
    }

    private sealed class CudaTheoryAttribute : TheoryAttribute
    {
        public CudaTheoryAttribute() => Skip = BlockSparseAttentionDeviceChecks.CudaSkip;
    }

    private sealed class VulkanTheoryAttribute : TheoryAttribute
    {
        public VulkanTheoryAttribute() => Skip = BlockSparseAttentionDeviceChecks.VulkanSkip;
    }
}
