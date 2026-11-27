using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Sparse;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The ERNIE-5.5 device ops against their CPU references on synthetic inputs: SiTU-GLU with any betas, the DSA top-k and
/// window merge, the block-diagonal <c>wo_a</c> linear, and the whole DSA indexer selection. Each test is reported as skipped
/// (never passed) when its device or its compiled kernels are absent.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class Ernie5OpsParityTests
{
    [CudaFact]
    public void Cuda_SituGlu_MatchesCpuReference_ForAnyBetas() => Run(Cuda, SituGluParity);

    [VulkanFact]
    public void Vulkan_SituGlu_MatchesCpuReference_ForAnyBetas() => Run(Vulkan, SituGluParity);

    [CudaFact]
    public void Cuda_SparseTopK_MatchesCpuExactly() => Run(Cuda, SparseTopKParity);

    [VulkanFact]
    public void Vulkan_SparseTopK_MatchesCpuExactly() => Run(Vulkan, SparseTopKParity);

    [CudaFact]
    public void Cuda_SparseTopK_ErnieShapeMatchesCpuExactly() => Run(Cuda, SparseTopKErnieShape);

    [VulkanFact]
    public void Vulkan_SparseTopK_ErnieShapeMatchesCpuExactly() => Run(Vulkan, SparseTopKErnieShape);

    [CudaFact]
    public void Cuda_MergeWindowTopK_MatchesCpuExactly() => Run(Cuda, MergeWindowTopKParity);

    [VulkanFact]
    public void Vulkan_MergeWindowTopK_MatchesCpuExactly() => Run(Vulkan, MergeWindowTopKParity);

    [CudaFact]
    public void Cuda_BlockDiagonalLinear_MatchesCpuReference() => Run(Cuda, BlockDiagonalLinearParity);

    [VulkanFact]
    public void Vulkan_BlockDiagonalLinear_MatchesCpuReference() => Run(Vulkan, BlockDiagonalLinearParity);

    [CudaFact]
    public void Cuda_DsaIndexerSelect_AgreesWithCpu() => Run(Cuda, DsaIndexerParity);

    [VulkanFact]
    public void Vulkan_DsaIndexerSelect_AgreesWithCpu() => Run(Vulkan, DsaIndexerParity);

    private static IBackend Cuda() => new CudaBackend(0, CudaPtxDirectory.Resolve());

    private static IBackend Vulkan() => new VulkanBackend();

    private static void Run(Func<IBackend> open, Action<IBackend> check)
    {
        using IBackend device = open();
        check(device);
    }

    private static void SituGluParity(IBackend device)
    {
        IBackend cpu = new CpuBackend();
        const int rows = 7, inner = 300;
        Random rng = new(3);
        // Wide values reach both soft caps: |g| well past beta and |u| past beta_lin.
        float[] x = Values(rng, rows * 2 * inner, 60f);
        foreach ((float beta, float linearBeta) in new[] { (SituDefaults.Beta, SituDefaults.LinearBeta), (2.5f, 10f), (7f, 40f) })
            foreach (GluGateHalf half in new[] { GluGateHalf.First, GluGateHalf.Second })
                foreach (DType dtype in new[] { DType.F32, DType.F16 })
                {
                    GluSpec spec = new(GluActivation.Situ, half, beta, linearBeta);
                    using Tensor f32 = Make(x, rows, 2 * inner);
                    using Tensor? f16 = dtype == DType.F16 ? Cast(f32, DType.F16) : null;
                    Tensor input = f16 ?? f32;
                    using Tensor expected = new(new TensorShape(rows, inner), dtype);
                    using Tensor actual = new(new TensorShape(rows, inner), dtype);
                    cpu.Glu(expected, input, spec);
                    device.Glu(actual, input, spec);
                    float tolerance = dtype == DType.F32 ? 2e-5f : 2e-3f;
                    AssertClose(ToF32(expected), ToF32(actual), tolerance, $"beta {beta}/{linearBeta} {half} {dtype.Name}");
                }
    }

    private static void SparseTopKParity(IBackend device)
    {
        CpuBackend cpu = new();
        Random rng = new(11);
        float[] specials = [float.NaN, float.NegativeInfinity, float.PositiveInfinity, -0f, 0f, 1f, 1f, -2f];
        (int Rows, int Cols, int TopK, int Offset)[] cases = [(5, 37, 9, 0), (3, 1000, 64, 7), (4, 600, 600, 0), (2, 10, 50, -3), (1, 513, 1, 0)];
        foreach ((int rows, int cols, int topK, int offset) in cases)
        {
            float[] scores = new float[rows * cols];
            for (int i = 0; i < scores.Length; i++)
                scores[i] = rng.Next(4) == 0 ? specials[rng.Next(specials.Length)] : rng.Next(-20, 21) * 0.25f;
            int[] limits = new int[rows];
            for (int r = 0; r < rows; r++) limits[r] = r == 0 ? cols : rng.Next(0, cols + 3);
            TopKMatches(cpu, device, scores, limits, rows, cols, topK, offset, $"{rows}x{cols} k{topK}");
        }
        // More rows than one dispatch's workgroups.
        const int manyRows = 65537;
        float[] tall = new float[manyRows * 4];
        for (int i = 0; i < tall.Length; i++) tall[i] = rng.Next(3);
        int[] tallLimits = Enumerable.Repeat(3, manyRows).ToArray();
        TopKMatches(cpu, device, tall, tallLimits, manyRows, 4, 2, 0, "65537 rows");
    }

    // The ERNIE indexer: top-2048 over a long context with heavy ties and a masked -inf tail per query.
    private static void SparseTopKErnieShape(IBackend device)
    {
        const int rows = 4, cols = 70000, topK = 2048;
        Random rng = new(19);
        float[] scores = new float[rows * cols];
        int[] limits = [cols, 50000, 1500, 69999];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                scores[r * cols + c] = c >= limits[r] ? float.NegativeInfinity : rng.Next(4000) * 0.01f;
        TopKMatches(new CpuBackend(), device, scores, limits, rows, cols, topK, 0, "ERNIE shape");
    }

    private static void TopKMatches(CpuBackend cpu, IBackend device, float[] scores, int[] limits, int rows, int cols, int topK, int offset,
        string label)
    {
        int k = Math.Min(topK, cols);
        using Tensor s = Make(scores, rows, cols), l = MakeInts(limits, rows);
        using Tensor expected = new(new TensorShape(rows, k), DType.I32), actual = new(new TensorShape(rows, k), DType.I32);
        cpu.SparseTopK(expected, s, l, topK, offset);
        device.SparseTopK(actual, s, l, topK, offset);
        Assert.True(expected.AsReadOnlySpan<int>().SequenceEqual(actual.AsReadOnlySpan<int>()), $"{label}: device top-k differs from the CPU");
    }

    private static void MergeWindowTopKParity(IBackend device)
    {
        CpuBackend cpu = new();
        Random rng = new(23);
        foreach ((int tokens, int cols, int window, int extra) in new[] { (9, 6, 4, 0), (40, 2048, 128, 3), (1, 1, 1, 0) })
        {
            int[] topk = new int[tokens * cols], positions = new int[tokens];
            for (int i = 0; i < topk.Length; i++) topk[i] = rng.Next(-1, 5000);
            for (int t = 0; t < tokens; t++) positions[t] = rng.Next(0, 3000);
            int width = cols + window + extra;
            using Tensor tk = MakeInts(topk, tokens, cols), p = MakeInts(positions, tokens);
            using Tensor expected = new(new TensorShape(tokens, width), DType.I32), actual = new(new TensorShape(tokens, width), DType.I32);
            cpu.MergeWindowTopK(expected, tk, p, window);
            device.MergeWindowTopK(actual, tk, p, window);
            Assert.True(expected.AsReadOnlySpan<int>().SequenceEqual(actual.AsReadOnlySpan<int>()), $"{tokens}x{cols} window {window}");
        }
    }

    // ERNIE's wo_a: 8 groups; the FP8 rank spans whole and partial 128-row blocks.
    private static void BlockDiagonalLinearParity(IBackend device)
    {
        CpuBackend cpu = new();
        Random rng = new(29);
        foreach ((int m, int groups, int r, int d, bool fp8) in new[] { (5, 8, 64, 96, false), (3, 8, 192, 160, true), (1, 1, 130, 300, true) })
        {
            float[] x = Values(rng, m * groups * d, 1f);
            using Tensor input = Make(x, m, groups * d);
            using Tensor weight = fp8
                ? MakeBytes(RandomCodes(rng, groups * r * d), groups * r, d)
                : Make(Values(rng, groups * r * d, 1f), groups * r, d);
            int gridRows = (groups * r + 127) / 128, gridCols = (d + 127) / 128;
            float[] scales = Values(rng, gridRows * gridCols, 1f).Select(v => 0.5f + MathF.Abs(v)).ToArray();
            using Tensor? scale = fp8 ? Make(scales, gridRows, gridCols) : null;
            using Tensor expected = new(new TensorShape(m, groups * r), DType.F32), actual = new(new TensorShape(m, groups * r), DType.F32);
            cpu.BlockDiagonalLinear(expected, input, weight, scale, groups);
            device.BlockDiagonalLinear(actual, input, weight, scale, groups);
            AssertClose(ToF32(expected), ToF32(actual), 2e-4f, $"{groups} groups {(fp8 ? "fp8" : "f32")}");
        }
    }

    // The whole indexer (projections, LayerNorm, rope, Hadamard, FP8x128 round trips, scores, top-k) on the device against the CPU.
    // The device projections round differently, which can move an FP8 code and so a near-tied pick; nearly every pick must agree.
    private static void DsaIndexerParity(IBackend device)
    {
        const int heads = 4, headDim = 128, ropeDim = 64, topK = 32, dim = 48, qLora = 40, window = 8, keys = 300;
        Random rng = new(37);
        DeepSeekV41RopeTable rope = DeepSeekV41RopeTable.Build(ropeDim, keys, 8e6, null);
        using Tensor wqB = Make(Values(rng, heads * headDim * qLora, 0.3f), heads * headDim, qLora);
        using Tensor wk = Make(Values(rng, headDim * dim, 0.3f), headDim, dim), wp = Make(Values(rng, heads * dim, 0.3f), heads, dim);
        using Tensor gamma = Make(Values(rng, headDim, 0.5f).Select(v => 1f + v).ToArray(), headDim);
        using Tensor beta = Make(Values(rng, headDim, 0.1f), headDim);
        DsaIndexerWeights weights = new() { WqB = wqB, Wk = wk, KeyNormWeight = gamma, KeyNormBias = beta, WeightsProj = wp };
        DsaIndexerSettings settings = new() { Heads = heads, HeadDim = headDim, RopeDim = ropeDim, TopK = topK };
        float[] x = Values(rng, keys * dim, 1f), qLatent = Values(rng, keys * qLora, 1f);
        int[] positions = [40, 41, 120, 255, 299];
        int[] limits = positions.Select(p => SparseWindowMerge.IndexerLimit(p, window)).ToArray();
        float[] xq = positions.SelectMany(p => x.AsSpan(p * dim, dim).ToArray()).ToArray();
        float[] ql = positions.SelectMany(p => qLatent.AsSpan(p * qLora, qLora).ToArray()).ToArray();

        int[] Select(IBackend backend, out int width)
        {
            using DsaIndexer indexer = new(backend, settings, weights, rope, dim, qLora);
            using Tensor keyStore = new(new TensorShape(keys, headDim), DType.F32);
            indexer.AppendKeys(x, keys, 0, keyStore);
            return indexer.Select(xq, ql, positions, limits, keyStore, out width);
        }

        int[] want = Select(new CpuBackend(), out int width);
        int[] got = Select(device, out int deviceWidth);
        Assert.Equal(width, deviceWidth);
        int agree = 0, total = 0;
        for (int t = 0; t < positions.Length; t++)
        {
            int valid = Math.Min(limits[t], topK);
            int[] row = got.AsSpan(t * width, width).ToArray();
            for (int i = 0; i < width; i++)
            {
                if (i >= valid) { Assert.Equal(-1, row[i]); continue; }
                Assert.InRange(row[i], 0, limits[t] - 1);
                if (i > 0) Assert.True(row[i] > row[i - 1], "rows must be strictly ascending");
            }
            agree += row.Take(valid).Intersect(want.AsSpan(t * width, valid).ToArray()).Count();
            total += valid;
        }
        Assert.True(agree >= total * 95 / 100, $"device kept {agree} of the {total} CPU picks");
    }

    private static Tensor Cast(Tensor f32, DType dtype)
    {
        Tensor t = new(f32.Shape, dtype);
        ReadOnlySpan<float> src = f32.AsReadOnlySpan<float>();
        Span<Half> dst = t.AsSpan<Half>();
        for (int i = 0; i < src.Length; i++) dst[i] = (Half)src[i];
        return t;
    }

    private static float[] ToF32(Tensor t)
    {
        if (t.DType == DType.F32) return t.AsReadOnlySpan<float>().ToArray();
        return t.AsReadOnlySpan<Half>().ToArray().Select(h => (float)h).ToArray();
    }

    private static void AssertClose(float[] want, float[] got, float tolerance, string label)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            Assert.True(MathF.Abs(want[i] - got[i]) <= tolerance * MathF.Max(1f, MathF.Abs(want[i])),
                $"{label} element {i}: cpu {want[i]} device {got[i]}");
    }

    private static byte[] RandomCodes(Random rng, int count)
    {
        byte[] codes = new byte[count];
        for (int i = 0; i < count; i++)
        {
            byte code = LatentCodec.EncodeE4M3((float)(rng.NextDouble() * 2 - 1));
            codes[i] = (code & 0x7F) == 0x7F ? (byte)0 : code;
        }
        return codes;
    }

    private static float[] Values(Random rng, int count, float amplitude)
    {
        float[] v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)((rng.NextDouble() * 2 - 1) * amplitude);
        return v;
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor MakeInts(int[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.I32);
        values.CopyTo(t.AsSpan<int>());
        return t;
    }

    private static Tensor MakeBytes(byte[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F8E4M3);
        values.CopyTo(t.AsSpan<byte>());
        return t;
    }
}
