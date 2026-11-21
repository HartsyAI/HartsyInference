using System.Diagnostics;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A descriptive timing of INT8 attention with u8 probabilities on Vulkan beside the backend's F32 attention
/// of the same shape. It asserts nothing about speed and is a <c>GpuBenchmark</c>, which the CPU lane and the GPU lane
/// both leave out: run it alone on an idle card.
/// <para>The INT8 timings are of the bit-exact passes as they stand, with every operand already on the device. The F32
/// comparators are the flash route and the dense route for the same unmasked call, the dense route with F16 allowed
/// (the other lossy path, and the one a speed claim has to beat), and the causal flash route for context (INT8 here
/// is unmasked only).</para>
/// <para>Expectation written before the first run: the INT8 path is slower, between 2 and 30 times, because its passes
/// are written to reproduce the reference bit for bit (about 30 operations for each fused multiply-add, one invocation a
/// row and tile or a row and channel), not for speed.</para></summary>
[Trait("Category", "GpuBenchmark")]
public sealed class Int8AttentionU8VulkanTimingTests(ITestOutputHelper output)
{
    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        return tensor;
    }

    /// <summary>A copy of <paramref name="host"/> made by the backend, so it lives on the device.</summary>
    private static Tensor OnDevice(VulkanBackend backend, Tensor host, TensorShape shape)
    {
        using Tensor shaped = new(shape, DType.F32);
        host.AsReadOnlySpan<float>().CopyTo(shaped.AsSpan<float>());
        Tensor device = new(shape, DType.F32);
        backend.Scale(device, shaped, 1f);
        backend.Sync();
        return device;
    }

    private static double MedianMs(VulkanBackend backend, int runs, Action body)
    {
        for (int i = 0; i < 2; i++)
        {
            body();
            backend.Sync();
        }
        double[] ms = new double[runs];
        for (int i = 0; i < runs; i++)
        {
            long start = Stopwatch.GetTimestamp();
            body();
            backend.Sync();
            ms[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        Array.Sort(ms);
        return ms[runs / 2];
    }

    [VulkanTheory]
    [InlineData("prefill", 8, 1024, 1024, 128)]
    [InlineData("prefill, D 64", 8, 1024, 1024, 64)]
    [InlineData("decode", 8, 1, 8192, 128)]
    public void TimesBesideF32(string name, int heads, int rows, int keys, int dim)
    {
        const int tile = 128, runs = 7;
        float scale = 1f / MathF.Sqrt(dim);
        using VulkanBackend backend = new();
        using Tensor qHost = Random(new TensorShape(1, heads, rows, dim), 1), kHost = Random(new TensorShape(1, heads, keys, dim), 2);
        using Tensor vHost = Random(kHost.Shape, 3), o4 = new(qHost.Shape, DType.F32), scratch = new(qHost.Shape, DType.F32);
        // Every operand is made on the device first, so no timing below includes an upload.
        using Tensor q4 = OnDevice(backend, qHost, qHost.Shape), k4 = OnDevice(backend, kHost, kHost.Shape), v4 = OnDevice(backend, vHost, vHost.Shape);
        double f32 = MedianMs(backend, runs, () => backend.FlashAttention(o4, q4, k4, v4, keys, 1, false, 0, scale));
        string flashRoute = VulkanAttentionRoute();
        double dense = MedianMs(backend, runs, () => backend.ScaledDotProductAttention(scratch, q4, k4, v4, null, scale));
        string denseRoute = VulkanAttentionRoute();
        using Tensor f16Out = new(qHost.Shape, DType.F32);
        double denseF16 = MedianMs(backend, runs, () => backend.ScaledDotProductAttention(f16Out, q4, k4, v4, null, scale, allowF16: true));
        string denseF16Route = VulkanAttentionRoute();
        double causal = rows > 1 ? MedianMs(backend, runs, () => backend.FlashAttention(scratch, q4, k4, v4, keys, 1, true, 0, scale)) : double.NaN;
        string causalRoute = rows > 1 ? VulkanAttentionRoute() : "none";

        using Tensor q = OnDevice(backend, qHost, new TensorShape(heads, rows, dim)), k = OnDevice(backend, kHost, new TensorShape(heads, keys, dim));
        using Tensor v = OnDevice(backend, vHost, k.Shape);
        int tiles = (keys + tile - 1) / tile;
        Int8Rotation rotation = dim == 64 ? Int8Rotation.H64 : Int8Rotation.SignedH128;
        using Tensor keyBytes = new(new TensorShape(heads * keys, dim), DType.I8), keyScales = new(new TensorShape(heads * keys), DType.F32);
        using Tensor means = new(new TensorShape(heads, dim), DType.F32);
        using Tensor valueBytes = new(keyBytes.Shape, DType.I8), valueScales = new(new TensorShape(heads, tiles, dim), DType.F32);
        using Tensor queryBytes = new(new TensorShape(heads * rows, dim), DType.I8), queryScales = new(new TensorShape(heads * rows), DType.F32);
        using Tensor dots = new(new TensorShape(heads, rows), DType.F32);
        using Tensor result = new(new TensorShape(heads, rows, dim), DType.F32);
        void Quantize()
        {
            backend.QuantizeInt8AttentionKeys(keyBytes, keyScales, means, k, rotation);
            backend.QuantizeInt8AttentionValues(valueBytes, valueScales, v, tile);
            backend.QuantizeInt8AttentionQueries(queryBytes, queryScales, dots, q, means, rotation);
        }
        double quantize = MedianMs(backend, runs, Quantize);
        long cells = (long)heads * rows * tiles;
        using Tensor tileMax = new(new TensorShape(cells), DType.F32), tileSums = new(new TensorShape(cells), DType.U32);
        using Tensor probabilities = new(new TensorShape(cells * tile), DType.U8);
        double tilePass = MedianMs(backend, runs, () => backend.Int8AttentionU8Tiles(tileMax, tileSums, probabilities, queryBytes,
            queryScales, dots, keyBytes, keyScales, heads, heads, tile, scale));
        double drain = MedianMs(backend, runs, () => backend.Int8AttentionU8Drain(result, tileMax, tileSums, probabilities, valueBytes,
            valueScales, heads, heads, tile, true));

        // That both paths attended the same call: INT8 is an approximation, so this is a loose sanity bound only.
        ReadOnlySpan<float> a = o4.AsReadOnlySpan<float>(), b = result.AsReadOnlySpan<float>();
        double worst = 0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        double total = quantize + tilePass + drain;
        ReadOnlySpan<float> h = f16Out.AsReadOnlySpan<float>();
        double worstF16 = 0;
        for (int i = 0; i < a.Length; i++) worstF16 = Math.Max(worstF16, Math.Abs(a[i] - h[i]));
        output.WriteLine($"{name}: heads {heads}, rows {rows}, keys {keys}, D {dim}; F32 flash {f32:F1} ms ({flashRoute}), "
            + $"F32 dense {dense:F1} ms ({denseRoute}), dense with F16 allowed {denseF16:F1} ms ({denseF16Route}, largest |F16 - F32| {worstF16:G3}), "
            + $"F32 causal flash {causal:F1} ms ({causalRoute}); INT8 quantize {quantize:F1} ms, "
            + $"tile pass {tilePass:F1} ms, drain {drain:F1} ms, total {total:F1} ms, attention alone {tilePass + drain:F1} ms; "
            + $"total over the faster unmasked F32 {total / Math.Min(f32, dense):F2}; largest |INT8 - F32| {worst:G3}");
        Assert.True(worst < 0.25, $"INT8 and F32 outputs differ by {worst}");
    }

    private static string VulkanAttentionRoute() =>
        typeof(VulkanBackend).GetProperty("LastAttentionRoute", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)
            ?.GetValue(null) as string ?? "unknown";
}
