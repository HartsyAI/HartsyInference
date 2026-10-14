using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>GPU-timed decode <c>gemv_quant</c> bandwidth at the dense <c>Linear</c> shapes of the decode perf set (llama
/// Q8_0, Qwen2.5 Q5_0, Qwen3 Q4_K and Q6_K, granite and OLMoE dense Q4_K and their F32 routers), plus two 9 to 16 row verify
/// shapes. Reports weight bytes read once per call over the timestamp time of a batch of calls, against the RTX 3060's
/// 360 GB/s peak. Timing only: correctness lives in <see cref="VulkanQuantGemvTests"/>. Opt-in:
/// <c>dotnet test --filter Category=GpuBenchmark</c>.</summary>
[Trait("Category", "GpuBenchmark")]
public sealed class VulkanGemvBenchmark(ITestOutputHelper output)
{
    private const int Iterations = 200;
    private const double PeakGbs = 360.0;

    public static IEnumerable<object[]> Shapes()
    {
        // type, N (output rows), K (inputs), M (activation rows)
        yield return new object[] { "Q8_0", 2048, 2048, 1 };      // llama q and o
        yield return new object[] { "Q8_0", 512, 2048, 1 };       // llama k and v
        yield return new object[] { "Q8_0", 8192, 2048, 1 };      // llama gate and up
        yield return new object[] { "Q8_0", 2048, 8192, 1 };      // llama down
        yield return new object[] { "Q8_0", 128256, 2048, 1 };    // llama lm_head (tied token embedding)
        yield return new object[] { "Q5_0", 4864, 896, 1 };       // Qwen2.5 gate and up
        yield return new object[] { "Q5_0", 896, 896, 1 };        // Qwen2.5 q and o
        yield return new object[] { "Q5_0", 128, 896, 1 };        // Qwen2.5 k and v
        yield return new object[] { "Q5_0", 896, 4864, 1 };       // Qwen2.5 down
        yield return new object[] { "Q4_K", 4096, 2560, 1 };      // Qwen3 q
        yield return new object[] { "Q4_K", 1024, 2560, 1 };      // Qwen3 k and v
        yield return new object[] { "Q4_K", 2560, 4096, 1 };      // Qwen3 o
        yield return new object[] { "Q4_K", 9728, 2560, 1 };      // Qwen3 gate and up
        yield return new object[] { "Q4_K", 9728, 2560, 16 };     // Qwen3 gate and up, 16-row cap
        yield return new object[] { "Q4_K", 2560, 9728, 1 };      // Qwen3 down (Q4_K share)
        yield return new object[] { "Q6_K", 2560, 9728, 1 };      // Qwen3 down (Q6_K share)
        yield return new object[] { "Q6_K", 151936, 2560, 1 };    // Qwen3 lm_head (tied token embedding)
        yield return new object[] { "Q4_K", 1024, 1024, 1 };      // granite q
        yield return new object[] { "Q4_K", 512, 1024, 1 };       // granite k and v
        yield return new object[] { "Q4_K", 2048, 2048, 1 };      // OLMoE q, k, v and o
        yield return new object[] { "Q6_K", 2048, 2048, 1 };      // OLMoE v (Q6_K share)
        yield return new object[] { "Q4_K", 2048, 2048, 9 };      // a 9-row speculative verify
        yield return new object[] { "Q4_0", 4096, 2048, 1 };      // parity format, no model here: split default
        yield return new object[] { "Q5_K", 4096, 2560, 1 };      // parity format, no model here: split default
        yield return new object[] { "F32", 32, 1024, 1 };         // granite MoE router
        yield return new object[] { "F32", 64, 2048, 1 };         // OLMoE MoE router
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void DecodeGemvGbPerSecond(string type, int n, int k, int m)
    {
        try
        {
            using VulkanInstance instance = new();
            if (instance.EnumeratePhysicalDevices().Length == 0) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        }
        catch (Exception e)
        {
            output.WriteLine($"SKIPPED: {e.Message}");
            return;
        }
        using VulkanBackend backend = VulkanTestDevice.Create();
        DType dtype = type == "F32" ? DType.F32 : VulkanQuantGemvTests.Named(type);
        using Tensor w = dtype == DType.F32
            ? Dsv41TestData.F32(Dsv41TestData.Random(n * k, seed: n + k, scale: 0.5f), n, k)
            : VulkanQuantGemvTests.RandomQuant(dtype, n, k, seed: n * 31 + k);
        using Tensor x = Dsv41TestData.F32(Dsv41TestData.Random(m * k, seed: k + m, scale: 1f), 1, m, k);
        using Tensor o = Dsv41TestData.EmptyF32(1, m, n);
        backend.PreloadWeights([w]);
        backend.Linear(o, x, w, null);   // warm the pipeline and the weight upload
        double totalMs = backend.MeasureGpuTimeMs(Iterations, () => backend.Linear(o, x, w, null));
        double ms = totalMs / Iterations;
        long bytes = dtype == DType.F32 ? (long)n * k * sizeof(float) : dtype.ComputeByteCount((long)n * k);
        double gbs = bytes / (ms * 1e6);
        output.WriteLine($"gemv {type} N={n} K={k} M={m}: {ms * 1000:F1} us per call, {bytes / 1e6:F2} MB, {gbs:F0} GB/s, {100 * gbs / PeakGbs:F0}% of {PeakGbs:F0}");
    }
}
