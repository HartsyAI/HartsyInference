using System.Diagnostics;
using System.Reflection;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The A/B that <c>numerics.vkTiledAttention</c> waits on: dense F32 attention on Vulkan through the scalar route
/// it takes by default and through the query-tiled F32 kernel the knob turns on, same call, same device buffers, the arms
/// alternated. It reports the time of each, the row each took, and how far each is from the host reference; it asserts
/// only that each arm took the row it is meant to measure and that the two outputs agree. It is a
/// <c>GpuBenchmark</c>, which the CPU lane and the GPU lane both leave out: run it alone on an idle card.
/// <para>Expectation written before the first run: the tiled kernel is at least 3 times faster at 1,024 by 1,024 tokens,
/// may be slower on the smallest shapes, and both outputs are within 2e-5 of the host reference.</para></summary>
[Trait("Category", "GpuBenchmark")]
public sealed class VulkanTiledAttentionAbTests(ITestOutputHelper output)
{
    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        return tensor;
    }

    private static Tensor OnDevice(VulkanBackend backend, Tensor host)
    {
        Tensor device = new(host.Shape, DType.F32);
        backend.Scale(device, host, 1f);
        backend.Sync();
        return device;
    }

    private static string Route() => typeof(VulkanBackend).GetProperty("LastAttentionRoute", BindingFlags.Static | BindingFlags.NonPublic)
        ?.GetValue(null) as string ?? "unknown";

    private static double Worst(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double worst = 0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    [VulkanTheory]
    [InlineData("self, 64 tokens", 1, 8, 8, 64, 64, 64, false)]
    [InlineData("self, 256 tokens", 1, 8, 8, 256, 256, 128, false)]
    [InlineData("self, 1,024 tokens, D 64", 1, 8, 8, 1024, 1024, 64, false)]
    [InlineData("self, 1,024 tokens, D 128", 1, 8, 8, 1024, 1024, 128, false)]
    [InlineData("self, 1,024 tokens, D 256", 1, 8, 8, 1024, 1024, 256, false)]
    [InlineData("self, 1,024 tokens, D 128, with a bias", 1, 8, 8, 1024, 1024, 128, true)]
    [InlineData("self, 1,024 tokens, D 128, 2 key heads", 1, 8, 2, 1024, 1024, 128, false)]
    [InlineData("self, 512 tokens, batch 2", 2, 8, 8, 512, 512, 128, false)]
    [InlineData("self, 4,096 tokens, 16 heads, D 64", 1, 16, 16, 4096, 4096, 64, false)]
    [InlineData("cross, 4,096 by 77", 1, 8, 8, 4096, 77, 128, false)]
    [InlineData("cross, 4,096 by 512", 1, 8, 8, 4096, 512, 128, false)]
    [InlineData("one row over 4,096 keys", 1, 8, 8, 1, 4096, 128, false)]
    public void ScalarAgainstTiled(string name, int batch, int heads, int kvHeads, int queryTokens, int keyTokens, int dim, bool withBias)
    {
        const int runs = 7;
        float scale = 1f / MathF.Sqrt(dim);
        using VulkanBackend backend = new();
        using Tensor qHost = Random(new TensorShape(batch, heads, queryTokens, dim), 1), kHost = Random(new TensorShape(batch, kvHeads, keyTokens, dim), 2);
        using Tensor vHost = Random(kHost.Shape, 3);
        using Tensor? biasHost = withBias ? Random(new TensorShape(queryTokens, keyTokens), 4) : null;
        using Tensor q = OnDevice(backend, qHost), k = OnDevice(backend, kHost), v = OnDevice(backend, vHost);
        using Tensor? bias = biasHost is null ? null : OnDevice(backend, biasHost);
        using Tensor scalarOut = new(qHost.Shape, DType.F32), tiledOut = new(qHost.Shape, DType.F32);

        string scalarRoute = "", tiledRoute = "";
        double[] scalarMs = new double[runs], tiledMs = new double[runs];
        for (int run = -2; run < runs; run++)
        {
            backend.EnableTiledAttention = false;
            long start = Stopwatch.GetTimestamp();
            backend.ScaledDotProductAttention(scalarOut, q, k, v, bias, scale);
            backend.Sync();
            if (run >= 0) scalarMs[run] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            scalarRoute = Route();

            backend.EnableTiledAttention = true;
            start = Stopwatch.GetTimestamp();
            backend.ScaledDotProductAttention(tiledOut, q, k, v, bias, scale);
            backend.Sync();
            if (run >= 0) tiledMs[run] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            tiledRoute = Route();
        }
        // A device that cannot run the tiled bucket sends the second arm down the default row again; that is not an A/B.
        Assert.True(tiledRoute == "A.2t", $"{name}: with the knob on the call took {tiledRoute}, not the tiled row A.2t");
        Assert.True(scalarRoute != "A.2t", $"{name}: with the knob off the call took the tiled row");
        Array.Sort(scalarMs);
        Array.Sort(tiledMs);
        double scalar = scalarMs[runs / 2], tiled = tiledMs[runs / 2];

        double between = Worst(scalarOut.AsReadOnlySpan<float>(), tiledOut.AsReadOnlySpan<float>());
        string reference = "not taken (too large for the host reference here)";
        if ((long)batch * heads * queryTokens * keyTokens <= 20_000_000)
        {
            using Tensor expected = new(qHost.Shape, DType.F32);
            AttentionCall call = AttentionCall.FromScaledDotProductAttention(expected, qHost, kHost, vHost, biasHost, scale, false);
            AttentionHostReference.Run(in call);
            reference = $"scalar {Worst(expected.AsReadOnlySpan<float>(), scalarOut.AsReadOnlySpan<float>()):G3}, "
                + $"tiled {Worst(expected.AsReadOnlySpan<float>(), tiledOut.AsReadOnlySpan<float>()):G3}";
        }
        output.WriteLine($"AB | {name} | B {batch}, H {heads}/{kvHeads}, {queryTokens} x {keyTokens}, D {dim} | {scalarRoute} | {scalar:F2} | "
            + $"{tiledRoute} | {tiled:F2} | {scalar / tiled:F2} | {between:G3} | {reference}");
        Assert.True(between <= 1e-4, $"{name}: the scalar and tiled outputs differ by {between}");
    }
}
