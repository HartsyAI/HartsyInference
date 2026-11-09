using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests;

/// <summary>Time of one single-query, bidirectional GQA attention (MOSS-VL's cross-attention decode shape: 32 query heads, 8 KV heads,
/// head 128) as the key count grows. Opt-in (<c>HARTSY_MOSS_VL_PROFILE=1</c>); prints only.</summary>
[Trait("Category", "GpuBenchmark")]
public sealed class CrossAttentionDecodeBenchmark
{
    [Fact]
    public void SingleQueryAttentionScalesWithKeys()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_MOSS_VL_PROFILE") != "1") return;
        using VulkanBackend backend = new();
        AttentionLayer attn = new(new AttentionLayerConfig { Heads = 32, KvHeads = 8, HeadDim = 128, ComputeDType = DType.F32, Scale = 1f / MathF.Sqrt(128) });
        Random rng = new(1);
        foreach (int keys in (int[])[64, 1024, 2048, 4096])
        {
            // Resident like the session's cross-attention cache: allocated on the device, filled once.
            using Tensor kHost = Random(rng, 1, 8, keys, 128);
            using Tensor vHost = Random(rng, 1, 8, keys, 128);
            using Tensor k = new(kHost.Shape, DType.F32);
            using Tensor v = new(vHost.Shape, DType.F32);
            IBackend device = backend;
            device.ResidentAllocateKv(k);
            device.ResidentAllocateKv(v);
            device.KvCacheAppend(k, kHost, 0);
            device.KvCacheAppend(v, vHost, 0);
            const int Reps = 100;
            for (int i = -5; i < Reps; i++)
            {
                if (i == 0) backend.Sync();
                using Tensor q = Random(rng, 1, 32, 1, 128);
                using Tensor o = attn.AttendOverKvHeadMajor(backend, q, k, v, null, 0, 0,
                    new KvAttentionCall { KvLength = keys, Bidirectional = true }, OperandOwnership.None);
                if (i == 0) StartClock();
            }
            backend.Sync();
            string? route = typeof(VulkanBackend).GetProperty("LastAttentionRoute",
                System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)?.GetValue(null) as string;
            Console.WriteLine($"{keys,5} keys: {_clock.Elapsed.TotalMilliseconds * 1000 / Reps:F0} µs per attention (incl. a query upload), route {route}");
        }
    }

    private Stopwatch _clock = new();

    private void StartClock() => _clock = Stopwatch.StartNew();

    private static Tensor Random(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}
