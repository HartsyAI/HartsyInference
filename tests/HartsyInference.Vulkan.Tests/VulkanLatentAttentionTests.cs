using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Vulkan.Tests.Dsv41TestData;
using static HartsyInference.Vulkan.Tests.LatentVulkanTestData;

namespace HartsyInference.Vulkan.Tests;

/// <summary>SparseLatentAttention and IndexerScores on Vulkan against the CPU reference over every encoding pairing,
/// ring/main addressing, skipped indices, sinks, an all-invalid row, compress-length and candidate masking, and byte
/// caches whose size is not a whole number of words. Skips without a Vulkan device. NVIDIA is plumbing evidence only.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanLatentAttentionTests(ITestOutputHelper log)
{
    private const float Tolerance = 1e-5f;

    private static float[] RunAttention(IBackend be, int tokens, int heads, int dim, int k, int slots, LatentEncoding winEnc,
        LatentEncoding mainEnc, int mainRows, int seed)
    {
        LatentSource window = MakeSource(winEnc, slots, dim, seed + 1), main = MakeSource(mainEnc, mainRows, dim, seed + 2);
        try
        {
            Random rng = new(seed);
            int total = slots + mainRows;
            int[] ids = new int[tokens * k];
            for (int i = 0; i < ids.Length; i++) ids[i] = rng.Next(-1, total);
            for (int j = 0; j < k; j++) ids[j] = -1;                       // token 0 attends to nothing
            using Tensor q = F32(Random(tokens * heads * dim, seed + 3), tokens, heads, dim);
            using Tensor sink = F32(Random(heads, seed + 4, 2f), heads);
            using Tensor idx = I32(ids, tokens, k);
            using Tensor o = EmptyF32(tokens, heads, dim);
            be.SparseLatentAttention(o, q, window, main, idx, slots, sink, 1f / MathF.Sqrt(dim));
            return ReadF32(o);
        }
        finally
        {
            Dispose(window);
            Dispose(main);
        }
    }

    [Theory]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.Fp4E2M1E4M3x16, 512, 64, 40, 128)]
    [InlineData(LatentEncoding.F32, LatentEncoding.F32, 64, 3, 9, 16)]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.Fp8E4M3Ue8m0x32, 128, 8, 33, 40)]
    [InlineData(LatentEncoding.Fp4E2M1E8M0x32, LatentEncoding.Fp4E2M1E4M3x16, 96, 5, 17, 64)]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.Fp4E2M1E4M3x16, 512, 128, 200, 300)]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.Fp8E4M3Ue8m0x32, 32, 2, 12, 9)]   // 9 scale bytes: padded upload
    public void Attention_MatchesCpuReference(LatentEncoding winEnc, LatentEncoding mainEnc, int dim, int heads, int k, int mainRows)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        log.WriteLine($"device: {vk.Vk.DeviceName}");
        const int Tokens = 6, Slots = 128;
        float[] cpu = RunAttention(new CpuBackend(), Tokens, heads, dim, k, Slots, winEnc, mainEnc, mainRows, 11);
        float[] gpu = RunAttention(vk, Tokens, heads, dim, k, Slots, winEnc, mainEnc, mainRows, 11);
        float diff = MaxAbsDiff(cpu, gpu);
        log.WriteLine($"{winEnc}/{mainEnc} dim={dim} heads={heads} k={k}: max diff {diff:E2}");
        Assert.True(diff < Tolerance, $"max diff {diff}");
        Assert.All(gpu.Take(heads * dim), v => Assert.Equal(0f, v));           // all-invalid row
    }

    [Fact]
    public void Attention_AtTheLargestSupportedK_MatchesCpu()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int K = 3800;
        float[] cpu = RunAttention(new CpuBackend(), 2, 2, 64, K, 128, LatentEncoding.F32, LatentEncoding.F32, 512, 21);
        float[] gpu = RunAttention(vk, 2, 2, 64, K, 128, LatentEncoding.F32, LatentEncoding.F32, 512, 21);
        Assert.True(MaxAbsDiff(cpu, gpu) < Tolerance);
    }

    [Fact]
    public void Attention_EmptyMainSource_MatchesCpu()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        float[] cpu = RunAttention(new CpuBackend(), 4, 8, 128, 12, 16, LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.F32, 0, 5);
        float[] gpu = RunAttention(vk, 4, 8, 128, 12, 16, LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.F32, 0, 5);
        Assert.True(MaxAbsDiff(cpu, gpu) < Tolerance);
    }

    [Fact]
    public void Attention_IndicesPastBothSources_AreSkippedLikeCpu()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Dim = 64;
        LatentSource window = MakeSource(LatentEncoding.Fp8E4M3Ue8m0x32, 4, Dim, 1);
        LatentSource main = MakeSource(LatentEncoding.Fp4E2M1E4M3x16, 3, Dim, 2);
        try
        {
            using Tensor q = F32(Random(2 * Dim, 3), 1, 2, Dim);
            using Tensor sink = F32(new[] { 0.5f, -1f }, 2);
            using Tensor idx = I32(new[] { 0, 6, 7, 99, -1, 2 }, 1, 6);
            using Tensor cpuOut = EmptyF32(1, 2, Dim), gpuOut = EmptyF32(1, 2, Dim);
            new CpuBackend().SparseLatentAttention(cpuOut, q, window, main, idx, 4, sink, 0.125f);
            vk.SparseLatentAttention(gpuOut, q, window, main, idx, 4, sink, 0.125f);
            Assert.True(MaxAbsDiff(ReadF32(cpuOut), ReadF32(gpuOut)) < Tolerance);
        }
        finally
        {
            Dispose(window);
            Dispose(main);
        }
    }

    [Fact]
    public void Attention_RejectsInvalidOperandsAndOversizedK()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Dim = 32;
        LatentSource window = MakeSource(LatentEncoding.F32, 2, Dim, 1);
        try
        {
            using Tensor q = F32(new float[Dim], 1, 1, Dim), sink = F32(new float[1], 1), o = EmptyF32(1, 1, Dim);
            using Tensor idx = I32(new[] { 0 }, 1, 1);
            Assert.Throws<ArgumentException>(() => vk.SparseLatentAttention(o, q, window, LatentSource.Empty, idx, 3, sink, 1f));
            const int K = 3801;
            using Tensor bigIdx = I32(new int[K], 1, K);
            Assert.Throws<NotSupportedException>(() => vk.SparseLatentAttention(o, q, window, LatentSource.Empty, bigIdx, 2, sink, 1f));
        }
        finally
        {
            Dispose(window);
        }
    }

    private static float[] RunIndexer(IBackend be, LatentEncoding enc, int tokens, int heads, int dim, int keysRows, bool candidates)
    {
        LatentSource keys = MakeSource(enc, keysRows, dim, 21, 1f);
        try
        {
            // Visible key count grows across the tokens, so early queries see few keys and the last sees all of them.
            int[] lens = Enumerable.Range(0, tokens).Select(t => Math.Min(1 + t * keysRows / tokens, keysRows)).ToArray();
            byte[] flags = Enumerable.Range(0, tokens * keysRows).Select(i => (byte)(i % 5 == 0 ? 0 : 1)).ToArray();
            using Tensor q = F32(Random(tokens * heads * dim, 22), tokens, heads, dim);
            using Tensor w = F32(Random(tokens * heads, 23), tokens, heads);
            using Tensor len = I32(lens, tokens);
            using Tensor? cand = candidates ? U8(flags, tokens, keysRows) : null;
            using Tensor scores = EmptyF32(tokens, keysRows);
            be.IndexerScores(scores, q, keys, w, len, cand, 0.25f);
            return ReadF32(scores);
        }
        finally
        {
            Dispose(keys);
        }
    }

    [Theory]
    [InlineData(LatentEncoding.Fp4E2M1E8M0x32, 128, 64, 200, false, 48)]
    [InlineData(LatentEncoding.Fp4E2M1E8M0x32, 128, 64, 200, true, 48)]
    [InlineData(LatentEncoding.F32, 64, 4, 17, true, 48)]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, 512, 8, 33, false, 48)]
    [InlineData(LatentEncoding.Fp4E2M1E4M3x16, 96, 3, 9, true, 48)]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, 32, 2, 9, true, 5)]   // 9 scale bytes and 45 candidate flags: padded uploads
    public void Indexer_MatchesCpuReference(LatentEncoding enc, int dim, int heads, int keys, bool candidates, int tokens)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        log.WriteLine($"device: {vk.Vk.DeviceName}");
        float[] cpu = RunIndexer(new CpuBackend(), enc, tokens, heads, dim, keys, candidates);
        float[] gpu = RunIndexer(vk, enc, tokens, heads, dim, keys, candidates);
        Assert.Equal(cpu.Length, gpu.Length);
        int masked = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.Equal(float.IsNegativeInfinity(cpu[i]), float.IsNegativeInfinity(gpu[i]));
            if (float.IsNegativeInfinity(cpu[i])) { masked++; continue; }
            Assert.True(MathF.Abs(cpu[i] - gpu[i]) < Tolerance, $"score {i}: cpu {cpu[i]:R} vk {gpu[i]:R}");
        }
        log.WriteLine($"{enc} dim={dim} heads={heads} keys={keys}: {masked}/{cpu.Length} masked");
        Assert.True(masked > 0 && masked < cpu.Length);
    }
}
