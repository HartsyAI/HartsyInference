using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wan-Animate-2 splices driving K/V onto the end of the generation keys, so it calls SDPA with a key
/// sequence LONGER than the query. If those tail keys are ignored, the driving video silently does nothing.</summary>
public unsafe class SdpaCrossLengthProbeTests
{
    private static Tensor Rand(TensorShape s, int seed, float scale = 1f)
    {
        Tensor t = new Tensor(s, DType.F32);
        Random r = new Random(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(r.NextDouble() * 2 - 1) * scale;
        return t;
    }

    [Fact]
    public void TailKeysBeyondTheQueryLength_AffectTheOutput()
    {
        const int heads = 2, sq = 6, sGen = 24, hw = 6, d = 16;
        const int skv = sGen + hw;
        using CpuBackend backend = new CpuBackend();
        using Tensor q = Rand(new TensorShape(1, heads, sq, d), 1);
        using Tensor k1 = Rand(new TensorShape(1, heads, skv, d), 2);
        using Tensor v1 = Rand(new TensorShape(1, heads, skv, d), 3);
        using Tensor k2 = new Tensor(k1.Shape, DType.F32);
        using Tensor v2 = new Tensor(v1.Shape, DType.F32);
        Buffer.MemoryCopy((void*)k1.DataPointer, (void*)k2.DataPointer, k1.ElementCount * 4, k1.ElementCount * 4);
        Buffer.MemoryCopy((void*)v1.DataPointer, (void*)v2.DataPointer, v1.ElementCount * 4, v1.ElementCount * 4);
        // Mutate ONLY the tail rows [sGen, skv) — the driving band — in every head.
        float* pk = (float*)k2.DataPointer; float* pv = (float*)v2.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int t = sGen; t < skv; t++)
                for (int e = 0; e < d; e++)
                {
                    long o = ((long)h * skv + t) * d + e;
                    pk[o] = -pk[o]; pv[o] = pv[o] + 5f;
                }

        using Tensor o1 = new Tensor(new TensorShape(1, heads, sq, d), DType.F32);
        using Tensor o2 = new Tensor(new TensorShape(1, heads, sq, d), DType.F32);
        float scale = 1f / MathF.Sqrt(d);
        backend.ScaledDotProductAttention(o1, q, k1, v1, null, scale, allowF16: false);
        backend.ScaledDotProductAttention(o2, q, k2, v2, null, scale, allowF16: false);

        double diff = 0;
        float* a = (float*)o1.DataPointer, b = (float*)o2.DataPointer;
        for (long i = 0; i < o1.ElementCount; i++) diff += Math.Abs(a[i] - b[i]);
        diff /= o1.ElementCount;
        Console.WriteLine($"[SDPA] sq={sq} skv={skv} mean|diff| from mutating tail keys = {diff:E3}");
        Assert.True(diff > 1e-3, $"SDPA IGNORES keys beyond the query length: mutating rows [{sGen},{skv}) changed the output by only {diff:E3}");
    }
}
