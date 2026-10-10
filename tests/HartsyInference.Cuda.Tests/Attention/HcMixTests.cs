using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cuda.Tests.MoePrimitiveTestData;
using static HartsyInference.Cuda.Tests.Attention.LatentGpuTestData;

namespace HartsyInference.Cuda.Tests.Attention;

/// <summary>HcSplitSinkhorn, HcPreMix and HcPostMix on CUDA against the CPU reference. The mixes are pure F32
/// mul/add chains built with round-to-nearest intrinsics, so they must agree bit for bit; the Sinkhorn split has one
/// exp per row and is held to 1e-6. Skips without CUDA.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed class HcMixTests(ITestOutputHelper output)
{
    private static (float[] Pre, float[] Post, float[] Comb) Split(IBackend be, int tokens, int hc, int iters)
    {
        int width = (2 + hc) * hc;
        using Tensor mixes = F32(Random(tokens * width, 31, 4f), tokens, width);
        using Tensor scale = F32(new[] { 0.7f, 1.3f, 0.9f }, 3);
        using Tensor bias = F32(Random(width, 32), width);
        using Tensor pre = EmptyF32(tokens, hc), post = EmptyF32(tokens, hc), comb = EmptyF32(tokens, hc, hc);
        be.HcSplitSinkhorn(pre, post, comb, mixes, scale, bias, hc, iters, 1e-6f);
        return (ReadF32(pre), ReadF32(post), ReadF32(comb));
    }

    [Theory]
    [InlineData(4, 20)]
    [InlineData(1, 20)]
    [InlineData(8, 20)]
    public void SplitSinkhorn_MatchesCpu(int hc, int iters)
    {
        if (!CudaContext.IsAvailable()) return;
        const int Tokens = 301;
        (float[] pre, float[] post, float[] comb) cpu = Split(new CpuBackend(), Tokens, hc, iters);
        using CudaBackend cuda = new(0, PtxDir());
        (float[] pre, float[] post, float[] comb) gpu = Split(cuda, Tokens, hc, iters);
        float dPre = MaxAbsDiff(cpu.pre, gpu.pre), dPost = MaxAbsDiff(cpu.post, gpu.post), dComb = MaxAbsDiff(cpu.comb, gpu.comb);
        output.WriteLine($"hc={hc} iters={iters}: pre {dPre:E2} post {dPost:E2} comb {dComb:E2}");
        Assert.True(dPre < 1e-6f && dPost < 1e-6f && dComb < 1e-6f, $"pre {dPre} post {dPost} comb {dComb}");
    }

    [Theory]
    [InlineData(4, 512)]
    [InlineData(8, 33)]
    public void PreMixAndPostMix_AreBitIdenticalToCpu(int hc, int dim)
    {
        if (!CudaContext.IsAvailable()) return;
        const int Tokens = 53;
        float[] xs = Random(Tokens * hc * dim, 41), pre = Random(Tokens * hc, 42), post = Random(Tokens * hc, 43);
        float[] comb = Random(Tokens * hc * hc, 44), x1 = Random(Tokens * dim, 45);

        (float[] Pre, float[] Post) Run(IBackend be)
        {
            using Tensor x = F32(xs, Tokens, hc, dim), p = F32(pre, Tokens, hc), po = F32(post, Tokens, hc);
            using Tensor c = F32(comb, Tokens, hc, hc), single = F32(x1, Tokens, dim);
            using Tensor collapsed = EmptyF32(Tokens, dim), expanded = EmptyF32(Tokens, hc, dim);
            be.HcPreMix(collapsed, x, p);
            be.HcPostMix(expanded, single, x, po, c);
            return (ReadF32(collapsed), ReadF32(expanded));
        }

        (float[] Pre, float[] Post) cpu = Run(new CpuBackend());
        using CudaBackend cuda = new(0, PtxDir());
        (float[] Pre, float[] Post) gpu = Run(cuda);
        AssertBitEqual(cpu.Pre, gpu.Pre, "hc pre mix");
        AssertBitEqual(cpu.Post, gpu.Post, "hc post mix");
    }

    [Fact]
    public void RejectsMoreThanEightStreams()
    {
        if (!CudaContext.IsAvailable()) return;
        using CudaBackend cuda = new(0, PtxDir());
        const int Hc = 9, Width = (2 + Hc) * Hc;
        using Tensor mixes = F32(new float[Width], 1, Width), scale = F32(new float[3], 3), bias = F32(new float[Width], Width);
        using Tensor pre = EmptyF32(1, Hc), post = EmptyF32(1, Hc), comb = EmptyF32(1, Hc, Hc);
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.HcSplitSinkhorn(pre, post, comb, mixes, scale, bias, Hc, 20, 1e-6f));
    }
}
