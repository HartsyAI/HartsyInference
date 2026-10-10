using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cuda.Tests.MoePrimitiveTestData;
using static HartsyInference.Cuda.Tests.Attention.LatentGpuTestData;

namespace HartsyInference.Cuda.Tests.Attention;

/// <summary>SparseLatentAttention on CUDA against the CPU reference over every encoding pairing, ring/main addressing,
/// skipped indices, sinks and an all-invalid row. Skips without CUDA.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed class SparseLatentAttentionTests(ITestOutputHelper output)
{
    private const float Tolerance = 1e-5f;

    private static float[] Run(IBackend be, int tokens, int heads, int dim, int k, int slots, LatentEncoding winEnc,
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
    [InlineData(LatentEncoding.Fp4E2M1E8M0x32, LatentEncoding.Fp4E2M1E4M3x16, 96, 5, 17, 64)]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.Fp4E2M1E4M3x16, 512, 128, 200, 300)]
    public void MatchesCpuReference(LatentEncoding winEnc, LatentEncoding mainEnc, int dim, int heads, int k, int mainRows)
    {
        if (!CudaContext.IsAvailable()) return;
        const int Tokens = 6, Slots = 128;
        float[] cpu = Run(new CpuBackend(), Tokens, heads, dim, k, Slots, winEnc, mainEnc, mainRows, 11);
        using CudaBackend cuda = new(0, PtxDir());
        float[] gpu = Run(cuda, Tokens, heads, dim, k, Slots, winEnc, mainEnc, mainRows, 11);
        float diff = MaxAbsDiff(cpu, gpu);
        output.WriteLine($"{winEnc}/{mainEnc} dim={dim} heads={heads} k={k}: max diff {diff:E2}");
        Assert.True(diff < Tolerance, $"max diff {diff}");
        Assert.All(gpu.Take(heads * dim), v => Assert.Equal(0f, v));           // all-invalid row
    }

    [Fact]
    public void EmptyMainSource_AndWindowOnlyAddressing_MatchCpu()
    {
        if (!CudaContext.IsAvailable()) return;
        float[] cpu = Run(new CpuBackend(), 4, 8, 128, 12, 16, LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.F32, 0, 5);
        using CudaBackend cuda = new(0, PtxDir());
        float[] gpu = Run(cuda, 4, 8, 128, 12, 16, LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.F32, 0, 5);
        Assert.True(MaxAbsDiff(cpu, gpu) < Tolerance);
    }

    [Fact]
    public void IndicesPastBothSources_AreSkippedLikeCpu()
    {
        if (!CudaContext.IsAvailable()) return;
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
            using CudaBackend cuda = new(0, PtxDir());
            cuda.SparseLatentAttention(gpuOut, q, window, main, idx, 4, sink, 0.125f);
            Assert.True(MaxAbsDiff(ReadF32(cpuOut), ReadF32(gpuOut)) < Tolerance);
        }
        finally
        {
            Dispose(window);
            Dispose(main);
        }
    }

    [Fact]
    public void RejectsInvalidOperandsAndOversizedK()
    {
        if (!CudaContext.IsAvailable()) return;
        const int Dim = 32;
        LatentSource window = MakeSource(LatentEncoding.F32, 2, Dim, 1);
        try
        {
            using CudaBackend cuda = new(0, PtxDir());
            using Tensor q = F32(new float[Dim], 1, 1, Dim), sink = F32(new float[1], 1), o = EmptyF32(1, 1, Dim);
            using Tensor idx = I32(new[] { 0 }, 1, 1);
            Assert.Throws<ArgumentException>(() => cuda.SparseLatentAttention(o, q, window, LatentSource.Empty, idx, 3, sink, 1f));
            int k = CudaKernels.LatentAttentionMaxK + 1;
            using Tensor bigIdx = I32(new int[k], 1, k);
            Assert.Throws<NotSupportedException>(() => cuda.SparseLatentAttention(o, q, window, LatentSource.Empty, bigIdx, 2, sink, 1f));
        }
        finally
        {
            Dispose(window);
        }
    }
}
