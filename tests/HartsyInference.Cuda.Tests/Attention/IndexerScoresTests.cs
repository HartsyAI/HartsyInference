using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cuda.Tests.MoePrimitiveTestData;
using static HartsyInference.Cuda.Tests.Attention.LatentGpuTestData;

namespace HartsyInference.Cuda.Tests.Attention;

/// <summary>IndexerScores on CUDA against the CPU reference, including mid-group compress-length masking and candidate
/// masks. Masked entries must be -infinity on both. Skips without CUDA.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class IndexerScoresTests(ITestOutputHelper output)
{
    private const float Tolerance = 1e-5f;

    // The CPU reference sums each dot serially while the kernel reduces across a warp, so the two differ by F32
    // summation-order noise that grows with the dot's magnitude (about 1e-5 once a 512-wide dot reaches ~20).
    // Unit-scale keys keep that noise well under the 1e-5 bound.
    private const float KeyAmplitude = 1f;

    private static float[] Run(IBackend be, LatentEncoding enc, int tokens, int heads, int dim, int keysRows, bool candidates)
    {
        LatentSource keys = MakeSource(enc, keysRows, dim, 21, KeyAmplitude);
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

    private static Tensor U8(byte[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.U8);
        data.AsSpan().CopyTo(new Span<byte>((byte*)t.DataPointer, data.Length));
        return t;
    }

    [Theory]
    [InlineData(LatentEncoding.Fp4E2M1E8M0x32, 128, 64, 200, false)]
    [InlineData(LatentEncoding.Fp4E2M1E8M0x32, 128, 64, 200, true)]
    [InlineData(LatentEncoding.F32, 64, 4, 17, true)]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, 512, 8, 33, false)]
    [InlineData(LatentEncoding.Fp4E2M1E4M3x16, 96, 3, 9, true)]
    public void MatchesCpuReference(LatentEncoding enc, int dim, int heads, int keys, bool candidates)
    {
        if (!CudaContext.IsAvailable()) return;
        const int Tokens = 48;
        float[] cpu = Run(new CpuBackend(), enc, Tokens, heads, dim, keys, candidates);
        using CudaBackend cuda = new(0, PtxDir());
        float[] gpu = Run(cuda, enc, Tokens, heads, dim, keys, candidates);
        Assert.Equal(cpu.Length, gpu.Length);
        int masked = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.Equal(float.IsNegativeInfinity(cpu[i]), float.IsNegativeInfinity(gpu[i]));
            if (float.IsNegativeInfinity(cpu[i])) { masked++; continue; }
            Assert.True(MathF.Abs(cpu[i] - gpu[i]) < Tolerance, $"score {i}: cpu {cpu[i]:R} cuda {gpu[i]:R}");
        }
        output.WriteLine($"{enc} dim={dim} heads={heads} keys={keys}: {masked}/{cpu.Length} masked");
        Assert.True(masked > 0 && masked < cpu.Length);
    }

    [Fact]
    public void RejectsDimBeyondTheKernelLimit()
    {
        if (!CudaContext.IsAvailable()) return;
        const int Dim = 544;
        LatentSource keys = MakeSource(LatentEncoding.F32, 2, Dim, 1);
        try
        {
            using CudaBackend cuda = new(0, PtxDir());
            using Tensor q = F32(new float[Dim], 1, 1, Dim), w = F32(new float[1], 1, 1), len = I32(new[] { 2 }, 1);
            using Tensor scores = EmptyF32(1, 2);
            Assert.Throws<NotSupportedException>(() => cuda.IndexerScores(scores, q, keys, w, len, null, 1f));
        }
        finally
        {
            Dispose(keys);
        }
    }
}
