using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cuda.Tests.MoePrimitiveTestData;
using static HartsyInference.Cuda.Tests.Attention.LatentGpuTestData;

namespace HartsyInference.Cuda.Tests.Attention;

/// <summary>QuantizeLatentRows and ActQuantDequantInPlace on CUDA against the CPU reference: code and scale bytes and
/// dequantized floats must be identical, including zero rows, tiny and huge magnitudes, skipped and duplicated
/// destinations. Skips without CUDA.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed class LatentQuantTests(ITestOutputHelper output)
{
    // Rows with the awkward magnitudes: zero, subnormal-range, saturating, mixed signs.
    private static float[] AwkwardRows(int count, int dim, int seed)
    {
        float[] v = Random(count * dim, seed, 6f);
        for (int i = 0; i < dim; i++) v[i] = 0f;                                       // row 0: all zero
        if (count > 1) for (int i = 0; i < dim; i++) v[dim + i] *= 1e-8f;              // row 1: tiny
        if (count > 2) for (int i = 0; i < dim; i++) v[2 * dim + i] *= 4e4f;           // row 2: beyond the e4m3 range
        if (count > 3) v[3 * dim + 5] = -0f;
        return v;
    }

    private static (byte[] Codes, byte[] Scales, float[] F32Codes) Quantize(IBackend be, LatentEncoding enc, int destRows,
        int dim, int[] targets)
    {
        LatentSource dest = MakeSource(enc, destRows, dim, 51);          // pre-populated: untouched rows must survive
        try
        {
            using Tensor rows = F32(AwkwardRows(targets.Length, dim, 52), targets.Length, dim);
            using Tensor phys = I32(targets, targets.Length);
            be.QuantizeLatentRows(dest, rows, phys);
            return enc == LatentEncoding.F32
                ? (Array.Empty<byte>(), Array.Empty<byte>(), ReadF32(dest.Codes!))
                : (ReadU8(dest.Codes!), ReadU8(dest.Scales!), Array.Empty<float>());
        }
        finally
        {
            Dispose(dest);
        }
    }

    [Theory]
    [InlineData(LatentEncoding.F32, 64)]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, 512)]
    [InlineData(LatentEncoding.Fp4E2M1E4M3x16, 512)]
    [InlineData(LatentEncoding.Fp4E2M1E8M0x32, 128)]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, 32)]
    public void QuantizeRows_IsByteIdenticalToCpu(LatentEncoding enc, int dim)
    {
        if (!CudaContext.IsAvailable()) return;
        int[] targets = { 3, -1, 0, 9, 3, 7, -1, 5, 1, 3 };               // row 3 written three times: the last source row wins
        (byte[] Codes, byte[] Scales, float[] F32Codes) cpu = Quantize(new CpuBackend(), enc, 12, dim, targets);
        using CudaBackend cuda = new(0, PtxDir());
        (byte[] Codes, byte[] Scales, float[] F32Codes) gpu = Quantize(cuda, enc, 12, dim, targets);
        Assert.Equal(cpu.Codes, gpu.Codes);
        Assert.Equal(cpu.Scales, gpu.Scales);
        AssertBitEqual(cpu.F32Codes, gpu.F32Codes, "F32 cache");
        output.WriteLine($"{enc} dim={dim}: {cpu.Codes.Length + cpu.F32Codes.Length} code elements identical");
    }

    [Theory]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, 512)]
    [InlineData(LatentEncoding.Fp4E2M1E4M3x16, 128)]
    [InlineData(LatentEncoding.Fp4E2M1E8M0x32, 96)]
    public void ActQuantDequant_IsBitIdenticalToCpu(LatentEncoding enc, int dim)
    {
        if (!CudaContext.IsAvailable()) return;
        const int Rows = 40;
        float[] data = AwkwardRows(Rows, dim, 61);

        float[] Run(IBackend be)
        {
            using Tensor x = F32(data, Rows, dim);
            be.ActQuantDequantInPlace(x, enc);
            return ReadF32(x);
        }

        float[] cpu = Run(new CpuBackend());
        using CudaBackend cuda = new(0, PtxDir());
        AssertBitEqual(cpu, Run(cuda), $"act quant {enc}");
        Assert.NotEqual(data, cpu);
    }

    [Fact]
    public void ActQuantDequant_F32IsANoOp_AndABadLastDimensionThrows()
    {
        if (!CudaContext.IsAvailable()) return;
        using CudaBackend cuda = new(0, PtxDir());
        float[] data = Random(64, 71);
        using Tensor x = F32(data, 2, 32);
        cuda.ActQuantDequantInPlace(x, LatentEncoding.F32);
        Assert.Equal(data, ReadF32(x));
        using Tensor bad = F32(new float[48], 3, 16);
        Assert.Throws<ArgumentException>(() => cuda.ActQuantDequantInPlace(bad, LatentEncoding.Fp8E4M3Ue8m0x32));
    }

    [Fact]
    public void QuantizeRows_TwoBatchesAccumulateInTheSameCache()
    {
        if (!CudaContext.IsAvailable()) return;
        const int Dim = 128;
        using CudaBackend cuda = new(0, PtxDir());
        LatentSource gpuDest = Empty(LatentEncoding.Fp8E4M3Ue8m0x32, 6, Dim), cpuDest = Empty(LatentEncoding.Fp8E4M3Ue8m0x32, 6, Dim);
        try
        {
            foreach ((int[] targets, int seed) in new[] { (new[] { 0, 1, 2 }, 81), (new[] { 4, 1 }, 82) })
            {
                using Tensor rows = F32(Random(targets.Length * Dim, seed, 3f), targets.Length, Dim);
                using Tensor phys = I32(targets, targets.Length);
                cuda.QuantizeLatentRows(gpuDest, rows, phys);
                new CpuBackend().QuantizeLatentRows(cpuDest, rows, phys);
            }
            Assert.Equal(ReadU8(cpuDest.Codes!), ReadU8(gpuDest.Codes!));
            Assert.Equal(ReadU8(cpuDest.Scales!), ReadU8(gpuDest.Scales!));
        }
        finally
        {
            Dispose(gpuDest);
            Dispose(cpuDest);
        }
    }
}
