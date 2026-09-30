using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Vulkan.Tests.Dsv41TestData;
using static HartsyInference.Vulkan.Tests.LatentVulkanTestData;

namespace HartsyInference.Vulkan.Tests;

/// <summary>QuantizeLatentRows and ActQuantDequantInPlace on Vulkan against the CPU reference: code and scale bytes and
/// dequantized floats must be identical, including zero rows, tiny and huge magnitudes, skipped and duplicated
/// destinations. Skips without a Vulkan device. NVIDIA is plumbing evidence only.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanLatentQuantTests(ITestOutputHelper log)
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
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        log.WriteLine($"device: {vk.Vk.DeviceName}");
        int[] targets = { 3, -1, 0, 9, 3, 7, -1, 5, 1, 3 };               // row 3 written three times: the last source row wins
        (byte[] Codes, byte[] Scales, float[] F32Codes) cpu = Quantize(new CpuBackend(), enc, 12, dim, targets);
        (byte[] Codes, byte[] Scales, float[] F32Codes) gpu = Quantize(vk, enc, 12, dim, targets);
        Assert.Equal(cpu.Codes, gpu.Codes);
        Assert.Equal(cpu.Scales, gpu.Scales);
        AssertBitsEqual(cpu.F32Codes, gpu.F32Codes, "F32 cache");
        log.WriteLine($"{enc} dim={dim}: {cpu.Codes.Length + cpu.F32Codes.Length} code elements identical");
    }

    [Theory]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32, 512)]
    [InlineData(LatentEncoding.Fp4E2M1E4M3x16, 128)]
    [InlineData(LatentEncoding.Fp4E2M1E8M0x32, 96)]
    public void ActQuantDequant_IsBitIdenticalToCpu(LatentEncoding enc, int dim)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Rows = 40;
        float[] data = AwkwardRows(Rows, dim, 61);

        float[] Run(IBackend be)
        {
            using Tensor x = F32(data, Rows, dim);
            be.ActQuantDequantInPlace(x, enc);
            return ReadF32(x);
        }

        float[] cpu = Run(new CpuBackend());
        AssertBitsEqual(cpu, Run(vk), $"act quant {enc}");
        Assert.NotEqual(data, cpu);
    }

    [Fact]
    public void ActQuantDequant_F32IsANoOp_AndABadLastDimensionThrows()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        float[] data = Random(64, 71);
        using Tensor x = F32(data, 2, 32);
        vk.ActQuantDequantInPlace(x, LatentEncoding.F32);
        Assert.Equal(data, ReadF32(x));
        using Tensor bad = F32(new float[48], 3, 16);
        Assert.Throws<ArgumentException>(() => vk.ActQuantDequantInPlace(bad, LatentEncoding.Fp8E4M3Ue8m0x32));
    }

    [Fact]
    public void QuantizeRows_TwoBatchesAccumulateInTheSameCache()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Dim = 128;
        LatentSource gpuDest = Empty(LatentEncoding.Fp8E4M3Ue8m0x32, 6, Dim), cpuDest = Empty(LatentEncoding.Fp8E4M3Ue8m0x32, 6, Dim);
        try
        {
            foreach ((int[] targets, int seed) in new[] { (new[] { 0, 1, 2 }, 81), (new[] { 4, 1 }, 82) })
            {
                using Tensor rows = F32(Random(targets.Length * Dim, seed, 3f), targets.Length, Dim);
                using Tensor phys = I32(targets, targets.Length);
                vk.QuantizeLatentRows(gpuDest, rows, phys);
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

    [Theory]
    [InlineData(LatentEncoding.F32)]
    [InlineData(LatentEncoding.Fp8E4M3Ue8m0x32)]
    [InlineData(LatentEncoding.Fp4E2M1E4M3x16)]
    public void QuantizeRows_SkipsDestinationsPastTheLastRow_AndKeepsTheRest(LatentEncoding enc)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Dim = 64, DestRows = 4;
        int[] targets = { 1, 4, DestRows + 300, -1, 3 };
        LatentSource gpuDest = MakeSource(enc, DestRows, Dim, 91), cpuDest = MakeSource(enc, DestRows, Dim, 91);
        try
        {
            using Tensor rows = F32(AwkwardRows(targets.Length, Dim, 92), targets.Length, Dim);
            using Tensor phys = I32(targets, targets.Length);
            vk.QuantizeLatentRows(gpuDest, rows, phys);
            // The CPU reference rejects out-of-range destinations, so compare against it with those rows removed.
            int[] valid = { 1, -1, 3 };
            float[] all = ReadF32(rows);
            float[] kept = new float[valid.Length * Dim];
            int[] src = { 0, 3, 4 };
            for (int r = 0; r < src.Length; r++) Array.Copy(all, src[r] * Dim, kept, r * Dim, Dim);
            using Tensor keptRows = F32(kept, valid.Length, Dim);
            using Tensor keptPhys = I32(valid, valid.Length);
            new CpuBackend().QuantizeLatentRows(cpuDest, keptRows, keptPhys);
            if (enc == LatentEncoding.F32)
                AssertBitsEqual(ReadF32(cpuDest.Codes!), ReadF32(gpuDest.Codes!), $"skip {enc}");
            else
            {
                Assert.Equal(ReadU8(cpuDest.Codes!), ReadU8(gpuDest.Codes!));
                Assert.Equal(ReadU8(cpuDest.Scales!), ReadU8(gpuDest.Scales!));
            }
        }
        finally
        {
            Dispose(gpuDest);
            Dispose(cpuDest);
        }
    }

    [Fact]
    public void QuantizeRows_RejectsADestinationThatIsNotWholeWords()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        LatentSource dest = Empty(LatentEncoding.Fp8E4M3Ue8m0x32, 3, 32);   // 3 scale bytes
        try
        {
            using Tensor rows = F32(new float[32], 1, 32);
            using Tensor phys = I32(new[] { 0 }, 1);
            Assert.Throws<NotSupportedException>(() => vk.QuantizeLatentRows(dest, rows, phys));
        }
        finally
        {
            Dispose(dest);
        }
    }
}
