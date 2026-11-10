using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Device Hadamard, FP8x128 round trip and block-FP8 linear against the CPU references on synthetic inputs.
/// Each test is reported as skipped (never passed) when its device or its compiled kernels are absent.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SharedBlockOpsParityTests
{
    [CudaFact]
    public void Cuda_Hadamard_MatchesCpuReference()
    {
        using IBackend device = new CudaBackend(0, CudaPtxDirectory.Resolve());
        HadamardParity(device);
    }

    [CudaFact]
    public void Cuda_ActQuant_MatchesCpuReference_ForEveryLatentEncoding()
    {
        using IBackend device = new CudaBackend(0, CudaPtxDirectory.Resolve());
        ActQuantParity(device);
    }

    [CudaFact]
    public void Cuda_Fp8BlockLinear_MatchesCpuReference()
    {
        using IBackend device = new CudaBackend(0, CudaPtxDirectory.Resolve());
        Fp8BlockLinearParity(device);
    }

    [CudaFact]
    public void Cuda_Fp8BlockLinear_WideOutputMatchesCpuReference()
    {
        using IBackend device = new CudaBackend(0, CudaPtxDirectory.Resolve());
        Fp8BlockLinearWideParity(device);
    }

    [VulkanFact]
    public void Vulkan_Fp8BlockLinear_WideOutputMatchesCpuReference()
    {
        using IBackend device = new VulkanBackend();
        Fp8BlockLinearWideParity(device);
    }

    [VulkanFact]
    public void Vulkan_Hadamard_MatchesCpuReference()
    {
        using IBackend device = new VulkanBackend();
        HadamardParity(device);
    }

    [VulkanFact]
    public void Vulkan_ActQuant_MatchesCpuReference_ForEveryLatentEncoding()
    {
        using IBackend device = new VulkanBackend();
        ActQuantParity(device);
    }

    [VulkanFact]
    public void Vulkan_Fp8BlockLinear_MatchesCpuReference()
    {
        using IBackend device = new VulkanBackend();
        Fp8BlockLinearParity(device);
    }

    private static void HadamardParity(IBackend device)
    {
        CpuBackend cpu = new();
        foreach (int width in new[] { 1, 2, 16, 128, 256 })
        {
            const int rows = 5;
            float[] x = Values(new Random(width), rows * width, 2f);
            float scale = 1f / MathF.Sqrt(width);
            using Tensor input = Make(x, rows, width);
            using Tensor expected = new(input.Shape, DType.F32);
            using Tensor actual = new(input.Shape, DType.F32);
            cpu.Hadamard(expected, input, scale);
            device.Hadamard(actual, input, scale);
            AssertClose(expected.AsReadOnlySpan<float>().ToArray(), actual.AsReadOnlySpan<float>().ToArray(), 2e-5f, $"width {width}");

            using Tensor inPlace = Make(x, rows, width);
            device.Hadamard(inPlace, inPlace, scale);
            AssertClose(expected.AsReadOnlySpan<float>().ToArray(), inPlace.AsReadOnlySpan<float>().ToArray(), 2e-5f, $"in-place width {width}");
        }
    }

    private static void ActQuantParity(IBackend device)
    {
        CpuBackend cpu = new();
        foreach (LatentEncoding encoding in new[]
                     { LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.Fp4E2M1E4M3x16, LatentEncoding.Fp4E2M1E8M0x32, LatentEncoding.Fp8E4M3Ue8m0x128 })
        {
            int group = LatentEncodings.GroupSize(encoding);
            float[] x = Values(new Random(7 + group), 4 * 256, 3f);
            // Outliers pin each group's scale, and one tiny group checks the amax floor.
            for (int g = 0; g < x.Length / group; g += 3) x[g * group + 1] = g % 2 == 0 ? 900f : -40f;
            for (int i = 0; i < 256; i++) x[3 * 256 + i] = 1e-5f * (i % 7 - 3);
            using Tensor expected = Make(x, 4, 256);
            using Tensor actual = Make(x, 4, 256);
            cpu.ActQuantDequantInPlace(expected, encoding);
            device.ActQuantDequantInPlace(actual, encoding);
            // The device round trip is IEEE arithmetic in the same order as the CPU codec, so it must agree bit for bit.
            float[] want = expected.AsReadOnlySpan<float>().ToArray(), got = actual.AsReadOnlySpan<float>().ToArray();
            for (int i = 0; i < want.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(got[i]),
                    $"{encoding} element {i}: cpu {want[i]} device {got[i]}");
        }
    }

    private static void Fp8BlockLinearParity(IBackend device)
    {
        CpuBackend cpu = new();
        const int m = 3, n = 130, k = 300, block = 128;
        Random rng = new(31);
        byte[] codes = RandomCodes(rng, n * k);
        float[] scales = Values(rng, 2 * 3, 1f);
        for (int i = 0; i < scales.Length; i++) scales[i] = 0.5f + 0.5f * MathF.Abs(scales[i]);
        float[] x = Values(rng, m * k, 1f);
        float[] bias = Values(rng, n, 1f);
        float[]?[] variants = { null, bias };
        foreach (float[]? b in variants)
        {
            using Tensor input = Make(x, m, k);
            using Tensor weight = MakeBytes(codes, n, k);
            using Tensor scale = Make(scales, 2, 3);
            using Tensor? biasTensor = b is null ? null : Make(b, n);
            using Tensor expected = new(new TensorShape(m, n), DType.F32);
            using Tensor actual = new(new TensorShape(m, n), DType.F32);
            cpu.Fp8BlockLinear(expected, input, weight, scale, biasTensor, block, block);
            device.Fp8BlockLinear(actual, input, weight, scale, biasTensor, block, block);
            AssertClose(expected.AsReadOnlySpan<float>().ToArray(), actual.AsReadOnlySpan<float>().ToArray(), 2e-4f, b is null ? "no bias" : "bias");
        }
    }

    // N above 65535 forces the Vulkan dispatch to split the output columns across workgroup chunks.
    private static void Fp8BlockLinearWideParity(IBackend device)
    {
        CpuBackend cpu = new();
        const int m = 1, n = 65600, k = 128, block = 128;
        Random rng = new(41);
        byte[] codes = RandomCodes(rng, n * k);
        int gridRows = (n + block - 1) / block;
        float[] scales = new float[gridRows];
        Array.Fill(scales, 1.25f);
        float[] x = Values(rng, m * k, 1f);
        using Tensor input = Make(x, m, k);
        using Tensor weight = MakeBytes(codes, n, k);
        using Tensor scale = Make(scales, gridRows, 1);
        using Tensor expected = new(new TensorShape(m, n), DType.F32);
        using Tensor actual = new(new TensorShape(m, n), DType.F32);
        cpu.Fp8BlockLinear(expected, input, weight, scale, null, block, block);
        device.Fp8BlockLinear(actual, input, weight, scale, null, block, block);
        AssertClose(expected.AsReadOnlySpan<float>().ToArray(), actual.AsReadOnlySpan<float>().ToArray(), 2e-4f, "wide output");
    }

    private static void AssertClose(float[] want, float[] got, float tolerance, string label)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            Assert.True(MathF.Abs(want[i] - got[i]) <= tolerance * MathF.Max(1f, MathF.Abs(want[i])),
                $"{label} element {i}: cpu {want[i]} device {got[i]}");
    }

    private static byte[] RandomCodes(Random rng, int count)
    {
        byte[] codes = new byte[count];
        for (int i = 0; i < count; i++)
        {
            // Finite codes only: a NaN code would make every product NaN.
            byte code = LatentCodec.EncodeE4M3((float)(rng.NextDouble() * 2 - 1));
            codes[i] = (code & 0x7F) == 0x7F ? (byte)0 : code;
        }
        return codes;
    }

    private static float[] Values(Random rng, int count, float amplitude)
    {
        float[] v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)((rng.NextDouble() * 2 - 1) * amplitude);
        return v;
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor MakeBytes(byte[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F8E4M3);
        values.CopyTo(t.AsSpan<byte>());
        return t;
    }
}
