using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Vulkan LeakyRelu and AdaInstanceNorm1d against the CPU references on synthetic inputs.
/// Each fact is reported as skipped (never passed) when no Vulkan device or compiled kernel is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanElementwiseParityTests
{
    [VulkanFact]
    public void Vulkan_LeakyRelu_MatchesCpuReference()
    {
        using IBackend device = new VulkanBackend();
        LeakyReluParity(device);
    }

    [VulkanFact]
    public void Vulkan_AdaInstanceNorm1d_MatchesCpuReference()
    {
        using IBackend device = new VulkanBackend();
        AdaInstanceNormParity(device);
    }

    private static void LeakyReluParity(IBackend device)
    {
        CpuBackend cpu = new();
        foreach (float slope in new[] { 0f, 0.01f, 0.1f, 0.2f })
        {
            float[] x = Values(new Random(7), 3 * 64 * 33, 4f);
            x[0] = 0f;
            x[1] = -0f;
            using Tensor input = Make(x, 3, 64, 33);
            using Tensor expected = new(input.Shape, DType.F32);
            using Tensor actual = new(input.Shape, DType.F32);
            cpu.LeakyRelu(expected, input, slope);
            device.LeakyRelu(actual, input, slope);
            AssertClose(expected.AsReadOnlySpan<float>().ToArray(), actual.AsReadOnlySpan<float>().ToArray(), 1e-6f, $"slope {slope}");

            using Tensor inPlace = Make(x, 3, 64, 33);
            device.LeakyRelu(inPlace, inPlace, slope);
            AssertClose(expected.AsReadOnlySpan<float>().ToArray(), inPlace.AsReadOnlySpan<float>().ToArray(), 1e-6f, $"in-place slope {slope}");
        }
    }

    private static void AdaInstanceNormParity(IBackend device)
    {
        CpuBackend cpu = new();
        // T=1 is not a parity point: zero variance makes the CPU reference cancel x*scale - mean*scale.
        foreach ((int batch, int channels, int t, bool perBatch) in new[]
        {
            (1, 8, 2, false), (2, 16, 300, true), (1, 64, 3000, false), (3, 5, 97, true),
        })
        {
            float[] x = Values(new Random(batch * 131 + t), batch * channels * t, 3f);
            for (int i = 0; i < x.Length; i++) x[i] += 50f;
            int affineCount = perBatch ? batch * channels : channels;
            long[] affineShape = perBatch ? new long[] { batch, channels } : new long[] { channels };
            float[] gamma = Values(new Random(11), affineCount, 0.5f);
            float[] beta = Values(new Random(13), affineCount, 1f);

            using Tensor input = Make(x, batch, channels, t);
            using Tensor gammaT = Make(gamma, affineShape);
            using Tensor betaT = Make(beta, affineShape);
            using Tensor expected = new(input.Shape, DType.F32);
            using Tensor actual = new(input.Shape, DType.F32);
            cpu.AdaInstanceNorm1d(expected, input, gammaT, betaT, 1e-5f);
            device.AdaInstanceNorm1d(actual, input, gammaT, betaT, 1e-5f);
            AssertClose(expected.AsReadOnlySpan<float>().ToArray(), actual.AsReadOnlySpan<float>().ToArray(), 1e-4f,
                $"B={batch} C={channels} T={t} perBatch={perBatch}");
        }
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

    private static void AssertClose(float[] want, float[] got, float tolerance, string label)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            Assert.True(MathF.Abs(want[i] - got[i]) <= tolerance * MathF.Max(1f, MathF.Abs(want[i])),
                $"{label} element {i}: cpu {want[i]} device {got[i]}");
    }
}
