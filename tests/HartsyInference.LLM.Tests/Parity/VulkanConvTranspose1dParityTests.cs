using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Vulkan ConvTranspose1d against the CPU reference: dense, depthwise (Kokoro's pool shape), and grouped with
/// unequal per-group widths. Each fact is reported as skipped (never passed) when no Vulkan device or compiled kernel is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanConvTranspose1dParityTests
{
    private static readonly Case[] DenseCases =
    {
        new(2, 6, 4, 1, 9, 5, 2, 1, 1, 1, true),
        new(1, 4, 8, 1, 7, 3, 3, 0, 2, 2, false),
    };

    // Kokoro's AdainResBlk1d pool: k=3, stride 2, padding 1, output_padding 1, so padRight = 0.
    private static readonly Case[] DepthwiseCases =
    {
        new(2, 512, 512, 512, 37, 3, 2, 1, 0, 1, true),
        new(1, 8, 8, 8, 5, 4, 3, 2, 1, 2, false),
    };

    // inPerGroup and outPerGroup differ and are both above 1, so a wrong weight-index order cannot pass.
    private static readonly Case[] GroupedCases =
    {
        new(2, 6, 12, 3, 8, 4, 2, 1, 1, 1, true),
        new(1, 8, 6, 2, 6, 3, 3, 0, 0, 2, false),
    };

    [VulkanFact]
    public void Vulkan_ConvTranspose1d_Dense_MatchesCpuReference()
    {
        using IBackend device = new VulkanBackend();
        foreach (Case c in DenseCases) AssertCase(device, c);
    }

    [VulkanFact]
    public void Vulkan_ConvTranspose1d_Depthwise_MatchesCpuReference()
    {
        using IBackend device = new VulkanBackend();
        foreach (Case c in DepthwiseCases) AssertCase(device, c);
    }

    [VulkanFact]
    public void Vulkan_ConvTranspose1d_Grouped_MatchesCpuReference()
    {
        using IBackend device = new VulkanBackend();
        foreach (Case c in GroupedCases) AssertCase(device, c);
    }

    private static void AssertCase(IBackend device, Case c)
    {
        int outPerGroup = c.COut / c.Groups;
        int tOut = (c.TIn - 1) * c.Stride + c.Dilation * (c.Kernel - 1) + 1 - c.PadLeft - c.PadRight;
        float[] x = Values(new Random(c.CIn * 31 + c.TIn), c.Batch * c.CIn * c.TIn, 2f);
        float[] w = Values(new Random(c.COut * 17 + c.Kernel), c.CIn * outPerGroup * c.Kernel, 0.5f);
        float[] b = Values(new Random(c.Batch + 3), c.COut, 1f);
        using Tensor input = Make(x, c.Batch, c.CIn, c.TIn);
        using Tensor weight = Make(w, c.CIn, outPerGroup, c.Kernel);
        using Tensor? bias = c.Bias ? Make(b, c.COut) : null;
        using Tensor expected = new(new TensorShape(c.Batch, c.COut, tOut), DType.F32);
        using Tensor actual = new(new TensorShape(c.Batch, c.COut, tOut), DType.F32);
        CpuBackend cpu = new();
        cpu.ConvTranspose1d(expected, input, weight, bias, c.Stride, c.PadLeft, c.PadRight, c.Dilation, c.Groups);
        device.ConvTranspose1d(actual, input, weight, bias, c.Stride, c.PadLeft, c.PadRight, c.Dilation, c.Groups);
        AssertClose(expected.AsReadOnlySpan<float>().ToArray(), actual.AsReadOnlySpan<float>().ToArray(), 1e-4f, c.ToString());
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

    private readonly record struct Case(int Batch, int CIn, int COut, int Groups, int TIn, int Kernel,
        int Stride, int PadLeft, int PadRight, int Dilation, bool Bias);
}
