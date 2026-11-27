using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Vulkan ConvTranspose2d against the interface's host reference, on the kernel shapes the engine's models use and on unequal
/// axes, plus the shape and dtype guards. Each fact is reported as skipped (never passed) when no Vulkan device or compiled kernel is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanConvTranspose2dParityTests
{
    private static readonly Case[] Cases =
    {
        // Kernel equals stride, no padding: Kim_Vocal_2's decoder and the YOLO-seg Proto (k2 s2), DPT (k4 s4).
        new(1, 96, 48, 8, 12, 16, 24, 2, 2, 2, 2, 0, 0, true),
        new(2, 5, 7, 3, 5, 12, 20, 4, 4, 4, 4, 0, 0, true),
        // Overlapping taps with padding: the line-art annotator (k4 s2 p1), RMVPE (k3 s2 p1 with output padding 1, no bias).
        new(1, 6, 4, 7, 5, 14, 10, 4, 4, 2, 2, 1, 1, true),
        new(2, 4, 6, 5, 9, 10, 18, 3, 3, 2, 2, 1, 1, false),
        // Every axis different, so a swapped stride, padding or kernel size cannot pass.
        new(1, 3, 5, 4, 6, 9, 11, 3, 2, 3, 2, 1, 0, true),
        // Stride one, and an output smaller than the full scatter, which crops it.
        new(1, 7, 3, 6, 4, 6, 4, 3, 3, 1, 1, 1, 1, false),
        new(1, 3, 2, 5, 5, 8, 7, 2, 2, 2, 2, 0, 0, true),
    };

    [VulkanFact]
    public void Vulkan_ConvTranspose2d_MatchesHostReference()
    {
        using IBackend device = new VulkanBackend();
        using IBackend host = new CpuBackend();
        foreach (Case c in Cases)
        {
            float[] x = Values(new Random(c.CIn * 31 + c.IH), c.Batch * c.CIn * c.IH * c.IW, 2f);
            float[] w = Values(new Random(c.COut * 17 + c.KH), c.CIn * c.COut * c.KH * c.KW, 0.5f);
            float[] b = Values(new Random(c.Batch + 3), c.COut, 1f);
            using Tensor input = Make(x, c.Batch, c.CIn, c.IH, c.IW);
            using Tensor weight = Make(w, c.CIn, c.COut, c.KH, c.KW);
            using Tensor? bias = c.Bias ? Make(b, c.COut) : null;
            using Tensor expected = new(new TensorShape(c.Batch, c.COut, c.OH, c.OW), DType.F32);
            using Tensor actual = new(new TensorShape(c.Batch, c.COut, c.OH, c.OW), DType.F32);
            host.ConvTranspose2d(expected, input, weight, bias, c.StrideH, c.StrideW, c.PadH, c.PadW);
            device.ConvTranspose2d(actual, input, weight, bias, c.StrideH, c.StrideW, c.PadH, c.PadW);
            float[] want = expected.AsReadOnlySpan<float>().ToArray(), got = actual.AsReadOnlySpan<float>().ToArray();
            for (int i = 0; i < want.Length; i++)
                Assert.True(MathF.Abs(want[i] - got[i]) <= 1e-4f * MathF.Max(1f, MathF.Abs(want[i])), $"{c} element {i}: host {want[i]} device {got[i]}");
        }
    }

    // The guards run before any dispatch, so this checks the exception, not the kernel.
    [VulkanFact]
    public void Vulkan_ConvTranspose2d_RejectsBadShapeDtypeOrBias()
    {
        using IBackend device = new VulkanBackend();
        using Tensor weight = new(new TensorShape(2, 3, 2, 2), DType.F32);
        using Tensor output = new(new TensorShape(1, 3, 4, 4), DType.F32);
        using Tensor input = new(new TensorShape(1, 2, 2, 2), DType.F32);
        using Tensor inputRank3 = new(new TensorShape(2, 2, 2), DType.F32);
        using Tensor inputWrongChannels = new(new TensorShape(1, 3, 2, 2), DType.F32);
        using Tensor inputF16 = new(new TensorShape(1, 2, 2, 2), DType.F16);
        using Tensor biasWrongLength = new(new TensorShape(2), DType.F32);
        Assert.Throws<ArgumentException>(() => device.ConvTranspose2d(output, inputRank3, weight, null, 2, 2, 0, 0));
        Assert.Throws<ArgumentException>(() => device.ConvTranspose2d(output, inputWrongChannels, weight, null, 2, 2, 0, 0));
        Assert.Throws<ArgumentException>(() => device.ConvTranspose2d(output, input, weight, null, 0, 2, 0, 0));
        Assert.Throws<ArgumentException>(() => device.ConvTranspose2d(output, input, weight, biasWrongLength, 2, 2, 0, 0));
        Assert.Throws<NotSupportedException>(() => device.ConvTranspose2d(output, inputF16, weight, null, 2, 2, 0, 0));
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

    private readonly record struct Case(int Batch, int CIn, int COut, int IH, int IW, int OH, int OW, int KH, int KW,
        int StrideH, int StrideW, int PadH, int PadW, bool Bias);
}
