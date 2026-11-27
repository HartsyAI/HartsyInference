using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Quant;

/// <summary>The Vulkan MXFP4-packed linear against the Core host reference, on random packed weights.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class PackedMxfp4VulkanTests
{
    private readonly ITestOutputHelper _output;
    public PackedMxfp4VulkanTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(3, 64, 6)]
    [InlineData(1, 224, 40)]
    public void Vulkan_MatchesHostReference(int n, int cols, int rows)
    {
        if (!VulkanContext.IsAvailable()) { _output.WriteLine("SKIPPED: Vulkan unavailable"); return; }
        Random rng = new(rows * 7 + cols);
        byte[] nibbles = new byte[rows * cols / 2];
        rng.NextBytes(nibbles);
        byte[] scales = new byte[rows * (cols / 32)];
        for (int i = 0; i < scales.Length; i++) scales[i] = (byte)rng.Next(118, 130);
        float[] x = new float[n * cols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        using Tensor weight = Bytes(nibbles, rows, cols / 2);
        using Tensor scale = Bytes(scales, rows, cols / 32);
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Mxfp4E8M0, Geometry = new BlockGeometry(1, 32), ScaleDType = DType.U8,
            LogicalRows = rows, LogicalCols = cols, Scale = scale,
        };
        using Tensor input = Floats(x, n, cols);
        using Tensor expected = new(new TensorShape(n, rows), DType.F32);
        using Tensor actual = new(new TensorShape(n, rows), DType.F32);

        Mxfp4LinearReference.Apply(expected, input, weight, recipe);
        using VulkanBackend backend = new();
        IBackend device = backend;
        device.Mxfp4LinearF32(actual, input, weight, recipe);
        backend.OffloadActivation(actual);

        float[] want = expected.AsReadOnlySpan<float>().ToArray();
        float[] got = actual.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < want.Length; i++)
            Assert.True(MathF.Abs(want[i] - got[i]) <= 1e-3f * MathF.Max(1f, MathF.Abs(want[i])), $"y[{i}] host {want[i]} vulkan {got[i]}");
    }

    private static Tensor Bytes(byte[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.U8);
        values.CopyTo(t.AsSpan<byte>());
        return t;
    }

    private static Tensor Floats(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
