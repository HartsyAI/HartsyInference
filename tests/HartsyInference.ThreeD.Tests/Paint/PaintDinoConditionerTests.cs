using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Unit lane: the DINO conditioning of Paint 2.1 around its projection.</summary>
public sealed class PaintDinoConditionerTests
{
    [Fact]
    public unsafe void UnconditionalTokens_AreTheLayerNormOfTheProjectionBiasNotZeros()
    {
        // Upstream zeroes the DINO hidden states and projects them like any others, so every unconditional token is layer_norm(bias).
        // The bias row is [0, 0.5, 1, 1.5]: mean 0.75, variance 0.3125, so the normalised row is (x - 0.75) / sqrt(0.3125 + 1e-5).
        const int Dim = 4;
        using Tensor projWeight = new(new TensorShape(4 * Dim, 1536), DType.F32), projBias = new(new TensorShape(4 * Dim), DType.F32),
            normWeight = new(new TensorShape(Dim), DType.F32), normBias = new(new TensorShape(Dim), DType.F32);
        float* bias = (float*)projBias.DataPointer, scale = (float*)normWeight.DataPointer;
        for (int i = 0; i < 4 * Dim; i++)
            bias[i] = (i % Dim) * 0.5f;
        for (int i = 0; i < Dim; i++)
            scale[i] = 1f;
        using CpuBackend backend = new();
        PaintDinoConditioner conditioner = new(new Dinov2VisionEncoder(Dinov2Preset.Giant), projWeight, projBias, normWeight, normBias);

        using Tensor tokens = conditioner.UnconditionalTokens(backend);

        Assert.Equal(new TensorShape(1, 257 * 4, Dim), tokens.Shape);
        float[] expected = [.. new[] { 0f, 0.5f, 1f, 1.5f }.Select(x => (float)((x - 0.75) / Math.Sqrt(0.3125 + 1e-5)))];
        float* values = (float*)tokens.DataPointer;
        for (int token = 0; token < 257 * 4; token++)
            for (int c = 0; c < Dim; c++)
                Assert.Equal(expected[c], values[token * Dim + c], 3);
    }
}
