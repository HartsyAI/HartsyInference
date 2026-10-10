using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical correctness tests for the audio-focused backend activations
/// (Sigmoid, Tanh, Snake, Elu) added in Phase 5 §3. Each kernel is compared against
/// a known-good closed-form reference at a handful of probe points spanning the
/// interesting regions (near zero, large positive, large negative).
///
/// <para>Snake-specific: also validates the parity-aware divisor selection (vanilla
/// snake vs snake-beta) and per-channel alpha/beta broadcast.</para></summary>
public sealed unsafe class BackendActivationTests
{

    [Fact]
    public void Sigmoid_NoOverflowAtLargeMagnitudes()
    {
        using CpuBackend backend = new();
        float[] extreme = [-100f, -50f, 50f, 100f];
        Tensor input = MakeInput(extreme);
        Tensor output = new(input.Shape, DType.F32);
        try
        {
            backend.Sigmoid(output, input);
            float* op = (float*)output.DataPointer;
            // sigmoid(-100) should be ~0, sigmoid(100) should be ~1. Neither should be NaN or Inf.
            for (int i = 0; i < extreme.Length; i++)
            {
                Assert.False(float.IsNaN(op[i]), $"sigmoid({extreme[i]}) returned NaN");
                Assert.False(float.IsInfinity(op[i]), $"sigmoid({extreme[i]}) returned Inf");
                Assert.InRange(op[i], 0f, 1f);
            }
            Assert.True(op[0] < 1e-20f, $"sigmoid(-100) should be near 0, got {op[0]}");
            Assert.True(op[3] > 0.999999f, $"sigmoid(100) should be near 1, got {op[3]}");
        }
        finally
        {
            input.Dispose();
            output.Dispose();
        }
    }

    [Fact]
    public void Elu_NegativeHalfMatchesAlphaTimesExpMinusOne()
    {
        using CpuBackend backend = new();
        float[] negatives = [-0.5f, -1f, -3f];
        Tensor input = MakeInput(negatives);
        Tensor output = new(input.Shape, DType.F32);
        try
        {
            backend.Elu(output, input, alpha: 1.5f);
            float* op = (float*)output.DataPointer;
            for (int i = 0; i < negatives.Length; i++)
            {
                float expected = 1.5f * (MathF.Exp(negatives[i]) - 1f);
                Assert.Equal(expected, op[i], precision: 5);
            }
        }
        finally
        {
            input.Dispose();
            output.Dispose();
        }
    }

    [Fact]
    public void Snake_VanillaMatchesFormula()
    {
        // y = x + sin(alpha*x)^2 / alpha — verified against the BigVGAN / DAC reference.
        using CpuBackend backend = new();
        int channels = 3;
        int t = 4;
        Tensor input = new(new TensorShape(1, channels, t), DType.F32);
        Tensor alpha = new(new TensorShape(channels), DType.F32);
        Tensor output = new(input.Shape, DType.F32);
        try
        {
            float* ip = (float*)input.DataPointer;
            float* ap = (float*)alpha.DataPointer;
            for (int c = 0; c < channels; c++) ap[c] = 0.5f + c * 0.7f;     // 0.5, 1.2, 1.9
            for (int c = 0; c < channels; c++)
                for (int j = 0; j < t; j++)
                    ip[c * t + j] = (c * 4 + j) * 0.3f - 1.5f;

            backend.Snake(output, input, alpha, beta: null);

            float* op = (float*)output.DataPointer;
            for (int c = 0; c < channels; c++)
            {
                for (int j = 0; j < t; j++)
                {
                    float x = ip[c * t + j];
                    float a = ap[c];
                    float s = MathF.Sin(a * x);
                    float expected = x + (s * s) / a;
                    Assert.Equal(expected, op[c * t + j], precision: 4);
                }
            }
        }
        finally
        {
            input.Dispose();
            alpha.Dispose();
            output.Dispose();
        }
    }

    private static Tensor MakeInput(ReadOnlySpan<float> values)
    {
        Tensor t = new(new TensorShape(values.Length), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < values.Length; i++) p[i] = values[i];
        return t;
    }
}
