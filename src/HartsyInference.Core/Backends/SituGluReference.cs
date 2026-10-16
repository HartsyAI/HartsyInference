using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Host reference for <see cref="IBackend.SituGlu"/>, MOSS-Image's SiTU-GLU
/// (<c>(4·tanh(g/4)·sigmoid(g)) · (25·tanh(u/25))</c> over the <c>[u | g]</c> halves of the last dimension).</summary>
public static class SituGluReference
{
    /// <summary>Soft cap on the gate branch.</summary>
    public const float GateCap = 4f;

    /// <summary>Soft cap on the value branch.</summary>
    public const float ValueCap = 25f;

    /// <summary>SiTU-GLU of one <paramref name="value"/>/<paramref name="gate"/> pair. The sigmoid takes the sign-aware
    /// form so neither tail overflows, and a saturated input stays finite.</summary>
    public static float Scalar(float value, float gate)
    {
        float sigmoid = gate >= 0f ? 1f / (1f + MathF.Exp(-gate)) : MathF.Exp(gate) / (1f + MathF.Exp(gate));
        return GateCap * MathF.Tanh(gate / GateCap) * sigmoid * (ValueCap * MathF.Tanh(value / ValueCap));
    }

    /// <summary>Validates the pair and returns the inner width <c>F</c> and the row count.</summary>
    public static (int Inner, long Rows) Validate(Tensor output, Tensor input)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        if (input.DType != output.DType || (input.DType != DType.F32 && input.DType != DType.F16))
            throw new NotSupportedException($"SituGlu needs matching F32 or F16 tensors, got input={input.DType}, output={output.DType}.");
        int inner = (int)output.Shape[output.Shape.Rank - 1];
        if (inner <= 0 || input.Shape[input.Shape.Rank - 1] != 2L * inner)
            throw new ArgumentException($"SituGlu input {input.Shape} must be twice the output's last dimension {output.Shape}.", nameof(input));
        long rows = output.ElementCount / inner;
        if (input.ElementCount != rows * 2 * inner)
            throw new ArgumentException($"SituGlu input {input.Shape} and output {output.Shape} disagree on the row count.", nameof(input));
        return (inner, rows);
    }

    /// <summary>Applies SiTU-GLU on the host.</summary>
    public static unsafe void Apply(Tensor output, Tensor input)
    {
        (int inner, long rows) = Validate(output, input);
        if (input.DType == DType.F32)
        {
            float* src = (float*)input.DataPointer;
            float* dst = (float*)output.DataPointer;
            Parallel.For(0L, rows, row =>
            {
                float* inRow = src + row * 2 * inner;
                float* outRow = dst + row * inner;
                for (int j = 0; j < inner; j++) outRow[j] = Scalar(inRow[j], inRow[inner + j]);
            });
            return;
        }
        Half* srcH = (Half*)input.DataPointer;
        Half* dstH = (Half*)output.DataPointer;
        Parallel.For(0L, rows, row =>
        {
            Half* inRow = srcH + row * 2 * inner;
            Half* outRow = dstH + row * inner;
            for (int j = 0; j < inner; j++) outRow[j] = (Half)Scalar((float)inRow[j], (float)inRow[inner + j]);
        });
    }
}
