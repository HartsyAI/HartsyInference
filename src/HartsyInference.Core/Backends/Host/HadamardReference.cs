using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>CPU reference for the Walsh-Hadamard transform of each row: <c>y = scale * (x @ H)</c>, with <c>H</c> the
/// Sylvester (natural order) Hadamard matrix of the row width, <c>H_2n = [[H_n, H_n], [H_n, -H_n]]</c>.</summary>
/// <remarks>With <c>scale = 1/sqrt(width)</c> the transform is orthonormal and its own inverse.</remarks>
public static class HadamardReference
{
    /// <summary>Checks the operands of <see cref="Apply"/>; shared by every backend.</summary>
    public static int Validate(Tensor output, Tensor input)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32) throw new NotSupportedException("Hadamard supports F32 only.");
        if (input.Shape.Rank < 1 || output.Shape.Rank < 1 || output.ElementCount != input.ElementCount)
            throw new ArgumentException($"Hadamard output has {output.ElementCount} elements, input {input.ElementCount}.");
        long width = input.Shape[input.Shape.Rank - 1];
        if (output.Shape[output.Shape.Rank - 1] != width)
            throw new ArgumentException($"Hadamard output rows must be {width} wide like the input; got {output.Shape}.", nameof(output));
        if (width < 1 || width > 1 << 20 || (width & (width - 1)) != 0)
            throw new ArgumentException($"Hadamard needs a power-of-two last dimension; got {width}.", nameof(input));
        return (int)width;
    }

    /// <summary>Transforms every row of <paramref name="input"/> into <paramref name="output"/>; in place is allowed.</summary>
    /// <remarks>The butterflies run in double and round once, so the result does not depend on the row width's factorization.</remarks>
    public static unsafe void Apply(Tensor output, Tensor input, float scale)
    {
        int width = Validate(output, input);
        float* src = (float*)input.DataPointer, dst = (float*)output.DataPointer;
        long rows = input.ElementCount / width;
        double[] row = new double[width];
        for (long r = 0; r < rows; r++)
        {
            float* x = src + r * width, y = dst + r * width;
            for (int i = 0; i < width; i++) row[i] = x[i];
            for (int h = 1; h < width; h <<= 1)
                for (int start = 0; start < width; start += h << 1)
                    for (int j = start; j < start + h; j++)
                    {
                        double a = row[j], b = row[j + h];
                        row[j] = a + b;
                        row[j + h] = a - b;
                    }
            for (int i = 0; i < width; i++) y[i] = (float)(row[i] * scale);
        }
    }
}
