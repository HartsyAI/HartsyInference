using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>Host reference for <see cref="IBackend.UcpeTransform"/>, one token at a time; tokens fan out.</summary>
public static class UcpeTransformReference
{
    /// <summary>Validates the operands of <see cref="IBackend.UcpeTransform"/>.</summary>
    public static void Validate(Tensor output, Tensor input, Tensor matrices, Tensor cos, Tensor sin, int tokens, int heads, int headDim)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(matrices);
        ArgumentNullException.ThrowIfNull(cos);
        ArgumentNullException.ThrowIfNull(sin);
        if (tokens < 1 || heads < 1 || headDim < 8 || headDim % 8 != 0)
            throw new ArgumentException($"UcpeTransform needs positive tokens and heads and a head dim divisible by 8, got {tokens}, {heads}, {headDim}.");
        foreach ((Tensor tensor, string name) in new[] { (output, nameof(output)), (input, nameof(input)), (matrices, nameof(matrices)),
            (cos, nameof(cos)), (sin, nameof(sin)) })
        {
            if (tensor.DType != DType.F32)
                throw new ArgumentException($"UcpeTransform {name} must be F32, got {tensor.DType.Name}.", name);
        }

        long vectors = (long)tokens * heads * headDim;
        if (input.ElementCount != vectors || output.ElementCount != vectors)
            throw new ArgumentException($"UcpeTransform input {input.Shape} and output {output.Shape} must hold {vectors} elements.", nameof(input));
        if (matrices.ElementCount != (long)tokens * 16)
            throw new ArgumentException($"UcpeTransform matrices {matrices.Shape} must be [{tokens}, 4, 4].", nameof(matrices));
        if (cos.ElementCount != (long)tokens * (headDim / 2) || sin.ElementCount != cos.ElementCount)
            throw new ArgumentException($"UcpeTransform tables {cos.Shape}, {sin.Shape} must be [{tokens}, {headDim / 2}].", nameof(cos));
    }

    /// <summary>Host <see cref="IBackend.UcpeTransform"/>.</summary>
    public static unsafe void Run(Tensor output, Tensor input, Tensor matrices, Tensor cos, Tensor sin, int tokens, int heads, int headDim)
    {
        Validate(output, input, matrices, cos, sin, tokens, heads, headDim);
        nint o = (nint)output.DataPointer, i = (nint)input.DataPointer, m = (nint)matrices.DataPointer;
        nint c = (nint)cos.DataPointer, s = (nint)sin.DataPointer;
        CpuParallel.For(tokens, (long)tokens * heads * headDim * 4, (o, i, m, c, s), (n, p) =>
        {
            int half = headDim / 2;
            float* outRow = (float*)p.o + (long)n * heads * headDim;
            float* inRow = (float*)p.i + (long)n * heads * headDim;
            float* mat = (float*)p.m + (long)n * 16;
            float* cosRow = (float*)p.c + (long)n * half;
            float* sinRow = (float*)p.s + (long)n * half;
            for (int h = 0; h < heads; h++)
            {
                float* x = inRow + h * headDim;
                float* y = outRow + h * headDim;
                for (int g = 0; g < half; g += 4)
                {
                    float x0 = x[g], x1 = x[g + 1], x2 = x[g + 2], x3 = x[g + 3];
                    for (int r = 0; r < 4; r++)
                        y[g + r] = mat[4 * r] * x0 + mat[4 * r + 1] * x1 + mat[4 * r + 2] * x2 + mat[4 * r + 3] * x3;
                }

                for (int e = 0; e < half; e += 2)
                {
                    float a = x[half + e], b = x[half + e + 1];
                    float cv = cosRow[e], sv = sinRow[e];
                    y[half + e] = a * cv - b * sv;
                    y[half + e + 1] = b * cv + a * sv;
                }
            }
        });
    }
}
