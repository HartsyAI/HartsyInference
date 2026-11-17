using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Upstream <c>PcdAbsolutePositionEmbedderv2</c>, the query-point embedding of both decoders: per axis the sines then the cosines of
/// <c>x · 2^e · π</c> for 170 exponents spaced evenly over [0, 10], zero-padded to the model width. The angle is formed as
/// <c>(x · f) · π</c> in float32 as torch does; the sine and cosine of that float32 angle are taken in double and rounded once.</summary>
internal static class IdolAnchorEmbedding
{
    private const float MaxResolutionExponent = 10f;

    /// <summary>The embedding <c>[1, N, channels]</c> of points <paramref name="points"/> (<c>N × 3</c> floats in [0, 1]); the caller owns the result.</summary>
    public static unsafe Tensor Build(ReadOnlySpan<float> points, int channels)
    {
        int count = points.Length / 3;
        int freqDim = channels / 3 / 2;
        float[] freqs = Frequencies(freqDim);
        Tensor result = new(new TensorShape(1, count, channels), DType.F32);
        float* o = (float*)result.DataPointer;
        float pi = (float)Math.PI;
        int axisWidth = 2 * freqDim;
        for (int i = 0; i < count; i++)
        {
            float* row = o + (long)i * channels;
            for (int axis = 0; axis < 3; axis++)
            {
                float x = points[i * 3 + axis];
                for (int j = 0; j < freqDim; j++)
                {
                    float angle = x * freqs[j] * pi;
                    row[axis * axisWidth + j] = (float)Math.Sin(angle);
                    row[axis * axisWidth + freqDim + j] = (float)Math.Cos(angle);
                }
            }
        }

        return result;
    }

    /// <summary><c>2 ** linspace(0, 10, freqDim)</c> in float32, with torch's two-sided linspace (the second half counts back from the end).</summary>
    internal static float[] Frequencies(int freqDim)
    {
        float[] f = new float[freqDim];
        float step = freqDim > 1 ? MaxResolutionExponent / (freqDim - 1) : 0f;
        for (int i = 0; i < freqDim; i++)
        {
            float exponent = i < freqDim / 2 ? step * i : MaxResolutionExponent - step * (freqDim - i - 1);
            f[i] = (float)Math.Pow(2.0, exponent);
        }

        return f;
    }
}
