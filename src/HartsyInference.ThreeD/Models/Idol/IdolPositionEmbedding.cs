using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Upstream <c>PcdAbsolutePositionEmbedder</c> over Idol's fixed token positions: per axis the sines then the cosines of
/// <c>2π · x · 2^e</c>, for 16 whole octaves and then 154 fractional ones up to 2^16, zero-padded to the model width. The positions never
/// change, so the embedding is built once on the host.
/// <para>The angles reach 4e5 rad, where a float32 angle is good to about 0.03 rad, so the arithmetic follows torch's float32 operations in
/// order: <c>((x · f) · 2) · π</c> with <c>π</c> rounded to float32. The sine and cosine of that float32 angle are taken in double and rounded once.</para></summary>
internal static class IdolPositionEmbedding
{
    private const int WholeOctaves = 16;

    /// <summary>The embedding <c>[1, tokens, channels]</c> of positions <paramref name="positions"/> <c>[1, tokens, 3]</c> in <c>[0, 1]</c>; the caller owns the result.</summary>
    public static unsafe Tensor Build(Tensor positions, int channels)
    {
        ArgumentNullException.ThrowIfNull(positions);
        int tokens = (int)positions.Shape[1];
        int freqDim = channels / 3 / 2;
        float[] freqs = Frequencies(freqDim);
        Tensor result = new(new TensorShape(1, tokens, channels), DType.F32);
        float* p = (float*)positions.DataPointer, o = (float*)result.DataPointer;
        float pi = (float)Math.PI;
        int axisWidth = 2 * freqDim;
        for (int t = 0; t < tokens; t++)
        {
            float* row = o + (long)t * channels;
            for (int axis = 0; axis < 3; axis++)
            {
                float x = p[t * 3 + axis];
                for (int j = 0; j < freqDim; j++)
                {
                    float angle = x * freqs[j] * 2f * pi;
                    row[axis * axisWidth + j] = (float)Math.Sin(angle);
                    row[axis * axisWidth + freqDim + j] = (float)Math.Cos(angle);
                }
            }
            // channels - 3 * axisWidth trailing channels stay zero, as upstream pads them.
        }

        return result;
    }

    /// <summary><c>2 ** cat([arange(16), arange(res) / res * 16])[:freqDim]</c> in float32.</summary>
    internal static float[] Frequencies(int freqDim)
    {
        int res = Math.Max(0, freqDim - WholeOctaves);
        float[] f = new float[freqDim];
        for (int j = 0; j < freqDim; j++)
        {
            float exponent = j < WholeOctaves ? j : (float)(j - WholeOctaves) / res * WholeOctaves;
            f[j] = (float)Math.Pow(2.0, exponent);
        }

        return f;
    }
}
