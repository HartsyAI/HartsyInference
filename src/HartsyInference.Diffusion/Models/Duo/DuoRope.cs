namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Duo's three-axis rotary tables, computed in F32 as upstream's <c>DuoJointMediaRotaryPosEmbed</c> does (<c>diffusers_model.py</c> @068e357): positions are cast to F32, the angle is the F32 product with <c>1 / theta^(2i/(2*16))</c>,
/// the t, h and w blocks are concatenated, then concatenated with themselves. Cos and sin cover the first <c>3 * 2 * 16 = 96</c> channels of each head; the rest is zero and is never rotated.</summary>
public static class DuoRope
{
    /// <summary>Frequencies per axis.</summary>
    public const int FreqDim = 16;

    /// <summary>Channels of each head that rotate.</summary>
    public const int RotaryDim = 3 * 2 * FreqDim;

    /// <summary>The F32 inverse frequencies, <c>1 / 10000^(2i / 32)</c> for <c>i</c> in <c>[0, 16)</c>.</summary>
    public static float[] InverseFrequencies(float theta = 10000.0f)
    {
        float[] inv = new float[FreqDim];
        for (int i = 0; i < FreqDim; i++)
        {
            float exponent = (float)(2 * i) / (2 * FreqDim);
            inv[i] = 1.0f / MathF.Pow(theta, exponent);
        }
        return inv;
    }

    /// <summary>Cos and sin tables <c>[rows, headDim]</c> for row-major <c>positions [rows, 3]</c>. Only the first 96 columns are set.</summary>
    public static (float[] Cos, float[] Sin) Tables(ReadOnlySpan<double> positions, int headDim, float theta = 10000.0f)
    {
        if (positions.Length % 3 != 0)
        {
            throw new ArgumentException("Positions must be (t, h, w) triples.", nameof(positions));
        }
        if (headDim < RotaryDim)
        {
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, $"The head dim must cover the {RotaryDim} rotary channels.");
        }
        float[] inv = InverseFrequencies(theta);
        int rows = positions.Length / 3;
        float[] cos = new float[rows * headDim];
        float[] sin = new float[rows * headDim];
        const int Half = RotaryDim / 2;
        for (int row = 0; row < rows; row++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                float pos = (float)positions[row * 3 + axis];
                for (int i = 0; i < FreqDim; i++)
                {
                    float angle = pos * inv[i];
                    int slot = row * headDim + axis * FreqDim + i;
                    cos[slot] = MathF.Cos(angle);
                    sin[slot] = MathF.Sin(angle);
                    cos[slot + Half] = cos[slot];
                    sin[slot + Half] = sin[slot];
                }
            }
        }
        return (cos, sin);
    }
}
