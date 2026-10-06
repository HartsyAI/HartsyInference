using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Rotary position table of one sequence: cos/sin of <c>pos * freq_f</c>, laid out <c>[position, pair]</c>.
/// Port of <c>compute_rope_rotations</c> / <c>apply_rope</c> (adjacent-pair rotation, float32 angles).</summary>
public sealed class ControlFoleyRope
{
    private ControlFoleyRope(int length, int pairs, float[] cos, float[] sin)
    {
        Length = length;
        Pairs = pairs;
        Cos = cos;
        Sin = sin;
    }

    /// <summary>Number of positions.</summary>
    public int Length { get; }

    /// <summary>Rotated pairs per head (head size / 2).</summary>
    public int Pairs { get; }

    /// <summary>Cosines, <c>[Length * Pairs]</c>.</summary>
    public float[] Cos { get; }

    /// <summary>Sines, <c>[Length * Pairs]</c>.</summary>
    public float[] Sin { get; }

    /// <summary>Builds the table for <paramref name="length"/> positions and head size <paramref name="dim"/>.</summary>
    public static ControlFoleyRope Compute(int length, int dim, float theta, float freqScaling)
    {
        if (length <= 0 || dim <= 0 || dim % 2 != 0)
        {
            throw new ArgumentException($"Invalid rope request (length {length}, dim {dim}).");
        }

        int pairs = dim / 2;
        float[] freqs = new float[pairs];
        for (int f = 0; f < pairs; f++)
        {
            freqs[f] = 1f / MathF.Pow(theta, 2f * f / dim);
            freqs[f] *= freqScaling;
        }

        float[] cos = new float[length * pairs], sin = new float[length * pairs];
        for (int p = 0; p < length; p++)
        {
            for (int f = 0; f < pairs; f++)
            {
                float angle = p * freqs[f];
                cos[p * pairs + f] = MathF.Cos(angle);
                sin[p * pairs + f] = MathF.Sin(angle);
            }
        }

        return new ControlFoleyRope(length, pairs, cos, sin);
    }

    /// <summary>Rotates every adjacent pair of <paramref name="x"/> (<c>[B, H, N, headDim]</c>) in place.</summary>
    internal unsafe void Apply(Tensor x)
    {
        int b = (int)x.Shape[0], h = (int)x.Shape[1], n = (int)x.Shape[2], d = (int)x.Shape[3];
        if (n != Length || d != Pairs * 2)
        {
            throw new ArgumentException($"Rope table [{Length}, {Pairs * 2}] does not fit tensor {x.Shape}.");
        }

        float* p = (float*)x.DataPointer;
        for (int bh = 0; bh < b * h; bh++)
        {
            for (int pos = 0; pos < n; pos++)
            {
                float* row = p + ((long)bh * n + pos) * d;
                for (int f = 0; f < Pairs; f++)
                {
                    float c = Cos[pos * Pairs + f], s = Sin[pos * Pairs + f];
                    float x0 = row[2 * f], x1 = row[2 * f + 1];
                    row[2 * f] = c * x0 - s * x1;
                    row[2 * f + 1] = s * x0 + c * x1;
                }
            }
        }
    }
}
