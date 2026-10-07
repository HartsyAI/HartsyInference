namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Interleaved-pair rotary tables for one V4.1 attention layer: cos and sin, row-major <c>[length, rotaryDim / 2]</c>.</summary>
/// <remarks>Matches upstream <c>precompute_freqs_cis</c>, whose YaRN floors and ceils the correction range and applies
/// no mscale, so it cannot reuse the shared <c>RopeFrequencyBuilder</c> (which does neither).</remarks>
public sealed class DeepSeekV41RopeTable
{
    /// <summary>Half the rotary width: one angle per adjacent element pair.</summary>
    public int HalfDim { get; }

    /// <summary>Number of positions in the table.</summary>
    public int Length { get; }

    /// <summary>Cosines, <c>[Length, HalfDim]</c>.</summary>
    public float[] Cos { get; }

    /// <summary>Sines, <c>[Length, HalfDim]</c>.</summary>
    public float[] Sin { get; }

    private DeepSeekV41RopeTable(int halfDim, int length, float[] cos, float[] sin)
    {
        HalfDim = halfDim;
        Length = length;
        Cos = cos;
        Sin = sin;
    }

    /// <summary>Builds the table for positions <c>[0, length)</c>.</summary>
    /// <param name="rotaryDim">The rope slice width, even and positive.</param>
    /// <param name="length">Positions to precompute.</param>
    /// <param name="theta">Rope base.</param>
    /// <param name="scaling">YaRN parameters, or null for plain rotary.</param>
    public static DeepSeekV41RopeTable Build(int rotaryDim, int length, double theta, DeepSeekV41RopeScaling? scaling)
    {
        if (rotaryDim < 2 || (rotaryDim & 1) != 0) throw new ArgumentOutOfRangeException(nameof(rotaryDim), "rotaryDim must be even and positive.");
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        int half = rotaryDim / 2;
        double[] invFreq = InverseFrequencies(rotaryDim, theta, scaling);
        float[] cos = new float[(long)length * half];
        float[] sin = new float[cos.Length];
        for (int pos = 0; pos < length; pos++)
            for (int i = 0; i < half; i++)
            {
                double angle = (double)pos * invFreq[i];
                cos[pos * half + i] = (float)Math.Cos(angle);
                sin[pos * half + i] = (float)Math.Sin(angle);
            }
        return new DeepSeekV41RopeTable(half, length, cos, sin);
    }

    /// <summary>The cosines of position <paramref name="pos"/>, <c>HalfDim</c> entries.</summary>
    public ReadOnlySpan<float> CosRow(int pos) => Cos.AsSpan(CheckedOffset(pos), HalfDim);

    /// <summary>The sines of position <paramref name="pos"/>, <c>HalfDim</c> entries.</summary>
    public ReadOnlySpan<float> SinRow(int pos) => Sin.AsSpan(CheckedOffset(pos), HalfDim);

    private int CheckedOffset(int pos)
    {
        if ((uint)pos >= (uint)Length) throw new ArgumentOutOfRangeException(nameof(pos));
        return pos * HalfDim;
    }

    /// <summary>Inverse frequencies with the optional YaRN blend; <c>original_seq_len &lt;= 0</c> disables it.</summary>
    internal static double[] InverseFrequencies(int rotaryDim, double theta, DeepSeekV41RopeScaling? scaling)
    {
        int half = rotaryDim / 2;
        double[] invFreq = new double[half];
        for (int i = 0; i < half; i++) invFreq[i] = 1.0 / Math.Pow(theta, (double)(2 * i) / rotaryDim);
        if (scaling is null || scaling.OriginalMaxPositionEmbeddings <= 0) return invFreq;

        double CorrectedDim(double rotations) =>
            rotaryDim * Math.Log(scaling.OriginalMaxPositionEmbeddings / (rotations * 2.0 * Math.PI)) / (2.0 * Math.Log(theta));

        double low = Math.Max(Math.Floor(CorrectedDim(scaling.BetaFast)), 0.0);
        double high = Math.Min(Math.Ceiling(CorrectedDim(scaling.BetaSlow)), rotaryDim - 1);
        double span = Math.Max(high - low, 1e-3);
        for (int i = 0; i < half; i++)
        {
            double ramp = Math.Clamp((i - low) / span, 0.0, 1.0);
            double smooth = 1.0 - ramp;
            invFreq[i] = invFreq[i] / scaling.Factor * (1.0 - smooth) + invFreq[i] * smooth;
        }
        return invFreq;
    }
}
