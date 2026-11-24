namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Round trip through float8 e4m3fn as torch does it: the per-vector round trip the Engram rows were trained to be read through.</summary>
public static class Hy4FlashFp8Rounding
{
    private const float E4M3Max = 448f;
    private static readonly float[] Values = BuildValues();

    /// <summary>The nearest e4m3fn value to <paramref name="value"/>, ties to the even code; magnitudes past 448 saturate.</summary>
    public static float RoundE4M3(float value)
    {
        if (float.IsNaN(value)) return value;
        float magnitude = MathF.Min(MathF.Abs(value), E4M3Max);
        // Values ascend with their codes, so the first code at or above the magnitude is found by bisection; the nearest is it or its neighbour below.
        int lo = 0, hi = Values.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (Values[mid] < magnitude) lo = mid + 1;
            else hi = mid;
        }
        int best = lo;
        if (lo > 0)
        {
            float below = magnitude - Values[lo - 1], above = Values[lo] - magnitude;
            if (below < above || (below == above && ((lo - 1) & 1) == 0)) best = lo - 1;
        }
        return value < 0f ? -Values[best] : Values[best];
    }

    /// <summary>Per-vector round trip in place: <c>scale = max(amax, 1e-12) / 448</c>, then <c>round(x / scale) * scale</c> (<c>_fp8_round_trip</c> of the reference).</summary>
    public static void RoundTripVector(Span<float> vector)
    {
        float amax = 0f;
        foreach (float v in vector) amax = MathF.Max(amax, MathF.Abs(v));
        float scale = MathF.Max(amax, 1e-12f) / E4M3Max;
        for (int i = 0; i < vector.Length; i++) vector[i] = RoundE4M3(Math.Clamp(vector[i] / scale, -E4M3Max, E4M3Max)) * scale;
    }

    // The 127 finite non-negative e4m3fn codes (0x7F is NaN): 4 exponent bits (bias 7), 3 mantissa bits, subnormals at exponent 0.
    private static float[] BuildValues()
    {
        float[] values = new float[127];
        for (int code = 0; code < values.Length; code++)
        {
            int exponent = code >> 3, mantissa = code & 7;
            values[code] = exponent == 0 ? mantissa / 8f * MathF.Pow(2f, -6f) : (1f + mantissa / 8f) * MathF.Pow(2f, exponent - 7);
        }
        return values;
    }
}
