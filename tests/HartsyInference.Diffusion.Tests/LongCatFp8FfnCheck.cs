using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>An FP8 E4M3 weight with one scale per tensor against the weight it was quantized from, element by element.</summary>
internal static class LongCatFp8FfnCheck
{
    /// <summary>The comparison of one weight. <c>Violations</c> counts elements farther from the source than their bound, <c>Worst</c> is the
    /// largest error as a share of the bound, and <c>EdgeRoundedDown</c> counts elements the encoder rounded down where up was nearer.</summary>
    /// <remarks>The bound is half an E4M3 step, except in the top half step below 2^-6: the engine's encoder caps a subnormal at 7 x 2^-9, so a
    /// value there goes down instead of up to 2^-6, and the bound is a full step.</remarks>
    public readonly record struct Report(long Violations, double Worst, double RelRms, bool ScaleIsAbsMaxOver448, long EdgeRoundedDown);

    /// <summary>Compares <paramref name="packed"/> (F8E4M3, its scale on <see cref="Tensor.Fp8ScaleFactor"/>) with the F32 values of
    /// <paramref name="source"/>.</summary>
    public static Report Compare(Tensor source, Tensor packed)
    {
        using Tensor wide = packed.CastTo(DType.F32);
        ReadOnlySpan<float> expected = source.AsReadOnlySpan<float>(), got = wide.AsReadOnlySpan<float>();
        double scale = packed.Fp8ScaleFactor, error = 0, power = 0, worst = 0;
        float absMax = 0f;
        foreach (float value in expected)
            absMax = MathF.Max(absMax, MathF.Abs(value));
        long violations = 0, edge = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double difference = Math.Abs((double)got[i] - expected[i]);
            error += difference * difference;
            power += (double)expected[i] * expected[i];
            // E4M3 keeps three mantissa bits, so a normal value's step is 2^(exponent - 3); below 2^-6 the step stays 2^-9.
            double scaled = Math.Abs(expected[i]) / scale;
            int exponent = scaled > 0 ? Math.Max(Math.ILogB(scaled), -6) : -6;
            bool belowSmallestNormal = scaled >= 7.5 / 512 && scaled < 1.0 / 64;
            if (belowSmallestNormal && difference > scale * (1.0 / 1024))
                edge++;
            // The scaling multiplies in F32 before the cast, which may move a value by a part in 2^23 across a tie.
            double bound = scale * ((belowSmallestNormal ? 1.0 / 512 : Math.ScaleB(1, exponent - 4)) + scaled * 2.4e-7);
            if (difference > bound)
                violations++;
            worst = Math.Max(worst, difference / bound);
        }
        return new Report(violations, worst, Math.Sqrt(error / power), packed.Fp8ScaleFactor == absMax / 448f, edge);
    }
}
