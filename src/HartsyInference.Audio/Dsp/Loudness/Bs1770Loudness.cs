namespace HartsyInference.Audio.Dsp.Loudness;

/// <summary>Integrated loudness of mono audio (the K-weighting and gating of ITU-R BS.1770) computed as pyloudnorm 0.1.1's
/// <c>Meter.integrated_loudness</c> computes it, and the gain that moves audio to a target loudness (its <c>normalize.loudness</c>).</summary>
public static class Bs1770Loudness
{
    private const double BlockSeconds = 0.4, StepFraction = 0.25, AbsoluteGateLufs = -70.0, RelativeGateLu = -10.0, OffsetDb = -0.691;

    /// <summary>The integrated loudness in LUFS; negative infinity when every block is gated out (silence).</summary>
    /// <exception cref="ArgumentException">The audio is shorter than one 400 ms gating block.</exception>
    public static double Integrated(ReadOnlySpan<float> samples, int sampleRate)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        if (samples.Length < BlockSeconds * sampleRate)
            throw new ArgumentException($"Loudness needs at least {BlockSeconds} s of audio; got {samples.Length} samples at {sampleRate} Hz.",
                nameof(samples));
        double[] weighted = new double[samples.Length];
        for (int i = 0; i < samples.Length; i++)
            weighted[i] = samples[i];
        // The two K-weighting stages: a +4 dB high shelf at 1.5 kHz, then a 38 Hz high pass.
        Filter(weighted, HighShelf(gainDb: 4.0, q: 1.0 / Math.Sqrt(2.0), frequency: 1500.0, sampleRate));
        Filter(weighted, HighPass(q: 0.5, frequency: 38.0, sampleRate));

        double duration = samples.Length / (double)sampleRate;
        // Math.Round rounds half to even, as numpy's round does.
        int blocks = (int)Math.Round((duration - BlockSeconds) / (BlockSeconds * StepFraction)) + 1;
        double[] energy = new double[blocks];
        for (int j = 0; j < blocks; j++)
        {
            // The same order of operations as the reference, so the truncated block edges agree.
            int from = (int)(BlockSeconds * (j * StepFraction) * sampleRate);
            int to = Math.Min((int)(BlockSeconds * (j * StepFraction + 1) * sampleRate), weighted.Length);
            double sum = 0;
            for (int i = from; i < to; i++)
                sum += weighted[i] * weighted[i];
            energy[j] = 1.0 / (BlockSeconds * sampleRate) * sum;
        }

        // The absolute gate, then a gate 10 LU under the loudness of what passed it.
        double relativeGate = Loudness(MeanAbove(energy, AbsoluteGateLufs, inclusive: true)) + RelativeGateLu;
        double gated = MeanAbove(energy, Math.Max(relativeGate, AbsoluteGateLufs), inclusive: false);
        return Loudness(double.IsNaN(gated) ? 0.0 : gated);
    }

    /// <summary>Scales <paramref name="samples"/> in place from <paramref name="measuredLufs"/> to <paramref name="targetLufs"/>. The result is
    /// not clipped, as the reference does not clip it.</summary>
    public static void ApplyGain(Span<float> samples, double measuredLufs, double targetLufs)
    {
        double gain = Math.Pow(10.0, (targetLufs - measuredLufs) / 20.0);
        for (int i = 0; i < samples.Length; i++)
            samples[i] = (float)(gain * samples[i]);
    }

    private static double Loudness(double meanSquare) => OffsetDb + 10.0 * Math.Log10(meanSquare);

    /// <summary>The mean block energy over the blocks whose loudness passes the gate; NaN when none does, or when the gate is NaN.</summary>
    private static double MeanAbove(double[] energy, double gateLufs, bool inclusive)
    {
        if (double.IsNaN(gateLufs))
            return double.NaN;
        double sum = 0;
        int count = 0;
        foreach (double block in energy)
        {
            double loudness = Loudness(block);
            if (inclusive ? loudness >= gateLufs : loudness > gateLufs)
            {
                sum += block;
                count++;
            }
        }
        return count == 0 ? double.NaN : sum / count;
    }

    /// <summary>A second-order section normalised to a0 = 1.</summary>
    private readonly record struct Section(double B0, double B1, double B2, double A1, double A2);

    private static Section HighShelf(double gainDb, double q, double frequency, int sampleRate)
    {
        double a = Math.Pow(10.0, gainDb / 40.0);
        double w0 = 2.0 * Math.PI * (frequency / sampleRate), cos = Math.Cos(w0), alpha = Math.Sin(w0) / (2.0 * q), root = 2.0 * Math.Sqrt(a) * alpha;
        double a0 = (a + 1) - (a - 1) * cos + root;
        return new Section(a * ((a + 1) + (a - 1) * cos + root) / a0, -2 * a * ((a - 1) + (a + 1) * cos) / a0,
            a * ((a + 1) + (a - 1) * cos - root) / a0, 2 * ((a - 1) - (a + 1) * cos) / a0, ((a + 1) - (a - 1) * cos - root) / a0);
    }

    private static Section HighPass(double q, double frequency, int sampleRate)
    {
        double w0 = 2.0 * Math.PI * (frequency / sampleRate), cos = Math.Cos(w0), alpha = Math.Sin(w0) / (2.0 * q);
        double a0 = 1 + alpha;
        return new Section((1 + cos) / 2 / a0, -(1 + cos) / a0, (1 + cos) / 2 / a0, -2 * cos / a0, (1 - alpha) / a0);
    }

    /// <summary>Direct form II transposed, from rest, in place.</summary>
    private static void Filter(double[] signal, Section section)
    {
        double z1 = 0, z2 = 0;
        for (int i = 0; i < signal.Length; i++)
        {
            double x = signal[i];
            double y = section.B0 * x + z1;
            z1 = section.B1 * x - section.A1 * y + z2;
            z2 = section.B2 * x - section.A2 * y;
            signal[i] = y;
        }
    }
}
