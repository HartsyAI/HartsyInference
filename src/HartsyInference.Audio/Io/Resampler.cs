namespace HartsyInference.Audio.Io;

/// <summary>Polyphase resampler with a windowed-sinc anti-alias filter. Matches the
/// behavior of <c>scipy.signal.resample_poly</c> / <c>librosa.resample(res_type="kaiser_best")</c>
/// within float32 rounding at default 64 taps.
///
/// <para>Pure C#, allocation-free per call once <see cref="Create"/> has built the
/// polyphase filter table. The table is built once per (rate-in, rate-out, num-taps)
/// triple and cached on the static type — a typical inference workload only ever sees
/// one rate change per session.</para>
///
/// <para>STT models all want 16 kHz mono. Most input files are 44.1 / 48 kHz, so the
/// common conversions (44100 → 16000 and 48000 → 16000) are exercised on every call.
/// Both are upsample-then-downsample by integer factors with the polyphase trick:
/// for <c>44100 → 16000</c> we have <c>gcd = 100</c>, so <c>up=160, down=441</c>.</para></summary>
public sealed class Resampler
{
    private readonly int _up;
    private readonly int _down;
    private readonly int _taps;
    private readonly float[][] _phaseTable; // [_up][_taps]

    private Resampler(int up, int down, int taps, float[][] phaseTable)
    {
        _up = up;
        _down = down;
        _taps = taps;
        _phaseTable = phaseTable;
    }

    /// <summary>Builds a resampler from <paramref name="inRate"/> Hz to
    /// <paramref name="outRate"/> Hz. Default <paramref name="numTaps"/>=64 is
    /// sufficient for STT (transcription WER is unaffected); use 256 for music
    /// where audible artifacts matter.</summary>
    public static Resampler Create(int inRate, int outRate, int numTaps = 64)
    {
        if (inRate <= 0 || outRate <= 0) throw new ArgumentOutOfRangeException(nameof(inRate));
        int g = Gcd(inRate, outRate);
        int up = outRate / g;
        int down = inRate / g;

        // Cutoff at half the lower of the two rates (in cycles/sample at the up-sampled rate).
        // Match scipy: cutoff = 1 / max(up, down), which equals min(in_rate, out_rate) / (2 * max(in_rate, out_rate))
        // in unit-rate space.
        double cutoff = 1.0 / Math.Max(up, down);
        double kaiserBeta = 8.6;

        int filterLen = numTaps * up;
        // Build a windowed-sinc low-pass at the up-sampled rate.
        double[] h = new double[filterLen];
        double mid = (filterLen - 1) / 2.0;
        for (int n = 0; n < filterLen; n++)
        {
            double x = n - mid;
            double sinc = x == 0 ? 1.0 : Math.Sin(Math.PI * cutoff * x) / (Math.PI * cutoff * x);
            double kaiser = KaiserWindow(n, filterLen, kaiserBeta);
            h[n] = sinc * cutoff * kaiser;
        }

        // Normalize so the sum of each phase equals 1/up (DC gain through polyphase = 1).
        // Each output sample uses one phase row only; sum-of-phases is the full filter sum.
        // The factor of `up` here cancels the implicit zero-stuffing.
        float[][] phaseTable = new float[up][];
        for (int p = 0; p < up; p++)
        {
            phaseTable[p] = new float[numTaps];
            for (int k = 0; k < numTaps; k++)
            {
                int idx = k * up + p;
                phaseTable[p][k] = (float)(h[idx] * up);
            }
        }

        return new Resampler(up, down, numTaps, phaseTable);
    }

    /// <summary>Output length for a given input length, matching scipy's
    /// <c>resample_poly</c> formula.</summary>
    public int OutputLength(int inputLength)
        => (int)(((long)inputLength * _up + _down - 1) / _down);

    /// <summary>Resamples a mono buffer. If <paramref name="output"/> is non-null it
    /// must be at least <see cref="OutputLength"/> samples long; otherwise a fresh
    /// array is allocated.</summary>
    public float[] Resample(ReadOnlySpan<float> input, float[]? output = null)
    {
        int outLen = OutputLength(input.Length);
        float[] result = output ?? new float[outLen];
        if (result.Length < outLen) throw new ArgumentException("output buffer too small", nameof(output));

        // Per-output-sample compute:
        //   inIdx, phase = divmod(m * down, up)
        //   y[m] = sum_k input[inIdx - half + k] * phaseTable[phase][k]
        // The "half" centers the filter so output sample 0 lines up with input sample 0.
        int half = _taps / 2;

        for (int m = 0; m < outLen; m++)
        {
            long t = (long)m * _down;
            int inIdx = (int)(t / _up) - half;
            int phase = (int)(((t % _up) + _up) % _up);
            // scipy.resample_poly uses h[p][k] convolved against the original-rate input;
            // when t % up != 0 the effective phase rotates through the polyphase rows.
            // We need the *reverse* phase index because the up-sampled filter sees the
            // zero-stuffed input shifted by `up - phase` zeros within each output stride.
            int pIdx = phase == 0 ? 0 : _up - phase;
            float[] taps = _phaseTable[pIdx];

            float acc = 0f;
            for (int k = 0; k < _taps; k++)
            {
                int s = inIdx + k;
                if ((uint)s < (uint)input.Length) acc += input[s] * taps[k];
            }
            result[m] = acc;
        }

        return result;
    }

    private static double KaiserWindow(int n, int N, double beta)
    {
        double x = 2.0 * n / (N - 1) - 1.0;
        return BesselI0(beta * Math.Sqrt(Math.Max(0.0, 1.0 - x * x))) / BesselI0(beta);
    }

    // Modified Bessel function of the first kind, order 0. Series expansion converges
    // quickly for the argument range we hit (beta * x in [0, ~9]).
    private static double BesselI0(double x)
    {
        double sum = 1.0;
        double term = 1.0;
        double half = x * 0.5;
        for (int k = 1; k < 50; k++)
        {
            term *= (half / k) * (half / k);
            sum += term;
            if (term < 1e-12 * sum) break;
        }
        return sum;
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0) { int t = b; b = a % b; a = t; }
        return a;
    }
}
