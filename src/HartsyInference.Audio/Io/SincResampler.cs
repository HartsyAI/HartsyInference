namespace HartsyInference.Audio.Io;

/// <summary>Windowed-sinc resampler that reproduces <c>torchaudio.functional.resample</c>
/// (<c>resampling_method="sinc_interp_hann"</c>) step for step: the rates are reduced by their gcd, one kernel row is
/// built per output phase (<c>sinc(t · base_freq)</c> under a squared-cosine window of
/// <c>lowpass_filter_width</c> zero crossings, scaled by <c>base_freq / orig</c>), the signal is padded by
/// <c>width</c> in front and <c>width + orig</c> behind, and every <c>orig</c> input samples yield <c>new</c> outputs.
/// IndexTTS-2's reference pipeline runs <c>torchaudio.transforms.Resample(sr, 16000)</c> on the clip, so its 16 kHz
/// w2v-bert / CAM++ inputs only match when this exact filter is used; <see cref="Resampler"/> (scipy-style, Kaiser) differs
/// by a few percent of waveform energy.</summary>
public static class SincResampler
{
    /// <summary>The Kaiser-windowed sinc that stands in for librosa's default <c>soxr_hq</c> resampler, fitted to the frequency
    /// response measured from python-soxr 1.1.0: -6 dB at <see cref="SoxrHqRolloff"/> of the lower rate's Nyquist, window shape
    /// <see cref="SoxrHqKaiserBeta"/>, <see cref="SoxrHqHalfWidth"/> samples of the lower rate on each side.</summary>
    public const double SoxrHqRolloff = 0.956872;
    public const double SoxrHqKaiserBeta = 13.04;
    public const double SoxrHqHalfWidth = 97;

    /// <summary>The window's half-width counted in zero crossings of the rolled-off sinc.</summary>
    private const double SoxrHqZeroCrossings = SoxrHqHalfWidth * SoxrHqRolloff;

    /// <summary>The stand-in for <c>librosa.resample(res_type="soxr_hq")</c> (also <c>librosa.load(sr=...)</c>). It is a close
    /// approximation, not a port: soxr interpolates its coefficients for ratios with many phases (44.1 kHz to 16 kHz), and
    /// that is not reproduced. Pass-through copy when the rates match.</summary>
    public static float[] ResampleSoxrHqStandIn(ReadOnlySpan<float> input, int inRate, int outRate) =>
        Run(input, inRate, outRate, SoxrHqZeroCrossings, SoxrHqRolloff, SoxrHqKaiserBeta);

    /// <summary>Whether <see cref="ResampleSoxrHqStandIn"/> accepts the rate pair: a pair whose reduced ratio needs too large a kernel table is refused.</summary>
    public static bool SupportsSoxrHqStandIn(int inRate, int outRate) =>
        inRate > 0 && outRate > 0 && (inRate == outRate || KernelShape(inRate, outRate, SoxrHqZeroCrossings, SoxrHqRolloff).Entries <= MaxKernelEntries);

    /// <summary>The torchaudio defaults: 6 zero crossings, 0.99 roll-off.</summary>
    public const int DefaultLowpassFilterWidth = 6;
    public const double DefaultRolloff = 0.99;

    /// <summary>Largest per-phase kernel table (<c>new × kernelLen</c> floats, 128 MB) the resampler will build.</summary>
    private const long MaxKernelEntries = 32_000_000;

    /// <summary>Resamples mono <paramref name="input"/> from <paramref name="inRate"/> to <paramref name="outRate"/> Hz.
    /// Output length is <c>ceil(outRate · length / inRate)</c>, as in torchaudio.</summary>
    public static float[] Resample(ReadOnlySpan<float> input, int inRate, int outRate,
        int lowpassFilterWidth = DefaultLowpassFilterWidth, double rolloff = DefaultRolloff) =>
        Run(input, inRate, outRate, lowpassFilterWidth, rolloff, kaiserBeta: null);

    /// <summary>The shared polyphase loop. The window spans <paramref name="zeroCrossings"/> on each side: torchaudio's squared
    /// cosine when <paramref name="kaiserBeta"/> is null, otherwise a Kaiser window of that shape.</summary>
    private static float[] Run(ReadOnlySpan<float> input, int inRate, int outRate, double zeroCrossings, double rolloff, double? kaiserBeta)
    {
        if (inRate <= 0 || outRate <= 0) throw new ArgumentOutOfRangeException(nameof(inRate));
        if (inRate == outRate) return input.ToArray();
        if (input.Length == 0) return [];

        (int orig, int @new, double baseFreq, int width, int kernelLen, long kernelEntries) =
            KernelShape(inRate, outRate, zeroCrossings, rolloff);

        // kernels[p, k] for output phase p and tap k: t = (-p / new + (k - width) / orig) * baseFreq.
        if (kernelEntries > MaxKernelEntries)
            throw new ArgumentException($"Resampling {inRate} -> {outRate} Hz needs a {kernelEntries:N0}-entry kernel table (limit {MaxKernelEntries:N0}); use a rate pair with a smaller reduced ratio.", nameof(outRate));
        float[] kernels = new float[kernelEntries];
        double scale = baseFreq / orig;
        double kaiserNorm = kaiserBeta is null ? 0.0 : 1.0 / Resampler.BesselI0(kaiserBeta.Value);
        for (int p = 0; p < @new; p++)
        {
            for (int k = 0; k < kernelLen; k++)
            {
                double t = (-(double)p / @new + (double)(k - width) / orig) * baseFreq;
                t = Math.Clamp(t, -zeroCrossings, zeroCrossings);
                double window;
                if (kaiserBeta is double beta)
                {
                    // A Kaiser window does not reach zero at its edge, so taps at or beyond the edge are dropped.
                    double r = t / zeroCrossings;
                    window = Math.Abs(r) < 1.0 ? Resampler.BesselI0(beta * Math.Sqrt(1.0 - r * r)) * kaiserNorm : 0.0;
                }
                else
                {
                    window = Math.Cos(t * Math.PI / zeroCrossings / 2.0);
                    window *= window;
                }
                double tp = t * Math.PI;
                double sinc = tp == 0.0 ? 1.0 : Math.Sin(tp) / tp;
                kernels[(long)p * kernelLen + k] = (float)(sinc * window * scale);
            }
        }

        // Padded signal: `width` zeros in front, `width + orig` behind (torchaudio's F.pad(waveform, (width, width + orig))).
        long paddedLen = (long)input.Length + 2L * width + orig;
        float[] padded = new float[paddedLen];
        input.CopyTo(padded.AsSpan(width));

        int frames = (int)((paddedLen - kernelLen) / orig + 1);
        int targetLength = (int)Math.Ceiling((double)@new * input.Length / orig);
        float[] result = new float[targetLength];

        float[] paddedLocal = padded;
        int newLocal = @new;
        Parallel.For(0, frames, frame =>
        {
            int start = frame * orig;
            ReadOnlySpan<float> window = paddedLocal.AsSpan(start, kernelLen);
            for (int p = 0; p < newLocal; p++)
            {
                long outIdx = (long)frame * newLocal + p;
                if (outIdx >= targetLength) break;
                ReadOnlySpan<float> kernel = kernels.AsSpan((int)((long)p * kernelLen), kernelLen);
                float acc = 0f;
                for (int k = 0; k < kernelLen; k++) acc += window[k] * kernel[k];
                result[outIdx] = acc;
            }
        });
        return result;
    }

    private static (int Orig, int New, double BaseFreq, int Width, int KernelLen, long Entries) KernelShape(int inRate, int outRate,
        double zeroCrossings, double rolloff)
    {
        int g = Gcd(inRate, outRate);
        int orig = inRate / g, @new = outRate / g;
        double baseFreq = Math.Min(orig, @new) * rolloff;
        int width = (int)Math.Ceiling(zeroCrossings * orig / baseFreq);
        int kernelLen = 2 * width + orig;
        return (orig, @new, baseFreq, width, kernelLen, (long)@new * kernelLen);
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }
}
