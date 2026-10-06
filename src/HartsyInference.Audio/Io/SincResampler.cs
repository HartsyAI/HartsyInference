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
    /// <summary>The torchaudio defaults: 6 zero crossings, 0.99 roll-off.</summary>
    public const int DefaultLowpassFilterWidth = 6;
    public const double DefaultRolloff = 0.99;

    /// <summary>Resamples mono <paramref name="input"/> from <paramref name="inRate"/> to <paramref name="outRate"/> Hz.
    /// Output length is <c>ceil(outRate · length / inRate)</c>, as in torchaudio.</summary>
    public static float[] Resample(ReadOnlySpan<float> input, int inRate, int outRate,
        int lowpassFilterWidth = DefaultLowpassFilterWidth, double rolloff = DefaultRolloff)
    {
        if (inRate <= 0 || outRate <= 0) throw new ArgumentOutOfRangeException(nameof(inRate));
        if (inRate == outRate) return input.ToArray();
        if (input.Length == 0) return [];

        int g = Gcd(inRate, outRate);
        int orig = inRate / g, @new = outRate / g;
        double baseFreq = Math.Min(orig, @new) * rolloff;
        int width = (int)Math.Ceiling(lowpassFilterWidth * (double)orig / baseFreq);
        int kernelLen = 2 * width + orig;

        // kernels[p, k] for output phase p and tap k: t = (-p / new + (k - width) / orig) * baseFreq.
        float[] kernels = new float[(long)@new * kernelLen];
        double scale = baseFreq / orig;
        for (int p = 0; p < @new; p++)
        {
            for (int k = 0; k < kernelLen; k++)
            {
                double t = (-(double)p / @new + (double)(k - width) / orig) * baseFreq;
                t = Math.Clamp(t, -lowpassFilterWidth, lowpassFilterWidth);
                double window = Math.Cos(t * Math.PI / lowpassFilterWidth / 2.0);
                window *= window;
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

    private static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }
}
