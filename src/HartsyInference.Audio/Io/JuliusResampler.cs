namespace HartsyInference.Audio.Io;

/// <summary>Reproduces <c>julius.resample_frac</c> (Smith's windowed-sinc resampler, 24 zero crossings, 0.945 roll-off) as
/// audiocraft's <c>convert_audio</c> applies it: both rates are reduced by their gcd, one Hann-squared sinc kernel is built per
/// output phase and normalised to unit sum, the signal is edge-replicated by <c>width</c> in front and <c>width + old</c>
/// behind, and every <c>old</c> input samples yield <c>new</c> outputs. The output holds <c>floor(new * length / old)</c> samples.</summary>
public static class JuliusResampler
{
    /// <summary>Default zero crossings kept in the sinc filter.</summary>
    public const int DefaultZeros = 24;

    /// <summary>Default roll-off of the low-pass cut-off relative to the lower Nyquist.</summary>
    public const double DefaultRolloff = 0.945;

    /// <summary>Resamples mono <paramref name="input"/> from <paramref name="inRate"/> to <paramref name="outRate"/> Hz.</summary>
    public static float[] Resample(ReadOnlySpan<float> input, int inRate, int outRate, int zeros = DefaultZeros,
        double rolloff = DefaultRolloff)
    {
        if (inRate <= 0 || outRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inRate), "Sample rates must be positive.");
        }

        if (inRate == outRate)
        {
            return input.ToArray();
        }

        int gcd = Gcd(inRate, outRate);
        int old = inRate / gcd, fresh = outRate / gcd;
        double sr = Math.Min(old, fresh) * rolloff;
        int width = (int)Math.Ceiling(zeros * (double)old / sr);
        int kernelLength = 2 * width + old;
        float[] kernels = new float[fresh * kernelLength];
        for (int phase = 0; phase < fresh; phase++)
        {
            double sum = 0.0;
            double[] taps = new double[kernelLength];
            for (int k = 0; k < kernelLength; k++)
            {
                double t = (-(double)phase / fresh + (double)(k - width) / old) * sr;
                t = Math.Clamp(t, -zeros, zeros) * Math.PI;
                double window = Math.Cos(t / zeros / 2.0);
                double sinc = t == 0.0 ? 1.0 : Math.Sin(t) / t;
                taps[k] = sinc * window * window;
                sum += taps[k];
            }

            for (int k = 0; k < kernelLength; k++)
            {
                kernels[phase * kernelLength + k] = (float)(taps[k] / sum);
            }
        }

        int length = input.Length;
        int blocks = length / old + 1;
        float[] padded = new float[width + length + width + old];
        for (int i = 0; i < padded.Length; i++)
        {
            padded[i] = input[Math.Clamp(i - width, 0, length - 1)];
        }

        int outLength = (int)((long)fresh * length / old);
        float[] result = new float[outLength];
        for (int block = 0; block < blocks; block++)
        {
            for (int phase = 0; phase < fresh; phase++)
            {
                int at = block * fresh + phase;
                if (at >= outLength)
                {
                    break;
                }

                float acc = 0f;
                int src = block * old;
                int row = phase * kernelLength;
                for (int k = 0; k < kernelLength; k++)
                {
                    acc += kernels[row + k] * padded[src + k];
                }

                result[at] = acc;
            }
        }

        return result;
    }

    private static int Gcd(int a, int b) => b == 0 ? a : Gcd(b, a % b);
}
