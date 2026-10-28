using System.Numerics;
using HartsyInference.Audio.Io;

namespace HartsyInference.Audio.Dsp.Channel;

/// <summary>Telephone voice channel: 300–3400 Hz band-pass, 24 kHz to 8 kHz resampling, then a G.711 round trip.</summary>
/// <remarks>The band-pass is a 6th-order Butterworth prototype, which is a 12th-order band-pass (six biquads).
/// Zero-phase forward-backward filtering squares the magnitude response, so the nominal 300 and 3400 Hz edges sit at
/// −6 dB. The filter runs in float64; the resampler is the shared float32 polyphase <see cref="Resampler"/>.</remarks>
public static class TelephoneChannel
{
    private const int InputRate = 24000;
    private const int FilterOrder = 6;
    private const int SectionCount = FilterOrder;
    private const int PadLength = 3 * (2 * SectionCount + 1);
    private const double LowHz = 300.0;
    private const double HighHz = 3400.0;

    private static readonly Resampler _resampler = Resampler.Create(InputRate, 8000);
    private static readonly double[] _sections = DesignBandPass();
    private static readonly double[] _initialStates = SteadyStateStates(_sections);

    /// <summary>Runs the channel and returns the 8 kHz PCM a G.711 leg delivers, each sample decoded from its code.</summary>
    /// <param name="pcm24k">Mono PCM in [-1, 1] at 24 kHz.</param>
    /// <param name="codec">The G.711 law to round-trip through.</param>
    /// <param name="sampleRate">Must be 24000.</param>
    /// <exception cref="ArgumentOutOfRangeException">The sample rate is not 24000 or the codec is not defined.</exception>
    /// <exception cref="ArgumentException">The input holds a NaN or infinite sample.</exception>
    public static float[] Process(ReadOnlySpan<float> pcm24k, TelephoneCodec codec, int sampleRate = InputRate)
    {
        ValidateArguments(codec, sampleRate);
        if (pcm24k.Length == 0)
        {
            return [];
        }

        short[] samples = Narrowband(pcm24k);
        byte[] encoded = new byte[samples.Length];
        G711.Encode(samples, encoded, ToLaw(codec));
        short[] decoded = new short[encoded.Length];
        G711.Decode(encoded, decoded, ToLaw(codec));
        float[] result = new float[decoded.Length];
        for (int i = 0; i < decoded.Length; i++)
        {
            result[i] = decoded[i] / 32768f;
        }
        return result;
    }

    /// <summary>Runs the channel and returns the raw G.711 bytes, one per 8 kHz sample.</summary>
    /// <inheritdoc cref="Process(ReadOnlySpan{float}, TelephoneCodec, int)"/>
    public static byte[] Encode(ReadOnlySpan<float> pcm24k, TelephoneCodec codec, int sampleRate = InputRate)
    {
        ValidateArguments(codec, sampleRate);
        if (pcm24k.Length == 0)
        {
            return [];
        }

        short[] samples = Narrowband(pcm24k);
        byte[] encoded = new byte[samples.Length];
        G711.Encode(samples, encoded, ToLaw(codec));
        return encoded;
    }

    private static void ValidateArguments(TelephoneCodec codec, int sampleRate)
    {
        if (sampleRate != InputRate)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "The telephone channel accepts 24000 Hz input only.");
        }
        if (codec is not (TelephoneCodec.MuLaw or TelephoneCodec.ALaw))
        {
            throw new ArgumentOutOfRangeException(nameof(codec), codec, "Unknown telephone codec.");
        }
    }

    private static G711Law ToLaw(TelephoneCodec codec) => codec == TelephoneCodec.MuLaw ? G711Law.MuLaw : G711Law.ALaw;

    /// <summary>Band-pass, resample to 8 kHz, clip and quantize to 16-bit PCM.</summary>
    private static short[] Narrowband(ReadOnlySpan<float> pcm24k)
    {
        double[] band = BandPass(pcm24k);
        float[] narrow = new float[band.Length];
        for (int i = 0; i < band.Length; i++)
        {
            narrow[i] = (float)band[i];
        }

        float[] resampled = _resampler.Resample(narrow);
        short[] samples = new short[resampled.Length];
        for (int i = 0; i < resampled.Length; i++)
        {
            double clipped = Math.Clamp((double)resampled[i], -1.0, 1.0);
            samples[i] = (short)Math.Clamp(Math.Round(clipped * 32768.0), -32768.0, 32767.0);
        }
        return samples;
    }

    /// <summary>Zero-phase band-pass as scipy's sosfiltfilt: odd extension, forward pass, reverse, backward pass.</summary>
    /// <remarks>Inputs shorter than <see cref="PadLength"/> use the largest odd extension they allow (length − 1), so an
    /// input of one sample is filtered with no extension at all.</remarks>
    private static double[] BandPass(ReadOnlySpan<float> pcm)
    {
        int n = pcm.Length;
        int padLength = Math.Min(PadLength, n - 1);
        double[] ext = new double[n + 2 * padLength];
        for (int i = 0; i < n; i++)
        {
            if (!float.IsFinite(pcm[i]))
            {
                throw new ArgumentException("The input holds a NaN or infinite sample.", nameof(pcm));
            }
            ext[padLength + i] = pcm[i];
        }

        double first = ext[padLength];
        double last = ext[padLength + n - 1];
        for (int i = 0; i < padLength; i++)
        {
            ext[padLength - 1 - i] = 2.0 * first - ext[padLength + 1 + i];
            ext[padLength + n + i] = 2.0 * last - ext[padLength + n - 2 - i];
        }

        FilterSections(ext);
        Array.Reverse(ext);
        FilterSections(ext);
        Array.Reverse(ext);

        double[] result = new double[n];
        Array.Copy(ext, padLength, result, 0, n);
        return result;
    }

    /// <summary>Applies every biquad in turn over the whole buffer, starting each from its steady state for a
    /// constant input equal to the buffer's first sample.</summary>
    private static void FilterSections(double[] y)
    {
        double x0 = y[0];
        for (int s = 0; s < SectionCount; s++)
        {
            int c = 5 * s;
            double b0 = _sections[c];
            double b1 = _sections[c + 1];
            double b2 = _sections[c + 2];
            double a1 = _sections[c + 3];
            double a2 = _sections[c + 4];
            double z0 = _initialStates[2 * s] * x0;
            double z1 = _initialStates[2 * s + 1] * x0;
            for (int i = 0; i < y.Length; i++)
            {
                double x = y[i];
                double yi = b0 * x + z0;
                z0 = b1 * x - a1 * yi + z1;
                z1 = b2 * x - a2 * yi;
                y[i] = yi;
            }
        }
    }

    /// <summary>Bilinear design of the band-pass: prewarped edges, lowpass-to-bandpass on the unit-cutoff prototype,
    /// then one biquad per conjugate pole pair with the numerator (z − 1)(z + 1).</summary>
    /// <returns>Five coefficients per section (b0, b1, b2, a1, a2), with the overall gain folded into section 0.</returns>
    private static double[] DesignBandPass()
    {
        const int n = FilterOrder;
        double fs2 = 2.0 * InputRate;
        double wl = fs2 * Math.Tan(Math.PI * LowHz / InputRate);
        double wh = fs2 * Math.Tan(Math.PI * HighHz / InputRate);
        double bw = wh - wl;
        double w0Sq = wl * wh;

        Complex[] analog = new Complex[2 * n];
        for (int k = 0; k < n; k++)
        {
            Complex prototype = -Complex.FromPolarCoordinates(1.0, Math.PI * (2 * k - n + 1) / (2.0 * n));
            Complex q = prototype * (bw / 2.0);
            Complex d = Complex.Sqrt(q * q - w0Sq);
            analog[2 * k] = q + d;
            analog[2 * k + 1] = q - d;
        }

        Complex denominator = Complex.One;
        Complex[] digital = new Complex[2 * n];
        for (int i = 0; i < 2 * n; i++)
        {
            digital[i] = (fs2 + analog[i]) / (fs2 - analog[i]);
            denominator *= fs2 - analog[i];
        }
        double gain = Math.Pow(bw, n) * (Complex.Pow(fs2, n) / denominator).Real;

        double[] sections = new double[5 * n];
        int count = 0;
        for (int i = 0; i < 2 * n; i++)
        {
            if (digital[i].Imaginary <= 0.0)
            {
                continue;
            }
            int c = 5 * count;
            sections[c] = 1.0;
            sections[c + 1] = 0.0;
            sections[c + 2] = -1.0;
            sections[c + 3] = -2.0 * digital[i].Real;
            sections[c + 4] = digital[i].Real * digital[i].Real + digital[i].Imaginary * digital[i].Imaginary;
            count++;
        }
        if (count != n)
        {
            throw new InvalidOperationException($"Band-pass design produced {count} conjugate pole pairs, expected {n}.");
        }

        sections[0] = gain;
        sections[2] = -gain;
        return sections;
    }

    /// <summary>Steady-state states of each section for a unit step through the cascade, as scipy's sosfilt_zi.</summary>
    private static double[] SteadyStateStates(double[] sections)
    {
        double[] states = new double[2 * SectionCount];
        double scale = 1.0;
        for (int s = 0; s < SectionCount; s++)
        {
            int c = 5 * s;
            double b0 = sections[c];
            double b1 = sections[c + 1];
            double b2 = sections[c + 2];
            double a1 = sections[c + 3];
            double a2 = sections[c + 4];
            double dcGain = (b0 + b1 + b2) / (1.0 + a1 + a2);
            states[2 * s] = scale * (b1 + b2 - (a1 + a2) * dcGain);
            states[2 * s + 1] = scale * (b2 - a2 * dcGain);
            scale *= dcGain;
        }
        return states;
    }
}
