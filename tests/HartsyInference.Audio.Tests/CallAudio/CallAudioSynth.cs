using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Io;
using HartsyInference.Tests.Common;

namespace HartsyInference.Audio.Tests.CallAudio;

/// <summary>Synthetic telephone audio at 16 kHz for the detector tests: keys, tone cadences, noise, speech-like
/// babble, and the in-repo speech clips.</summary>
internal sealed class CallAudioSynth
{
    public const int Rate = 16_000;

    private static readonly double[] Low = [697, 770, 852, 941];
    private static readonly double[] High = [1209, 1336, 1477, 1633];
    private static readonly string Keys = "123A456B789C*0#D";

    private readonly List<float> _samples = [];

    public int Count => _samples.Count;

    public static int Ms(int ms) => ms * Rate / 1000;

    public float[] ToArray() => [.. _samples];

    public CallAudioSynth Silence(int ms)
    {
        for (int i = 0; i < Ms(ms); i++)
        {
            _samples.Add(0f);
        }
        return this;
    }

    /// <summary>Sum of sines, each with its own amplitude.</summary>
    public CallAudioSynth Tones(int ms, params (double Hz, float Amplitude)[] components)
    {
        int n = Ms(ms);
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            foreach ((double hz, float amplitude) in components)
            {
                sum += amplitude * Math.Sin(2 * Math.PI * hz * i / Rate);
            }
            _samples.Add((float)sum);
        }
        return this;
    }

    public CallAudioSynth Key(char key, int ms, float amplitude = 0.25f, double highTwistDb = 0)
    {
        int index = Keys.IndexOf(key);
        float high = amplitude * (float)Math.Pow(10, highTwistDb / 20);
        return Tones(ms, (Low[index / 4], amplitude), (High[index % 4], high));
    }

    public CallAudioSynth Append(ReadOnlySpan<float> samples)
    {
        foreach (float s in samples)
        {
            _samples.Add(s);
        }
        return this;
    }

    public static float[] Noise(int samples, float rms, int seed)
    {
        uint state = DeterministicRng.Seed(seed);
        float[] noise = new float[samples];
        for (int i = 0; i < noise.Length; i++)
        {
            noise[i] = DeterministicRng.NextGaussian(ref state) * rms;
        }
        return noise;
    }

    public static float Rms(ReadOnlySpan<float> x)
    {
        double e = 0;
        foreach (float s in x)
        {
            e += (double)s * s;
        }
        return (float)Math.Sqrt(e / Math.Max(1, x.Length));
    }

    /// <summary>Adds white noise at <paramref name="snrDb"/> relative to the RMS of the non-silent part of the signal.</summary>
    public static float[] AddNoise(float[] signal, double snrDb, int seed)
    {
        double e = 0;
        int active = 0;
        foreach (float s in signal)
        {
            if (s != 0f)
            {
                e += (double)s * s;
                active++;
            }
        }
        float rms = (float)Math.Sqrt(e / Math.Max(1, active)) * (float)Math.Pow(10, -snrDb / 20);
        float[] noise = Noise(signal.Length, rms, seed);
        float[] mixed = new float[signal.Length];
        for (int i = 0; i < mixed.Length; i++)
        {
            mixed[i] = signal[i] + noise[i];
        }
        return mixed;
    }

    /// <summary>Voiced-speech-like babble: a glottal-pulse harmonic series whose pitch wanders, shaped by formants that
    /// sweep, so partials cross every DTMF frequency over time. Deterministic.</summary>
    public static float[] Babble(int samples, int seed, float rms)
    {
        uint state = DeterministicRng.Seed(seed);
        float[] x = new float[samples];
        double phase = 0;
        double f0 = 110 + 90 * DeterministicRng.NextUniform(ref state);
        double f1 = 500, f2 = 1500, f3 = 2500;
        double t1 = 700, t2 = 1800;
        for (int i = 0; i < samples; i++)
        {
            if (i % 1600 == 0)
            {
                f0 = Math.Clamp(f0 + 40 * (DeterministicRng.NextUniform(ref state) - 0.5), 80, 280);
                t1 = 300 + 700 * DeterministicRng.NextUniform(ref state);
                t2 = 900 + 1700 * DeterministicRng.NextUniform(ref state);
            }
            f1 += (t1 - f1) * 0.0004;
            f2 += (t2 - f2) * 0.0004;
            phase += f0 / Rate;
            phase -= Math.Floor(phase);
            double sum = 0;
            for (int h = 1; h * f0 < 4000; h++)
            {
                double f = h * f0;
                double gain = 1.0 / h + 1.5 * Resonance(f, f1, 90) + 1.2 * Resonance(f, f2, 130) + 0.6 * Resonance(f, f3, 200);
                sum += gain * Math.Sin(2 * Math.PI * h * phase);
            }
            x[i] = (float)sum;
        }
        float scale = rms / Math.Max(1e-9f, Rms(x));
        for (int i = 0; i < x.Length; i++)
        {
            x[i] *= scale;
        }
        return x;
    }

    private static double Resonance(double f, double center, double width) => 1.0 / (1.0 + Math.Pow((f - center) / width, 2));

    public static float[] Speech(string name)
    {
        string path = Path.Combine(RepoRoot.Path, "tests", "python-reference", name);
        WavFile.DecodedAudio audio = WavFile.Read(path);
        if (audio.SampleRate != Rate)
        {
            throw new InvalidOperationException($"{name} is {audio.SampleRate} Hz, expected {Rate}.");
        }
        return audio.ToMono();
    }

    public static float[] Jfk() => Speech(Path.Combine("silerovad_reference", "jfk.wav"));

    public static float[] Alexa() => Speech(Path.Combine("wake_reference", "alexa_16k.wav"));

    /// <summary>Scales <paramref name="x"/> so its RMS is <paramref name="rms"/>.</summary>
    public static float[] Normalize(float[] x, float rms)
    {
        float scale = rms / Math.Max(1e-9f, Rms(x));
        float[] y = new float[x.Length];
        for (int i = 0; i < y.Length; i++)
        {
            y[i] = x[i] * scale;
        }
        return y;
    }
}
