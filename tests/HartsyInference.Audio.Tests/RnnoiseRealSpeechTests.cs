using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>RNNoise on real speech the way the voice front end runs it: 16 kHz in and out through
/// <see cref="RnnoiseStream"/>'s resamplers, int16-scaled. Parity with the C build is
/// <c>RnnoiseParityTests</c>' job, at 48 kHz; this checks what a caller relies on at the rate it actually uses —
/// noise goes down, the speech survives, and the stream's declared latency is where the output really lands.
///
/// <para>jfk.wav (committed) with white noise mixed in at 5 dB SNR. Needs the converted weights, from
/// <c>HARTSYINFERENCE_RNNOISE_WEIGHTS</c> or the wake model root under <c>HARTSYINFERENCE_MODELS_DIR</c>.</para></summary>
public sealed class RnnoiseRealSpeechTests(ITestOutputHelper log)
{
    private const int Rate = 16_000;
    private const int Chunk = 320;
    private const float Int16Scale = 32768f;

    [Fact]
    [Trait("Category", "Integration")]
    public void Stream16k_RemovesNoise_KeepsSpeech_AtTheDeclaredLatency()
    {
        string weightsPath = WeightsPath();
        string clipPath = Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(log.WriteLine, weightsPath, clipPath)) return;

        float[] clean = WavFile.Read(clipPath).ToMono();
        for (int i = 0; i < clean.Length; i++) clean[i] *= Int16Scale;
        float[] noisy = AddWhiteNoise(clean, snrDb: 5, seed: 1234);

        using RnnoiseWeights weights = LoadWeights(weightsPath);
        using CpuBackend backend = new();
        using RnnoiseStream stream = new(weights, Rate);
        int latency = stream.LatencySamples;
        float[] output = Run(stream, backend, noisy, latency);

        int lag = BestLag(clean, output, latency - 160, latency + 160);
        log.WriteLine($"declared latency {latency} samples, measured best alignment {lag}");
        Assert.Equal(latency, lag);

        // 20 ms frames ranked by the clean clip's level: the quietest fifth are the gaps between words, where the mix
        // is nearly all noise, and the loudest half are speech.
        int frames = clean.Length / Chunk;
        int[] byLevel = [.. Enumerable.Range(0, frames).OrderBy(f => Energy(clean, f * Chunk, Chunk))];
        double gapIn = 0, gapOut = 0, speechClean = 0, speechOut = 0;
        foreach (int f in byLevel[..(frames / 5)])
        {
            gapIn += Energy(noisy, f * Chunk, Chunk);
            gapOut += Energy(output, f * Chunk + lag, Chunk);
        }
        foreach (int f in byLevel[(frames / 2)..])
        {
            speechClean += Energy(clean, f * Chunk, Chunk);
            speechOut += Energy(output, f * Chunk + lag, Chunk);
        }
        double signal = 0, residualIn = 0, residualOut = 0;
        for (int i = 0; i < frames * Chunk; i++)
        {
            signal += Square(clean[i]);
            residualIn += Square(noisy[i] - clean[i]);
            residualOut += Square(output[i + lag] - clean[i]);
        }
        double snrIn = Db(signal / residualIn), snrOut = Db(signal / residualOut);
        double gapSuppression = Db(gapOut / gapIn), speechKept = Db(speechOut / speechClean);
        log.WriteLine($"SNR {snrIn:F1} dB in, {snrOut:F1} dB out; gaps between words {gapSuppression:F1} dB; "
            + $"speech {speechKept:+0.0;-0.0} dB against the clean clip");

        Assert.True(snrOut > snrIn + 4, $"SNR only went from {snrIn:F1} dB to {snrOut:F1} dB");
        Assert.True(gapSuppression < -10, $"noise between words dropped by only {gapSuppression:F1} dB");
        Assert.True(Math.Abs(speechKept) < 2, $"speech level moved {speechKept:F1} dB");
    }

    /// <summary>Pushes the clip through in 20 ms chunks, then enough silence to flush the stream's latency, and
    /// returns the output with the same indexing: sample <c>i + latency</c> is the denoised input sample <c>i</c>.</summary>
    private static float[] Run(RnnoiseStream stream, CpuBackend backend, float[] input, int latency)
    {
        int tail = latency + stream.FrameSize;
        float[] padded = new float[input.Length + tail + Chunk];
        input.CopyTo(padded, 0);
        float[] output = new float[padded.Length + stream.FrameSize];
        float[] scratch = new float[Chunk + stream.FrameSize];
        int written = 0;
        for (int offset = 0; offset + Chunk <= padded.Length; offset += Chunk)
        {
            int n = stream.Process(backend, padded.AsSpan(offset, Chunk), scratch);
            scratch.AsSpan(0, n).CopyTo(output.AsSpan(written));
            written += n;
        }
        return output;
    }

    private static int BestLag(float[] reference, float[] output, int from, int to)
    {
        int best = from;
        double bestDot = double.MinValue;
        for (int lag = Math.Max(0, from); lag <= to; lag++)
        {
            double dot = 0;
            for (int i = 0; i < reference.Length; i++) dot += (double)reference[i] * output[i + lag];
            if (dot > bestDot)
            {
                bestDot = dot;
                best = lag;
            }
        }
        return best;
    }

    private static float[] AddWhiteNoise(float[] clean, double snrDb, int seed)
    {
        Random rng = new(seed);
        double[] noise = new double[clean.Length];
        double power = 0, signal = 0;
        for (int i = 0; i < noise.Length; i++)
        {
            // Box-Muller: Gaussian, so the level is a true RMS rather than a uniform distribution's.
            noise[i] = Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
            power += noise[i] * noise[i];
            signal += (double)clean[i] * clean[i];
        }
        double gain = Math.Sqrt(signal / power / Math.Pow(10, snrDb / 10));
        float[] noisy = new float[clean.Length];
        for (int i = 0; i < noisy.Length; i++) noisy[i] = (float)(clean[i] + gain * noise[i]);
        return noisy;
    }

    internal static RnnoiseWeights LoadWeights(string path)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        RnnoiseWeights weights = new();
        weights.Load(tensors);
        foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        return weights;
    }

    /// <summary>The file <c>WakeModelSet.LoadDenoiser</c> opens, unless overridden.</summary>
    internal static string WeightsPath() =>
        Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_WEIGHTS")
        ?? Path.Combine(TestPaths.ModelsDir, "audio", "wake", "denoise", "rnnoise.safetensors");

    private static double Energy(float[] x, int start, int count)
    {
        double sum = 0;
        for (int i = start; i < start + count; i++) sum += (double)x[i] * x[i];
        return sum;
    }

    private static double Square(float x) => (double)x * x;

    private static double Db(double ratio) => 10 * Math.Log10(ratio);
}
