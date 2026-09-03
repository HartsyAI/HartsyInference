using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>Real-weight parity for the RNNoise port against the upstream C implementation.
///
/// <para>Env-gated and skips cleanly, per the Integration tier: it needs converted weights and a reference
/// waveform produced by the C build, neither of which is committed.</para>
///
/// <para><b>Regenerating the fixtures</b> — build upstream (<c>xiph/rnnoise</c>) after
/// <c>./download_model.sh</c>, then:</para>
/// <code>
///   ./examples/rnnoise_demo input48k.raw reference48k.raw     # 48 kHz mono s16le, both files
///   python tools/convert_pth_to_safetensors.py models/rnnoise10Ga_12.pth -o rnnoise.safetensors
///   export HARTSYINFERENCE_RNNOISE_WEIGHTS=/path/to/rnnoise.safetensors
///   export HARTSYINFERENCE_RNNOISE_REF_DIR=/path/containing/input48k.raw+reference48k.raw
/// </code>
///
/// <para><b>On the tolerance.</b> Agreement is not bit-exact and cannot be: upstream's kiss_fft is mixed-radix
/// over 960 points where <see cref="Preprocessing.Fft"/> falls back to Bluestein, the high-pass is a recursive
/// biquad that accumulates the difference, and the pitch search takes an <i>integer</i> argmax over correlations
/// computed from those spectra. When a tie tips, that frame's comb filter mixes a different harmonic structure
/// and the error spikes for a few frames before the gain smoothing reconverges. Measured over 10 s of real
/// speech: median per-frame error 0.04% of signal RMS, with 7 frames of 1014 above 5%. The assertions below are
/// therefore on the <b>distribution</b> — overall energy, and a median — rather than a max-abs bound, which
/// would only be testing whether a pitch tie happened to tip on this particular clip.</para></summary>
public sealed class RnnoiseParityTests
{
    private const int Frame = RnnoiseDenoiser.FrameSize;

    private static string? WeightsPath => Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_WEIGHTS");
    private static string? RefDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_REF_DIR");

    private static float[] ReadS16(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        float[] samples = new float[raw.Length / 2];
        // int16 scale, not +/-1: RNNoise's silence floor and log offsets are absolute.
        for (int i = 0; i < samples.Length; i++) samples[i] = BitConverter.ToInt16(raw, i * 2);
        return samples;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Denoiser_MatchesUpstreamC_OnRealSpeech()
    {
        string? weights = WeightsPath;
        string? dir = RefDir;
        if (weights is null || dir is null) return;   // tier-lint: guarded
        string inputPath = Path.Combine(dir, "input48k.raw");
        string referencePath = Path.Combine(dir, "reference48k.raw");
        if (!File.Exists(weights) || !File.Exists(inputPath) || !File.Exists(referencePath)) return;

        float[] input = ReadS16(inputPath);
        float[] reference = ReadS16(referencePath);

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(weights);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        using RnnoiseWeights shared = new RnnoiseWeights();
        shared.Load(tensors);
        foreach (Tensor t in tensors.Values) t.Dispose();
        using IBackend backend = new CpuBackend();
        using RnnoiseDenoiser denoiser = new RnnoiseDenoiser(shared);

        int frames = input.Length / Frame;
        float[] actual = new float[frames * Frame];
        for (int f = 0; f < frames; f++)
            denoiser.Process(backend, input.AsSpan(f * Frame, Frame), actual.AsSpan(f * Frame, Frame));

        // rnnoise_demo discards its first output frame, so its sample j is ours at j + Frame.
        int count = Math.Min(reference.Length, actual.Length - Frame);
        Assert.True(count > 48_000, $"need at least a second of reference audio, got {count} samples");

        double sumSqRef = 0, sumSqOut = 0;
        for (int i = 0; i < count; i++)
        {
            sumSqRef += (double)reference[i] * reference[i];
            sumSqOut += (double)actual[i + Frame] * actual[i + Frame];
        }
        double rmsRef = Math.Sqrt(sumSqRef / count);
        double rmsOut = Math.Sqrt(sumSqOut / count);

        // Same amount of noise removed, to within a couple of percent.
        Assert.True(Math.Abs(rmsOut - rmsRef) / rmsRef < 0.05,
            $"output RMS {rmsOut:F1} differs from reference {rmsRef:F1} by more than 5%");

        int frameCount = count / Frame;
        double[] frameError = new double[frameCount];
        for (int f = 0; f < frameCount; f++)
        {
            double sum = 0;
            for (int i = 0; i < Frame; i++)
            {
                double d = actual[f * Frame + Frame + i] - reference[f * Frame + i];
                sum += d * d;
            }
            // Normalized by the whole clip's RMS, so silent frames don't divide by ~0.
            frameError[f] = Math.Sqrt(sum / Frame) / rmsRef;
        }
        Array.Sort(frameError);
        double median = frameError[frameCount / 2];
        double p99 = frameError[(int)(frameCount * 0.99)];

        Assert.True(median < 0.01, $"median per-frame error {median:P2} exceeds 1% of signal RMS");
        Assert.True(p99 < 0.25, $"99th-percentile per-frame error {p99:P2} exceeds 25% of signal RMS");
    }

    /// <summary>Guards the input-scale contract independently of the reference waveform. At ±1 scale every frame
    /// falls under the absolute silence floor, the network never runs, and the denoiser degrades to a passthrough
    /// — which looks like "working" until you check whether anything was actually suppressed.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void Denoiser_Suppresses_Noise_At_Int16_Scale()
    {
        string? weights = WeightsPath;
        if (weights is null || !File.Exists(weights)) return;   // tier-lint: guarded

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(weights);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        using RnnoiseWeights shared = new RnnoiseWeights();
        shared.Load(tensors);
        foreach (Tensor t in tensors.Values) t.Dispose();
        using IBackend backend = new CpuBackend();
        using RnnoiseDenoiser denoiser = new RnnoiseDenoiser(shared);

        Random rng = new Random(4242);
        const int Frames = 150;
        float[] output = new float[Frame];
        double sumIn = 0, sumOut = 0;
        int counted = 0;
        for (int f = 0; f < Frames; f++)
        {
            float[] noise = new float[Frame];
            for (int i = 0; i < Frame; i++) noise[i] = (float)((rng.NextDouble() * 2 - 1) * 3000);
            denoiser.Process(backend, noise, output);
            if (f < 20) continue;   // let the GRUs and the gain smoother settle
            for (int i = 0; i < Frame; i++)
            {
                sumIn += (double)noise[i] * noise[i];
                sumOut += (double)output[i] * output[i];
            }
            counted++;
        }
        Assert.True(counted > 0);
        double suppression = 20 * Math.Log10(Math.Sqrt(sumOut / sumIn));
        Assert.True(suppression < -6.0, $"white noise suppressed by only {suppression:F1} dB");
    }
}
