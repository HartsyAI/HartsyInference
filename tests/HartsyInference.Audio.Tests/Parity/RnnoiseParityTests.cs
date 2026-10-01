using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>Real-weight parity for the RNNoise port against the upstream C implementation.
///
/// <para>Env-gated and skips cleanly, per the Integration tier: it needs converted weights and a reference
/// waveform produced by the C build, neither of which is committed.</para>
///
/// <para><b>Regenerating the fixtures</b> — build upstream (<c>xiph/rnnoise</c>) after
/// <c>./download_model.sh</c> with <c>./configure --enable-dnn-debug-float</c>: that build runs every layer in
/// float from the same checkpoint the weights are converted from, so it is the like-for-like reference. A stock
/// <c>./configure</c> runs conv2 and the GRUs on int8 copies, and differs from the float build by about as much
/// as this port differs from it — a comparison against it measures upstream's quantization, not the port. Then:</para>
/// <code>
///   ./examples/rnnoise_demo input48k.raw reference48k.raw     # 48 kHz mono s16le, both files
///   python tools/convert_rnnoise.py rnnoise_data-*.tar.gz rnnoise.safetensors   # or RnnoiseInstaller
///   export HARTSYINFERENCE_RNNOISE_WEIGHTS=/path/to/rnnoise.safetensors
///   export HARTSYINFERENCE_RNNOISE_REF_DIR=/path/containing/input48k.raw+reference48k.raw
/// </code>
///
/// <para><b>On the tolerance.</b> Agreement is not bit-exact and cannot be. The FFT is a port of upstream's
/// kiss_fft (<see cref="Preprocessing.FftPlan"/>), but the spectrum is scaled after the transform rather than
/// before it, the network's products sum in a different order, and upstream approximates tanh and sigmoid; the
/// high-pass is a recursive biquad that accumulates any difference, and the pitch search takes an <i>integer</i>
/// argmax over correlations computed from those spectra. When a tie tips, that frame's comb filter mixes a different harmonic structure
/// and the error spikes for a few frames before the gain smoothing reconverges. The assertions below are
/// therefore on the <b>distribution</b> — overall energy, and a median — rather than a max-abs bound, which
/// would only be testing whether a pitch tie happened to tip on this particular clip. Measured distributions are
/// in docs/Checklists/PARITY_VERIFICATION.md.</para></summary>
public sealed class RnnoiseParityTests(ITestOutputHelper log)
{
    private const int Frame = RnnoiseDenoiser.FrameSize;

    private static string? WeightsPath => Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_WEIGHTS");
    private static string? RefDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_REF_DIR");

    /// <summary>Output of upstream's default build, whose conv2 and GRUs run on int8: the reference for
    /// <see cref="RnnoisePrecision.Int8"/>. Same file names as <see cref="RefDir"/>, from a stock <c>./configure</c>
    /// build of the same tree.</summary>
    private static string? Int8RefDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_INT8_REF_DIR");

    /// <summary>The int8 tables; by default <see cref="RnnoiseInt8Tables.FileName"/> beside the weights.</summary>
    private static string? Int8TablesPath => Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_INT8_TABLES");

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
        if (!File.Exists(weights) || !HasFixture(dir)) return;

        using RnnoiseWeights shared = RnnoiseWeights.LoadFile(weights);
        CompareWithReference(shared, dir, "F32 port vs upstream's float build");
    }

    /// <summary>The int8 port against upstream's default (int8) build. Their int8 products are the same exact sums
    /// of the same tables and codes, so what remains is what separates the F32 port from upstream's float build:
    /// conv1 and the heads sum in a different order, the activations are exact here where upstream approximates tanh
    /// and sigmoid, and the front end (FFT scaling, pitch search) differs as described above. A slightly different
    /// activation can move a uint8 code by one step, so the int8 path cannot be closer than that.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void Denoiser_Int8_MatchesUpstreamsDefaultBuild_OnRealSpeech()
    {
        string? weights = WeightsPath;
        string? dir = Int8RefDir;
        if (weights is null || dir is null) return;   // tier-lint: guarded
        if (!File.Exists(weights) || !HasFixture(dir)) return;
        string tables = Int8TablesPath
            ?? Path.Combine(Path.GetDirectoryName(Path.GetFullPath(weights))!, RnnoiseInt8Tables.FileName);
        if (!File.Exists(tables)) return;

        using RnnoiseWeights shared = RnnoiseWeights.LoadFile(weights, RnnoisePrecision.Int8, tables);
        Assert.Equal(RnnoisePrecision.Int8, shared.Precision);
        CompareWithReference(shared, dir, "int8 port vs upstream's default (int8) build");

        // How far apart upstream's own two builds are, on the same input, for scale.
        if (RefDir is string floatDir && HasFixture(floatDir)
            && File.ReadAllBytes(Path.Combine(floatDir, "input48k.raw")).AsSpan()
                .SequenceEqual(File.ReadAllBytes(Path.Combine(dir, "input48k.raw"))))
        {
            float[] floatBuild = ReadS16(Path.Combine(floatDir, "reference48k.raw"));
            float[] int8Build = ReadS16(Path.Combine(dir, "reference48k.raw"));
            (double median, double p99, double max) = FrameErrors(floatBuild, 0, int8Build);
            log.WriteLine($"for scale, upstream's float build vs its int8 build: per-frame error median {median:P3}, "
                + $"p99 {p99:P2}, max {max:P2}");
        }
    }

    private static bool HasFixture(string dir) =>
        File.Exists(Path.Combine(dir, "input48k.raw")) && File.Exists(Path.Combine(dir, "reference48k.raw"));

    private void CompareWithReference(RnnoiseWeights shared, string dir, string label)
    {
        float[] input = ReadS16(Path.Combine(dir, "input48k.raw"));
        float[] reference = ReadS16(Path.Combine(dir, "reference48k.raw"));
        using IBackend backend = new CpuBackend();
        using RnnoiseDenoiser denoiser = new RnnoiseDenoiser(shared);

        int frames = input.Length / Frame;
        float[] actual = new float[frames * Frame];
        for (int f = 0; f < frames; f++)
            denoiser.Process(backend, input.AsSpan(f * Frame, Frame), actual.AsSpan(f * Frame, Frame));

        // rnnoise_demo discards its first output frame, so its sample j is ours at j + Frame.
        int count = Math.Min(reference.Length, actual.Length - Frame);
        Assert.True(count > 48_000, $"need at least a second of reference audio, got {count} samples");

        double rmsRef = Rms(reference, 0, count);
        double rmsOut = Rms(actual, Frame, count);

        // Same amount of noise removed, to within a couple of percent.
        Assert.True(Math.Abs(rmsOut - rmsRef) / rmsRef < 0.05,
            $"output RMS {rmsOut:F1} differs from reference {rmsRef:F1} by more than 5%");

        (double median, double p99, double max) = FrameErrors(actual, Frame, reference);
        log.WriteLine($"{label}: {count / Frame} frames, RMS out {rmsOut:F1} vs reference {rmsRef:F1}; per-frame error "
            + $"median {median:P3}, p99 {p99:P2}, max {max:P2}");

        Assert.True(median < 0.01, $"median per-frame error {median:P2} exceeds 1% of signal RMS");
        Assert.True(p99 < 0.25, $"99th-percentile per-frame error {p99:P2} exceeds 25% of signal RMS");
    }

    private static double Rms(float[] x, int offset, int count)
    {
        double sum = 0;
        for (int i = 0; i < count; i++) sum += (double)x[offset + i] * x[offset + i];
        return Math.Sqrt(sum / count);
    }

    /// <summary>Per-frame RMS difference between <paramref name="actual"/> (from <paramref name="offset"/>) and
    /// <paramref name="reference"/>, normalized by the reference clip's RMS so silent frames do not divide by ~0.</summary>
    private static (double Median, double P99, double Max) FrameErrors(float[] actual, int offset, float[] reference)
    {
        int count = Math.Min(reference.Length, actual.Length - offset);
        double rmsRef = Rms(reference, 0, count);
        int frameCount = count / Frame;
        double[] frameError = new double[frameCount];
        for (int f = 0; f < frameCount; f++)
        {
            double sum = 0;
            for (int i = 0; i < Frame; i++)
            {
                double d = actual[offset + f * Frame + i] - reference[f * Frame + i];
                sum += d * d;
            }
            frameError[f] = Math.Sqrt(sum / Frame) / rmsRef;
        }
        Array.Sort(frameError);
        return (frameError[frameCount / 2], frameError[(int)(frameCount * 0.99)], frameError[^1]);
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
        log.WriteLine($"white noise at int16 scale suppressed by {suppression:F1} dB");
        Assert.True(suppression < -6.0, $"white noise suppressed by only {suppression:F1} dB");
    }
}
