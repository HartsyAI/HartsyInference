using HartsyInference.Audio.Models.Vits;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Onnx;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numeric parity of the C# VITS synthesizer against onnxruntime for the same Piper voice. With the noise
/// scales zeroed (<c>noise_scale=0, length_scale=1, noise_w=0</c>) both implementations are fully deterministic, so the
/// audio must match closely — this proves the weight loading + VITS forward math, independent of the RNG. The
/// reference (<c>Fixtures/Piper/lessac_medium_det_*</c>) was generated offline from the same ONNX via onnxruntime.
/// Gated on <c>PIPER_ONNX</c> pointing at the matching <c>en_US-lessac-medium.onnx</c>.</summary>
public sealed class PiperVitsParityTests
{
    private readonly ITestOutputHelper _out;
    public PiperVitsParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void MatchesOnnxRuntimeDeterministic()
    {
        string? onnx = Environment.GetEnvironmentVariable("PIPER_ONNX");
        if (string.IsNullOrEmpty(onnx) || !File.Exists(onnx)) return; // gated

        string fixDir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Piper");
        string idsPath = Path.Combine(fixDir, "lessac_medium_det_ids.txt");
        string audioPath = Path.Combine(fixDir, "lessac_medium_det_audio.f32");
        if (!File.Exists(idsPath) || !File.Exists(audioPath)) return;

        int[] ids = File.ReadAllText(idsPath).Split(',').Select(int.Parse).ToArray();
        byte[] refBytes = File.ReadAllBytes(audioPath);
        float[] reference = new float[refBytes.Length / 4];
        Buffer.BlockCopy(refBytes, 0, reference, 0, refBytes.Length);

        using OnnxWeightLoader loader = new();
        loader.Load(onnx);
        // Deterministic config: zero the stochastic-duration noise so durations are reproducible.
        VitsConfig cfg = VitsConfig.PiperMedium with { NoiseScaleW = 0f };
        using VitsSynthesizer synth = new(cfg);
        synth.LoadWeights(loader.GetResolvedTensors());

        using CpuBackend backend = new();
        float[] audio = synth.Infer(backend, ids, lengthScale: 1f, noiseScale: 0f, seed: 0);

        _out.WriteLine($"ref {reference.Length} samples, got {audio.Length} samples");
        // Durations come from the stochastic predictor; ceil-boundary rounding can shift total length by a frame or
        // two even when deterministic. Allow a small length delta and correlate the overlapping prefix.
        // Durations are deterministic here, so the length must match exactly.
        Assert.Equal(reference.Length, audio.Length);
        int n = reference.Length;

        double sumSq = 0, refSq = 0, dot = 0, gotSq = 0;
        float maxAbs = 0;
        for (int i = 0; i < n; i++)
        {
            float d = audio[i] - reference[i];
            sumSq += (double)d * d;
            refSq += (double)reference[i] * reference[i];
            gotSq += (double)audio[i] * audio[i];
            dot += (double)audio[i] * reference[i];
            maxAbs = MathF.Max(maxAbs, MathF.Abs(d));
        }
        double mae = Math.Sqrt(sumSq / n);
        double corr = dot / (Math.Sqrt(refSq) * Math.Sqrt(gotSq));
        _out.WriteLine($"RMSE={mae:E3} maxAbs={maxAbs:E3} corr={corr:F6}");

        Assert.True(corr > 0.999, $"correlation {corr:F6} too low");
        Assert.True(mae < 1e-2, $"RMSE {mae:E3} too high");
    }
}
