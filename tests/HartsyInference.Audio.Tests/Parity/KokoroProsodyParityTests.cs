using System.Text.Json;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Kokoro's prosody against the official PyTorch <c>KModel</c>: for the same phonemes and <c>af_heart</c>
/// voice, the predicted durations and the F0 and energy curves the decoder receives. These are deterministic in the
/// reference (only the vocoder's source is random), so they pin the predictor wiring — which features the F0/N
/// branch reads and which voice-pack row is used. Gated on <c>REF_KOKORO_PROSODY</c> (written by
/// <c>tools/kokoro/prosody_reference.py</c>) and the cached Kokoro weights.</summary>
public sealed class KokoroProsodyParityTests
{
    private readonly ITestOutputHelper _out;

    public KokoroProsodyParityTests(ITestOutputHelper output) => _out = output;

    [Fact]
    [Trait("Category", "Integration")]
    public async Task DurationsAndF0N_MatchTheReferenceModel()
    {
        string? refPath = Environment.GetEnvironmentVariable("REF_KOKORO_PROSODY");
        if (string.IsNullOrEmpty(refPath) || !File.Exists(refPath)) return;
        if (!RealWeightGate.Require(_out.WriteLine, GpuBenchSupport.KokoroFiles())) return;

        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(refPath));
        using CpuBackend backend = new();
        using KokoroPipeline kokoro = await KokoroPipeline.LoadAsync();
        foreach (JsonElement row in reference.RootElement.EnumerateArray())
        {
            int[] refDurations = row.GetProperty("durations").EnumerateArray().Select(static e => e.GetInt32()).ToArray();
            float[] refF0 = row.GetProperty("f0").EnumerateArray().Select(static e => e.GetSingle()).ToArray();
            float[] refN = row.GetProperty("n").EnumerateArray().Select(static e => e.GetSingle()).ToArray();
            int[]? durations = null;
            float[]? f0 = null, n = null;
            kokoro.TestProsodyObserver = (d, f, e) => (durations, f0, n) = (d, f.AsSpan<float>().ToArray(), e.AsSpan<float>().ToArray());
            kokoro.Synthesize(backend, row.GetProperty("phonemes").GetString()!, "af_heart");

            Assert.NotNull(durations);
            Assert.Equal(refDurations, durations);
            double f0Err = MaxAbs(refF0, f0!), nErr = MaxAbs(refN, n!);
            _out.WriteLine($"T={durations!.Length} T_total={durations.Sum()} F0 maxAbs={f0Err:F4} Hz (peak {refF0.Max():F1}) " +
                $"N maxAbs={nErr:F4} (range {refN.Min():F2}..{refN.Max():F2})");
            Assert.True(f0Err < 0.5, $"F0 diverges from the reference (maxAbs {f0Err:F4} Hz).");
            Assert.True(nErr < 0.05, $"Energy diverges from the reference (maxAbs {nErr:F4}).");
        }
    }

    private static double MaxAbs(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double max = 0;
        for (int i = 0; i < expected.Length; i++) max = Math.Max(max, Math.Abs(expected[i] - actual[i]));
        return max;
    }
}
