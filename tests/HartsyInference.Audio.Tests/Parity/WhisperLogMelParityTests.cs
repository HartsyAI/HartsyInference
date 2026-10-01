using System.Globalization;
using System.Text;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>Whisper's log-mel against HF transformers' <c>WhisperFeatureExtractor</c> (its default torch path, with the
/// numpy path beside it as the reference's own noise floor), 80 and 128 bins, on <see cref="WhisperParityClips"/>,
/// through the pipeline's own <see cref="WhisperPipeline.ComputeWindowMel"/>.
///
/// <para>Integration: set <c>HARTSYINFERENCE_WHISPER_PARITY_DIR</c>. The first run writes the clips there and skips;
/// then <c>whisper_reference.py features --dir &lt;dir&gt;</c> writes <c>hf/{clip}.mel{80|128}[.np].f32</c>, and the
/// next run compares. The clips on disk must still be the ones built now, or the reference ran on other
/// samples.</para></summary>
[Trait("Category", "Integration")]
public sealed class WhisperLogMelParityTests(ITestOutputHelper output)
{
    /// <summary>Float rounding on the normalized log-mel: the 400-point FFT, power and filterbank all run in float32
    /// on both sides, in different summation orders.</summary>
    private const float Tolerance = 1e-4f;

    private const string OutEnvVar = "HARTSY_WHISPER_PARITY_OUT";

    [Fact]
    public void LogMel_MatchesHfWhisperFeatureExtractor()
    {
        string? dir = Environment.GetEnvironmentVariable(WhisperParityClips.DirEnvVar);
        if (string.IsNullOrEmpty(dir))
        {
            output.WriteLine($"SKIPPED: set {WhisperParityClips.DirEnvVar}.");
            return;   // tier-lint: guarded
        }
        IReadOnlyList<(string Name, float[] Audio)> clips = WhisperParityClips.Build();
        if (!File.Exists(Path.Combine(dir, "clips", "clips.tsv")))
        {
            WhisperParityClips.Write(dir, clips);
            output.WriteLine($"SKIPPED: wrote the clips to {dir}/clips; run whisper_reference.py features, then rerun.");
            return;
        }
        foreach ((string name, float[] audio) in clips)
        {
            float[] onDisk = WhisperParityClips.ReadF32(Path.Combine(dir, "clips", name + ".f32"));
            Assert.True(audio.AsSpan().SequenceEqual(onDisk), $"{name}: the clip on disk is not the one built now");
        }

        StringBuilder table = new();
        table.AppendLine("| Clip | samples | mels | max abs | mean abs | at (mel, frame) | max abs vs HF numpy | HF torch vs numpy |");
        table.AppendLine("|---|---:|---:|---:|---:|---|---:|---:|");
        List<string> failures = [];
        foreach (int nMels in (int[])[80, 128])
        {
            MelSpectrogramExtractor extractor = new(MelSpectrogramExtractor.WhisperConfig(nMels));
            int frames = extractor.OutputFrames(WhisperPipeline.WindowSamples);
            float[] mel = new float[nMels * frames];
            foreach ((string name, float[] audio) in clips)
            {
                string torchPath = Path.Combine(dir, "hf", $"{name}.mel{nMels}.f32");
                string numpyPath = Path.Combine(dir, "hf", $"{name}.mel{nMels}.np.f32");
                Assert.True(File.Exists(torchPath) && File.Exists(numpyPath), $"missing {torchPath} or its .np twin");
                float[] reference = WhisperParityClips.ReadF32(torchPath);
                float[] numpy = WhisperParityClips.ReadF32(numpyPath);
                Assert.Equal(mel.Length, reference.Length);
                Assert.Equal(mel.Length, numpy.Length);

                WhisperPipeline.ComputeWindowMel(extractor, audio, mel);
                (float maxAbs, double meanAbs, int at) = Diff(mel, reference);
                (float maxVsNumpy, _, _) = Diff(mel, numpy);
                (float referenceFloor, _, _) = Diff(reference, numpy);
                table.AppendLine(CultureInfo.InvariantCulture, $"| {name} | {audio.Length} | {nMels} | {maxAbs:E2} | {meanAbs:E2} | "
                    + $"({at / frames}, {at % frames}) | {maxVsNumpy:E2} | {referenceFloor:E2} |");
                if (!(maxAbs <= Tolerance))
                {
                    failures.Add($"{name} mel{nMels}: max abs {maxAbs:E2}");
                }
            }
        }
        string text = table.ToString();
        output.WriteLine(text);
        string? outPath = Environment.GetEnvironmentVariable(OutEnvVar);
        if (!string.IsNullOrEmpty(outPath))
        {
            File.AppendAllText(outPath, text + Environment.NewLine);
        }
        Assert.True(failures.Count == 0, $"above {Tolerance:E0}: {string.Join("; ", failures)}");
    }

    /// <summary>Max and mean absolute difference, and the flat index of the max.</summary>
    private static (float MaxAbs, double MeanAbs, int At) Diff(float[] actual, float[] expected)
    {
        float maxAbs = 0f;
        double sum = 0;
        int at = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            float d = MathF.Abs(actual[i] - expected[i]);
            sum += d;
            if (d > maxAbs || float.IsNaN(d))
            {
                maxAbs = d;
                at = i;
            }
        }
        return (maxAbs, sum / actual.Length, at);
    }
}
