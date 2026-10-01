using System.Diagnostics;
using System.Text;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Does the int8 network denoise as well as the F32 one where it counts, in what the recognizer hears?
///
/// <para>Probe C of <see cref="VoiceTurnBenchTests"/>, with RNNoise at both precisions side by side. The inputs are
/// the full JFK clip and the same clip under white noise at 5 dB SNR (as <see cref="RnnoiseRealSpeechTests"/> mixes
/// it), each at 16 kHz and after the narrowband 16 k → 8 k → 16 k round trip. Each goes through RNNoise at F32 and at
/// int8, then Whisper small.en, and is scored by content-word recall against the transcript. Whisper runs on the CPU
/// here, so the GPUs stay free; it is deterministic, so one transcription per row is the answer.</para>
///
/// <para>Opt in with <c>HARTSY_RNNOISE_QUALITY=1</c>: it holds every core for a few minutes. Needs Whisper small.en in the
/// audio model cache, the RNNoise weights and the int8 tables beside them.</para></summary>
public sealed class RnnoiseInt8QualityTests(ITestOutputHelper log)
{
    /// <summary>How far int8's recall may fall below F32's on any row.</summary>
    private const double AllowedRecallDrop = 0.10;

    [Fact]
    [Trait("Category", "Integration")]
    public async Task Int8_KeepsF32sRecall_ThroughRnnoiseAndWhisper()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_RNNOISE_QUALITY") != "1")
        {
            log.WriteLine("SKIPPED: set HARTSY_RNNOISE_QUALITY=1 to run Whisper small.en on the CPU over both precisions");
            return;
        }
        string jfk = VoiceTurnBenchTests.JfkPath();
        string weightsPath = RnnoiseRealSpeechTests.WeightsPath();
        string[] required =
        [
            .. VoiceTurnBenchTests.WhisperFiles(VoiceTurnBenchTests.WhisperSmallEn), jfk, weightsPath,
            RnnoiseRealSpeechTests.Int8TablesPath(),
        ];
        if (!RealWeightGate.Require(log.WriteLine, required)) return;

        float[] clean = VoiceTurnBenchTests.LoadJfk16k(jfk);
        float[] noisy = RnnoiseRealSpeechTests.AddWhiteNoise(clean, snrDb: 5, seed: 1234);
        using WhisperPipeline whisper = await WhisperPipeline.LoadAsync(VoiceTurnBenchTests.WhisperSmallEn);
        WhisperOptions options = new() { Language = null };
        using CpuBackend backend = new();
        using RnnoiseWeights f32 = RnnoiseRealSpeechTests.LoadWeights(weightsPath);
        using RnnoiseWeights int8 = RnnoiseRealSpeechTests.LoadWeights(weightsPath, RnnoisePrecision.Int8);

        StringBuilder table = new();
        table.AppendLine("| Input | no RNNoise | RNNoise F32 | RNNoise int8 | int8 transcript |");
        table.AppendLine("|---|---:|---:|---:|---|");
        List<string> drops = [];
        foreach ((string name, float[] audio) in new[] { ("clean", clean), ("5 dB white noise", noisy) })
        {
            foreach ((string band, float[] input) in new[]
                { ("16 kHz", audio), ("narrowband", VoiceTurnBenchTests.NarrowbandRoundTrip(audio)) })
            {
                Stopwatch timer = Stopwatch.StartNew();
                double raw = Recall(whisper.TranscribeAudio(backend, input, VoiceTurnBenchTests.SampleRate, options));
                double viaF32 = Recall(whisper.TranscribeAudio(backend,
                    VoiceTurnBenchTests.Denoise(backend, f32, input), VoiceTurnBenchTests.SampleRate, options));
                string heard = whisper.TranscribeAudio(backend, VoiceTurnBenchTests.Denoise(backend, int8, input),
                    VoiceTurnBenchTests.SampleRate, options);
                double viaInt8 = Recall(heard);
                timer.Stop();
                table.AppendLine($"| {name}, {band} | {raw:P0} | {viaF32:P0} | {viaInt8:P0} | {heard.Trim()} |");
                log.WriteLine($"{name}, {band}: recall without RNNoise {raw:P0}, F32 {viaF32:P0}, int8 {viaInt8:P0} "
                    + $"({timer.Elapsed.TotalSeconds:F0} s)");
                if (viaInt8 < viaF32 - AllowedRecallDrop)
                    drops.Add($"{name}, {band}: int8 {viaInt8:P0} against F32 {viaF32:P0}");
            }
        }
        log.WriteLine(table.ToString());
        Assert.True(drops.Count == 0, $"int8 recall fell more than {AllowedRecallDrop:P0} below F32: {string.Join("; ", drops)}");
    }

    private static double Recall(string heard) =>
        VoiceTurnBenchTests.ContentWordRecall(VoiceTurnBenchTests.JfkTranscript, heard);
}
