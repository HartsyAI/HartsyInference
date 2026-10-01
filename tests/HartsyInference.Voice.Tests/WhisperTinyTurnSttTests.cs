using HartsyInference.Engine;
using HartsyInference.Tests.Common;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>The session's recognition path on CPU with real weights: JFK through the audio thread (Silero endpointing),
/// each closed utterance through a Whisper-tiny lease on the GPU thread (here a CPU engine), answered by a scripted
/// model with Kokoro speaking. The joined user transcripts must hold at least 80 % of the line's content words.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WhisperTinyTurnSttTests
{
    internal const string WhisperTiny = "openai/whisper-tiny";

    private readonly ITestOutputHelper _output;

    public WhisperTinyTurnSttTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task EveryJfkTurnIsTranscribedWithEightyPercentRecall()
    {
        if (!RealWeightGate.Require(_output.WriteLine,
            [VoiceAssets.SileroWeights, VoiceAssets.Jfk, .. VoiceAssets.WhisperFiles(WhisperTiny), .. VoiceAssets.KokoroFiles()]))
        {
            return;
        }
        float[] jfk = VoiceAssets.Jfk16k();
        VoiceAgentOptions options = CpuOptions();
        int turns = TurnEndpointerRealVadTests.Run(jfk, options).Utterances.Count;
        using InferenceEngine engine = new("cpu");
        await using VoiceModelSet models = await VoiceModelSet.LoadAsync(engine, options, VoiceAssets.WakeRoot);

        (string heard, List<VoiceTurnMetrics> metrics) = await SessionRuns.HearAsync(models, new ScriptedTextService { DefaultReply = "Okay." },
            jfk, turns, _output.WriteLine);

        double recall = VoiceAssets.Recall(heard, VoiceAssets.JfkWords);
        _output.WriteLine($"{turns} turn(s); content-word recall {recall:P0}");
        Assert.True(recall >= 0.8, $"recall {recall:P0} on \"{heard}\"");
        Assert.All(metrics, m => Assert.NotNull(m.SttMs));
    }

    /// <summary>Whisper tiny and Kokoro on a CPU engine, 16 kHz out.</summary>
    internal static VoiceAgentOptions CpuOptions() => new()
    {
        SttModel = "whisper:" + WhisperTiny,
        AudioDevice = "cpu",
        LlmDevice = "cpu",
        OutboundSampleRate = 16_000,
    };
}
