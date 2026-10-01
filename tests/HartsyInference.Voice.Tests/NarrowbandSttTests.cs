using HartsyInference.Engine;
using HartsyInference.Tests.Common;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>Telephone-band audio through the session: JFK taken 16 kHz → 8 kHz → 16 kHz with the gateway's streaming
/// resamplers, then endpointed and recognized as a call would be (Whisper tiny on CPU). The plan's gate is recall within
/// 10 points of the wideband baseline; with RNNoise installed, the denoised narrowband path is held to the same gate.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class NarrowbandSttTests
{
    private const double AllowedDropPoints = 0.10;

    private readonly ITestOutputHelper _output;

    public NarrowbandSttTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task NarrowbandRecallIsWithinTenPointsOfWideband()
    {
        if (!RealWeightGate.Require(_output.WriteLine,
            [VoiceAssets.SileroWeights, VoiceAssets.Jfk, .. VoiceAssets.WhisperFiles(WhisperTinyTurnSttTests.WhisperTiny), .. VoiceAssets.KokoroFiles()]))
        {
            return;
        }
        float[] wideband = VoiceAssets.Jfk16k();
        float[] narrowband = VoiceAssets.Narrowband(wideband);
        VoiceAgentOptions options = WhisperTinyTurnSttTests.CpuOptions();
        using InferenceEngine engine = new("cpu");
        await using VoiceModelSet models = await VoiceModelSet.LoadAsync(engine, options, VoiceAssets.WakeRoot);

        double baseline = await RecallAsync(models, options, wideband, "16 kHz");
        double narrow = await RecallAsync(models, options, narrowband, "narrowband");

        Assert.True(narrow >= baseline - AllowedDropPoints, $"narrowband recall {narrow:P0} is more than 10 points under the 16 kHz {baseline:P0}.");

        if (!File.Exists(VoiceAssets.RnnoiseWeights))
        {
            _output.WriteLine($"SKIPPED optional row: no RNNoise weights at {VoiceAssets.RnnoiseWeights}");
            return;
        }
        VoiceAgentOptions denoised = options with { Denoise = true };
        await using VoiceModelSet denoisedModels = await VoiceModelSet.LoadAsync(engine, denoised, VoiceAssets.WakeRoot);
        double withRnnoise = await RecallAsync(denoisedModels, denoised, narrowband, "narrowband + RNNoise");
        Assert.True(withRnnoise >= baseline - AllowedDropPoints,
            $"narrowband recall through RNNoise {withRnnoise:P0} is more than 10 points under the 16 kHz {baseline:P0}.");
    }

    private async Task<double> RecallAsync(VoiceModelSet models, VoiceAgentOptions options, float[] audio, string label)
    {
        int turns = TurnEndpointerRealVadTests.Run(audio, options).Utterances.Count;
        (string heard, _) = await SessionRuns.HearAsync(models, new ScriptedTextService { DefaultReply = "Okay." }, audio, turns, _output.WriteLine);
        double recall = VoiceAssets.Recall(heard, VoiceAssets.JfkWords);
        _output.WriteLine($"{label}: {turns} turn(s), content-word recall {recall:P0}: \"{heard.Trim()}\"");
        return recall;
    }
}
