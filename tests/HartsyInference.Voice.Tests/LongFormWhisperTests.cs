using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>A clip longer than Whisper's 30 s window goes through the engine's window plan: three JFK repeats with
/// 4 s of silence between them are cut in a gap, and every repeat must come back in the joined transcript.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class LongFormWhisperTests
{
    private const int SampleRate = 16_000;

    private readonly ITestOutputHelper _output;

    public LongFormWhisperTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task ThreeJfkRepeatsAcrossSilenceAreAllTranscribed()
    {
        if (!RealWeightGate.Require(_output.WriteLine,
            [VoiceAssets.Jfk, .. VoiceAssets.WhisperFiles(WhisperTinyTurnSttTests.WhisperTiny)]))
        {
            return;
        }
        float[] jfk = VoiceAssets.Jfk16k();
        float[] gap = new float[4 * SampleRate];
        float[] clip = [.. jfk, .. gap, .. jfk, .. gap, .. jfk];
        byte[] wav = AudioClipCodec.EncodeWav(clip, null, SampleRate);
        ModelSpec tiny = ModelResolver.Resolve("whisper:" + WhisperTinyTurnSttTests.WhisperTiny, null, Modality.Transcribe);

        using InferenceEngine engine = new("cpu");
        TranscriptResult result = await engine.Transcribe.RunAsync(tiny, new AudioRequest { Audio = new AudioClip { Data = wav, Format = "wav" } });

        double recall = VoiceAssets.Recall(result.Text, VoiceAssets.JfkWords);
        int countries = result.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Count(w => w.Trim('.', ',').Equals("country", StringComparison.OrdinalIgnoreCase));
        _output.WriteLine($"{result.WindowCount} windows over {result.Duration:0.0} s; recall {recall:P0}; \"country\" x{countries}");
        Assert.True(result.WindowCount >= 2, "a clip past the 30 s window must be planned into more than one window");
        Assert.True(recall >= 0.8, $"recall {recall:P0} on \"{result.Text}\"");
        Assert.True(countries >= 4, $"the three repeats say \"country\" six times; got {countries} in \"{result.Text}\"");
    }
}
