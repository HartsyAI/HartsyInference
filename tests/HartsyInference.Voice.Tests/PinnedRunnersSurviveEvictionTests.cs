using System.Diagnostics;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Voice.Gpu;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>The voice model set's leases on the RTX 3060 under forced host-memory pressure: other service calls on the
/// same engine (Whisper tiny) load and evict around them, and the session's pinned Whisper small.en and Kokoro runners
/// must keep answering on the GPU thread without a reload (no lease revocation, no reopen, no load-sized slowdown), and a
/// full turn must still work afterwards. Run alone, in a quiet window:
/// <c>CUDA_VISIBLE_DEVICES=1 HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models dotnet test tests/HartsyInference.Voice.Tests -c Release --filter "FullyQualifiedName~PinnedRunnersSurviveEvictionTests"</c>.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class PinnedRunnersSurviveEvictionTests
{
    private const string Sentence = "The pinned voice is still here after every switch.";

    private readonly ITestOutputHelper _output;

    public PinnedRunnersSurviveEvictionTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task PinnedSpeechModelsSurviveMemoryPressureSwitches()
    {
        if (!RealWeightGate.Require(_output.WriteLine, [.. GpuVoiceRig.Assets(), .. VoiceAssets.WhisperFiles(WhisperTinyTurnSttTests.WhisperTiny)]))
        {
            return;
        }
        using InferenceEngine? engine = GpuVoiceRig.OpenAudioEngine(_output.WriteLine);
        if (engine is null)
        {
            return;
        }
        VoiceAgentOptions options = GpuVoiceRig.Options(engine);
        await using VoiceModelSet models = await VoiceModelSet.LoadAsync(engine, options, VoiceAssets.WakeRoot);
        ScriptedTextService text = new() { DefaultReply = "Still listening." };
        await models.WarmAsync(text);
        float[] jfk = VoiceAssets.Jfk16k();
        AudioClip jfkClip = new() { Data = await File.ReadAllBytesAsync(VoiceAssets.Jfk), Format = "wav" };
        ModelSpec tiny = ModelResolver.Resolve("whisper:" + WhisperTinyTurnSttTests.WhisperTiny, null, Modality.Transcribe);
        double warmMs = (await TranscribeTimedAsync(models, jfk)).Ms;

        bool hadOverride = KnobStore.HasOverride(EngineKnobs.AudioEvictBelowGb);
        long previous = EngineKnobs.AudioEvictBelowGb.Value;
        KnobStore.Set(EngineKnobs.AudioEvictBelowGb, 1_000_000_000L);
        try
        {
            for (int round = 1; round <= 3; round++)
            {
                TranscriptResult other = await engine.Transcribe.RunAsync(tiny, new AudioRequest { Audio = jfkClip });
                Assert.Contains("country", other.Text, StringComparison.OrdinalIgnoreCase);
                (string heard, double ms) = await TranscribeTimedAsync(models, jfk);
                float[] spoken = await models.Gpu.RunAsync(VoiceGpuJobKind.Synthesize, () => models.Synthesize(Sentence), CancellationToken.None);
                _output.WriteLine($"switch {round}: pinned Whisper {ms:F0} ms (warm {warmMs:F0} ms), Kokoro {spoken.Length} samples: \"{heard}\"");
                Assert.True(VoiceAssets.Recall(heard, VoiceAssets.JfkWords) >= 0.8, $"the pinned recognizer misheard after switch {round}: \"{heard}\"");
                Assert.NotEmpty(spoken);
                Assert.True(ms <= 3 * warmMs + 150, $"the pinned recognizer took {ms:F0} ms after switch {round} (warm {warmMs:F0} ms): it was reloaded.");
            }
        }
        finally
        {
            if (hadOverride)
            {
                KnobStore.Set(EngineKnobs.AudioEvictBelowGb, previous);
            }
            else
            {
                KnobStore.Clear(EngineKnobs.AudioEvictBelowGb);
            }
        }
        Assert.Equal(0, models.Gpu.Reopens);

        await using VoiceHarness harness = await VoiceHarness.StartAsync(models, text, TimeSpan.FromMilliseconds(2));
        harness.Session.PushDtmf('7');
        VoiceTurnMetrics turn = (await harness.TurnCompletedAsync(1, 60)).Metrics!.Value;
        _output.WriteLine(turn.ToLogLine());
        Assert.True(harness.Reader.Samples.Length > 0);
        Assert.Equal(0, models.Gpu.Reopens);
    }

    private static async Task<(string Text, double Ms)> TranscribeTimedAsync(VoiceModelSet models, float[] audio) =>
        await models.Gpu.RunAsync(VoiceGpuJobKind.Transcribe, () =>
        {
            long started = Stopwatch.GetTimestamp();
            string text = models.Transcribe(audio);
            return (text, Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        }, CancellationToken.None);
}
