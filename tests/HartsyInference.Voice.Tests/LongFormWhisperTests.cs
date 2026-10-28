using HartsyInference.Core.Backends;
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

    /// <summary>Backends for the transcribe path. Vulkan and CUDA are skipped when the device is missing; BackendGate
    /// fails instead when HARTSY_REQUIRE_BACKENDS=1.</summary>
    public static IEnumerable<object[]> Backends => [["cpu"], ["vulkan"], ["cuda"]];

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task ThreeJfkRepeatsAcrossSilenceAreAllTranscribed(string backend)
    {
        if (!RealWeightGate.Require(_output.WriteLine,
            [VoiceAssets.Jfk, .. VoiceAssets.WhisperFiles(WhisperTinyTurnSttTests.WhisperTiny)]))
        {
            return;
        }
        if (backend != "cpu")
        {
            if (!BackendGate.TryOpen(backend, _output.WriteLine, out IBackend? probe))
            {
                return;
            }
            (probe as IDisposable)?.Dispose();
        }
        float[] jfk = VoiceAssets.Jfk16k();
        float[] gap = new float[4 * SampleRate];
        float[] clip = [.. jfk, .. gap, .. jfk, .. gap, .. jfk];
        byte[] wav = AudioClipCodec.EncodeWav(clip, null, SampleRate);
        ModelSpec tiny = ModelResolver.Resolve("whisper:" + WhisperTinyTurnSttTests.WhisperTiny, null, Modality.Transcribe);

        // The gate accepted the device HARTSY_TEST_VULKAN_DEVICE names, so the engine must open that one too; a bare
        // "vulkan" would take the engine's own ranking, which can be the card the pin is there to avoid.
        string selector = backend == "vulkan" && VulkanTestDeviceSelection.FromEnvironment() is int pinned ? $"vulkan:{pinned}" : backend;
        using InferenceEngine engine = new(selector);
        TranscriptResult result = await engine.Transcribe.RunAsync(tiny, new AudioRequest { Audio = new AudioClip { Data = wav, Format = "wav" } });

        double recall = VoiceAssets.Recall(result.Text, VoiceAssets.JfkWords);
        int countries = VoiceAssets.Words(result.Text).Count(w => w == "country");
        _output.WriteLine($"[{backend}] {result.WindowCount} windows over {result.Duration:0.0} s; recall {recall:P0}; \"country\" x{countries}");
        Assert.True(result.WindowCount >= 2, "a clip past the 30 s window must be planned into more than one window");
        Assert.True(recall >= 0.8, $"recall {recall:P0} on \"{result.Text}\"");
        Assert.True(countries >= 4,$"the three repeats say \"country\" six times; got {countries} in \"{result.Text}\"");
    }
}
