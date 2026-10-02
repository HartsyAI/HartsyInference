using System.Collections.Concurrent;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A switch back to a runner that is already loaded keeps it through the eviction sweep, and the sweep then
/// releases every device buffer the backend caches, the kept runner's included. The kept runner must still produce
/// correct output on its next run, which re-uploads what it needs. Whisper base and Kokoro load side by side, then
/// the free-VRAM floor is raised past any card so the switch back to Whisper evicts Kokoro, keeps Whisper and releases
/// the device: the order of the production failure.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class AudioKeptRunnerReleaseRealWeightTests
{
    private readonly ITestOutputHelper _output;

    public AudioKeptRunnerReleaseRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task SwitchBackToALoadedRunner_AfterTheSweepReleasesTheDevice_StillTranscribes()
    {
        string? unavailable = BackendGate.UnavailableReason("cuda");
        if (unavailable is not null)
        {
            _output.WriteLine($"SKIPPED: {unavailable}");
            return;
        }
        string kokoroDir = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M", "tts");
        string whisperDir = AudioModelCache.GetRepoDirectory("openai/whisper-base", "stt");
        string jfk = Path.Combine(RepoPaths.RepoRoot(), "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(kokoroDir, "config.json"),
            Path.Combine(kokoroDir, "voices", "af_heart.bin"), Path.Combine(whisperDir, "model.safetensors"), jfk))
        {
            return;
        }

        ModelSpec tts = ModelResolver.Resolve("kokoro", modelPathArg: null, Modality.Speech);
        ModelSpec stt = ModelResolver.Resolve("whisper:openai/whisper-base", modelPathArg: null, Modality.Transcribe);
        SpeechRequest speech = new() { Text = "Hello there, this is a test of the resident audio cache.", Voice = "af_heart" };
        AudioRequest transcribe = new() { Audio = new AudioClip { Data = await File.ReadAllBytesAsync(jfk), Format = "wav" } };

        ConcurrentQueue<string> lines = new();
        Logs.SetLogger((level, message) =>
        {
            lines.Enqueue(message);
            Console.Error.WriteLine($"[{level}] {message}");
        });
        try
        {
            using InferenceEngine engine = new("cuda");
            TranscriptResult first = await engine.Transcribe.RunAsync(stt, transcribe);
            AudioResult spoken = await engine.Speech.SynthesizeAsync(tts, speech);
            Assert.True(engine.AudioRuntime.Stt.IsResident("openai/whisper-base"), "Whisper was evicted before the switch back.");

            KnobStore.Set(EngineKnobs.AudioEvictFreeVramFloorMb, 1L << 30);
            TranscriptResult again = await engine.Transcribe.RunAsync(stt, transcribe);
            TranscriptResult repeat = await engine.Transcribe.RunAsync(stt, transcribe);
            _output.WriteLine($"first: {first.Text}");
            _output.WriteLine($"again: {again.Text}");

            Assert.Contains("country", first.Text, StringComparison.OrdinalIgnoreCase);
            Assert.True(spoken.DurationSeconds > 0.5, "Kokoro produced no usable audio.");
            Assert.Equal(first.Text, again.Text);
            Assert.Equal(first.Text, repeat.Text);
            Assert.Contains(lines, line => line.StartsWith("[Audio] Switching to 'stt:openai/whisper-base'", StringComparison.Ordinal)
                && line.Contains("unloaded tts:hexgrad/Kokoro-82M", StringComparison.Ordinal));
            Assert.True(engine.AudioRuntime.Stt.IsResident("openai/whisper-base"));
            Assert.False(engine.AudioRuntime.Tts.IsResident("hexgrad/Kokoro-82M"));
        }
        finally
        {
            Logs.SetLogger(null!);
            KnobStore.Clear(EngineKnobs.AudioEvictFreeVramFloorMb);
        }
    }
}
