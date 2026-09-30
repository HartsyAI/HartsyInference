using Xunit;
using Xunit.Abstractions;
using HartsyInference.Audio.Cache;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight, on-device check of the memory-pressure sweep through the public engine services. With Kokoro and Whisper-tiny both resident, a switch under forced host pressure must keep the model about to run (the sweep used to evict it and reload it), and a pinned runner must survive a switch. Runs on whichever CUDA device is visible (the plan runs it on the 3060 with <c>CUDA_VISIBLE_DEVICES=1</c>); the two models together stay well under 2 GB, and the whole test is a few Kokoro sentences plus three short transcriptions.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class AudioRuntimeEvictionRealWeightTests
{
    private readonly ITestOutputHelper _output;

    public AudioRuntimeEvictionRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task SwitchUnderPressure_KeepsIncomingAndPinnedRunners_OnDevice()
    {
        string? unavailable = BackendGate.UnavailableReason("cuda");
        if (unavailable is not null)
        {
            _output.WriteLine($"SKIPPED: {unavailable}");
            return;
        }
        string kokoroDir = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M", "tts");
        string whisperDir = AudioModelCache.GetRepoDirectory("openai/whisper-tiny", "stt");
        string jfk = Path.Combine(RepoPaths.RepoRoot(), "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(kokoroDir, "config.json"),
            Path.Combine(kokoroDir, "voices", "af_heart.bin"), Path.Combine(whisperDir, "model.safetensors"), jfk))
        {
            return;
        }

        ModelSpec tts = ModelResolver.Resolve("kokoro", modelPathArg: null, Modality.Speech);
        ModelSpec stt = ModelResolver.Resolve("whisper:openai/whisper-tiny", modelPathArg: null, Modality.Transcribe);
        SpeechRequest speech = new() { Text = "Hello there, this is a test of the resident audio cache.", Voice = "af_heart" };
        AudioRequest transcribe = new() { Audio = new AudioClip { Data = await File.ReadAllBytesAsync(jfk), Format = "wav" } };

        using InferenceEngine engine = new("cuda");
        AudioResult first = await engine.Speech.SynthesizeAsync(tts, speech);
        TranscriptResult heard = await engine.Transcribe.RunAsync(stt, transcribe);
        _output.WriteLine($"kokoro {first.DurationSeconds:0.00}s of audio; whisper heard: {heard.Text}");
        Assert.True(first.DurationSeconds > 0.5, "Kokoro produced no usable audio.");
        Assert.Contains("country", heard.Text, StringComparison.OrdinalIgnoreCase);

        AudioRuntime runtime = engine.AudioRuntime;
        if (runtime.Tts.ResidentKeys.Count != 1 || runtime.Stt.ResidentKeys.Count != 1)
        {
            _output.WriteLine($"SKIPPED: host pressure was already real on this box (tts={runtime.Tts.ResidentKeys.Count}, "
                + $"stt={runtime.Stt.ResidentKeys.Count} resident after the warm-up); the premise needs both resident.");
            return;
        }
        string ttsKey = runtime.Tts.ResidentKeys.Single();
        string sttKey = runtime.Stt.ResidentKeys.Single();

        using (AudioEvictionPressure.Force())
        {
            // The incoming model is already resident: it must survive the sweep that drops the other one.
            long started = Environment.TickCount64;
            AudioResult second = await engine.Speech.SynthesizeAsync(tts, speech);
            _output.WriteLine($"kokoro again under forced pressure: {Environment.TickCount64 - started}ms wall");
            Assert.True(second.DurationSeconds > 0.5, "Kokoro produced no usable audio under pressure.");
            Assert.True(runtime.Tts.IsResident(ttsKey), "the incoming Kokoro runner was evicted by its own switch.");
            if (AudioEvictionPressure.HostPressureObservable)
            {
                Assert.False(runtime.Stt.IsResident(sttKey), "the other resident model should have been evicted under pressure.");
            }

            // A pinned runner reloads once and then survives a switch away from it.
            using (runtime.Stt.Pin(sttKey))
            {
                TranscriptResult again = await engine.Transcribe.RunAsync(stt, transcribe);
                Assert.Contains("country", again.Text, StringComparison.OrdinalIgnoreCase);
                await engine.Speech.SynthesizeAsync(tts, speech);
                Assert.True(runtime.Stt.IsResident(sttKey), "a pinned runner must survive memory-pressure eviction.");
                Assert.True(runtime.Tts.IsResident(ttsKey), "the incoming Kokoro runner must be resident after its own run.");
            }
        }
    }
}
