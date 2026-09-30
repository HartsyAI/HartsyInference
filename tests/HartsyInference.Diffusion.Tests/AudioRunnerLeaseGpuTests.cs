using System.Diagnostics;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Backends;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight leases on a CUDA device, driven the way the voice session drives them: Whisper small.en and
/// Kokoro held open, one call at a time under <see cref="DeviceGate"/>. Whisper must hear the JFK clip and Kokoro's
/// sentence, match the service's transcript on the same samples, and both runners must stay resident through
/// forced-pressure switches to a third model. The plan runs it on the 3060 with <c>CUDA_VISIBLE_DEVICES=1</c>; the
/// three models together stay under 2 GB.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class AudioRunnerLeaseGpuTests
{
    private const string Sentence = "The weather tomorrow is clear and mild with a gentle breeze.";
    private const string KokoroKey = "hexgrad/Kokoro-82M";
    private const string SmallEnKey = "openai/whisper-small.en";
    private const string TinyKey = "openai/whisper-tiny";

    private static readonly string[] SentenceWords = ["weather", "tomorrow", "clear", "mild", "gentle", "breeze"];
    private static readonly string[] JfkWords =
        ["fellow", "americans", "ask", "not", "what", "your", "country", "can", "do", "for", "you"];

    private readonly ITestOutputHelper _output;

    public AudioRunnerLeaseGpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task KokoroAndWhisperSmallEnLeases_OnDevice()
    {
        string? unavailable = BackendGate.UnavailableReason("cuda");
        if (unavailable is not null)
        {
            _output.WriteLine($"SKIPPED: {unavailable}");
            return;
        }
        string kokoroDir = AudioModelCache.GetRepoDirectory(KokoroKey, "tts");
        string smallEnDir = AudioModelCache.GetRepoDirectory(SmallEnKey, "stt");
        string tinyDir = AudioModelCache.GetRepoDirectory(TinyKey, "stt");
        string jfk = Path.Combine(RepoPaths.RepoRoot(), "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(kokoroDir, "config.json"),
            Path.Combine(kokoroDir, "voices", "af_heart.bin"), AudioModelRoot.SharedFile("cmudict.dict"),
            Path.Combine(smallEnDir, "model.safetensors"), Path.Combine(tinyDir, "model.safetensors"), jfk))
        {
            return;
        }

        using InferenceEngine engine = new("cuda");
        IBackend device = engine.ComputeBackend;
        _output.WriteLine($"device {engine.DeviceKey}: {device.Capabilities.DeviceName}");
        ModelSpec kokoro = ModelResolver.Resolve("kokoro", modelPathArg: null, Modality.Speech);
        ModelSpec smallEn = ModelResolver.Resolve("whisper:openai/whisper-small.en", modelPathArg: null, Modality.Transcribe);
        ModelSpec tiny = ModelResolver.Resolve("whisper:openai/whisper-tiny", modelPathArg: null, Modality.Transcribe);
        AudioClip jfkClip = new() { Data = await File.ReadAllBytesAsync(jfk), Format = "wav" };
        float[] jfk16k = AudioClipCodec.DecodeMono(jfkClip, 16_000);
        AudioRequest sttOptions = new() { Audio = jfkClip };
        SpeechRequest ttsOptions = new() { Text = "", Voice = "af_heart" };

        // Opened without holding the gate: opening takes it itself.
        using ISynthesizerLease tts = await engine.Speech.OpenSynthesizerAsync(kokoro);
        using ITranscriberLease stt = await engine.Transcribe.OpenTranscriberAsync(smallEn);

        string heardJfk = Gated(device, "whisper small.en JFK, cold", () => stt.Transcribe(jfk16k, 16_000, sttOptions));
        heardJfk = Gated(device, "whisper small.en JFK, warm", () => stt.Transcribe(jfk16k, 16_000, sttOptions));
        _output.WriteLine($"lease heard: \"{heardJfk}\"");
        AssertRecall(heardJfk, JfkWords);
        TranscriptResult served = await engine.Transcribe.RunAsync(smallEn, new AudioRequest { Audio = jfkClip });
        Assert.Equal(served.Text, heardJfk);

        float[] speech = Gated(device, "kokoro sentence, cold", () => tts.Synthesize(Sentence, ttsOptions));
        speech = Gated(device, "kokoro sentence, warm", () => tts.Synthesize(Sentence, ttsOptions));
        string heardSpeech = Gated(device, "whisper small.en on kokoro (24 kHz in)",
            () => stt.Transcribe(speech, tts.SampleRate, sttOptions));
        _output.WriteLine($"kokoro {speech.Length / (double)tts.SampleRate:0.00} s; whisper heard: \"{heardSpeech}\"");
        AssertRecall(heardSpeech, SentenceWords);

        AudioRuntime runtime = engine.AudioRuntime;
        using (AudioEvictionPressure.Force())
        {
            for (int turn = 0; turn < 3; turn++)
            {
                TranscriptResult other = await engine.Transcribe.RunAsync(tiny, new AudioRequest { Audio = jfkClip });
                Assert.Contains("country", other.Text, StringComparison.OrdinalIgnoreCase);
                await engine.Speech.SynthesizeAsync(kokoro, ttsOptions with { Text = Sentence });
                Assert.True(runtime.Stt.IsResident(SmallEnKey), "the pinned Whisper runner was evicted by a switch.");
                Assert.True(runtime.Tts.IsResident(KokoroKey), "the pinned Kokoro runner was evicted by a switch.");
                AssertRecall(Gated(device, $"whisper small.en JFK after switch {turn + 1}",
                    () => stt.Transcribe(jfk16k, 16_000, sttOptions)), JfkWords);
            }
        }
        if (AudioEvictionPressure.HostPressureObservable)
        {
            Assert.False(runtime.Stt.IsResident(TinyKey), "the unpinned third model should have been evicted by the last switch.");
        }
    }

    /// <summary>Runs one lease call under the device gate, as the voice session's GPU thread does, and logs its wall time.</summary>
    private T Gated<T>(IBackend device, string label, Func<T> call)
    {
        Stopwatch watch = Stopwatch.StartNew();
        T result;
        using (DeviceGate.Acquire(device))
        {
            result = call();
        }
        _output.WriteLine($"{label}: {watch.Elapsed.TotalMilliseconds:0} ms");
        return result;
    }

    private void AssertRecall(string heard, string[] expected)
    {
        HashSet<string> words = new(heard.ToLowerInvariant()
            .Split([' ', ',', '.', '!', '?', ';', ':', '-'], StringSplitOptions.RemoveEmptyEntries), StringComparer.Ordinal);
        int hits = expected.Count(words.Contains);
        double recall = hits / (double)expected.Length;
        _output.WriteLine($"content-word recall {hits}/{expected.Length} ({recall:P0})");
        Assert.True(recall >= 0.8, $"recall {recall:P0} on \"{heard}\"");
    }
}
