using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tools;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>The model set's load-time contract without weights: a missing denoiser or VAD fails loudly (never a silent
/// passthrough), the engine must sit on the audio device, sessions must match its models, the CPU thread cap lives
/// exactly as long as the set, the warm-up synthesizes one text per length bucket, each as its own GPU job, and the
/// speech models are released on the GPU thread.</summary>
public sealed class VoiceModelSetTests
{
    [Fact]
    public void DenoiseWithoutWeightsFailsLoudly()
    {
        string root = EmptyDirectory();
        try
        {
            FileNotFoundException error = Assert.Throws<FileNotFoundException>(() =>
                VoiceModelSet.LoadFrontEnd(root, new VoiceAgentOptions { Denoise = true }, out _, out _));
            Assert.Contains("rnnoise.safetensors", error.Message, StringComparison.Ordinal);
            Assert.Contains("rnnoise_int8.safetensors", error.Message, StringComparison.Ordinal);
            Assert.Contains("never substitutes unprocessed audio", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void DenoiseWithOnlyFloatWeightsFailsLoudlyNamingTheInt8Tables()
    {
        // VoiceModelSet always asks for Int8 (the front-end gate was only met at that precision), so a directory
        // that has the F32 weights xiph's installer always writes but not yet the int8 tables must still fail
        // loudly, naming the tables rather than quietly running on a precision the gate never cleared.
        string root = EmptyDirectory();
        try
        {
            string denoiseDir = Path.Combine(root, "denoise");
            Directory.CreateDirectory(denoiseDir);
            File.WriteAllBytes(Path.Combine(denoiseDir, "rnnoise.safetensors"), []);
            FileNotFoundException error = Assert.Throws<FileNotFoundException>(() =>
                VoiceModelSet.LoadFrontEnd(root, new VoiceAgentOptions { Denoise = true }, out _, out _));
            Assert.Contains("rnnoise_int8.safetensors", error.Message, StringComparison.Ordinal);
            Assert.Contains("never substitutes unprocessed audio", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void AMissingVadFailsLoudly()
    {
        string root = EmptyDirectory();
        try
        {
            // Denoise off explicitly: it defaults on now, and the denoiser is checked first in LoadFrontEnd, which
            // would otherwise throw about RNNoise instead of exercising the VAD path this test is about.
            FileNotFoundException error = Assert.Throws<FileNotFoundException>(() =>
                VoiceModelSet.LoadFrontEnd(root, new VoiceAgentOptions { Denoise = false }, out _, out _));
            Assert.Contains("Silero VAD", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task LoadRefusesAnEngineOnAnotherDevice()
    {
        using InferenceEngine engine = new("cpu");
        ArgumentException error = await Assert.ThrowsAsync<ArgumentException>(() =>
            VoiceModelSet.LoadAsync(engine, new VoiceAgentOptions { AudioDevice = "cuda:1" }, wakeModelRoot: "unused"));
        Assert.Contains("cuda:1", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ASessionMustUseTheModelSetsModels()
    {
        using CpuBackend device = new();
        VoiceAgentOptions loaded = VoiceHarness.DefaultOptions();
        await using VoiceModelSet models = new(loaded, new FakeSpeech(), device, () => new LevelVadModel(), createDenoiser: null);
        Assert.Throws<ArgumentException>(() => new VoiceAgentSession(models, new ScriptedTextService(), new ToolRegistry(),
            loaded with { TtsModel = "kokoro:am_adam" }));
        Assert.Throws<ArgumentException>(() => new VoiceAgentSession(models, new ScriptedTextService(), new ToolRegistry(),
            loaded with { Denoise = true }));
    }

    [Fact]
    public async Task TheCpuThreadCapLastsAsLongAsTheModelSet()
    {
        bool hadOverride = KnobStore.HasOverride(EngineKnobs.CpuThreads);
        int previous = EngineKnobs.CpuThreads.Value;
        using CpuBackend device = new();
        VoiceModelSet models = new(VoiceHarness.DefaultOptions() with { CpuThreadCap = 3 }, new FakeSpeech(), device,
            () => new LevelVadModel(), createDenoiser: null);
        Assert.Equal(3, EngineKnobs.CpuThreads.Value);
        await models.DisposeAsync();
        Assert.Equal(previous, EngineKnobs.CpuThreads.Value);
        Assert.Equal(hadOverride, KnobStore.HasOverride(EngineKnobs.CpuThreads));
    }

    [Fact]
    public async Task TheSpeechModelsAreReleasedOnTheGpuThread()
    {
        using CpuBackend device = new();
        FakeSpeech speech = new();
        VoiceModelSet models = new(VoiceHarness.DefaultOptions(), speech, device, () => new LevelVadModel(), createDenoiser: null);
        int gpuThread = models.Gpu.ManagedThreadId;
        await models.DisposeAsync();
        Assert.True(speech.Disposed);
        Assert.Equal(gpuThread, speech.DisposedOnThread);
    }

    [Fact]
    public async Task WarmUpRunsSpeechOnTheGpuThreadAndOneThinkingOffTokenOnTheLlmDevice()
    {
        using CpuBackend device = new();
        FakeSpeech speech = new();
        ScriptedTextService text = new();
        VoiceAgentOptions options = VoiceHarness.DefaultOptions() with { LlmDevice = "cuda:0" };
        await using VoiceModelSet models = new(options, speech, device, () => new LevelVadModel(), createDenoiser: null);
        List<string> log = [];
        Logs.SetLogger((_, message) =>
        {
            lock (log)
            {
                log.Add(message);
            }
        });
        try
        {
            await models.WarmAsync(text).WaitAsync(TimeSpan.FromSeconds(10));
        }
        finally
        {
            Logs.SetLogger(null!);
        }

        string warmLine = Assert.Single(log, message => message.StartsWith("[Voice] Warm-up on cpu: synthesized 1 / 3 / 6 / 13 / 30 words in ", StringComparison.Ordinal));
        Assert.Contains("recognized 1 s of silence in ", warmLine, StringComparison.Ordinal);
        Assert.Equal(VoiceModelSet.WarmTexts, speech.Synthesized);
        Assert.Single(speech.TranscribedSamples, 16_000);
        Assert.All(speech.Threads, thread => Assert.Equal(models.Gpu.ManagedThreadId, thread));
        TextRequest warm = Assert.Single(text.Requests);
        Assert.Equal(1, warm.MaxTokens);
        Assert.False(warm.EnableThinking);
        Assert.Equal("cuda:0", warm.Device);
        Assert.False(warm.AlwaysFreeMemory);
    }

    [Fact]
    public async Task WarmUpSynthesizesOneTextPerLengthBucketOnceThroughTheLeaseEachAsItsOwnGpuJob()
    {
        FakeSynthesizerLease synthesizer = new(1);
        FakeTranscriberLease transcriber = new(1);
        VoiceLeaseSpeech speech = await VoiceLeaseSpeech.OpenAsync(_ => Task.FromResult<ISynthesizerLease>(synthesizer),
            _ => Task.FromResult<ITranscriberLease>(transcriber), voice: "af_heart", CancellationToken.None);
        using CpuBackend cpu = new();
        RecordingDevice device = RecordingDevice.Wrap(cpu);
        await using VoiceModelSet models = new(VoiceHarness.DefaultOptions(), speech, device.Backend, () => new LevelVadModel(), createDenoiser: null);

        await models.WarmAsync(new ScriptedTextService()).WaitAsync(TimeSpan.FromSeconds(10));

        // Five texts of growing length, one per power-of-two frame bucket from 32 to 512, each synthesized once.
        int[] words = [.. VoiceModelSet.WarmTexts.Select(text => text.Split(' ').Length)];
        Assert.Equal(5, words.Length);
        Assert.All(words.Zip(words.Skip(1)), pair => Assert.True(pair.Second >= 2 * pair.First, $"{pair.First} then {pair.Second} words"));
        Assert.Equal(VoiceModelSet.WarmTexts, synthesizer.Texts);
        Assert.All(synthesizer.Threads, thread => Assert.Equal(models.Gpu.ManagedThreadId, thread));
        // Each synthesis and the recognition ran as its own GPU job: one activation free per job.
        Assert.Equal(Enumerable.Repeat(RecordingDevice.FreeKeepingPool, words.Length + 1), device.Calls);
    }

    private static string EmptyDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "voice-wake-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }
}
