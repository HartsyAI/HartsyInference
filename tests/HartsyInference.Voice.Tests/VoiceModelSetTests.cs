using HartsyInference.Core.Configuration;
using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using HartsyInference.Tools;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>The model set's load-time contract without weights: a missing denoiser or VAD fails loudly (never a silent
/// passthrough), the engine must sit on the audio device, sessions must match its models, the CPU thread cap lives
/// exactly as long as the set, and the speech models are released on the GPU thread.</summary>
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
            FileNotFoundException error = Assert.Throws<FileNotFoundException>(() =>
                VoiceModelSet.LoadFrontEnd(root, new VoiceAgentOptions(), out _, out _));
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

        await models.WarmAsync(text).WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Single(speech.Synthesized);
        Assert.Single(speech.TranscribedSamples, 16_000);
        Assert.All(speech.Threads, thread => Assert.Equal(models.Gpu.ManagedThreadId, thread));
        TextRequest warm = Assert.Single(text.Requests);
        Assert.Equal(1, warm.MaxTokens);
        Assert.False(warm.EnableThinking);
        Assert.Equal("cuda:0", warm.Device);
        Assert.False(warm.AlwaysFreeMemory);
    }

    private static string EmptyDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "voice-wake-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }
}
