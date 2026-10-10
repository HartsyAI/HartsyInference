using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;
using HartsyInference.Tools;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>The model set's load-time contract without weights: a missing denoiser fails loudly (never a silent
/// passthrough), the engine must sit on the audio device, sessions must match its models, the CPU thread cap lives
/// exactly as long as the set, the warm-up runs on the GPU thread, and the speech models are released on it.</summary>
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
    public async Task LoadRefusesAnEngineOnAnotherDevice()
    {
        using InferenceEngine engine = new("cpu");
        ArgumentException error = await Assert.ThrowsAsync<ArgumentException>(() =>
            VoiceModelSet.LoadAsync(engine, new VoiceAgentOptions { AudioDevice = "cuda:1" }, wakeModelRoot: "unused"));
        Assert.Contains("cuda:1", error.Message, StringComparison.Ordinal);
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
        Assert.Null(warm.Tools);
    }

    private static string EmptyDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "voice-wake-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }
}
