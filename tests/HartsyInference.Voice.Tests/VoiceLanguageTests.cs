using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Engine.ToolCalling;
using HartsyInference.Engine.Services;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>The recognition language is a model-set field: it reaches the transcriber, is validated, and a session must
/// match the set it runs on.</summary>
public sealed class VoiceLanguageTests
{
    [Theory]
    [InlineData("en")]
    [InlineData("fr")]
    [InlineData("yue")]
    public void ALanguageCodeOfTwoOrThreeLettersIsAccepted(string code) => new VoiceAgentOptions { Language = code }.Validate();

    [Theory]
    [InlineData("")]
    [InlineData("e")]
    [InlineData("english")]
    [InlineData("e1")]
    [InlineData("en-US")]
    public void AnythingElseIsRefused(string code) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new VoiceAgentOptions { Language = code }.Validate());

    [Fact]
    public async Task TheLanguageReachesTheTranscriber()
    {
        FakeTranscriberLease? transcriber = null;
        using VoiceLeaseSpeech speech = await VoiceLeaseSpeech.OpenAsync(
            _ => Task.FromResult<ISynthesizerLease>(new FakeSynthesizerLease(1)),
            _ => Task.FromResult<ITranscriberLease>(transcriber = new FakeTranscriberLease(1)),
            voice: "ff_siwis", language: "fr", CancellationToken.None);

        speech.Transcribe(new float[16_000]);

        Assert.Equal("fr", transcriber!.LastOptions!.Language);
    }

    [Fact]
    public async Task ASessionInAnotherLanguageThanItsModelSetIsRefused()
    {
        VoiceAgentOptions loaded = VoiceHarness.DefaultOptions();
        using CpuBackend device = new();
        await using VoiceModelSet models = new(loaded, new FakeSpeech(), device, () => new LevelVadModel(), createDenoiser: null);

        ArgumentException error = Assert.Throws<ArgumentException>(() =>
            new VoiceAgentSession(models, new ScriptedTextService(), new ToolRegistry(), loaded with { Language = "fr" }));
        Assert.Contains("fr", error.Message);
    }
}
