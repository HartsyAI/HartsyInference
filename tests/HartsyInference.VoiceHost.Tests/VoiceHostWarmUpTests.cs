using HartsyInference.Audio.Models.Wake;
using HartsyInference.Cpu;
using HartsyInference.Engine.Requests;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Tests.Support;
using HartsyInference.VoiceHost.Tools;
using Xunit;

namespace HartsyInference.VoiceHost.Tests;

/// <summary>The boot warm-up (<see cref="VoiceHostService.WarmModelsAsync"/>) offers the model the same tool
/// definitions a real call would (<see cref="VoiceHostTools.Build"/>, via <see cref="VoiceHostTools.WarmDefinitions"/>),
/// so the tool-call grammar sampler, its stream filter/parser and the chat template's tools branch are hot before the
/// first caller; with none enabled it stays on the cold one-token path.</summary>
public sealed class VoiceHostWarmUpTests
{
    [Fact]
    public async Task TheBootWarmUpOffersTheHostsRealToolDefinitions()
    {
        using CpuBackend device = new();
        ScriptedRounds text = new();
        await using VoiceModelSet models = NewFakeModelSet(device);

        await VoiceHostService.WarmModelsAsync(models, text, VoiceHostTools.Names, CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(10));

        TextRequest request = Assert.Single(text.Requests);
        Assert.NotNull(request.Tools);
        Assert.Equal(VoiceHostTools.Names, request.Tools!.Select(d => d.Name));
        Assert.Equal(8, request.MaxTokens); // VoiceModelSet.WarmToolMaxTokens: real decode steps, not just prefill.
    }

    [Fact]
    public async Task OnlyTheEnabledToolsAreWarmed()
    {
        using CpuBackend device = new();
        ScriptedRounds text = new();
        await using VoiceModelSet models = NewFakeModelSet(device);
        string[] enabled = [VoiceHostTools.GetTime, VoiceHostTools.Hangup];

        await VoiceHostService.WarmModelsAsync(models, text, enabled, CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(10));

        TextRequest request = Assert.Single(text.Requests);
        Assert.Equal([VoiceHostTools.Hangup, VoiceHostTools.GetTime], request.Tools!.Select(d => d.Name));
    }

    [Fact]
    public async Task NoEnabledToolsStaysOnTheColdOneTokenPath()
    {
        using CpuBackend device = new();
        ScriptedRounds text = new();
        await using VoiceModelSet models = NewFakeModelSet(device);

        await VoiceHostService.WarmModelsAsync(models, text, [], CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(10));

        TextRequest request = Assert.Single(text.Requests);
        Assert.Null(request.Tools);
        Assert.Equal(1, request.MaxTokens);
    }

    [Fact]
    public void WarmDefinitionsMirrorsWhatARealCallWouldOffer()
    {
        IReadOnlyList<ToolDefinition> warm = VoiceHostTools.WarmDefinitions(VoiceHostTools.Names);

        Assert.Equal(VoiceHostTools.Names, warm.Select(d => d.Name));
        foreach (ToolDefinition definition in warm)
        {
            Assert.False(string.IsNullOrWhiteSpace(definition.Description));
            Assert.False(string.IsNullOrWhiteSpace(definition.JsonSchema));
        }
    }

    /// <summary>Borrows <paramref name="device"/> (the caller disposes it, after the model set, which is how
    /// <see cref="VoiceModelSet"/> itself documents the parameter).</summary>
    private static VoiceModelSet NewFakeModelSet(CpuBackend device)
    {
        // Denoise defaults to true (int8 RNNoise); this fake model set never reaches a denoiser, so turn it off.
        VoiceAgentOptions options = new() { AudioDevice = "cpu", LlmDevice = "cpu", Denoise = false };
        return new VoiceModelSet(options, new FakeVoiceSpeech(), device,
            static () => throw new InvalidOperationException("The warm-up never touches the VAD."), createDenoiser: null);
    }

    /// <summary>Speech without weights: synthesis returns a fixed buffer, recognition returns "". Enough for
    /// <see cref="VoiceModelSet.WarmAsync"/>, which never inspects either result.</summary>
    private sealed class FakeVoiceSpeech : IVoiceSpeech
    {
        public int SynthesisSampleRate => 16_000;

        public string Transcribe(float[] audio) => "";

        public float[] Synthesize(string text, CancellationToken cancel) => new float[160];

        public void Reopen()
        {
        }

        public void Dispose()
        {
        }
    }
}
