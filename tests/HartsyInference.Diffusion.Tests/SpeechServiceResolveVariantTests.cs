using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit tests for <see cref="SpeechService.ResolveVariant"/> -- the fix for an unparameterized Piper
/// request 404ing on <c>rhasspy/piper-voices/piper.onnx</c>.
///
/// <para>Root cause: <see cref="AudioModelSelector.Parse"/> falls back to the bare catalog token (e.g.
/// <c>"piper"</c>) for <see cref="AudioModelSelector.Variant"/> whenever the request token has no <c>':'</c> --
/// intentional for a descriptor that treats a bare id as its own repo/model identifier. <see cref="SpeechService.ResolveTarget"/>
/// used to pass that bare-token <c>Variant</c> straight through as the load variant for ANY descriptor,
/// including a <see cref="TtsModelDescriptor.VoiceSelectsWeights"/> one (Piper ships one <c>.onnx</c> per
/// voice), which then had no way to tell "the catalog id leaked through as a voice" from "the caller really
/// asked for a voice named 'piper'". <see cref="SpeechService.ResolveVariant"/> is the single place that now
/// makes that distinction, by comparing <see cref="AudioModelSelector.Variant"/> against
/// <see cref="AudioModelSelector.Id"/> rather than hardcoding Piper's name -- so the fix covers any other
/// <c>VoiceSelectsWeights</c> model with the same shape, not just Piper.</para></summary>
public sealed class SpeechServiceResolveVariantTests
{
    private const string PiperId = "piper";

    private static TtsModelDescriptor VoiceSelectsWeightsDescriptor() => new()
    {
        ResolveRepo = _ => "rhasspy/piper-voices",
        LoadAsync = (_, _, _) => throw new NotSupportedException("not exercised by this test"),
        VoiceSelectsWeights = true,
    };

    private static TtsModelDescriptor PlainDescriptor() => new()
    {
        ResolveRepo = _ => "some/repo",
        LoadAsync = (_, _, _) => throw new NotSupportedException("not exercised by this test"),
        // VoiceSelectsWeights left false -- the default, matching every non-Piper TTS model today.
    };

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("default")]
    [InlineData("DEFAULT")]
    public void BareCatalogIdWithNoNamedVoice_ResolvesToTheEmptyDefaultSentinel(string? voice)
    {
        // The exact shape AudioModelSelector.Parse produces for an unparameterized request: no ':' in the
        // token, so Variant falls back to the bare token, which equals Id.
        AudioModelSelector selector = new(PiperId, PiperId, LocalPath: null);

        string variant = SpeechService.ResolveVariant(selector, VoiceSelectsWeightsDescriptor(), voice);

        Assert.Equal("", variant);
    }

    [Theory]
    [InlineData("en_US-amy-medium")]
    [InlineData("en_GB-vctk-medium")]
    public void ExplicitVoiceParameter_IsUnchanged_EvenWithABareCatalogIdToken(string voice)
    {
        // A real voice supplied through the separate `voice` argument always wins, regardless of what the
        // model token itself resolved to.
        AudioModelSelector selector = new(PiperId, PiperId, LocalPath: null);

        string variant = SpeechService.ResolveVariant(selector, VoiceSelectsWeightsDescriptor(), voice);

        Assert.Equal(voice, variant);
    }

    [Fact]
    public void ExplicitColonVariant_ThatIsARealVoice_IsUnchanged()
    {
        // "piper:en_US-amy-medium" -- ModelSelector.Parse gives Variant="en_US-amy-medium", distinct from
        // Id="piper", so this is NOT the bare-token-fallback shape and must pass through untouched.
        AudioModelSelector selector = new(PiperId, "en_US-amy-medium", LocalPath: null);

        string variant = SpeechService.ResolveVariant(selector, VoiceSelectsWeightsDescriptor(), voice: null);

        Assert.Equal("en_US-amy-medium", variant);
    }

    [Fact]
    public void ExplicitColonVariant_ThatIsTheLiteralWordDefault_IsUnchanged()
    {
        // "piper:default" -- a contrived but real request: Variant="default", Id="piper", still distinct, so
        // the bare-token detection (Variant == Id) must not eat it either. It flows through to the
        // descriptor's own LoadAsync exactly as before this fix, which already treats the literal "default"
        // as "use my own default" -- this test only pins that ResolveVariant itself leaves it alone.
        AudioModelSelector selector = new(PiperId, "default", LocalPath: null);

        string variant = SpeechService.ResolveVariant(selector, VoiceSelectsWeightsDescriptor(), voice: null);

        Assert.Equal("default", variant);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("some_voice")]
    public void NonVoiceSelectsWeightsDescriptor_AlwaysReturnsTheSelectorVariantUnchanged(string? voice)
    {
        // Every other TTS model (Kokoro, VibeVoice, Dia, ...) is VoiceSelectsWeights=false: the whole
        // bare-token question never arises for them, and the `voice` argument is -- as before this fix --
        // simply irrelevant to what gets loaded.
        AudioModelSelector bareTokenSelector = new("whisper_like", "whisper_like", LocalPath: null);

        Assert.Equal("whisper_like", SpeechService.ResolveVariant(bareTokenSelector, PlainDescriptor(), voice));
    }

    [Fact]
    public void RealPiperDescriptor_BareCatalogId_ResolvesToTheEmptyDefaultSentinel()
    {
        // Glues the fix to the real catalog entries (TtsCatalog.Resolve("piper") is the actual
        // PiperModel.Descriptor, not a stand-in) so this doesn't only prove the fake shape works.
        AudioModelSelector selector = AudioModelSelector.Parse(new ModelSpec { Requested = PiperId, Modality = Modality.Speech });
        TtsModelDescriptor piper = TtsCatalog.Resolve(selector.Id);

        Assert.True(piper.VoiceSelectsWeights);
        Assert.Equal("", SpeechService.ResolveVariant(selector, piper, voice: null));
    }

    [Fact]
    public void RealKokoroDescriptor_BareCatalogId_IsUnaffected()
    {
        AudioModelSelector selector = AudioModelSelector.Parse(new ModelSpec { Requested = "kokoro", Modality = Modality.Speech });
        TtsModelDescriptor kokoro = TtsCatalog.Resolve(selector.Id);

        Assert.False(kokoro.VoiceSelectsWeights);
        Assert.Equal("kokoro", SpeechService.ResolveVariant(selector, kokoro, voice: null));
    }

    [Fact]
    public void AudioModelSelectorParse_BareToken_VariantEqualsId()
    {
        // Pins the actual root-cause precondition ResolveVariant's bare-token detection relies on: for a
        // request with no ':', AudioModelSelector.Parse's Variant really does fall back to equal Id.
        AudioModelSelector selector = AudioModelSelector.Parse(new ModelSpec { Requested = "piper", Modality = Modality.Speech });

        Assert.Equal("piper", selector.Id);
        Assert.Equal("piper", selector.Variant);
    }

    [Fact]
    public void AudioModelSelectorParse_WithColonVariant_VariantDiffersFromId()
    {
        AudioModelSelector selector = AudioModelSelector.Parse(new ModelSpec { Requested = "piper:en_US-amy-medium", Modality = Modality.Speech });

        Assert.Equal("piper", selector.Id);
        Assert.Equal("en_US-amy-medium", selector.Variant);
    }
}
