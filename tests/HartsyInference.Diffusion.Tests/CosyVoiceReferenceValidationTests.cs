using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A voice reference CosyVoice 2 cannot clone from is rejected through its descriptor, before the model is loaded.</summary>
public sealed class CosyVoiceReferenceValidationTests
{
    private const int Rate = SpeechService.ReferenceSampleRate;

    [Fact]
    public void ALongReference_IsRejectedBeforeAnyLoad()
    {
        TtsModelDescriptor descriptor = TtsCatalog.Resolve("cosyvoice");
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => SpeechService.ValidateReference(descriptor, new float[21 * Rate]));
        Assert.Contains("the limit is 20 s", error.Message);
    }

    [Theory]
    [InlineData(5)]
    [InlineData(15)]
    [InlineData(20)]
    public void AReferenceWithinTheLimit_Passes(int seconds) =>
        SpeechService.ValidateReference(TtsCatalog.Resolve("cosyvoice"), new float[seconds * Rate]);

    [Fact]
    public void NoReference_Passes()
    {
        TtsModelDescriptor descriptor = TtsCatalog.Resolve("cosyvoice");
        SpeechService.ValidateReference(descriptor, null);
        SpeechService.ValidateReference(descriptor, []);
    }

    [Fact]
    public void AFamilyWithoutAValidator_AcceptsAnyReference() =>
        SpeechService.ValidateReference(TtsCatalog.Resolve("auk"), new float[60 * Rate]);
}
