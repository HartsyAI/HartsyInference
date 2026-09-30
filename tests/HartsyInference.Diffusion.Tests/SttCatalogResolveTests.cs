using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Whisper variant → repo resolution in the STT catalog, pinning the <c>.en</c> suffix rule: it selects the
/// English-only release for the four sizes that have one and is ignored for the large family, which has none.</summary>
public sealed class SttCatalogResolveTests
{
    [Theory]
    [InlineData("tiny.en", "openai/whisper-tiny.en")]
    [InlineData("base.en", "openai/whisper-base.en")]
    [InlineData("small.en", "openai/whisper-small.en")]
    [InlineData("medium.en", "openai/whisper-medium.en")]
    [InlineData("Small.EN", "openai/whisper-small.en")]
    [InlineData("small", "openai/whisper-small")]
    [InlineData("", "openai/whisper-base")]
    [InlineData("large.en", "openai/whisper-large-v3")]
    [InlineData("large-v2.en", "openai/whisper-large-v2")]
    [InlineData("turbo.en", "openai/whisper-large-v3-turbo")]
    [InlineData("distil-small.en", "distil-whisper/distil-small.en")]
    [InlineData("someorg/whisper-fork.en", "someorg/whisper-fork.en")]
    public void ResolveWhisperRepo_HonorsTheEnglishOnlySuffix(string variant, string expectedRepo)
        => Assert.Equal(expectedRepo, SttCatalog.Whisper.ResolveRepo(variant));
}
