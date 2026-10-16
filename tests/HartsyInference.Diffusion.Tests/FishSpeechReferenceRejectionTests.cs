using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fish-Speech takes no voice reference yet, so a reference clip is refused before the job is queued. The refusal does not
/// depend on the clip's container: a clip the WAV header cannot measure is refused too, which the duration hook alone missed.</summary>
public sealed class FishSpeechReferenceRejectionTests
{
    // An MP3 ID3 tag, not a RIFF header, so the WAV duration reader gives up on it.
    private static readonly byte[] NotAWav = [0x49, 0x44, 0x33, 0x04, 0x00, 0x00, 0x00, 0x00];

    [Theory]
    [InlineData("mp3")]
    [InlineData("flac")]
    [InlineData("wav")]
    public void ClipIsRefusedWhateverItsContainer(string format) =>
        Assert.Throws<NotSupportedException>(() =>
            SpeechService.ValidateReference(FishSpeechModel.Descriptor, new AudioClip { Data = NotAWav, Format = format }));

    [Fact]
    public void DecodedReferenceIsRefusedOnTheStreamingPath() =>
        Assert.Throws<NotSupportedException>(() => SpeechService.ValidateReference(FishSpeechModel.Descriptor, new float[480]));

    [Fact]
    public void UndecodableReferenceIsRefusedBeforeItIsDecoded() =>
        Assert.Throws<NotSupportedException>(() => SpeechService.RejectUnsupportedReferences(FishSpeechModel.Descriptor,
            new SpeechRequest { Text = "hi", Reference = new AudioClip { Data = NotAWav, Format = "mp3" } }));

    [Fact]
    public void ReferencesListIsRefused() =>
        Assert.Throws<NotSupportedException>(() => SpeechService.RejectUnsupportedReferences(FishSpeechModel.Descriptor,
            new SpeechRequest { Text = "hi", References = [new SpeechTurn { Text = "hello" }] }));

    [Fact]
    public void RequestWithoutReferencesIsAccepted() =>
        SpeechService.RejectUnsupportedReferences(FishSpeechModel.Descriptor, new SpeechRequest { Text = "hi" });

    [Fact]
    public void NoReferenceIsAccepted()
    {
        SpeechService.ValidateReference(FishSpeechModel.Descriptor, (AudioClip?)null);
        SpeechService.ValidateReference(FishSpeechModel.Descriptor, new AudioClip { Data = [] });
        SpeechService.ValidateReference(FishSpeechModel.Descriptor, (float[]?)null);
        SpeechService.ValidateReference(FishSpeechModel.Descriptor, []);
    }
}
