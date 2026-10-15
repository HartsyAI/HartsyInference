using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A voice reference CosyVoice 2 cannot clone from is rejected through its descriptor before the job is queued, and
/// a family without a validator never decodes the reference for it.</summary>
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
        SpeechService.ValidateReference(descriptor, (float[]?)null);
        SpeechService.ValidateReference(descriptor, []);
        SpeechService.ValidateReference(descriptor, (AudioClip?)null);
    }

    [Fact]
    public void AnEncodedLongClip_IsRejectedBeforeAdmission()
    {
        AudioClip clip = new() { Data = AudioClipCodec.EncodeWav(new float[22 * Rate], null, Rate), Format = "wav" };
        Assert.Throws<InvalidOperationException>(() => SpeechService.ValidateReference(TtsCatalog.Resolve("cosyvoice"), clip));
        SpeechService.ValidateReference(TtsCatalog.Resolve("cosyvoice"),
            new AudioClip { Data = AudioClipCodec.EncodeWav(new float[8 * Rate], null, Rate), Format = "wav" });
    }

    [Fact]
    public void AClipTheHeaderCannotSize_IsLeftToTheJob()
    {
        byte[] wav = AudioClipCodec.EncodeWav(new float[22 * Rate], null, Rate);
        BitConverter.GetBytes(uint.MaxValue).CopyTo(wav, 40);
        SpeechService.ValidateReference(TtsCatalog.Resolve("cosyvoice"), new AudioClip { Data = wav, Format = "wav" });
        SpeechService.ValidateReference(TtsCatalog.Resolve("cosyvoice"), new AudioClip { Data = [1, 2, 3] });
    }

    [Fact]
    public void AFamilyWithoutAValidator_DoesNotReadTheReference() =>
        SpeechService.ValidateReference(TtsCatalog.Resolve("auk"), new AudioClip { Data = [1, 2, 3] });
}
