using HartsyInference.Audio.Models.FishAudio;
using Xunit;

namespace HartsyInference.Audio.Tests;

public sealed class FishAudioS2ConfigTests
{
    [Fact]
    public void S2ProKeepsDecoderAndCodecVocabulariesDistinct()
    {
        FishAudioS2Config config = FishAudioS2Config.S2Pro;

        Assert.Equal(10, config.NumCodebooks);
        Assert.Equal(4_096, config.CodebookSize);
        Assert.Equal(4_096, config.Fast.VocabSize);
        Assert.Equal(4_096, config.SemanticCodebookSize);
        Assert.Equal(1_024, config.ResidualCodebookSize);
    }
}
