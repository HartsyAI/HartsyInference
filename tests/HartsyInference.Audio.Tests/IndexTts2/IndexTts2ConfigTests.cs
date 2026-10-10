using HartsyInference.Audio.Models.IndexTts2;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Pins the two version presets against the values in the real <c>config.yaml</c> files: 2.0 and 2.5 are
/// the same stack except the text vocabulary, the semantic codec's resample scale and the version tag itself.</summary>
public sealed class IndexTts2ConfigTests
{
    /// <summary>The reference's <c>GPT2InferenceModel</c> looks the mel position up as <c>attention_mask.shape[1] - mel_len</c>,
    /// so the start token sits at 0 and the first generated code at 2 — position 1 is never used.</summary>
    [Fact]
    public void MelPositionAtStep_SkipsPositionOne_LikeTheReference()
    {
        Assert.Equal(0, IndexTts2T2sDecoder.MelPositionAtStep(0));
        Assert.Equal(2, IndexTts2T2sDecoder.MelPositionAtStep(1));
        Assert.Equal(3, IndexTts2T2sDecoder.MelPositionAtStep(2));
        Assert.Equal(101, IndexTts2T2sDecoder.MelPositionAtStep(100));
    }
}
