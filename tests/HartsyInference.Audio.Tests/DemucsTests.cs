using HartsyInference.Audio.Models.Demucs;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>HTDemucs config sanity. The component-level forwards are covered by the real-weight parity tests
/// (<see cref="DemucsSpecParityTests"/>, <see cref="DemucsStageDebugTests"/>, <see cref="DemucsFullParityTests"/>).</summary>
public sealed class DemucsTests
{
    [Fact]
    public void Config_FourStereoStems()
    {
        HtDemucsConfig c = HtDemucsConfig.Htdemucs;
        Assert.Equal(4, c.NumSources);
        Assert.Equal(new[] { "drums", "bass", "other", "vocals" }, c.Sources.ToArray());
        Assert.Equal(4, c.SpecInChannels);       // 2 (re/im) × 2 stereo
        Assert.Equal(512, c.BottomChannels);
        Assert.Equal(2048, c.TransformerFfn);
    }
}
