using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Diarization spans for models that emit no timestamps: bounded in count, and tiling the whole clip.</summary>
public sealed class SpeakerSpanTests
{
    [Fact]
    public void BuildSpans_ShortClip_KeepsTheStandardSweep()
    {
        List<SttSegment> spans = SpeakerDiarizer.BuildSpans(null, 60.0);

        Assert.Equal(0.0, spans[0].Start, precision: 9);
        Assert.Equal(0.75, spans[1].Start, precision: 9);
        Assert.Equal(1.5, spans[0].End - spans[0].Start, precision: 9);
    }

    [Fact]
    public void BuildSpans_TwentyMinuteClip_StaysBoundedAndCoversTheClip()
    {
        const double total = 1200.0;

        List<SttSegment> spans = SpeakerDiarizer.BuildSpans(null, total);

        Assert.True(spans.Count <= 512);
        Assert.Equal(0.0, spans[0].Start, precision: 9);
        for (int i = 1; i < spans.Count; i++)
        {
            Assert.True(spans[i].Start <= spans[i - 1].End, "spans must not leave gaps");
        }
        Assert.True(spans[^1].End >= total - 3.0, "the sweep must reach the end of the clip");
    }
}
