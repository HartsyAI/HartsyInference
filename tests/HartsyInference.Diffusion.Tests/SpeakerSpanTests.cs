using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Diarization spans: bounded in count whether the model emitted timestamps or not, and covering the whole clip.</summary>
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

    [Fact]
    public void BuildSpans_ManyTimestampSegments_AreJoinedToTheCapWithoutLosingCoverage()
    {
        List<SttSegment> segments = Enumerable.Range(0, 2000)
            .Select(i => new SttSegment { Text = $"s{i}", Start = i * 3.0, End = i * 3.0 + 2.5 })
            .ToList();

        List<SttSegment> spans = SpeakerDiarizer.BuildSpans(segments, 6000.0);

        Assert.True(spans.Count <= 512);
        Assert.Equal(0.0, spans[0].Start, precision: 9);
        Assert.Equal(1999 * 3.0 + 2.5, spans[^1].End, precision: 9);
        Assert.Contains("s0", spans[0].Text);
        Assert.Contains("s1999", spans[^1].Text);
        for (int i = 1; i < spans.Count; i++)
        {
            Assert.True(spans[i].Start >= spans[i - 1].End, "joined spans must stay in order");
        }
    }

    [Fact]
    public void BuildSpans_JoinedTimestampSegments_NeverStretchAcrossALongGap()
    {
        const double total = 8000.0;
        List<SttSegment> segments = [];
        for (int i = 0; i < 600; i++)
        {
            segments.Add(new SttSegment { Text = "early", Start = i * 3.0, End = i * 3.0 + 2.5 });
        }
        for (int i = 0; i < 600; i++)
        {
            segments.Add(new SttSegment { Text = "late", Start = 6000.0 + i * 3.0, End = 6000.0 + i * 3.0 + 2.5 });
        }

        List<SttSegment> spans = SpeakerDiarizer.BuildSpans(segments, total);

        Assert.True(spans.Count <= 512);
        Assert.All(spans, span => Assert.True(span.End - span.Start <= 4.0 * total / 512, "a joined span must not cross the silent stretch"));
    }

    [Fact]
    public void BuildSpans_FewTimestampSegments_AreKeptAsTheyAre()
    {
        List<SttSegment> segments = [new SttSegment { Text = "a", Start = 0.0, End = 2.0 }, new SttSegment { Text = "b", Start = 2.0, End = 5.0 }];

        List<SttSegment> spans = SpeakerDiarizer.BuildSpans(segments, 5.0);

        Assert.Equal(2, spans.Count);
        Assert.Equal("b", spans[1].Text);
    }
}
