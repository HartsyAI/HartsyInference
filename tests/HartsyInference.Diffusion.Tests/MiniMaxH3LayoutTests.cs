using Xunit;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Structural gates for the MiniMax-H3 packed sequence, ported from ComfyUI PR #15224's <c>PackedLayout</c>.
/// These are checkable without weights: segment order, row accounting, target masks, and the position grids —
/// the pieces whose ordering errors would decode to plausible-but-wrong output.</summary>
public class MiniMaxH3LayoutTests
{
    private const int TextLen = 7, LatentT = 5, LatentH = 8, LatentW = 12, AudioT = 10;

    private static MiniMaxH3PackedLayout T2va() => new MiniMaxH3PackedLayout(TextLen, LatentT, LatentH, LatentW, AudioT);

    [Fact]
    public void T2vaPacksTextThenAudioThenVideo()
    {
        MiniMaxH3PackedLayout l = T2va();
        Assert.Equal(
            [MiniMaxH3SegmentKind.Text, MiniMaxH3SegmentKind.Audio, MiniMaxH3SegmentKind.Video],
            l.Segments.Select(s => s.Kind));
        // Target audio and target video are always the last two segments.
        Assert.Equal(MiniMaxH3SegmentKind.Audio, l.Segments[^2].Kind);
        Assert.Equal(MiniMaxH3SegmentKind.Video, l.Segments[^1].Kind);
    }

    [Fact]
    public void RowAccountingIsContiguousAndComplete()
    {
        MiniMaxH3PackedLayout l = T2va();
        int frameRows = (LatentH / 2) * (LatentW / 2);
        Assert.Equal(TextLen + AudioT * 2 + LatentT * frameRows, l.SequenceLength);
        Assert.Equal(l.SequenceLength * 3, l.PositionIds.Length);
        int cursor = 0;
        foreach (MiniMaxH3Segment s in l.Segments)
        {
            Assert.Equal(cursor, s.Start);
            cursor = s.Stop;
        }
        Assert.Equal(l.SequenceLength, cursor);
    }

    [Fact]
    public void AllTargetRowsAreMarkedForUpdateWhenThereIsNoConditioning()
    {
        MiniMaxH3PackedLayout l = T2va();
        Assert.All(l.ImageUpdate, u => Assert.True(u));
        Assert.All(l.AudioUpdate, u => Assert.True(u));
        Assert.Equal(AudioT * 2, l.AudioUpdate.Length);
        Assert.Equal(LatentT * (LatentH / 2) * (LatentW / 2), l.ImageUpdate.Length);
    }

    [Fact]
    public void KeyframeConditioningRowsPrecedeTargetsAndAreNotUpdated()
    {
        MiniMaxH3PackedLayout l = new MiniMaxH3PackedLayout(TextLen, LatentT, LatentH, LatentW, AudioT,
            keyframes: [new MiniMaxH3Keyframe { ResolvedFrameIndex = 0 }], frameCount: 17);
        Assert.Equal(MiniMaxH3SegmentKind.Cond, l.Segments[1].Kind);
        int frameRows = (LatentH / 2) * (LatentW / 2);
        // Conditioning rows come first in the video row stream and are not denoise targets.
        Assert.Equal(frameRows + LatentT * frameRows, l.ImageUpdate.Length);
        Assert.All(l.ImageUpdate.Take(frameRows), u => Assert.False(u));
        Assert.All(l.ImageUpdate.Skip(frameRows), u => Assert.True(u));
    }

    [Fact]
    public void OnlyFirstAndLastKeyframeAnchorsAreAccepted()
    {
        Assert.Throws<ArgumentException>(() => new MiniMaxH3PackedLayout(TextLen, LatentT, LatentH, LatentW, AudioT,
            keyframes: [new MiniMaxH3Keyframe { ResolvedFrameIndex = 3 }], frameCount: 17));
    }

    [Fact]
    public void ReferenceBlocksPackAudioBeforeVideoAndAdvanceTheCursor()
    {
        MiniMaxH3PackedLayout l = new MiniMaxH3PackedLayout(TextLen, LatentT, LatentH, LatentW, AudioT,
            refs: [new MiniMaxH3RefBlock { Kind = "video_audio", LatentT = 2, LatentH = 8, LatentW = 8, RefAudioT = 3 }]);
        Assert.Equal(MiniMaxH3SegmentKind.RefAudio, l.Segments[1].Kind);
        Assert.Equal(MiniMaxH3SegmentKind.RefImage, l.Segments[2].Kind);
        Assert.Equal(MiniMaxH3SegmentKind.Audio, l.Segments[^2].Kind);
        Assert.Equal(MiniMaxH3SegmentKind.Video, l.Segments[^1].Kind);
    }

    [Fact]
    public void AudioRowsAreChannelMajorWithWPinnedToGridExtremes()
    {
        MiniMaxH3PackedLayout l = T2va();
        MiniMaxH3Segment audio = l.Segments.Single(s => s.Kind == MiniMaxH3SegmentKind.Audio);
        double At(int r, int c) => l.PositionIds[(audio.Start + r) * 3 + c];
        // Channel-major: rows 0..T-1 are channel 0, rows T..2T-1 channel 1, each advancing t by one latent frame.
        Assert.Equal(At(0, 0) + 1.0, At(1, 0), 12);
        Assert.Equal(At(0, 0), At(AudioT, 0), 12);
        // h stays 0; w pins low for channel 0 and high for channel 1.
        Assert.Equal(0.0, At(0, 1), 12);
        Assert.Equal(0.0, At(AudioT, 1), 12);
        Assert.True(At(AudioT, 2) > At(0, 2));
    }

    [Fact]
    public void VideoTimeGridUsesTheCyclicFrameSpans()
    {
        MiniMaxH3PackedLayout l = T2va();
        MiniMaxH3Segment video = l.Segments.Single(s => s.Kind == MiniMaxH3SegmentKind.Video);
        int frameRows = (LatentH / 2) * (LatentW / 2);
        double TAt(int frame) => l.PositionIds[(video.Start + frame * frameRows) * 3];
        // Spans cycle (1,4,4,4,4) scaled by 5/3, applied as an exclusive cumulative sum.
        Assert.Equal(5.0 / 3.0 * 1, TAt(1) - TAt(0), 12);
        Assert.Equal(5.0 / 3.0 * 4, TAt(2) - TAt(1), 12);
        Assert.Equal(5.0 / 3.0 * 4, TAt(3) - TAt(2), 12);
    }

    [Fact]
    public void FrameGridIsAreaNormalizedAndSpansTheSameExtentOnBothAxes()
    {
        // A square latent must produce identical h and w axes; a wide one must widen w relative to h.
        MiniMaxH3PackedLayout square = new MiniMaxH3PackedLayout(1, 1, 8, 8, 1);
        MiniMaxH3Segment v = square.Segments.Single(s => s.Kind == MiniMaxH3SegmentKind.Video);
        double h0 = square.PositionIds[v.Start * 3 + 1], w0 = square.PositionIds[v.Start * 3 + 2];
        Assert.Equal(h0, w0, 12);

        MiniMaxH3PackedLayout wide = new MiniMaxH3PackedLayout(1, 1, 8, 32, 1);
        MiniMaxH3Segment vw = wide.Segments.Single(s => s.Kind == MiniMaxH3SegmentKind.Video);
        int cols = 32 / 2;
        double wFirst = wide.PositionIds[vw.Start * 3 + 2];
        double wLast = wide.PositionIds[(vw.Start + cols - 1) * 3 + 2];
        double hFirst = wide.PositionIds[vw.Start * 3 + 1];
        double hLast = wide.PositionIds[(vw.Start + (8 / 2 - 1) * cols) * 3 + 1];
        Assert.True(wLast - wFirst > hLast - hFirst);
    }
}
