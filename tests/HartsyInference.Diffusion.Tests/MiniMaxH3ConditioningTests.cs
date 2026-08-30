using Xunit;
using HartsyInference.Core.Exceptions;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Gates for the conditioned pack's per-step modulation rows and its conditioning-row accounting, ported from
/// ComfyUI PR #15224's <c>_forward</c>. Both are checkable without weights, and both fail silently in generation if
/// wrong: a bad row index mis-modulates whole segments, and a bad row count shifts every packed row.</summary>
public class MiniMaxH3ConditioningTests
{
    private const int TextLen = 7, LatentT = 5, LatentH = 8, LatentW = 12, AudioT = 10;
    private const float VisAug = MiniMaxH3Schedule.VisualCondTimestep, AudAug = MiniMaxH3Schedule.AudioCondTimestep;

    private static MiniMaxH3PackedLayout T2va() => new MiniMaxH3PackedLayout(TextLen, LatentT, LatentH, LatentW, AudioT);

    private static MiniMaxH3PackedLayout Fl2va() => new MiniMaxH3PackedLayout(TextLen, LatentT, LatentH, LatentW, AudioT,
        keyframes: [new MiniMaxH3Keyframe { ResolvedFrameIndex = 0 }], frameCount: 73);

    /// <summary>An image ref, a standalone audio ref, and a video+audio ref — the mix that exposes per-kind
    /// interleaving, since the video block's audio rows pack before its image rows.</summary>
    private static MiniMaxH3PackedLayout Ref2va() => new MiniMaxH3PackedLayout(TextLen, LatentT, LatentH, LatentW, AudioT,
        refs:
        [
            new MiniMaxH3RefBlock { Kind = "image", LatentH = 4, LatentW = 6 },
            new MiniMaxH3RefBlock { Kind = "audio", RefAudioT = 3 },
            new MiniMaxH3RefBlock { Kind = "video_audio", LatentT = 2, LatentH = 4, LatentW = 6, RefAudioT = 4 },
        ]);

    [Fact]
    public void T2vaKeepsTwoRowsWithVideoFirst()
    {
        (float[] t, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(T2va(), tVideo: 0.5f, tAudio: 0.8f, VisAug, AudAug);
        Assert.Equal([0.5f, 0.8f], t);
        Assert.Equal(0, rowOf[MiniMaxH3SegmentKind.Video]);
        Assert.Equal(0, rowOf[MiniMaxH3SegmentKind.Text]);
        Assert.Equal(1, rowOf[MiniMaxH3SegmentKind.Audio]);
        // Absent kinds get no entry at all, so a stale map cannot silently resolve to a wrong row.
        Assert.False(rowOf.ContainsKey(MiniMaxH3SegmentKind.Cond));
        Assert.False(rowOf.ContainsKey(MiniMaxH3SegmentKind.RefAudio));
    }

    [Fact]
    public void EqualStreamTimestepsCollapseToOneRow()
    {
        // Step 0 runs at sigma 1.0, where both schedules map to t=0 — the reference dedups rather than
        // carrying a duplicate row.
        (float[] t, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(T2va(), tVideo: 0f, tAudio: 0f, VisAug, AudAug);
        Assert.Equal([0f], t);
        Assert.Equal(0, rowOf[MiniMaxH3SegmentKind.Video]);
        Assert.Equal(0, rowOf[MiniMaxH3SegmentKind.Audio]);
    }

    [Fact]
    public void KeyframeConditioningPinsItsOwnRowNearOne()
    {
        (float[] t, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(Fl2va(), tVideo: 0.5f, tAudio: 0.8f, VisAug, AudAug);
        Assert.Equal([0.5f, 0.8f, VisAug], t);
        Assert.Equal(2, rowOf[MiniMaxH3SegmentKind.Cond]);
        Assert.Equal(0, rowOf[MiniMaxH3SegmentKind.Video]);
        Assert.False(rowOf.ContainsKey(MiniMaxH3SegmentKind.RefImage));
        Assert.False(rowOf.ContainsKey(MiniMaxH3SegmentKind.RefAudio));
    }

    [Fact]
    public void ReferenceBlocksAddBothConditioningRows()
    {
        (float[] t, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(Ref2va(), tVideo: 0.5f, tAudio: 0.8f, VisAug, AudAug);
        Assert.Equal([0.5f, 0.8f, VisAug, AudAug], t);
        Assert.Equal(2, rowOf[MiniMaxH3SegmentKind.RefImage]);
        Assert.Equal(3, rowOf[MiniMaxH3SegmentKind.RefAudio]);
    }

    [Fact]
    public void PairedGuideAudioUsesThePinnedAudioTimestep()
    {
        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(TextLen, LatentT, LatentH, LatentW, AudioT,
            keyframes:
            [
                new MiniMaxH3Keyframe
                {
                    ResolvedFrameIndex = 2,
                    VideoLatentFrames = 0,
                    AudioLatentFrames = 3,
                },
            ], frameCount: 17);
        (float[] timesteps, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(layout, tVideo: 0.5f, tAudio: 0.8f, VisAug, AudAug);
        Assert.Equal([0.5f, 0.8f, AudAug], timesteps);
        Assert.Equal(2, rowOf[MiniMaxH3SegmentKind.CondAudio]);
        Assert.False(rowOf.ContainsKey(MiniMaxH3SegmentKind.RefAudio));
        Assert.Equal((0, 6), MiniMaxH3Conditioning.ConditioningRowCounts(layout));
    }

    [Fact]
    public void ConditioningRowsFoldIntoTheStreamRowOnceDenoisingPassesThem()
    {
        // Late in the schedule t_video overtakes the 0.999 pin, so cond stops being distinct.
        (float[] t, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(Fl2va(), tVideo: 0.9995f, tAudio: 0.9998f, VisAug, AudAug);
        Assert.Equal([0.9995f, 0.9998f], t);
        Assert.Equal(rowOf[MiniMaxH3SegmentKind.Video], rowOf[MiniMaxH3SegmentKind.Cond]);
    }

    [Fact]
    public void RowOrderFollowsTheTimestepsNotTheStreams()
    {
        // A video shift below the audio one reverses which stream is cleaner; the rows must follow.
        (float[] t, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(T2va(), tVideo: 0.8f, tAudio: 0.5f, VisAug, AudAug);
        Assert.Equal([0.5f, 0.8f], t);
        Assert.Equal(0, rowOf[MiniMaxH3SegmentKind.Audio]);
        Assert.Equal(1, rowOf[MiniMaxH3SegmentKind.Video]);
    }

    [Fact]
    public void ContinuousMasksProducePerTargetRowTimestepsAndClampBlackRowsToTheConditionPin()
    {
        MiniMaxH3PackedLayout layout = T2va();
        int videoRows = LatentT * (LatentH / 2) * (LatentW / 2);
        int audioRows = AudioT * 2;
        float[] videoMask = Enumerable.Repeat(1f, videoRows).ToArray();
        float[] audioMask = Enumerable.Repeat(1f, audioRows).ToArray();
        videoMask[0] = 0f;
        videoMask[1] = 0.5f;
        audioMask[0] = 0f;
        audioMask[1] = 0.5f;

        MiniMaxH3TimestepPlan plan = MiniMaxH3Conditioning.BuildMaskedTimestepRows(
            layout, tVideo: 0.5f, tAudio: 0.8f, VisAug, AudAug, videoMask, audioMask);

        Assert.Equal([0.5f, 0.75f, 0.8f, 0.9f, VisAug, AudAug], plan.Timesteps);
        Assert.NotNull(plan.VideoRowOf);
        Assert.NotNull(plan.AudioRowOf);
        Assert.Equal(4, plan.VideoRowOf![0]);
        Assert.Equal(1, plan.VideoRowOf[1]);
        Assert.Equal(0, plan.VideoRowOf[2]);
        Assert.Equal(5, plan.AudioRowOf![0]);
        Assert.Equal(3, plan.AudioRowOf[1]);
        Assert.Equal(2, plan.AudioRowOf[2]);
    }

    [Fact]
    public void AllWhiteMasksCollapseExactlyToTheScalarTimestepPath()
    {
        MiniMaxH3PackedLayout layout = T2va();
        float[] videoMask = Enumerable.Repeat(1f, LatentT * (LatentH / 2) * (LatentW / 2)).ToArray();
        float[] audioMask = Enumerable.Repeat(1f, AudioT * 2).ToArray();

        MiniMaxH3TimestepPlan plan = MiniMaxH3Conditioning.BuildMaskedTimestepRows(
            layout, tVideo: 0.5f, tAudio: 0.8f, VisAug, AudAug, videoMask, audioMask);

        Assert.Equal([0.5f, 0.8f], plan.Timesteps);
        Assert.Null(plan.VideoRowOf);
        Assert.Null(plan.AudioRowOf);
    }

    [Fact]
    public void MaskRowCountAndRangeAreValidatedBeforeEmbedding()
    {
        MiniMaxH3PackedLayout layout = T2va();
        Assert.Throws<ArgumentException>(() => MiniMaxH3Conditioning.BuildMaskedTimestepRows(
            layout, 0.5f, 0.8f, VisAug, AudAug, [1f], null));

        float[] audioMask = Enumerable.Repeat(1f, AudioT * 2).ToArray();
        audioMask[3] = -0.01f;
        Assert.Throws<ArgumentOutOfRangeException>(() => MiniMaxH3Conditioning.BuildMaskedTimestepRows(
            layout, 0.5f, 0.8f, VisAug, AudAug, null, audioMask));
    }

    [Fact]
    public void T2vaHasNoConditioningRows()
    {
        Assert.Equal((0, 0), MiniMaxH3Conditioning.ConditioningRowCounts(T2va()));
    }

    [Fact]
    public void KeyframeContributesOneFrameOfVideoRows()
    {
        (int video, int audio) = MiniMaxH3Conditioning.ConditioningRowCounts(Fl2va());
        Assert.Equal((LatentH / 2) * (LatentW / 2), video);
        Assert.Equal(0, audio);
    }

    /// <summary>The counts must match a straight walk of the segment table, which is the order the packed rows are
    /// assembled in — not a per-kind grouping, since a video_audio block interleaves its two segment kinds.</summary>
    [Fact]
    public void ReferenceCountsMatchASegmentTableWalk()
    {
        MiniMaxH3PackedLayout l = Ref2va();
        int walkedVideo = 0, walkedAudio = 0;
        foreach (MiniMaxH3Segment s in l.Segments)
        {
            switch (s.Kind)
            {
                case MiniMaxH3SegmentKind.Cond or MiniMaxH3SegmentKind.RefImage: walkedVideo += s.Length; break;
                case MiniMaxH3SegmentKind.RefAudio: walkedAudio += s.Length; break;
            }
        }
        Assert.Equal((walkedVideo, walkedAudio), MiniMaxH3Conditioning.ConditioningRowCounts(l));

        int refImageRows = (4 / 2) * (6 / 2);
        Assert.Equal(refImageRows + 2 * refImageRows, walkedVideo);
        Assert.Equal((3 + 4) * 2, walkedAudio);
    }

    /// <summary>The video_audio block's audio segment packs before its image segment; both precede the targets.</summary>
    [Fact]
    public void ReferenceSegmentsInterleaveByKindAndPrecedeTheTargets()
    {
        Assert.Equal(
            [
                MiniMaxH3SegmentKind.Text,
                MiniMaxH3SegmentKind.RefImage,
                MiniMaxH3SegmentKind.RefAudio,
                MiniMaxH3SegmentKind.RefAudio,
                MiniMaxH3SegmentKind.RefImage,
                MiniMaxH3SegmentKind.Audio,
                MiniMaxH3SegmentKind.Video,
            ],
            Ref2va().Segments.Select(s => s.Kind));
    }

    [Fact]
    public void ConditioningRowsAlwaysPrecedeTheDenoiseTarget()
    {
        foreach (MiniMaxH3PackedLayout l in new[] { T2va(), Fl2va(), Ref2va() })
        {
            (int video, int audio) = MiniMaxH3Conditioning.ConditioningRowCounts(l);
            Assert.All(l.ImageUpdate.Take(video), u => Assert.False(u));
            Assert.All(l.ImageUpdate.Skip(video), u => Assert.True(u));
            Assert.All(l.AudioUpdate.Take(audio), u => Assert.False(u));
            Assert.All(l.AudioUpdate.Skip(audio), u => Assert.True(u));
        }
    }

    [Fact]
    public void AConditioningRowAfterATargetIsRejected()
    {
        MiniMaxH3PackedLayout l = Fl2va();
        l.ImageUpdate[^1] = false;
        HartsyInferenceException ex =
            Assert.Throws<HartsyInferenceException>(() => MiniMaxH3Conditioning.ConditioningRowCounts(l));
        Assert.Contains("conditioning row", ex.Message);
    }
}
