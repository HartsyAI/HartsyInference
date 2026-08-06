using HartsyInference.Core.Exceptions;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Per-step modulation bookkeeping for a conditioned MiniMax-H3 pack. Keyframe (fl2va) and reference
/// (ref2va) rows carry already-clean latents, so they modulate at a timestep pinned near 1 rather than at the
/// stream's own — giving the pack up to four distinct timesteps where plain t2va has two.</summary>
public static class MiniMaxH3Conditioning
{
    /// <summary>The distinct timesteps present in the pack, ascending, plus each segment kind's row into them.
    /// Rows are derived fresh per step: the stream timesteps sweep the schedule while the conditioning ones stay
    /// pinned, so their relative order — and therefore every row index — changes as denoising proceeds. Kinds absent
    /// from <paramref name="layout"/> get no entry, so a stale mapping cannot silently select a wrong row.</summary>
    public static (float[] Timesteps, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> RowOf) BuildTimestepRows(
        MiniMaxH3PackedLayout layout, float tVideo, float tAudio, float visualCondTimestep, float audioCondTimestep)
    {
        ArgumentNullException.ThrowIfNull(layout);
        bool hasVisualCond = false, hasAudioCond = false;
        foreach (MiniMaxH3Segment seg in layout.Segments)
        {
            hasVisualCond |= seg.Kind is MiniMaxH3SegmentKind.Cond or MiniMaxH3SegmentKind.RefImage;
            hasAudioCond |= seg.Kind is MiniMaxH3SegmentKind.RefAudio;
        }

        float tCond = Math.Max(tVideo, visualCondTimestep);
        float tRefAudio = Math.Max(tAudio, audioCondTimestep);
        SortedSet<float> distinct = new SortedSet<float> { tVideo, tAudio };
        if (hasVisualCond)
        {
            distinct.Add(tCond);
        }
        if (hasAudioCond)
        {
            distinct.Add(tRefAudio);
        }

        float[] timesteps = new float[distinct.Count];
        distinct.CopyTo(timesteps);
        Dictionary<float, int> rowOfTimestep = new Dictionary<float, int>(timesteps.Length);
        for (int i = 0; i < timesteps.Length; i++)
        {
            rowOfTimestep[timesteps[i]] = i;
        }

        Dictionary<MiniMaxH3SegmentKind, int> rowOf = new Dictionary<MiniMaxH3SegmentKind, int>
        {
            [MiniMaxH3SegmentKind.Text] = rowOfTimestep[tVideo],
            [MiniMaxH3SegmentKind.Video] = rowOfTimestep[tVideo],
            [MiniMaxH3SegmentKind.Audio] = rowOfTimestep[tAudio],
        };
        if (hasVisualCond)
        {
            rowOf[MiniMaxH3SegmentKind.Cond] = rowOfTimestep[tCond];
            rowOf[MiniMaxH3SegmentKind.RefImage] = rowOfTimestep[tCond];
        }
        if (hasAudioCond)
        {
            rowOf[MiniMaxH3SegmentKind.RefAudio] = rowOfTimestep[tRefAudio];
        }
        return (timesteps, rowOf);
    }

    /// <summary>Conditioning row counts for the video and audio streams, and the assertion that earns the packed-row
    /// assembly the right to concatenate instead of scatter: every conditioning row must precede its stream's denoise
    /// target. Reference blocks interleave per kind — a <c>video_audio</c> block emits its RefAudio segment before its
    /// RefImage one — so the two streams are counted independently over their own update masks, never by segment
    /// grouping.</summary>
    public static (int Video, int Audio) ConditioningRowCounts(MiniMaxH3PackedLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        return (LeadingConditioningRows(layout.ImageUpdate, "video"),
            LeadingConditioningRows(layout.AudioUpdate, "audio"));
    }

    private static int LeadingConditioningRows(bool[] update, string stream)
    {
        int leading = 0;
        while (leading < update.Length && !update[leading])
        {
            leading++;
        }
        for (int i = leading; i < update.Length; i++)
        {
            if (!update[i])
            {
                throw new HartsyInferenceException(
                    $"MiniMax-H3 {stream} conditioning row {i} of {update.Length} follows a denoise target; the packed "
                    + "assembly requires every conditioning row to precede its stream's target segment.");
            }
        }
        return leading;
    }
}
