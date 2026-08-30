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
        MiniMaxH3TimestepPlan plan = BuildMaskedTimestepRows(
            layout, tVideo, tAudio, visualCondTimestep, audioCondTimestep, null, null);
        return (plan.Timesteps, plan.RowOf);
    }

    /// <summary>Builds scalar conditioning rows and optional per-target-row timesteps for continuous AV denoise
    /// masks. A target mask value <c>m</c> uses <c>t = min(1 - m·sigma, max(tStream, conditionPin))</c>: white
    /// follows the stream exactly, black is pinned as clean conditioning, and intermediate values remain continuous.
    /// All-white inputs collapse to the scalar path so an omitted mask and a white mask execute identically.</summary>
    public static MiniMaxH3TimestepPlan BuildMaskedTimestepRows(
        MiniMaxH3PackedLayout layout, float tVideo, float tAudio, float visualCondTimestep,
        float audioCondTimestep, IReadOnlyList<float>? videoMaskRows, IReadOnlyList<float>? audioMaskRows)
    {
        ArgumentNullException.ThrowIfNull(layout);
        HashSet<MiniMaxH3SegmentKind> kinds = new HashSet<MiniMaxH3SegmentKind>();
        foreach (MiniMaxH3Segment seg in layout.Segments)
        {
            kinds.Add(seg.Kind);
        }
        bool hasVisualCond = kinds.Contains(MiniMaxH3SegmentKind.Cond)
            || kinds.Contains(MiniMaxH3SegmentKind.RefImage);
        bool hasAudioCond = kinds.Contains(MiniMaxH3SegmentKind.CondAudio)
            || kinds.Contains(MiniMaxH3SegmentKind.RefAudio);

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

        int videoTargetRows = TargetRows(layout, MiniMaxH3SegmentKind.Video);
        int audioTargetRows = TargetRows(layout, MiniMaxH3SegmentKind.Audio);
        IReadOnlyList<float>? effectiveVideoMask = ValidateMask(videoMaskRows, videoTargetRows, nameof(videoMaskRows));
        IReadOnlyList<float>? effectiveAudioMask = ValidateMask(audioMaskRows, audioTargetRows, nameof(audioMaskRows));
        float[]? videoRowTimesteps = effectiveVideoMask is null ? null
            : MaskedRowTimesteps(effectiveVideoMask, 1f - tVideo, tCond);
        float[]? audioRowTimesteps = effectiveAudioMask is null ? null
            : MaskedRowTimesteps(effectiveAudioMask, 1f - tAudio, tRefAudio);
        if (videoRowTimesteps is not null)
        {
            distinct.UnionWith(videoRowTimesteps);
        }
        if (audioRowTimesteps is not null)
        {
            distinct.UnionWith(audioRowTimesteps);
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
        if (kinds.Contains(MiniMaxH3SegmentKind.Cond))
        {
            rowOf[MiniMaxH3SegmentKind.Cond] = rowOfTimestep[tCond];
        }
        if (kinds.Contains(MiniMaxH3SegmentKind.RefImage))
        {
            rowOf[MiniMaxH3SegmentKind.RefImage] = rowOfTimestep[tCond];
        }
        if (kinds.Contains(MiniMaxH3SegmentKind.CondAudio))
        {
            rowOf[MiniMaxH3SegmentKind.CondAudio] = rowOfTimestep[tRefAudio];
        }
        if (kinds.Contains(MiniMaxH3SegmentKind.RefAudio))
        {
            rowOf[MiniMaxH3SegmentKind.RefAudio] = rowOfTimestep[tRefAudio];
        }
        return new MiniMaxH3TimestepPlan
        {
            Timesteps = timesteps,
            RowOf = rowOf,
            VideoRowOf = IndexRows(videoRowTimesteps, rowOfTimestep),
            AudioRowOf = IndexRows(audioRowTimesteps, rowOfTimestep),
        };
    }

    private static int TargetRows(MiniMaxH3PackedLayout layout, MiniMaxH3SegmentKind kind)
    {
        foreach (MiniMaxH3Segment segment in layout.Segments)
        {
            if (segment.Kind == kind)
            {
                return segment.Stop - segment.Start;
            }
        }
        return 0;
    }

    private static IReadOnlyList<float>? ValidateMask(
        IReadOnlyList<float>? values, int expectedRows, string parameterName)
    {
        if (values is null)
        {
            return null;
        }
        if (values.Count != expectedRows)
        {
            throw new ArgumentException(
                $"MiniMax-H3 {parameterName} has {values.Count} values, expected {expectedRows} target rows.",
                parameterName);
        }
        bool allWhite = true;
        for (int i = 0; i < values.Count; i++)
        {
            float value = values[i];
            if (!float.IsFinite(value) || value < 0f || value > 1f)
            {
                throw new ArgumentOutOfRangeException(parameterName, value,
                    $"MiniMax-H3 mask values must be finite and in [0,1]; row {i} was {value}.");
            }
            allWhite &= value == 1f;
        }
        return allWhite ? null : values;
    }

    private static float[] MaskedRowTimesteps(IReadOnlyList<float> mask, float sigma, float conditionPin)
    {
        float[] rows = new float[mask.Count];
        for (int i = 0; i < mask.Count; i++)
        {
            rows[i] = Math.Min(1f - mask[i] * sigma, conditionPin);
        }
        return rows;
    }

    private static int[]? IndexRows(float[]? values, IReadOnlyDictionary<float, int> rowOfTimestep)
    {
        if (values is null)
        {
            return null;
        }
        int[] rows = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            rows[i] = rowOfTimestep[values[i]];
        }
        return rows;
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
