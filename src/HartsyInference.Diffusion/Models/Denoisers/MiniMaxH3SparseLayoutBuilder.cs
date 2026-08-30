using HartsyInference.Core.Backends;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Builds the immutable generation-scoped 64-token routing layout used by MiniMax-H3 VSA.</summary>
internal static class MiniMaxH3SparseLayoutBuilder
{
    private const int BlockEdge = 4;
    private const int BlockTokens = BlockEdge * BlockEdge * BlockEdge;

    /// <summary>Splits every non-target segment into segment-pure 64-row blocks, then tiles the target video into
    /// <c>4x4x4</c> cubes. Prefix queries remain exact while target-video queries use checkpoint-bound routing.</summary>
    internal static VideoSparseAttentionPlan Build(MiniMaxH3PackedLayout layout,
        VideoSparseAttentionProfileKind profile)
    {
        ArgumentNullException.ThrowIfNull(layout);
        MiniMaxH3Segment video = layout.Segments.Last(segment => segment.Kind == MiniMaxH3SegmentKind.Video);
        if (video.Stop != layout.SequenceLength)
        {
            throw new ArgumentException("MiniMax-H3 VSA requires the target video to be the final packed segment.",
                nameof(layout));
        }

        List<int> offsets = new List<int> { 0 };
        List<int> sources = new List<int>(layout.SequenceLength);
        List<int> classes = new List<int>();
        foreach (MiniMaxH3Segment segment in layout.Segments)
        {
            if (segment.Kind == MiniMaxH3SegmentKind.Video)
            {
                break;
            }
            for (int start = segment.Start; start < segment.Stop; start += BlockTokens)
            {
                int stop = Math.Min(start + BlockTokens, segment.Stop);
                for (int row = start; row < stop; row++)
                {
                    sources.Add(row);
                }
                classes.Add((int)segment.Kind);
                offsets.Add(sources.Count);
            }
        }
        int prefixBlocks = classes.Count;

        (int _, int latentT, int latentH, int latentW, int _) = layout.Signature;
        int rowsH = latentH / 2;
        int rowsW = latentW / 2;
        if (latentT <= 0 || rowsH <= 0 || rowsW <= 0 || video.Length != latentT * rowsH * rowsW)
        {
            throw new ArgumentException("MiniMax-H3 VSA target-video geometry does not match the packed layout.",
                nameof(layout));
        }
        for (int t0 = 0; t0 < latentT; t0 += BlockEdge)
        {
            int tStop = Math.Min(t0 + BlockEdge, latentT);
            for (int y0 = 0; y0 < rowsH; y0 += BlockEdge)
            {
                int yStop = Math.Min(y0 + BlockEdge, rowsH);
                for (int x0 = 0; x0 < rowsW; x0 += BlockEdge)
                {
                    int xStop = Math.Min(x0 + BlockEdge, rowsW);
                    for (int t = t0; t < tStop; t++)
                    {
                        for (int y = y0; y < yStop; y++)
                        {
                            int row = video.Start + (t * rowsH + y) * rowsW + x0;
                            for (int x = x0; x < xStop; x++)
                            {
                                sources.Add(row + x - x0);
                            }
                        }
                    }
                    classes.Add((int)MiniMaxH3SegmentKind.Video);
                    offsets.Add(sources.Count);
                }
            }
        }

        VideoSparseAttentionPlan plan = new VideoSparseAttentionPlan
        {
            Profile = profile,
            SequenceLength = layout.SequenceLength,
            BlockOffsets = offsets.ToArray(),
            SourceIndices = sources.ToArray(),
            SegmentClasses = classes.ToArray(),
            PrefixSinkBlocks = prefixBlocks,
            KeepFraction = 0.10f,
        };
        plan.Validate();
        return plan;
    }
}
