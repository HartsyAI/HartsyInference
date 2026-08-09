namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>One contiguous run of rows in the packed sequence.</summary>
public readonly record struct MiniMaxH3Segment(int Start, int Stop, MiniMaxH3SegmentKind Kind)
{
    public int Length => Stop - Start;
}
