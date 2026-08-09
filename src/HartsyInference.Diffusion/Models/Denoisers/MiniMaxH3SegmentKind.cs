namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>What a packed-sequence segment carries. The packed order is text, then any conditioning/reference blocks,
/// then the target audio and target video — which are always the last two segments.</summary>
public enum MiniMaxH3SegmentKind
{
    Text,
    Cond,
    RefImage,
    RefAudio,
    Audio,
    Video,
}
