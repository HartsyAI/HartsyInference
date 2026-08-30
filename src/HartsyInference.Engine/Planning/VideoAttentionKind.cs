namespace HartsyInference.Engine.Planning;

/// <summary>The attention semantics required by a resolved video checkpoint.</summary>
public enum VideoAttentionKind
{
    /// <summary>Ordinary exact attention.</summary>
    Dense = 0,

    /// <summary>Kijai/Comfy 64-token segment-pure sparse routing semantics.</summary>
    ComfySol64V1 = 1,

    /// <summary>FastVideo 64-token pooled-routing sparse semantics.</summary>
    FastVideoVsa64V1 = 2,
}
