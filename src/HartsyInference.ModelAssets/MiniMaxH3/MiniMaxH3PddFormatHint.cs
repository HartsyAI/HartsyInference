namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Hash/profile-derived hint used only when a flattened PDD bank is structurally ambiguous.</summary>
public enum MiniMaxH3PddFormatHint
{
    /// <summary>Infer only formats whose tensor rank or metadata proves their semantics.</summary>
    Auto,

    /// <summary>The official three-dimensional bank contains complete heads.</summary>
    OfficialFullHeads,

    /// <summary>A known converted flattened bank stores one base head followed by offsets.</summary>
    KnownFlattenedOffsets,
}
