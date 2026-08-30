namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Semantic encoding used by the rows of a PDD final-head bank.</summary>
public enum MiniMaxH3PddHeadLayout
{
    /// <summary>Every row is a complete independently trained output head.</summary>
    FullHeads,

    /// <summary>Row zero is the complete base head and later rows are additive offsets from it.</summary>
    BasePlusOffsets,
}
