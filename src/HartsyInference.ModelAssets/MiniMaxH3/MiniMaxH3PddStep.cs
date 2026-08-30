namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>One legal coarse PDD interval and its modality-specific fine-head weights.</summary>
public readonly record struct MiniMaxH3PddStep(int Index, int FineStart, int FineCount, double Sigma,
    double SigmaNext, IReadOnlyList<float> VideoWeights, IReadOnlyList<float> AudioWeights);
