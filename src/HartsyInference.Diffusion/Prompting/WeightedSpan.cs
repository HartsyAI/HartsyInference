namespace HartsyInference.Diffusion.Prompting;

/// <summary>A contiguous run of prompt text plus the emphasis weight that applies to every token it produces.</summary>
public readonly record struct WeightedSpan(string Text, float Weight);
