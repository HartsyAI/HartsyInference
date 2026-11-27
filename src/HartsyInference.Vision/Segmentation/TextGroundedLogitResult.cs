namespace HartsyInference.Vision.Segmentation;

/// <summary>Queries of one text prompt above a score floor, highest score first, with mask logits of <see cref="LogitSide"/> squared.</summary>
public sealed record TextGroundedLogitResult(string Prompt, int LogitSide, IReadOnlyList<TextGroundedLogit> Instances);
