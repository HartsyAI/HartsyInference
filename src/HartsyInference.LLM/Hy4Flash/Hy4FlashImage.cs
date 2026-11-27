namespace HartsyInference.LLM.Hy4Flash;

/// <summary>One image already run through the vision tower, in the order its placeholder run appears in the prompt.</summary>
/// <param name="GridHeight">Patch rows (before the 2x2 merge).</param>
/// <param name="GridWidth">Patch columns.</param>
/// <param name="Features">Projected tokens, <c>[tokens, hidden]</c>, the newline token closing each merged row included (<see cref="Hy4FlashVisionTower.Forward"/>).</param>
public sealed record Hy4FlashImage(int GridHeight, int GridWidth, float[] Features);
