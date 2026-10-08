namespace HartsyInference.LLM.Generation.Speculative;

/// <summary>Tokens a proposer guesses to follow a context.</summary>
/// <param name="Tokens">The drafted token ids, in order.</param>
/// <param name="Probs">For each drafted position, the full distribution the proposer drew that token from, or null when the proposer is deterministic (a point mass on each drafted token).</param>
public sealed record DraftBlock(int[] Tokens, float[][]? Probs);
