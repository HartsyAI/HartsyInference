using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>A tokenized Iris-Z prompt.</summary>
/// <param name="Tokens">Token ids, right-padded when <paramref name="ValidLength"/> is set.</param>
/// <param name="ValidLength">Real tokens before the padding, which the encoder then masks as transformers does; 0 for
/// an unpadded prompt.</param>
/// <param name="Weights">Per-token emphasis weights, or null for an unweighted prompt.</param>
public sealed record IrisZPrompt(int[] Tokens, int ValidLength = 0, WeightedTokenSequence? Weights = null);
