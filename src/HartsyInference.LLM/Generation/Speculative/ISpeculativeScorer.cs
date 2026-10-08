namespace HartsyInference.LLM.Generation.Speculative;

/// <summary>Scores a draft with the target model in one pass.</summary>
public interface ISpeculativeScorer
{
    /// <summary>Writes the target logits into <paramref name="rows"/>: row j is the next-token logits after <paramref name="context"/> followed by the first j drafted tokens.
    /// <paramref name="rows"/> holds <c>draft.Length + 1</c> rows, each as wide as the vocabulary.</summary>
    void Score(ReadOnlySpan<int> context, ReadOnlySpan<int> draft, float[][] rows);
}
