namespace HartsyInference.LLM.Generation.Speculative;

/// <summary>Guesses the tokens that follow a context, so the target model can check several at once.</summary>
public interface IDraftProposer
{
    /// <summary>Up to <paramref name="maxTokens"/> tokens expected to follow <paramref name="context"/>; empty when there is no guess.</summary>
    DraftBlock Propose(ReadOnlySpan<int> context, int maxTokens);
}
