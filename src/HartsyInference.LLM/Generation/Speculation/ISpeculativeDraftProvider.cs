namespace HartsyInference.LLM.Generation.Speculation;

/// <summary>Proposes draft tokens that are expected to follow the current context.
/// The target model verifies every proposal, so a provider only affects speed, never the output.</summary>
public interface ISpeculativeDraftProvider
{
    /// <summary>A stable name, unique within a <see cref="SpeculationSelector"/>, used for measurements and logs.</summary>
    string Name { get; }

    /// <summary>Up to <paramref name="maxDraftLen"/> tokens expected to follow <paramref name="promptIds"/> followed by <paramref name="generated"/>.
    /// Empty when there is no guess.</summary>
    /// <param name="promptIds">The prompt token ids.</param>
    /// <param name="generated">The tokens generated so far, after the prompt.</param>
    /// <param name="maxDraftLen">The most tokens the caller can verify this round.</param>
    int[] Propose(int[] promptIds, IReadOnlyList<int> generated, int maxDraftLen);
}
