namespace HartsyInference.LLM.Generation.Speculation;

/// <summary>What one speculative round measured, reported to <see cref="SpeculationSelector.Record"/>.
/// The caller measures it; the selector never reads a clock.</summary>
/// <param name="Proposed">Draft tokens the provider proposed this round; 0 when it had no guess.</param>
/// <param name="Accepted">Draft tokens the target kept, from the start of the draft.</param>
/// <param name="TokensEmitted">Tokens the round added to the output, including the correction or bonus token.</param>
/// <param name="ElapsedMilliseconds">Wall time for the whole round (drafting and verification), measured by the caller.
/// Zero adds no throughput sample.</param>
public readonly record struct SpeculationRound(int Proposed, int Accepted, int TokensEmitted, double ElapsedMilliseconds);
