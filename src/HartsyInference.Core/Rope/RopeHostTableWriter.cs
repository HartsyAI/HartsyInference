namespace HartsyInference.Core.Rope;

/// <summary>Fills compact per-pair host tables for a batch of token positions. Writers run under the cache gate and must
/// not call back into the cache.</summary>
public delegate void RopeHostTableWriter(ReadOnlySpan<float> positions, int tokens, Span<float> cos, Span<float> sin);
