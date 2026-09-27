namespace HartsyInference.Engine.Variants;

/// <summary>What a variant is resolved from: the checkpoint on disk and the caller's hints, strongest hint first.</summary>
/// <param name="CheckpointPath">Checkpoint file or bundle directory; null resolves from hints alone.</param>
/// <param name="Hints">Caller hints in priority order (explicit variant, selector suffix, requested family id); blanks are ignored.</param>
public sealed record ModelVariantEvidence(string? CheckpointPath, IReadOnlyList<string?> Hints);
