using HartsyInference.ThreeD.Document;

namespace HartsyInference.ThreeD.Operations;

/// <summary>What an operation produced.</summary>
/// <param name="Document">The edited model (the input is not changed).</param>
/// <param name="Notes">What the operation could not keep or did not do, in words a host can show; empty when nothing was lost.</param>
public sealed record MeshOperationOutcome(MeshDocument Document, IReadOnlyList<string> Notes);
