using HartsyInference.ThreeD.Document;

namespace HartsyInference.ThreeD.Operations;

/// <summary>The result of running a list of steps.</summary>
/// <param name="Document">The model after the last step.</param>
/// <param name="Steps">One report per step, in order.</param>
public sealed record MeshOperationRun(MeshDocument Document, IReadOnlyList<MeshOperationReport> Steps);
