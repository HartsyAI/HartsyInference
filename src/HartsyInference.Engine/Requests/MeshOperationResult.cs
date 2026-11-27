using HartsyInference.ThreeD.Operations;

namespace HartsyInference.Engine.Requests;

/// <summary>The result of editing a model file: the edited model in the requested format, and what each step did.</summary>
public sealed record MeshOperationResult
{
    /// <summary>The edited model as files (see <see cref="MeshConversion.Files"/>; an OBJ comes with its MTL and images), the counts of what was written, and what the output format dropped.</summary>
    public required MeshConversion Output { get; init; }

    /// <summary>One report per step, in order: vertex and triangle counts before and after, and what the step could not keep (a primitive that lost its textures, pieces removed).</summary>
    public required IReadOnlyList<MeshOperationReport> Steps { get; init; }
}
