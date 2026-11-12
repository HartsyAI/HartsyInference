using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Planning.Memory;

namespace HartsyInference.API.Endpoints;

/// <summary>Client-safe mesh preflight result: the plan without the server's checkpoint paths.</summary>
public sealed class NativeMeshPlanResponse
{
    /// <summary>The model's capabilities.</summary>
    public required MeshModelDescriptor Model { get; init; }

    /// <summary>The settings that will run and where each came from.</summary>
    public required IReadOnlyDictionary<string, ResolvedParameter> Settings { get; init; }

    /// <summary>Blocking errors and non-blocking warnings.</summary>
    public required IReadOnlyList<MeshPlanIssue> Issues { get; init; }

    /// <summary>Catalog files not at their catalog path, relative to the models root.</summary>
    public required IReadOnlyList<MeshMissingArtifact> MissingArtifacts { get; init; }

    /// <summary>Whether the request fits the device; <c>Unknown</c> with the reason when the model has no recorded peak.</summary>
    public required MemoryFit Memory { get; init; }

    /// <summary>Whether preflight found no blocking issue.</summary>
    public required bool IsValid { get; init; }

    /// <summary>Projects the engine plan onto the HTTP contract.</summary>
    internal static NativeMeshPlanResponse Create(MeshPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        return new NativeMeshPlanResponse
        {
            Model = plan.Descriptor,
            Settings = plan.Settings,
            Issues = plan.Issues,
            MissingArtifacts = plan.MissingArtifacts,
            Memory = plan.Memory,
            IsValid = plan.IsValid,
        };
    }
}
