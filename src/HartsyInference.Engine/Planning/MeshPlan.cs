using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning.Memory;

namespace HartsyInference.Engine.Planning;

/// <summary>The preflight of one mesh request: what will run, what is wrong with it, what is missing, and whether it fits the device.</summary>
public sealed record MeshPlan
{
    /// <summary>The model request this plan resolved.</summary>
    public required ModelSpec Model { get; init; }

    /// <summary>The model's capabilities.</summary>
    public required MeshModelDescriptor Descriptor { get; init; }

    /// <summary>The settings that will run and where each came from; a null value is chosen by the model's checkpoint.</summary>
    public required IReadOnlyDictionary<string, ResolvedParameter> Settings { get; init; }

    /// <summary>Blocking errors and non-blocking warnings.</summary>
    public required IReadOnlyList<MeshPlanIssue> Issues { get; init; }

    /// <summary>The model's file inventory: what is present, what is missing, and what the engine cannot fetch.</summary>
    public required MeshAssetStatus Assets { get; init; }

    /// <summary>Whether the request fits this engine's device; null when the model has no measured peak or the device does not report memory.</summary>
    public MemoryFit? Memory { get; init; }

    /// <summary>Whether preflight found no blocking issue.</summary>
    public bool IsValid => !Issues.Any(issue => issue.Severity == MeshPlanIssueSeverity.Error);
}
