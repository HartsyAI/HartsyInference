using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Mesh preflight without loading weights: what each 3D model accepts, and whether one request would run.</summary>
public interface IMeshPlanningService
{
    /// <summary>Every 3D model with a wired pipeline.</summary>
    IReadOnlyList<MeshModelDescriptor> DescribeAll();

    /// <summary>One model's capabilities.</summary>
    /// <exception cref="NotSupportedException"><paramref name="modelId"/> has no wired 3D pipeline.</exception>
    MeshModelDescriptor Describe(string modelId);

    /// <summary>Checks <paramref name="request"/> against the model's rules, the installed files and the device, with the same validation generation runs; nothing is loaded.</summary>
    /// <exception cref="NotSupportedException">The spec names a model with no wired 3D pipeline.</exception>
    Task<MeshPlan> PlanAsync(ModelSpec spec, MeshRequest request, CancellationToken cancel = default);
}
