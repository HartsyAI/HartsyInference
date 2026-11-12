using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Which 3D pipelines the engine holds resident, and freeing them so a host can give the card back after a job (the 3D models do not unload on their own).</summary>
public interface IMeshLifecycleService
{
    /// <summary>The pipelines resident now.</summary>
    IReadOnlyList<MeshLoadedModel> Loaded();

    /// <summary>Frees the pipeline loaded from <paramref name="name"/> (<see cref="MeshLoadedModel.Name"/>, case-insensitive), or every one when it is null, and returns how many were freed. A generation running on one finishes first; the next request loads it again.</summary>
    int Free(string? name = null);
}
