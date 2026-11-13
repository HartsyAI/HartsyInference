using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Which 3D pipelines the engine holds resident, and freeing them so a host can give the card back after a job (the 3D models do not unload on their own).</summary>
public interface IMeshLifecycleService
{
    /// <summary>The pipelines resident now.</summary>
    IReadOnlyList<MeshLoadedModel> Loaded();

    /// <summary>Frees the pipeline loaded from <paramref name="name"/> (<see cref="MeshLoadedModel.Name"/>, case-insensitive), or every one when it is null, and returns how many were freed. A generation running on one finishes first, and for the models without a run gate of their own (image, scene, rig, paint and part pipelines) the wait covers the running generations of the pipelines being freed, not those of other pipelines, and holds new ones on the freed pipelines off until the free is done; the call returns only then, which can be minutes. The next request loads it again. The matting model every image preparation shares is listed and freed like a pipeline (kind <c>matting</c>), and is freed with the rest when <paramref name="name"/> is null. Two kinds cached from the same folder share a name and are freed together.</summary>
    int Free(string? name = null);
}
