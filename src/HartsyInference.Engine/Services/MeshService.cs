using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>3D-mesh service. Wired alongside the recipe lift; not yet available.</summary>
public sealed class MeshService : IMeshService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal MeshService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<MeshResult> GenerateAsync(ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default) =>
        throw new NotSupportedException("3D mesh generation is not yet wired on the typed service surface.");
}
