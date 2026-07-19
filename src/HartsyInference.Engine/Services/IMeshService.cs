using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed 3D-mesh surface: text- or image-to-3D.</summary>
public interface IMeshService
{
    /// <summary>Generates a mesh for <paramref name="request"/>.</summary>
    Task<MeshResult> GenerateAsync(ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default);
}
