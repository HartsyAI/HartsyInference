using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Vision service (embed/detect/segment). Wired by the vision detect/segment lift (E-IMG-6); not yet available.</summary>
public sealed class VisionService : IVisionService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal VisionService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<VisionResult> RunAsync(ModelSpec spec, VisionRequest request, CancellationToken cancel = default) =>
        throw new NotSupportedException("Vision detect/segment is wired by E-IMG-6; not yet available.");
}
