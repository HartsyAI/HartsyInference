using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Interactive-world service. Wired alongside the recipe lift; not yet available.</summary>
public sealed class WorldService : IWorldService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal WorldService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public IWorldSession Open(ModelSpec spec, WorldRequest request) =>
        throw new NotSupportedException("Interactive world sessions are not yet wired on the typed service surface.");
}
