using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Audio-effects service (stem separation / enhancement). Wired by the audio lift (E-AUD-2); not yet available.</summary>
public sealed class FxService : IFxService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal FxService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<StemsResult> SeparateAsync(ModelSpec spec, FxSeparateRequest request, CancellationToken cancel = default) =>
        throw new NotSupportedException("Stem separation is wired by the audio lift (E-AUD-2); not yet available.");

    /// <inheritdoc/>
    public Task<AudioResult> EnhanceAsync(ModelSpec spec, FxEnhanceRequest request, CancellationToken cancel = default) =>
        throw new NotSupportedException("Audio enhancement is wired by the audio lift (E-AUD-2); not yet available.");
}
