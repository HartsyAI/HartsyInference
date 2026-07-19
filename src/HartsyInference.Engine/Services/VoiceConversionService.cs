using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Voice-conversion service (RVC / OpenVoice). Wired by the audio lift (E-AUD-2); not yet available.</summary>
public sealed class VoiceConversionService : IVoiceConversionService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal VoiceConversionService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<AudioResult> ConvertAsync(ModelSpec spec, VoiceConversionRequest request, CancellationToken cancel = default) =>
        throw new NotSupportedException("Voice conversion is wired by the audio lift (E-AUD-2); not yet available.");
}
