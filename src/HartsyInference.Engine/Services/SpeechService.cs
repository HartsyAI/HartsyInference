using System.Globalization;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Text-to-speech service. Phase 1 drives the existing speech handler over the flat request fields; the full
/// descriptor/runner lift (VibeVoice/Kokoro/Bark/… and zero-shot cloning) lands with E-AUD-2.</summary>
public sealed class SpeechService : ISpeechService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal SpeechService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<AudioResult> SynthesizeAsync(ModelSpec spec, SpeechRequest request, CancellationToken cancel = default)
    {
        ParamState parameters = new ParamState(Modality.Speech);
        if (!string.IsNullOrEmpty(request.Voice))
            parameters.Put("voice", request.Voice);
        if (request.Speed is double speed)
            parameters.Put("speed", speed.ToString(CultureInfo.InvariantCulture));

        SinkBridge sink = new SinkBridge(null, null);
        return Task.Run(
            () =>
            {
                GeneratedArtifact artifact = _engine.RunHandler(spec, request.Text, parameters, sink, cancel);
                return AudioResults.From(artifact);
            },
            cancel);
    }
}
