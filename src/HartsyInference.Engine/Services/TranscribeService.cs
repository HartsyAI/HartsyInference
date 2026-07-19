using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Speech-to-text service. Wired by the audio lift (E-AUD-2), including word timestamps and diarization;
/// not yet available.</summary>
public sealed class TranscribeService : ITranscribeService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal TranscribeService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<TranscriptResult> RunAsync(ModelSpec spec, AudioRequest request, CancellationToken cancel = default) =>
        throw new NotSupportedException("Transcription is wired by the audio lift (E-AUD-2); not yet available.");
}
