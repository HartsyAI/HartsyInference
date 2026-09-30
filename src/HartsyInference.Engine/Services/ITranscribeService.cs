using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed speech-to-text surface, including optional word timestamps and speaker diarization.</summary>
public interface ITranscribeService
{
    /// <summary>Transcribes the audio in <paramref name="request"/>.</summary>
    Task<TranscriptResult> RunAsync(ModelSpec spec, AudioRequest request, CancellationToken cancel = default);

    /// <summary>Transcribes music into a symbolic score, returning it both with and without chord symbols.
    /// Throws <see cref="NotSupportedException"/> for a model that does not write scores.</summary>
    Task<ScoreTranscriptResult> RunScoreAsync(ModelSpec spec, AudioRequest request, CancellationToken cancel = default);

    /// <summary>Loads the model for <paramref name="spec"/> exactly as <see cref="RunAsync"/> would (same catalog, cache
    /// key and download) and returns a lease that keeps it resident on this engine's backend until disposed; see
    /// <see cref="ITranscriberLease"/> for the contract.</summary>
    /// <remarks>Opening runs the service path, with the engine's audio generation lock, the device gate and the
    /// memory-pressure sweep, so it waits for a generation in flight; never await it while holding
    /// <see cref="DeviceGate"/>.</remarks>
    Task<ITranscriberLease> OpenTranscriberAsync(ModelSpec spec, CancellationToken cancel = default) =>
        throw new NotSupportedException("This transcription-service implementation does not expose transcriber leases.");
}
