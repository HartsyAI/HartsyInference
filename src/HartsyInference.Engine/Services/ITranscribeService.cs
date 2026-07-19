using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed speech-to-text surface, including optional word timestamps and speaker diarization.</summary>
public interface ITranscribeService
{
    /// <summary>Transcribes the audio in <paramref name="request"/>.</summary>
    Task<TranscriptResult> RunAsync(ModelSpec spec, AudioRequest request, CancellationToken cancel = default);
}
